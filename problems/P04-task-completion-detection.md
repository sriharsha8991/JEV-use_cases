# P04 — Task-Completion & Loop-Termination Detection

| | |
|---|---|
| **Theme** | A · Agent & inference-loop optimization |
| **Primitives** | `noul`, `choice` |
| **Dominant win** | `$` cost — eliminates wasted turns, the least-visible line item in agent spend |
| **Latency budget** | <150 ms, once per turn |
| **Volume profile** | Once per agent turn |
| **Blast radius if wrong** | Moderate — premature stop delivers incomplete work; late stop wastes money |

---

## Problem

Agents are bad at knowing when they are finished. Two failure modes, both expensive:

- **Overrun.** The task is done at turn 8, but the agent keeps going to turn 25 — re-verifying, re-reading files, restating conclusions, "double-checking". Every one of those turns pays full prefill over the accumulated context (see [P03](P03-context-compaction.md)), so the tail turns are the *most* expensive ones in the task.
- **Spin.** The agent is stuck: same tool, similar arguments, no new information, three turns running. It will not stop, because nothing in the loop is watching for a lack of progress. It runs to the max-turn limit and returns a partial result with a confident summary.

The usual guard is a hard turn cap. A cap prevents infinite loops and does nothing about either problem: it truncates good work and fails to notice bad work early.

## Today's pattern

| Approach | Problem |
|---|---|
| **Max-turn cap** | Blunt. Fires too late for spin, too early for genuinely long tasks. |
| **Agent self-declares done** (a `finish` tool) | The model's judgement of its own completeness is unreliable and systematically optimistic; it also *does not fire at all* during spin, because a spinning agent believes it is making progress. |
| **LLM-as-supervisor each turn** | Correct idea, unaffordable: doubles per-turn cost and adds seconds to every turn. |
| **Heuristics on repeated tool calls** | Catches literal repetition, misses semantic repetition (three differently-worded queries for the same fact). |

The right instrument is a cheap **supervisor that runs every turn and is independent of the agent's own optimism**. That is only architecturally possible at ~100 ms and ~$0.0001.

## Jev design

Split the decision into two orthogonal questions: **is the goal satisfied?** and **is progress still being made?** They are independent, and conflating them is why self-declaration fails.

### State

```python
state = {
    "goal": original_request,
    "acceptance_criteria": explicit_criteria or None,   # if the caller gave any
    "deliverables_so_far": [d.summary for d in produced_artifacts],
    "last_turns": [
        {"action": t.tool, "args": brief(t.args), "result": truncate(t.result, 400)}
        for t in trace[-4:]
    ],
    "turn_number": len(trace),
}
```

### Questions

```python
from typesafe_sdk import Choice, Noul

TERMINATION_QUESTIONS = {
    "goal_satisfied": Noul(
        instructions="Everything the user asked for has been produced. Judge only "
                     "against the stated goal, not against what would be ideal"),
    "unaddressed_part": Noul(
        instructions="Some explicit part of the user's request has not been "
                     "addressed at all yet"),
    "progress_last_turns": Noul(
        instructions="The most recent turns produced new information or moved "
                     "the task forward, as opposed to repeating, re-reading, or "
                     "re-verifying work already done"),
    "repeating": Noul(
        instructions="The agent is attempting substantially the same action as "
                     "an earlier turn, even if worded or parameterised differently"),
    "blocked": Noul(
        instructions="The agent cannot proceed without something it does not "
                     "have: a permission, a credential, a missing file, or a "
                     "decision only the user can make"),
    "state": Choice(
        instructions="The overall state of this task",
        criteria={
            "complete":       "The goal is met; further work adds nothing",
            "progressing":    "Incomplete, but the last turns advanced it",
            "stalled":        "Incomplete and the last turns added nothing new",
            "blocked":        "Cannot proceed without external input",
            "overreaching":   "The goal is met and the agent has moved on to "
                              "work that was not requested",
        },
    ),
}
```

`overreaching` is worth its own category. It is the most common and least noticed waste in production agents: the goal was met at turn 8 and the agent is now doing unrequested adjacent work — often helpfully, always unbilled to anyone who asked for it.

## Integration

```python
STOP, CONTINUE, ASK, ABORT = "stop", "continue", "ask", "abort"

def supervise(trace, goal, turn):
    a = client.system_one(state=build_state(trace, goal),
                          questions=TERMINATION_QUESTIONS).answers

    if a["blocked"].noul > 0.6:
        return ASK, "blocked"

    # Completion requires agreement from two independent signals.
    if a["goal_satisfied"].noul > 0.85 and a["unaddressed_part"].noul < 0.20:
        return STOP, "complete"

    if a["state"].choice == "overreaching" and a["state"].confidence > 0.7:
        return STOP, "goal met, scope exceeded"

    # Stall detection, escalating with persistence.
    if a["repeating"].noul > 0.6 or a["progress_last_turns"].noul < 0.3:
        stalls[trace.id] += 1
        if stalls[trace.id] >= 2:
            return (ASK, "stalled") if a["unaddressed_part"].noul > 0.5 \
                   else (ABORT, "stalled with nothing outstanding")
    else:
        stalls[trace.id] = 0

    return CONTINUE, None
```

Two structural points:

- **Completion needs two signals to agree.** `goal_satisfied` high *and* `unaddressed_part` low. A single optimistic signal is exactly the failure mode of agent self-declaration, and asking both questions independently is what makes the check honest.
- **Stall requires persistence.** One unproductive turn is normal; two consecutive are a pattern. A counter in your code, not a question to the model.

## Thresholds & escalation

| Signal | Threshold | Action |
|---|---|---|
| `goal_satisfied` | > 0.85 **and** `unaddressed_part` < 0.20 | Stop and deliver |
| `state == overreaching`, conf > 0.7 | — | Stop; log the scope excursion |
| `blocked` | > 0.6 | Ask the user; do not burn turns |
| Stall counter | ≥ 2 | Ask if work outstanding, else abort |
| Turn cap | absolute | Keep it. This is a supervisor, not a replacement for a hard limit |

The completion bar is deliberately high (0.85). Stopping early delivers incomplete work to a user, which is far more damaging than three extra turns.

## Impact model

*Illustrative.* 50k agent tasks/month, mean 22 turns, of which ~5 are overrun or spin. Turn cost rises with context; assume $0.05 average for tail turns.

```
wasted turns        50k × 5 × $0.05      = $12,500/month
supervisor cost     50k × 22 × $0.00011  = $   121/month
net saving                                 $12,379/month  (~100× return)
```

Plus effects that do not appear in that arithmetic: tasks finish sooner (better UX), stalled tasks surface as questions instead of silent partial results, and the scope-excursion log becomes real data about where your agent wanders.

## Failure modes

- **Optimistic completion.** Jev judges the goal satisfied when a subtle requirement is unmet. Mitigated by the two-signal rule and the high threshold. If the caller supplied explicit acceptance criteria, put them in `state` — completion judgement against explicit criteria is markedly more reliable than against a prose goal.
- **Vague goals make this unanswerable.** "Improve the codebase" has no completion condition. Detect goal vagueness at intake (see [P07](P07-clarify-vs-proceed.md)) and refuse to run an unbounded agent on it.
- **Legitimate repetition read as spin.** Polling a job, retrying with backoff, paginating. Exempt known-idempotent polling tools from `repeating` in code rather than trying to explain the exception in the question.
- **Missing the slow stall.** An agent making *tiny* progress every turn never trips the counter. Add a code-side rule: total turns > 2× the median for this task type triggers review.
- **Supervisor cost on short tasks.** Skip the check for the first 3 turns.

## Evaluation

1. Take 200 completed production traces. Have a human mark the turn at which the task was *actually* done. Run the supervisor in shadow and measure the gap: turns saved (positive) versus turns cut short (negative, weighted much more heavily).
2. Separately evaluate spin detection on traces that hit the turn cap — did the supervisor fire, and how many turns earlier?
3. Track in production: mean turns per task before/after, premature-stop complaints (the metric that matters most), ask-rate from `blocked`, and the distribution of `state` values as a profile of how your agent actually behaves.
4. Watch the `overreaching` rate specifically. A high value is a prompt problem, not a supervisor problem — fix it upstream.

## Related

- [P03](P03-context-compaction.md) — the other lever on long-loop cost
- [P06](P06-failure-recovery-decision.md) — what to do about a stall rather than just detecting it
- [P07](P07-clarify-vs-proceed.md) — where `blocked` routes to
- [P17](P17-agent-trace-triage.md) — offline classification of the traces that ran long
