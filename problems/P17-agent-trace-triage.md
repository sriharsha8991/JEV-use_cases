# P17 — Agent Trace Failure Triage at Scale

| | |
|---|---|
| **Theme** | C · Evaluation & observability |
| **Primitives** | `choice`, `score`, `noul` |
| **Dominant win** | `📈` coverage — classify every trace instead of reading a handful — plus `$` |
| **Latency budget** | Batch/streaming, minutes not milliseconds |
| **Volume profile** | Every agent run, or every failed run |
| **Blast radius if wrong** | Low — output is engineering triage, not a user-facing action |

---

## Problem

Agent observability produces traces, and traces are unreadable at volume. A single failed run is 40 steps, 60k tokens of reasoning and tool output. An engineer can read maybe twenty traces a day. At 5,000 failures a week, coverage is 0.5%, and the sample is biased toward whatever was escalated loudly.

The consequence is that **teams do not know why their agents fail.** They have anecdotes: "it seems to loop on multi-step lookups", "it gets confused when the tool returns an empty list". They do not have a distribution, so they cannot prioritise. Engineering effort goes to the failure that the loudest customer reported, which is rarely the most common one.

Standard observability tools give you latency, token counts, error rates and step counts. None of those answer the actual question, which is semantic: *what went wrong, and at which step did it start going wrong?*

## Today's pattern

| Approach | Problem |
|---|---|
| **Manual trace reading** | <1% coverage, biased sample, does not scale with volume |
| **Error-rate dashboards** | Tell you failures happened, never why |
| **Clustering on embeddings** | Produces clusters nobody can label; topical rather than causal |
| **LLM trace summarization** | Accurate, ~$0.30–2.00 per trace at 60k tokens, so it is run on a sample — which recreates the coverage problem |
| **User-reported issues only** | Extreme survivorship bias: silent failures never surface |

## Jev design

Two stages, because trace length exceeds the ~32k state budget and because the interesting question is *where* things went wrong, not just *whether*.

**Stage 1 — step-level.** Judge each step independently against the goal. Cheap, parallel, and locates the first bad step.

**Stage 2 — run-level.** Judge a compact summary built from stage 1 plus the goal and outcome.

### Stage 1 — per-step

```python
state = {
    "goal": trace.goal,
    "step": {"n": s.index, "reasoning": truncate(s.reasoning, 600),
             "action": s.tool, "args": brief(s.args),
             "result": truncate(s.result, 700)},
    "prior_step_summaries": [x.summary for x in trace[:s.index]][-4:],
}
```

```python
from typesafe_sdk import Choice, Noul, Score

STEP_QUESTIONS = {
    "step_quality": Choice(
        instructions="How this step contributed to the goal",
        criteria={
            "productive":   "Advanced the task",
            "neutral":      "Neither advanced nor harmed it",
            "redundant":    "Repeated work already done",
            "misdirected":  "Pursued something not needed for the goal",
            "erroneous":    "Used a tool wrongly, or misread a previous result",
            "recovery":     "Correctly recovered from an earlier problem",
        },
    ),
    "misread_prior": Noul(
        instructions="The reasoning in this step misinterprets the result of an "
                     "earlier step"),
    "wrong_tool": Noul(
        instructions="A different available tool would have been clearly more "
                     "appropriate for what this step was trying to do"),
    "first_divergence": Noul(
        instructions="This is the first step at which the run started going wrong, "
                     "as opposed to a later consequence of an earlier mistake"),
}
```

`first_divergence` is the highest-value question in this document. Debugging effort is wasted on symptoms: the run visibly fell apart at step 31, but the cause was a misread result at step 9. Locating divergence converts a 40-step trace into a one-step bug report.

### Stage 2 — per-run

```python
RUN_QUESTIONS = {
    "root_cause": Choice(
        instructions="The primary reason this run did not succeed",
        criteria={
            "goal_ambiguous":     "The request was too unclear to act on",
            "missing_capability": "No available tool could do what was needed",
            "tool_failure":       "A tool failed or was unavailable",
            "tool_misuse":        "A tool was available but used incorrectly",
            "misread_result":     "A tool result was misinterpreted",
            "lost_context":       "Earlier information needed later was lost or ignored",
            "looped":             "Repeated the same approach without progress",
            "premature_stop":     "Stopped before completing the goal",
            "overreach":          "Did unrequested work instead of, or as well as, the goal",
            "permission_denied":  "Lacked access to something required",
            "succeeded":          "The run actually succeeded",
        },
    ),
    "preventable_by_prompt": Noul(
        instructions="A clearer system prompt or tool description would plausibly "
                     "have prevented this failure"),
    "preventable_by_tooling": Noul(
        instructions="A new or fixed tool would have prevented this failure"),
    "user_impact": Score(
        instructions="The impact on the user",
        criteria=["None: user unaffected or unaware",
                  "Mild: needed to rephrase or retry",
                  "Significant: got a wrong answer or gave up",
                  "Severe: a wrong action was taken on their behalf"],
    ),
}
```

`preventable_by_prompt` versus `preventable_by_tooling` splits the backlog along the line that matters for planning: prompt work versus platform work. That single split is usually what turns this analysis into a roadmap.

## Integration

```python
def triage(trace):
    steps = fan_out([judge_step(s, trace) for s in trace.steps])

    divergence = next((s.index for s, a in steps
                       if a["first_divergence"].noul > 0.6), None)

    run = judge_run(state={
        "goal": trace.goal,
        "outcome": trace.outcome,
        "step_labels": [{"n": s.index, "q": a["step_quality"].choice}
                        for s, a in steps],
        "divergence_step": divergence,
        "divergence_detail": detail_of(trace, divergence),
        "final_output": truncate(trace.final_output, 900),
    }).answers

    return TraceReport(
        root_cause=run["root_cause"].choice,
        confidence=run["root_cause"].confidence,
        divergence_step=divergence,
        fix_class=("prompt" if run["preventable_by_prompt"].noul > 0.6
                   else "tooling" if run["preventable_by_tooling"].noul > 0.6
                   else "design"),
        impact=run["user_impact"].score,
        wasted_steps=sum(1 for _, a in steps
                         if a["step_quality"].choice in ("redundant", "misdirected")),
    )
```

Note that stage 2's state is a **compact derived summary**, not the raw trace. That is what keeps a 60k-token trace inside the 32k budget, and it is a better input anyway: the step labels are exactly the signal the run-level question needs.

## Thresholds & escalation

| Signal | Action |
|---|---|
| `root_cause.confidence < 0.5` | Route to the human queue — this is the sample worth reading |
| `user_impact ≥ 2.5` | Alert regardless of cause; severe impact needs a person |
| `root_cause == "missing_capability"`, rate rising | Product gap; route to roadmap |
| `preventable_by_prompt > 0.6`, clustered by cause | Prompt work item with the traces attached as fixtures |
| `wasted_steps` high across many runs | Cost opportunity → [P04](P04-task-completion-detection.md) |
| Any cause class up >2× week-over-week | Regression alert |

The human queue is now the low-confidence tail rather than a random sample — which is exactly the right place for scarce expert attention.

## Impact model

*Illustrative.* 200k agent runs/month, 15% fail = 30k failures, ~25 steps each.

```
step stage    30k × 25 × $0.00006  = $   45
run stage     30k × $0.00015       = $    4.50
                                     ─────────
                                     $   50/month for 100% coverage

LLM triage    30k × $0.60          = $18,000/month  (so it is run on ~200 traces)
human triage  30k × 20 min         = ~10,000 hours  (does not happen)
```

The output is what matters: a **ranked distribution of root causes with linked example traces and a prompt-versus-tooling split.** That is a prioritised engineering backlog derived from all failures rather than from the twenty someone read. Teams routinely discover that their assumed top failure mode is third or fourth, and that a cause nobody had a name for is the largest.

The wasted-step count also gives a hard number for the [P04](P04-task-completion-detection.md) business case.

## Failure modes

- **`first_divergence` fires on several steps or none.** It is judged per step without global view, so multiple steps may claim it. Take the earliest above threshold; if none fires, fall back to the first `erroneous` or `misread_prior` step.
- **Misattributing cause to the last visible symptom.** Mitigated by feeding divergence into stage 2 rather than letting the run-level question infer it from the outcome.
- **Taxonomy fit.** Derive `root_cause` categories from 100 hand-read traces first. An unfamiliar taxonomy produces a distribution nobody believes.
- **Long steps truncated.** A tool result that is 20k tokens of JSON loses its tail. Summarise structured results in code before judging, and keep the summary deterministic.
- **Succeeded-runs blindness.** Triaging only failures misses degraded successes — right answer, twelve wasted steps. Sample successful runs too; that is where cost savings hide.
- **Injection via trace content.** Tool results containing text aimed at the analyser. Low severity (it corrupts analytics, not a control) but it will skew your distribution if an upstream source does it systematically.

## Evaluation

1. Have engineers hand-triage 150 traces: root cause, divergence step, fix class. Compare against Jev. Report root-cause accuracy and **divergence-step accuracy within ±1 step** — that tolerance is what makes the output actionable.
2. Check that the aggregate distribution matches the hand-labelled distribution. A triage system can be mediocre per-trace and still produce correct aggregates, which for prioritisation is sufficient — but you need to know which regime you are in.
3. Validate the prompt/tooling split by acting on the top cause and measuring whether the failure rate for that class actually falls. This is the only real proof the analysis is causal.
4. Track over time: cause distribution, divergence-step histogram (clusters reveal specific brittle steps), mean wasted steps, and human-queue size.

## Related

- [P06](P06-failure-recovery-decision.md) — the live version of this; `root_cause` classes overlap with its `cause`
- [P04](P04-task-completion-detection.md) — consumes `looped`, `premature_stop`, `overreach` and `wasted_steps`
- [P16](P16-regression-gating-ci.md) — triaged traces become CI fixtures
- [P18](P18-drift-detection.md) — cause distribution shifts are a drift signal
- [P03](P03-context-compaction.md) — `lost_context` is its failure mode
