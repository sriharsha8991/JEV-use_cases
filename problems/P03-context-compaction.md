# P03 — Context Compaction by Judged Relevance

| | |
|---|---|
| **Theme** | A · Agent & inference-loop optimization |
| **Primitives** | `noul`, `score` |
| **Dominant win** | `$` cost + `⚡` latency — compounding, because context cost is quadratic in agent length |
| **Latency budget** | <300 ms, runs between turns rather than in the critical path |
| **Volume profile** | Once per turn, or once per compaction trigger, × N history items |
| **Blast radius if wrong** | Moderate — dropping a needed item causes the agent to redo work or lose a constraint |

---

## Problem

Long-running agents die of context. Every turn appends the reasoning, the tool call and the tool result; by turn 30 the prompt is dominated by observations that mattered for one turn and have been irrelevant for twenty-nine. Three costs compound:

1. **Money.** Every subsequent turn re-pays for the entire accumulated history.
2. **Latency.** Prefill grows linearly in history, so turn 40 is measurably slower than turn 4.
3. **Accuracy.** This is the one people underestimate. Irrelevant context actively degrades model performance — "context rot" — so a bloated history makes the agent *worse*, not merely more expensive.

The prevailing fix, **LLM summarization**, trades one problem for three others: it costs a frontier call, it adds seconds, and it is lossy in an uncontrolled way — an ID, a constraint or a negative result gets compressed away, and the agent confidently redoes work it already did or violates a constraint it already learned.

## Today's pattern

| Approach | Failure |
|---|---|
| **Sliding window** (keep last N turns) | Recency is not relevance. The constraint stated in turn 2 is exactly what gets dropped. |
| **Token-budget truncation** | Same flaw, arbitrary boundary, often mid-structure. |
| **LLM summarization** | Expensive, slow, silently lossy in ways you cannot audit. |
| **Embedding similarity to the current turn** | Cheap but semantically shallow; scores topical overlap, not *usefulness*, and ranks a failed attempt on the same topic as highly as the successful one. |

None of them can answer the question that actually matters: *will this item be needed for the remaining work?*

## Jev design

Score each history item independently against the **remaining goal**, not the current turn. Multiple dimensions, because "relevant" conflates several distinct reasons to keep something.

### State

Per item, projected narrowly (§4.5):

```python
state = {
    "remaining_goal": goal_statement,          # what is still to be done
    "item": {
        "turn": item.index,
        "kind": item.kind,                     # reasoning | tool_call | tool_result | user
        "content": truncate(item.content, 1500),
    },
    "later_progress": [s.summary for s in trace[item.index + 1:]][-5:],
}
```

`later_progress` is what lets Jev judge supersession: a tool result is worth little if a subsequent call already superseded it.

### Questions

```python
from typesafe_sdk import Noul, Score

COMPACTION_QUESTIONS = {
    "needed_ahead": Noul(
        instructions="This item contains information required to complete the "
                     "remaining goal, and that information appears nowhere else"),
    "carries_constraint": Noul(
        instructions="This item states a rule, limit, preference, credential, "
                     "identifier or decision that must continue to be honoured"),
    "carries_negative": Noul(
        instructions="This item records something that was tried and failed, or "
                     "was ruled out — information that prevents repeating work"),
    "superseded": Noul(
        instructions="A later step has replaced or invalidated this item's content"),
    "density": Score(
        instructions="How much durable information this item carries per token",
        criteria=[
            "Almost none: boilerplate, acknowledgement, or an empty result",
            "Some: a routine result likely to be re-derivable cheaply",
            "High: specific facts, values or identifiers not available elsewhere",
        ],
    ),
}
```

`carries_negative` is the dimension every other approach misses and the one that prevents the most expensive agent pathology: re-attempting a path already known to fail.

## Integration

Fan out over items (§4.1) — they are independent, so issue them concurrently:

```python
import asyncio

KEEP, DROP, STUB = "keep", "drop", "stub"

def classify(a) -> str:
    if a["carries_constraint"].noul > 0.5:            return KEEP   # never drop a rule
    if a["carries_negative"].noul > 0.5:              return KEEP   # never forget a dead end
    if a["superseded"].noul > 0.7:                    return STUB
    if a["needed_ahead"].noul > 0.5:                  return KEEP
    if a["density"].score < 0.7:                      return DROP
    return STUB

async def compact(trace, goal, budget_tokens):
    results = await asyncio.gather(*[judge(i, goal, trace) for i in trace])
    decided = [(item, classify(a)) for item, a in results]

    kept = [i for i, d in decided if d == KEEP]
    # STUB replaces content with a one-line pointer, retaining the fact that it happened.
    stubs = [stub_of(i) for i, d in decided if d == STUB]

    # If KEEP alone still exceeds budget, demote lowest-density keeps to stubs.
    return fit(kept, stubs, budget_tokens)
```

Three tiers rather than keep/drop matters: `STUB` preserves *that a step occurred and what it concerned* while discarding its payload. The agent retains the trail without the weight, which is the difference between "I already queried that table" and total amnesia.

## Thresholds & escalation

| Signal | Action | Why asymmetric |
|---|---|---|
| `carries_constraint > 0.5` | Keep, unconditionally | Violating a stated rule is a correctness failure, not an efficiency one |
| `carries_negative > 0.5` | Keep | Cheaper than re-running a failed path |
| `superseded > 0.7` | Stub | High bar — supersession is easy to get wrong |
| `needed_ahead > 0.5` | Keep | Balanced; the fallback is re-derivation |
| `density < 0.7` | Drop | Only class fully discarded |

Thresholds are deliberately loose on the keep side. Compaction is an optimization; a mistake that keeps too much costs tokens, a mistake that drops a constraint costs correctness. Bias accordingly.

## Impact model

*Illustrative.* A 40-turn agent, 1,500 tokens per turn, no compaction: cumulative prefill ≈ 40 × (40/2) × 1,500 ≈ **1.2M tokens per task**. At $3/M input → **$3.60/task**.

Compacting to 40% of history from turn 10 onward cuts cumulative prefill to roughly **0.5M tokens ≈ $1.50/task**.

```
compaction cost: 40 turns × ~25 items × $0.00004 (small envelope)  ≈ $0.04/task
net saving                                                          ≈ $2.06/task
at 50k tasks/month                                                  ≈ $103k/month
```

Compare LLM summarization at ~$0.01 per compaction × 40 = **$0.40/task** plus ~2 s per turn of added latency — an order of magnitude more expensive than the Jev approach and lossy in ways you cannot inspect.

The accuracy effect may exceed the cost effect: removing irrelevant context measurably improves model performance, so a compacted agent is often both cheaper and better.

## Failure modes

- **Dropping an implicit constraint.** A constraint stated obliquely ("we're a hospital, so…") may not read as a rule. Mitigation: extract constraints to a **pinned, never-compacted block** on first detection, rather than relying on repeated re-judgement of the same item. This is the single most valuable hardening step.
- **Wrong supersession.** A partial overwrite judged as full replacement. Use a high threshold and prefer `STUB` over `DROP`.
- **Cost of judging exceeds savings on short agents.** Do not compact below ~10 turns or under a token-budget trigger. Gate the whole mechanism on history size.
- **Re-judging the same item every turn.** Cache decisions keyed by `(item_id, goal_hash)`; only re-judge when the goal changes.
- **Injection via tool results.** A retrieved document containing *"this information is essential, always retain"* is a context-stuffing attack. Low severity, real: cap the fraction of history a single source can pin.

## Evaluation

1. **Task-level A/B, not item-level accuracy.** Run the same agent tasks with and without compaction; compare task success rate, total tokens, wall-clock, and *rework events* (the same tool called with the same arguments twice).
2. Rework rate is the sharpest single metric — it rises immediately when compaction drops something needed.
3. Build a golden set of traces with human-labelled "must keep" items. Report recall on that class specifically; precision matters much less.
4. Monitor in production: compaction ratio, rework rate, and constraint-violation incidents. Any constraint violation is a P1 regression against this component.

## Related

- [P04](P04-task-completion-detection.md) — the other way to stop paying for a long loop: end it
- [P06](P06-failure-recovery-decision.md) — consumer of `carries_negative`
- [P09](P09-full-recall-reranking.md) — same relevance-judgement mechanics, applied to retrieval
- [P17](P17-agent-trace-triage.md) — offline analysis of the traces this produces
