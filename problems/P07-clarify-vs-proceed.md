# P07 — Clarify vs. Proceed

| | |
|---|---|
| **Theme** | A · Agent & inference-loop optimization |
| **Primitives** | `noul`, `score`, `choice` |
| **Dominant win** | `🛡` reliability + `$` cost — avoids doing the wrong work expensively |
| **Latency budget** | <150 ms, at intake |
| **Volume profile** | Every incoming request |
| **Blast radius if wrong** | Asymmetric — a needless question is mild friction; a wrong assumption on a consequential task is expensive rework |

---

## Problem

Agents guess. Handed *"cancel the subscription"* with three active subscriptions on the account, an agent picks one — usually the most recent — and cancels it. Handed *"update the pricing page"* with no indication of what to change, it invents something plausible.

The inverse failure is just as common and more annoying: agents that ask clarifying questions about things they could trivially resolve, or that ask three questions where one would do. Both failures come from the same gap — **nothing in the pipeline measures whether ambiguity actually matters here.**

The decision has two independent parts, and systems that collapse them get it wrong:

1. Is the request ambiguous?
2. Does the ambiguity *change the outcome*, and how costly is being wrong?

"Summarize this document — briefly or in detail?" is ambiguous and harmless. "Delete the old records — which ones?" is ambiguous and irreversible. Same ambiguity level, opposite correct behaviour.

## Today's pattern

| Approach | Problem |
|---|---|
| **Always proceed with a best guess** | Silent wrong work. Worst on irreversible actions |
| **Always ask on any uncertainty** | Users abandon. Trains them to over-specify, which defeats the point of natural language |
| **Prompt instruction: "ask if unclear"** | The most common approach and the least reliable: entirely at the model's discretion, wildly inconsistent run to run, and not tunable |
| **Required slot-filling forms** | Deterministic and reliable, but it is a form. You built a conversational interface to avoid forms |

## Jev design

Separate **ambiguity** from **consequence**, then combine them in code. The product decision — how much friction to accept for how much risk — belongs in your arithmetic, not inside a model's judgement.

### State

Include what the agent *can already resolve*, so Jev judges residual ambiguity rather than surface ambiguity. This is the highest-leverage part of the design.

```python
state = {
    "request": user_message,
    "recent_turns": history[-3:],
    "resolvable_context": {
        "subscriptions": [{"id": s.id, "plan": s.plan} for s in account.subs],
        "default_env": user.default_environment,
        "prior_similar_request": last_similar_request_summary,
    },
}
```

### Questions

```python
from typesafe_sdk import Choice, Noul, Score

CLARIFY_QUESTIONS = {
    "ambiguity": Score(
        instructions="How ambiguous this request is, given the context provided",
        criteria=[
            "Unambiguous: exactly one reasonable interpretation",
            "Minor ambiguity about style or format only",
            "Real ambiguity about scope or target, but one reading is clearly most likely",
            "Genuinely ambiguous: several readings are equally plausible",
        ],
    ),
    "target_identified": Noul(
        instructions="The specific object of this request is uniquely identified "
                     "by the request plus the context provided"),
    "consequence": Score(
        instructions="How costly it would be to act on the wrong interpretation",
        criteria=[
            "Trivial: easily noticed and redone",
            "Wasted effort, but no lasting effect",
            "Lasting effect that requires deliberate work to undo",
            "Irreversible, externally visible, or affecting other people",
        ],
    ),
    "missing_required": Noul(
        instructions="A piece of information required to act at all is absent "
                     "and cannot be inferred from the context provided"),
    "resolvable_by_lookup": Noul(
        instructions="The missing information could be obtained by the system "
                     "itself, without asking the user"),
    "single_question_suffices": Noul(
        instructions="One question would resolve all the outstanding ambiguity"),
}
```

`resolvable_by_lookup` is the anti-annoyance guard: never ask the user for something you can look up. It converts a would-be clarification into a tool call.

## Integration

Combine the two axes multiplicatively — a risk score, not a threshold on either alone:

```python
ASK, LOOKUP, PROCEED, PROCEED_STATED = range(4)

def decide(request, history, ctx):
    a = client.system_one(state=build_state(request, history, ctx),
                          questions=CLARIFY_QUESTIONS).answers

    amb  = a["ambiguity"].score / 3.0        # normalise to 0..1
    cons = a["consequence"].score / 3.0
    risk = amb * cons                        # the product is the point

    if a["missing_required"].noul > 0.6:
        return LOOKUP if a["resolvable_by_lookup"].noul > 0.6 else ASK

    if risk > 0.30:
        return ASK
    if risk > 0.12:
        return PROCEED_STATED     # proceed, but state the assumption in the reply
    if a["target_identified"].noul < 0.4 and cons > 0.5:
        return ASK                # unidentified target on a consequential action
    return PROCEED
```

**`PROCEED_STATED` is the most useful of the four.** It covers the large middle band: act on the most likely reading but surface the assumption — *"Cancelling the Pro subscription (the only active paid one). Say the word if you meant a different one."* The user gets a result immediately and an easy correction path. Most systems have only ASK and PROCEED and therefore have to be wrong in one direction.

## Thresholds & escalation

| Condition | Action |
|---|---|
| `missing_required` and resolvable | Look it up; do not ask |
| `missing_required`, not resolvable | Ask |
| `risk > 0.30` | Ask — bundle into one question if `single_question_suffices` |
| `0.12 < risk ≤ 0.30` | Proceed and state the assumption |
| Unidentified target + high consequence | Ask, regardless of score |
| `risk ≤ 0.12` | Proceed silently |

Tune the 0.30 boundary against a friction budget: measure the ask-rate your users tolerate and set the threshold to hit it, rather than picking a number that feels right.

## Impact model

*Illustrative.* 500k requests/month. Suppose 6% are consequentially ambiguous today and half of those produce wrong work costing ~$2 in rework (agent turns plus a support touch).

```
today       500k × 6% × 50% × $2      = $30,000/month
with Jev    500k × $0.00011            = $    55
          + questions on ~3% of traffic (friction, not cost)
```

The real return is trust. An assistant that occasionally says "I assumed X" is trusted; one that silently does the wrong thing on irreversible actions gets switched off — and adoption, not inference cost, is what usually determines whether these systems survive.

## Failure modes

- **Over-asking.** The most likely failure and the most damaging to adoption. Monitor ask-rate weekly; treat a rise as a regression. Prefer `PROCEED_STATED` when in doubt.
- **Bundling failure.** Asking three questions in sequence when one compound question would do. Use `single_question_suffices` and compose one question in code.
- **Ambiguity the model cannot see.** Organisation-specific overloaded terms ("the platform", "prod") read as unambiguous. Fix by enriching `resolvable_context` with a glossary, not by rewording the question.
- **Consequence underestimated for domain reasons.** A read that triggers a compliance notification is not trivial. Override `consequence` in code for known-sensitive tool paths.
- **Clarification loops.** User answers vaguely, agent asks again. Cap clarification rounds at one, then proceed with a stated assumption.

## Evaluation

1. Label 400 requests with the outcome of proceeding on the most likely reading: right / wrong-and-cheap / wrong-and-expensive. Measure whether ASK concentrates on the third bucket — that concentration, not accuracy, is the metric.
2. Tune the risk threshold against a target ask-rate derived from user tolerance, not intuition.
3. A/B the three-way policy (ASK / PROCEED_STATED / PROCEED) against your current two-way one. Measure task success, user corrections, and abandonment.
4. In production: ask-rate, correction-after-`PROCEED_STATED` rate (validates the middle band is calibrated), and rework incidents traced to a silent wrong assumption.

## Related

- [P01](P01-tiered-model-routing.md) — `ambiguity` also routes difficulty
- [P04](P04-task-completion-detection.md) — vague goals make completion undetectable; catch them here
- [P02](P02-tool-call-risk-gating.md) — the consequence axis reappears at execution time
- [P26](P26-deflection-eligibility.md) — the customer-facing analogue of "can I answer this as asked?"
