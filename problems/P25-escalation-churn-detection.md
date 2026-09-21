# P25 — Live Escalation & Churn-Risk Detection

| | |
|---|---|
| **Theme** | E · Customer & revenue operations |
| **Primitives** | `score`, `noul`, `choice` |
| **Dominant win** | `⚡` latency + `📈` coverage — intervention while the conversation is still live |
| **Latency budget** | <200 ms per turn |
| **Volume profile** | Every turn of every live conversation |
| **Blast radius if wrong** | Moderate — a missed escalation loses a customer; a false one wastes supervisor time |

---

## Problem

[P24](P24-ticket-triage.md) judges a conversation at intake. Conversations then *change*, and the change is where customers are lost. A ticket that arrives calm and routine becomes a churn event over six turns: the first answer misses the point, the second repeats the first, the customer says "I've already told you this", and by turn five they are asking for a manager or writing a review.

Every signal needed to intervene is present by turn three. Nobody acts on it, because nothing is watching. The existing controls are all retrospective:

- **Post-conversation CSAT** arrives after the damage, from the minority who respond.
- **Supervisor monitoring** covers a tiny sample of live conversations.
- **Escalation on customer request** means the customer had to ask, which is already a failure.

The intervention window is minutes wide and requires per-turn judgement. That is only possible at ~100 ms and ~$0.0001 — an LLM watching every turn of every conversation is both too slow to be inside the turn and too expensive to run at that multiplier.

## Today's pattern

| Approach | Problem |
|---|---|
| **Post-hoc CSAT / NPS** | Lagging, sparse, response-biased. Diagnoses, never prevents |
| **Keyword triggers** ("manager", "cancel", "lawyer") | Fires late — by the time those words appear the customer has decided — and misses polite escalation entirely |
| **Sentiment APIs** | Turn-level polarity without trajectory. Cannot distinguish "frustrated and improving" from "calm and deteriorating" |
| **Supervisor live monitoring** | Covers a small sample; attention goes where it is noticed, not where it is needed |
| **LLM per-turn analysis** | Right capability; cost multiplies by turn count and latency sits inside the response path |

## Jev design

The essential design point: **judge trajectory, not state.** A single turn's frustration level is weak signal. Frustration *rising* across three turns while the agent repeats itself is strong signal. So `state` must carry the conversation, and the questions must ask about change.

### State

```python
state = {
    "conversation": [
        {"role": m.role, "text": truncate(m.text, 500)} for m in conv.messages[-8:]
    ],
    "meta": {
        "turn_count": len(conv.messages),
        "elapsed_minutes": conv.elapsed_min,
        "agent_type": "ai" if conv.is_ai else "human",
        "prior_escalations_90d": acct.escalations_90d,
    },
    "account": {"tier": acct.tier, "arr_band": acct.arr_band,
                "renewal_in_days": acct.renewal_days},
}
```

`renewal_in_days` transforms the economics of an intervention. The same frustration 30 days before a renewal is worth far more supervisor attention than 300 days out.

### Questions

```python
from typesafe_sdk import Choice, Noul, Score

ESCALATION_QUESTIONS = {
    "frustration_now": Score(
        instructions="How frustrated the customer is in their most recent message",
        criteria=["Neutral or positive", "Mildly impatient",
                  "Clearly frustrated", "Angry or hostile"],
    ),
    "frustration_trend": Choice(
        instructions="How the customer's frustration has changed across this "
                     "conversation",
        criteria={"improving": "Calmer than earlier",
                  "stable": "Roughly unchanged",
                  "worsening": "More frustrated than earlier",
                  "sharp_worsening": "A marked deterioration in the last turn or two"},
    ),
    "progress_made": Noul(
        instructions="The conversation has moved measurably towards resolving the "
                     "customer's problem, as opposed to circling it"),
    "repeating_self": Noul(
        instructions="The customer has had to restate information they already "
                     "gave earlier in this conversation"),
    "agent_missed_point": Noul(
        instructions="The most recent agent response failed to address what the "
                     "customer actually asked"),
    "churn_language": Choice(
        instructions="If the customer signals risk to the relationship, how",
        criteria={
            "none":        "No such signal",
            "dissatisfaction": "Expresses dissatisfaction without any threat",
            "comparison":  "Mentions a competitor or alternative",
            "cancellation":"Raises cancelling, not renewing, or downgrading",
            "public":      "Threatens a review, social post, or complaint to a regulator",
            "legal":       "Raises legal action or a lawyer",
        },
    ),
    "wants_human": Noul(
        instructions="The customer has asked for a human, a manager, or a "
                     "different agent, whether explicitly or by clear implication"),
    "resolvable_by_agent": Noul(
        instructions="The current agent could plausibly still resolve this without "
                     "escalation, given what has been discussed"),
}
```

`agent_missed_point` is the diagnostic question, and it is the one worth building the whole system for. It identifies the *cause* of the deterioration rather than the symptom, which makes the output actionable in two directions: intervene now, and fix the agent or the content later.

## Integration

```python
def watch(conv, account):
    a = client.system_one(state=build_state(conv, account),
                          questions=ESCALATION_QUESTIONS).answers

    # Hard escalations — no scoring arithmetic.
    if a["churn_language"].choice in ("legal", "public") \
            and a["churn_language"].confidence > 0.6:
        return Escalate("urgent", notify=["supervisor", "account_owner", "legal"])
    if a["wants_human"].noul > 0.6 and conv.is_ai:
        return HandOffToHuman(reason="requested", context=summarise(conv))

    risk = (0.30 * a["frustration_now"].score / 3.0
          + 0.25 * TREND_WEIGHT[a["frustration_trend"].choice]
          + 0.15 * (1 - a["progress_made"].noul)
          + 0.15 * a["repeating_self"].noul
          + 0.15 * a["agent_missed_point"].noul)

    risk *= renewal_multiplier(account.renewal_days)      # code, not model
    risk *= TIER_MULTIPLIER[account.tier]

    if a["churn_language"].choice == "cancellation":
        risk = max(risk, 0.70)

    if risk > 0.70 or a["frustration_trend"].choice == "sharp_worsening":
        if a["resolvable_by_agent"].noul > 0.6 and conv.is_ai:
            return CoachAgent(hint=coaching_hint(a))       # inject guidance, no handoff
        return Escalate("standard", notify=["supervisor"])

    if risk > 0.45:
        return Flag(visible_to="supervisor_dashboard", risk=risk)

    if a["agent_missed_point"].noul > 0.6:
        log_quality_signal(conv, a)                        # → P27 / P13
    return Continue
```

`CoachAgent` is the intervention most organisations lack. Not every deteriorating conversation needs a handoff — many need the AI agent to be told "you have missed the point; the customer asked about X twice". Because the watcher is cheap, you can attempt a correction before spending a human.

Note that `agent_missed_point` is logged even when risk is low. Over time that log is the highest-value dataset here: it identifies which questions your agent consistently mishandles, which feeds [P27](P27-conversation-qa-scoring.md) and content gaps in [P13](P13-knowledge-conflict-detection.md).

## Thresholds & escalation

| Signal | Threshold | Action |
|---|---|---|
| `churn_language ∈ {legal, public}` | conf > 0.6 | Urgent escalation; notify legal and the account owner |
| `wants_human` on an AI conversation | > 0.6 | Hand off immediately with context. Never argue |
| `churn_language == cancellation` | — | Risk floor 0.70 |
| `frustration_trend == sharp_worsening` | — | Escalate regardless of the composite |
| composite risk | > 0.70 | Coach the agent if still resolvable, else escalate |
| composite risk | 0.45–0.70 | Flag on the supervisor dashboard |
| `agent_missed_point` | > 0.6 | Log for quality analysis regardless of risk |

Multipliers for renewal proximity and account tier are commercial policy and belong in code where a revenue leader can see and change them.

## Impact model

*Illustrative.* 500k conversations/month, mean 6 turns = 3M turn evaluations.

```
Jev, every turn      3M × $0.00012  = $  360/month
LLM, every turn      3M × $0.008    = $24,000/month + latency inside each turn
today (post-hoc)     ~$0, zero prevention
```

The business case is retention arithmetic. If this surfaces 2,000 at-risk conversations a month that would previously have gone unnoticed, and intervention saves even a small fraction of the accounts among them, the value dwarfs every other number in this catalogue. That said, **the saved-churn figure is the one most likely to be overstated** — an escalation is not a save, and attribution is genuinely hard. Measure it as a controlled experiment (below), not as an assumption.

The reliable, non-speculative wins: handoffs happen when the customer asks rather than after they have asked three times, and the `agent_missed_point` log gives concrete evidence of where the AI agent fails.

## Failure modes

- **Alert fatigue destroys this faster than anything else.** A supervisor dashboard flagging 15% of conversations gets ignored. Tune the 0.45 threshold to your actual supervisor capacity, and treat flag volume as a hard constraint rather than an output.
- **Cultural variation in frustration expression.** Directness norms differ substantially across languages and regions. Calibrate per locale; a rubric tuned on US English will over-flag some locales and under-flag others.
- **Politeness masking.** Some customers stay courteous while deciding to leave. `churn_language` and `progress_made` catch more of this than sentiment alone, but this class will remain under-detected. Be honest about it.
- **Escalating what the agent could have fixed.** The `resolvable_by_agent` + `CoachAgent` path exists for this. Measure how often coaching resolves without handoff.
- **Self-fulfilling escalation.** Flagging a conversation then handling it differently makes clean measurement hard. Hold out a control group.
- **Injection / gaming.** A customer learning that "lawyer" triggers escalation will use it. Monitor for abuse; the hard escalation is worth keeping anyway, since the cost of honouring a false one is low.
- **Stale renewal data.** A wrong `renewal_in_days` misprices every intervention. Read live.

## Evaluation

1. **Backtest against outcomes.** Take 2,000 historical conversations where you know what happened next: CSAT score, escalation, churn within 90 days. Measure whether the risk score at turn 3 predicts the outcome. That predictive lift is the whole validity claim.
2. Measure **lead time**: how many turns before the escalation actually occurred does the detector fire? Lead time is the value; accuracy without lead time is just a post-hoc label.
3. Validate `agent_missed_point` against human review of the same turns — this is the dimension most likely to disagree with humans, and the one most used downstream.
4. **Run a controlled trial for the churn claim.** Randomise flagged conversations into intervene/control arms and measure retention difference. Without this, "churn saved" is a story, not a number.
5. In production: flag rate against supervisor capacity, escalation precision (did the supervisor agree it needed them?), coaching resolution rate, lead time distribution, and CSAT for intervened versus control conversations.

## Related

- [P24](P24-ticket-triage.md) — intake-time judgement; this continues it through the conversation
- [P27](P27-conversation-qa-scoring.md) — consumes `agent_missed_point` for systematic quality scoring
- [P21](P21-brand-policy-gating.md) — `empathy_fit` on the outbound side of the same conversations
- [P15](P15-online-output-qa.md) — catches the bad agent response that triggers deterioration
- [P18](P18-drift-detection.md) — aggregate frustration trends as a drift signal
