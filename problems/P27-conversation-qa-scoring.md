# P27 — Conversation QA Scoring for Human and AI Agents

| | |
|---|---|
| **Theme** | E · Customer & revenue operations |
| **Primitives** | `score`, `noul`, `choice` |
| **Dominant win** | `📈` coverage — every conversation instead of 1–3% — plus `$` |
| **Latency budget** | Batch; hourly or nightly |
| **Volume profile** | Every closed conversation |
| **Blast radius if wrong** | **Sensitive** — this scores people's work. Misuse is an HR and trust problem, not just an accuracy problem |

---

## Problem

Contact-centre quality assurance is a sampling exercise. A QA analyst scores 3–5 conversations per agent per month against a rubric — perhaps 1–3% of the work. Everything that follows from QA is built on that sample: coaching, performance reviews, process improvement, compliance attestations.

The sample is too small to be fair or useful. An agent handling 400 conversations a month is coached on four. Which four is partly luck, and the agents know it. QA becomes something that happens *to* people rather than something that helps them.

Two changes make this worse. **Volume has grown** and QA headcount has not. And **AI agents now handle a growing share of conversations**, which need the same scrutiny but at machine volume — and their failures are systematic rather than individual, so a 1% sample is exactly the wrong instrument for finding them.

At ~$0.0001 per conversation, scoring 100% becomes ordinary. But the shift from sampling to census changes what this *is*, and that has to be handled deliberately — see the governance note below.

> **Governance note.** Scoring 100% of a person's conversations is a materially different thing from sampling. In many jurisdictions automated evaluation of employee performance triggers consultation requirements, transparency obligations, and rights of explanation ([P22](P22-regulated-review-routing.md)). Involve HR, works councils and employee representatives **before** building this, and tell agents what is measured and how. Deploying it as a surprise is both a legal risk and a guaranteed way to destroy the trust that makes coaching work.

## Today's pattern

| Approach | Problem |
|---|---|
| **Human QA on a 1–3% sample** | Too sparse to be fair; inconsistent between analysts; expensive per unit |
| **Keyword compliance checks** ("did they say the disclosure?") | Catches literal scripts only; no judgement of quality |
| **Speech analytics / sentiment tools** | Surface metrics: talk time, silence, polarity. Weak on whether the customer was actually helped |
| **CSAT as a proxy** | Response-biased, conflates agent performance with product problems |
| **LLM scoring of every conversation** | Right capability; ~$0.05–0.30 per conversation at length, so it is sampled |

## Jev design

Reuse the [P14](P14-llm-judge-replacement.md) discipline: **the QA rubric becomes explicit ordered `score` criteria**, authored and signed off by the QA function. This is a genuine improvement over a prose rubric in an analyst's head — the same definition is applied to every conversation, which is the fairness property sampling can never provide.

Separate **agent-controllable** dimensions from **outcome** dimensions. Conflating them is the classic QA injustice: an agent penalised because the product was broken.

### State

```python
state = {
    "conversation": [
        {"role": m.role, "text": truncate(m.text, 600)} for m in conv.messages
    ],
    "meta": {"channel": conv.channel, "agent_type": "human" | "ai",
             "duration_min": conv.duration, "transfers": conv.transfer_count,
             "resolved_flag": conv.resolved},
    "rubric_version": QA_RUBRIC_VERSION,
    "required_disclosures": DISCLOSURES_FOR[conv.channel, conv.jurisdiction],
}
```

Long conversations exceed the state budget. Score in segments and combine in code, or score the first and last N turns plus a deterministic middle summary — but be explicit about which, and keep it consistent, or your scores are not comparable across conversation lengths.

### Questions

```python
from typesafe_sdk import Choice, Noul, Score

QA_QUESTIONS = {
    # --- Agent-controllable ---
    "understood_issue": Score(
        instructions="How well the agent understood what the customer needed",
        criteria=["Never grasped it",
                  "Partially; some responses were off-target",
                  "Understood after clarification",
                  "Understood immediately and accurately"],
    ),
    "accuracy": Score(
        instructions="Accuracy of the information the agent provided",
        criteria=["Gave incorrect information",
                  "Gave incomplete or ambiguous information",
                  "Gave accurate, complete information"],
    ),
    "empathy": Score(
        instructions="How appropriately the agent acknowledged the customer's "
                     "situation, given what the customer expressed",
        criteria=["Dismissive or tone-deaf",
                  "Neutral; no acknowledgement where some was warranted",
                  "Acknowledged appropriately",
                  "Warm and specific to their situation"],
    ),
    "efficiency": Score(
        instructions="How efficiently the agent moved towards resolution, "
                     "excluding delays caused by the customer or by systems",
        criteria=["Substantial avoidable back-and-forth",
                  "Some avoidable steps",
                  "Direct and efficient"],
    ),
    "ownership": Noul(
        instructions="The agent took responsibility for driving the issue to "
                     "resolution rather than deflecting it elsewhere"),
    "disclosures_given": Noul(
        instructions="The agent gave the required disclosures listed, where the "
                     "conversation made them applicable"),
    "made_unauthorised_commitment": Noul(
        instructions="The agent promised something outside normal policy: a "
                     "refund, an exception, a date, or a guarantee"),

    # --- Outcome (NOT attributed to the agent) ---
    "customer_resolved": Noul(
        instructions="The customer's problem was actually resolved in this "
                     "conversation, judged from the conversation itself rather "
                     "than from any system flag"),
    "blocked_by_product": Noul(
        instructions="Resolution was prevented by a product limitation, bug or "
                     "outage rather than by anything the agent did"),
    "blocked_by_policy": Noul(
        instructions="Resolution was prevented by a policy the agent correctly "
                     "applied"),
    "root_cause": Choice(
        instructions="The underlying reason the customer needed to make contact",
        criteria=ROOT_CAUSE_TAXONOMY,     # product gaps, docs gaps, UX, billing, …
    ),
}
```

`blocked_by_product` and `blocked_by_policy` are the fairness mechanism. An agent who correctly applies a policy the customer dislikes should score well on a conversation that resolved badly — and without these dimensions every QA system punishes them for it.

`root_cause` is the dimension that makes this pay for itself outside QA. Across 100% of conversations it produces a ranked list of *why customers contact you at all*, which is a product and content backlog rather than an agent metric.

## Integration

```python
def score(conv):
    a = client.system_one(state=build_state(conv), questions=QA_QUESTIONS).answers

    # Agent score uses ONLY agent-controllable dimensions.
    agent = (0.25 * a["understood_issue"].score / 3.0
           + 0.30 * a["accuracy"].score / 2.0
           + 0.20 * a["empathy"].score / 3.0
           + 0.15 * a["efficiency"].score / 2.0
           + 0.10 * a["ownership"].noul)

    flags = []
    if a["made_unauthorised_commitment"].noul > 0.5: flags.append("commitment")
    if a["disclosures_given"].noul < 0.5 and disclosures_required(conv):
        flags.append("missing_disclosure")

    # Low confidence → human QA. This is the sample worth an analyst's time.
    low_conf = min(a["understood_issue"].confidence, a["accuracy"].confidence) < 0.55

    return QARow(
        agent_score=agent,
        flags=flags,
        needs_human_qa=low_conf or bool(flags) or agent < 0.5,
        # Outcome dimensions recorded separately, never folded into agent_score.
        outcome={"resolved": a["customer_resolved"].noul,
                 "blocked_product": a["blocked_by_product"].noul,
                 "blocked_policy": a["blocked_by_policy"].noul,
                 "root_cause": a["root_cause"].choice},
        rubric_version=QA_RUBRIC_VERSION, model_version=response.model,
    )
```

The architectural commitment: **`agent_score` never includes an outcome dimension.** Outcome data goes to product and policy owners; agent data goes to coaching. Mixing them is the single most common way QA systems become unfair and then resented.

## Thresholds & escalation

| Signal | Action |
|---|---|
| `made_unauthorised_commitment > 0.5` | Route to human QA; do not act on the automated finding alone |
| Missing required disclosure | Compliance review ([P22](P22-regulated-review-routing.md)) |
| Confidence < 0.55 on a weighted dimension | Human QA — this is the high-value analyst sample |
| `agent_score < 0.5` | **Human QA before any coaching conversation.** Never coach from an unreviewed automated score |
| `blocked_by_product > 0.6`, clustered | Product backlog item, not an agent finding |
| `root_cause` distribution shift | Content or product signal ([P18](P18-drift-detection.md)) |

The rule in bold is the one to hold firm on: **no automated score should reach a performance conversation without a human having reviewed that conversation.** Use the census for coverage, trends and targeting; use humans for judgements about people.

## Impact model

*Illustrative.* 500k conversations/month, ~3,500-token envelope ≈ $0.00015.

```
Jev, 100%             500k × $0.00015 = $    75/month
human QA, 1.5%        7,500 × 20 min × $30/h = $75,000/month  (1.5% coverage)
LLM, 100%             500k × $0.08    = $40,000/month
```

The comparison to make is not cost. **Same QA budget, 100% coverage instead of 1.5%, plus analyst time redirected from random sampling to a targeted queue.** Concretely:

- Coaching becomes based on an agent's actual distribution rather than four conversations.
- AI-agent quality becomes measurable at the population level, where its systematic failures are visible.
- The `root_cause` census produces a contact-driver ranking across all conversations — usually the most valuable output, and entirely outside QA's remit.
- Compliance attestation moves from "we sampled" to "we checked every conversation", which is a materially stronger position.

## Failure modes

- **Deployed without HR and employee consultation** — the top risk, and it is not technical. It will generate legitimate grievance, and in some jurisdictions legal exposure.
- **Used for discipline rather than coaching.** The fastest way to make agents game the rubric. Keep the human-review requirement absolute for any performance use.
- **Rubric encodes taste rather than behaviour.** If QA cannot articulate a level, it should not be scored. Vague criteria produce unstable scores and arguments about the tool.
- **Attribution errors.** Agents penalised for product or policy limits. The outcome/controllable split addresses this; audit it specifically, because it is the thing agents will check first.
- **Long-conversation truncation** makes scores non-comparable across lengths. Pick one segmentation strategy and hold it constant; monitor score-versus-length correlation as a diagnostic.
- **Rubric and model versioning.** A rubric edit or model bump silently reinterprets history. Record both on every row; re-baseline on change (§4.6); never compare scores across versions.
- **Cross-language and cross-channel bias.** Validate per language and channel, and check score distributions for systematic differences between agent cohorts. If a cohort scores consistently lower, investigate the rubric before the cohort.

## Evaluation

1. **Inter-rater baseline first.** Have three QA analysts score the same 100 conversations. Measure their agreement with each other. That number is the ceiling — and it is usually lower than people expect, which reframes the whole accuracy conversation.
2. Measure Jev-versus-analyst agreement per dimension against that baseline. Parity with human inter-rater agreement is success.
3. **Audit for bias explicitly**: compare score distributions across agent cohorts, languages, channels and shifts, controlling for conversation mix. Do this before deployment and on a schedule after.
4. Validate the outcome/controllable split by checking that `agent_score` is uncorrelated with `blocked_by_product` on matched conversation types.
5. Verify calibration to license the human-QA routing.
6. In production: score distributions, human-QA queue size and agreement, flag rates, agent-raised disputes and their outcomes (a rising dispute rate is a rubric problem), and the `root_cause` census as a monthly report to product.

## Related

- [P14](P14-llm-judge-replacement.md) — same rubric-as-criteria method, applied to model output
- [P25](P25-escalation-churn-detection.md) — live version; its `agent_missed_point` log feeds this
- [P22](P22-regulated-review-routing.md) — employment-related automated evaluation obligations
- [P21](P21-brand-policy-gating.md) — `made_unauthorised_commitment` is its `exceeds_authority`, after the fact
- [P18](P18-drift-detection.md) — consumes the `root_cause` census
