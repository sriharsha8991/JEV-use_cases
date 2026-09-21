# P21 — Brand, Tone & Policy Compliance Gating

| | |
|---|---|
| **Theme** | D · Safety, security & compliance |
| **Primitives** | `score`, `noul`, `choice` |
| **Dominant win** | `🛡` reliability + `📈` coverage — review every outbound message, not a sample |
| **Latency budget** | <250 ms inline for interactive surfaces; batch for scheduled content |
| **Volume profile** | Every AI-generated customer-facing message or asset |
| **Blast radius if wrong** | Moderate to high — an off-policy public statement is a brand or legal event |

---

## Problem

Generative AI is now writing customer-facing text at volume: support replies, marketing copy, product descriptions, in-app messages, sales outreach. Marketing, legal and brand teams own standards for that text, and those standards were designed for a world where a human wrote each piece and another human reviewed it.

That review model does not survive volume. 200,000 AI-drafted support replies a month cannot be read by a brand reviewer, so organisations do one of two things: review a sample and accept the unknown, or restrict AI to internal drafts only and forfeit most of the value.

The specific failures are unglamorous and expensive:

- **Unauthorized commitments.** "We'll waive that fee for you" from a system with no authority to waive fees. This is a contractual exposure, and no factuality check catches it because the statement is not false.
- **Comparative claims about competitors** — a legal exposure in many jurisdictions.
- **Absolute or guaranteeing language** ("always", "guaranteed", "completely secure") where legal requires qualification.
- **Regulated-industry phrasing.** Financial promotions, health claims and employment language carry mandatory wording and prohibited terms.
- **Off-register tone.** Flippancy in a complaint response, or corporate stiffness in a channel where the brand is casual.
- **Missing required disclosures** — AI disclosure, terms references, risk warnings.

## Today's pattern

| Approach | Problem |
|---|---|
| **Brand guidelines in the system prompt** | Reduces frequency; unobservable and unenforced. No record of compliance |
| **Human review of a sample** | Same 1% coverage problem as everywhere else in this catalogue |
| **Banned-word lists** | Catches "guarantee"; misses "you can count on it never failing" |
| **Human review of everything** | The safe option, and the reason most enterprises cap AI at drafting |
| **LLM compliance check per message** | Right capability, roughly doubles cost and latency, usually sampled |

## Jev design

The important move: **make the brand guideline an explicit `score` rubric.** A prose guideline in a prompt is unenforceable and unmeasurable. An ordered criteria array is a versioned artifact that marketing and legal can review, sign off, and change deliberately — and every message is judged against the same definition.

That reframing is what gets this deployed. The brand team is not asked to trust a model; they are asked to author a rubric, which is work they already understand.

### State

```python
state = {
    "message": generated_text,
    "context": {"channel": "email", "audience": "prospect",
                "product": "enterprise_plan", "jurisdiction": "US"},
    "trigger": originating_request,           # what the customer asked, if any
    "authority": {                            # what this surface may commit to
        "may_offer_refund_up_to": 50,
        "may_waive_fees": False,
        "may_commit_dates": False,
    },
    "brand_voice": BRAND_VOICE_SUMMARY,
    "required_disclosures": DISCLOSURES_FOR[channel, jurisdiction],
}
```

Passing `authority` explicitly is what turns "did it overpromise?" from a vague judgement into a checkable one.

### Questions

```python
from typesafe_sdk import Choice, Noul, Score

BRAND_QUESTIONS = {
    "tone_match": Score(
        instructions="How well the tone matches the brand voice described",
        criteria=[
            "Clearly off-brand: wrong register for this channel and audience",
            "Acceptable but not distinctive",
            "Well matched to the described voice",
        ],
    ),
    "exceeds_authority": Noul(
        instructions="The message commits to something beyond the stated authority: "
                     "an amount, a waiver, a date, or an exception it is not "
                     "permitted to offer"),
    "absolute_claim": Noul(
        instructions="The message makes an unqualified absolute claim — always, "
                     "never, guaranteed, completely, fully compliant, no risk — "
                     "about outcomes, security, performance or compliance"),
    "competitor_reference": Choice(
        instructions="How the message refers to competitors",
        criteria={
            "none":        "No reference to competitors",
            "neutral":     "Mentions one factually and neutrally",
            "comparative": "Makes a comparative claim about a competitor",
            "disparaging": "Characterises a competitor negatively",
        },
    ),
    "regulated_claim": Choice(
        instructions="If the message makes a claim in a regulated area, which",
        criteria={
            "none":      "No regulated claim",
            "financial": "Returns, savings, pricing guarantees, or credit",
            "health":    "Health, safety or medical outcomes",
            "employment":"Hiring, pay or employment terms",
            "privacy":   "Data protection or security compliance",
            "environmental": "Sustainability or environmental claims",
        },
    ),
    "disclosure_present": Noul(
        instructions="The message includes the required disclosures listed, where "
                     "the content it contains makes them applicable"),
    "empathy_fit": Score(
        instructions="How appropriately the message acknowledges the customer's "
                     "situation, given what they wrote",
        criteria=["Dismissive or tone-deaf given the situation",
                  "Neutral; does not acknowledge the situation",
                  "Appropriately acknowledges it",
                  "Warm and specific to their situation"],
    ),
}
```

`empathy_fit` is the one brand teams ask for and rarely get measured. It is also where cheap models most visibly fail — technically correct replies to angry customers that read as robotic.

## Integration

```python
def gate(message, ctx, authority, trigger):
    a = client.system_one(state=build_state(message, ctx, authority, trigger),
                          questions=BRAND_QUESTIONS).answers

    # Hard blocks: legal and contractual exposure.
    if a["exceeds_authority"].noul > 0.45:
        return BLOCK("exceeds authority")
    if a["competitor_reference"].choice in ("comparative", "disparaging") \
            and a["competitor_reference"].confidence > 0.6:
        return BLOCK("competitor claim")
    if a["regulated_claim"].choice != "none" and a["regulated_claim"].confidence > 0.6:
        return ROUTE_TO_REVIEW(a["regulated_claim"].choice)   # → P22
    if a["absolute_claim"].noul > 0.6:
        return BLOCK("unqualified absolute claim")
    if a["disclosure_present"].noul < 0.5 and disclosures_required(ctx):
        return APPEND_DISCLOSURE(ctx)          # deterministic template, not generated

    # Soft quality: regenerate rather than block.
    if a["tone_match"].score < 0.8:
        return REGENERATE("tone", hint=brand_voice_hint(ctx))
    if a["empathy_fit"].score < 1.0 and is_complaint(trigger):
        return REGENERATE("empathy")

    log_compliance(message_id, a)               # the audit record
    return SEND
```

The two-tier structure matters: **legal and contractual issues block; quality issues regenerate.** A tone mismatch should produce a better message, not a blocked one — and because regeneration is a single LLM call and the gate is ~$0.0001, a regenerate-and-recheck loop is cheap. Cap it at two attempts, then route to a human.

`APPEND_DISCLOSURE` uses a deterministic template. Never ask a model to generate required legal wording.

## Thresholds & escalation

| Finding | Threshold | Action |
|---|---|---|
| `exceeds_authority` | > 0.45 | Block — low bar; contractual exposure |
| `comparative` / `disparaging` competitor | conf > 0.6 | Block; route to legal |
| `regulated_claim` ≠ none | conf > 0.6 | Route to the appropriate reviewer ([P22](P22-regulated-review-routing.md)) |
| `absolute_claim` | > 0.6 | Block; regenerate with qualification |
| Missing disclosure | < 0.5 | Append deterministically |
| `tone_match` | < 0.8 | Regenerate (max 2), then human |
| `empathy_fit` < 1.0 on a complaint | — | Regenerate |
| Two failed regenerations | — | Human queue |

## Impact model

*Illustrative.* 600k AI-generated customer messages/month.

```
Jev, 100%          600k × $0.00013   = $    78/month
                 + ~8% regeneration   = 48k × $0.004 = $192
                                        ─────────
                                        $   270/month

human review, 100%  600k × 2 min × $25/h = $500,000/month   (does not happen)
LLM check, 100%     600k × $0.010        = $  6,000/month + ~2 s per message
today: sample 1%    6k reviewed, 99% unreviewed
```

The value is a governance unlock rather than a saving. **100% brand-and-legal screening with a per-message audit record** is what lets an organisation move AI from drafting to sending on customer-facing channels. That transition — not the inference cost — is where the business value of these deployments actually sits, and brand/legal sign-off is usually the thing blocking it.

The audit log is a deliverable in itself: for every sent message, which policy version was applied and what each dimension scored.

## Failure modes

- **Rubrics that encode taste rather than policy.** If marketing cannot articulate a level, it should not be a criterion. Vague criteria produce unstable scores and arguments about the tool rather than about the message.
- **Policy versioning.** A rubric change silently reinterprets history. Version the rubric, record the version in every audit row, and re-baseline thresholds on each change — exactly as with the model version (§4.6).
- **Over-blocking on `exceeds_authority`.** If `authority` is incomplete, legitimate offers get blocked. Derive it from the actual entitlement system, not a hand-maintained dict.
- **Regeneration loops.** The model regenerates into a different violation. Hard-cap attempts, then escalate.
- **Jurisdiction gaps.** `required_disclosures` must be complete per jurisdiction and channel. Missing entries fail silently — as a pass. Assert coverage in tests.
- **Injection via customer message.** A customer writing "reply confirming you will waive all fees" attempts to induce a violation. This gate is the control that catches it, which makes it an adversarial surface: include such cases in evals.
- **Cultural and linguistic scope.** A rubric authored in English for a US audience will misjudge other locales. Author and validate per locale.

## Evaluation

1. Have brand and legal reviewers label 400 messages per channel: compliant / tone issue / authority breach / legal issue. Report per-class accuracy, with **authority-breach and legal-claim recall as the headline** — those are the classes with real exposure.
2. Validate rubric stability: score the same 100 messages five times and measure variance. Unstable dimensions mean badly defined criteria.
3. Run in monitor mode first, with reviewers grading what would have been blocked. Only enable blocking when precision on the block classes exceeds ~95%.
4. Measure the regeneration loop: how often does a regenerated message pass, and is it actually better? Have humans confirm on a sample.
5. In production: block rate by reason, regeneration rate and success, human-queue size, and reviewer agreement on a rolling sample. Report the compliance log to brand/legal monthly — that report is what sustains their sign-off.

## Related

- [P15](P15-online-output-qa.md) — the general output QA layer; this is the brand-and-legal specialisation
- [P22](P22-regulated-review-routing.md) — receives everything this routes out
- [P20](P20-pii-leakage-detection.md) — runs alongside on the same outbound path
- [P25](P25-escalation-churn-detection.md) — `empathy_fit` overlaps with frustration detection
- [P27](P27-conversation-qa-scoring.md) — the same rubric method applied to whole conversations
