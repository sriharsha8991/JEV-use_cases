# P22 — Regulated-Review Routing & AI Risk Tiering

| | |
|---|---|
| **Theme** | D · Safety, security & compliance |
| **Primitives** | `choice`, `noul` |
| **Dominant win** | `🛡` compliance — makes obligation-matching systematic rather than judgement-by-availability |
| **Latency budget** | <200 ms inline, or batch for queued review |
| **Volume profile** | Every AI decision or output in a regulated domain |
| **Blast radius if wrong** | **Severe** — an unreviewed regulated decision is a compliance breach |

---

## Problem

Regulation now attaches obligations to *specific kinds* of automated decision. The EU AI Act imposes duties by risk tier and grants rights around decisions affecting employment, credit and essential services. GDPR Art. 22 gives individuals rights regarding solely-automated decisions with legal or significant effects. Sectoral rules add more: credit adverse-action notices, medical-device software, financial-promotion approval, insurance underwriting justification.

The engineering consequence is unglamorous but hard: **every AI output has to be classified against a matrix of obligations, and routed accordingly.** Which reviewer, which retention period, which disclosure, which appeal path.

Organisations handle this by over-applying or under-applying. Over-applying routes everything through legal and creates a queue that destroys the throughput AI was supposed to provide. Under-applying means an AI system makes decisions in scope of Art. 22 with no human review, no disclosure and no appeal path — usually discovered during an audit.

The reason it is not done systematically is that the classification is a judgement call about *what kind of decision this is*, made per output, and there has been no affordable way to make it at volume.

> **Scope note:** this describes an engineering control for routing and record-keeping. It is not legal advice, and the criteria below must be authored and signed off by your own counsel and compliance function. What follows is the mechanism, not the obligations.

## Today's pattern

| Approach | Problem |
|---|---|
| **Route entire systems by static classification** | Coarse. One assistant handles both a benefits question (in scope) and a printer query (not) |
| **Route everything to legal review** | Queue collapse; AI throughput lost; reviewers desensitised |
| **Manual reviewer judgement at intake** | Inconsistent, unrecorded, dependent on who is on shift |
| **Rules on keywords or department** | Misses semantics; "can I appeal my performance rating?" is routed by the word "appeal", not by what it is |
| **Nothing; rely on annual audit** | Common, and the audit is where it surfaces |

## Jev design

Classify the **decision type and its effect on the person**, then map to obligations in a code table your compliance team owns. The separation is essential: Jev determines what kind of decision this is; **code determines what obligations follow.** Never encode the obligation in the question — obligations change with regulation and must be reviewable as a table, not buried in prompt text.

### State

```python
state = {
    "request": user_request,
    "output": ai_output,
    "action_taken": action_description or None,
    "subject": {"is_individual": True, "is_customer": True,
                "jurisdiction": "EU", "is_minor": False},
    "automation": {"human_in_loop": False, "human_can_override": True},
    "domain": surface_domain,
}
```

`automation` is what determines whether Art. 22-style "solely automated" framing applies, and it must come from your architecture, not from a model's inference.

### Questions

```python
from typesafe_sdk import Choice, Noul

RISK_TIER_QUESTIONS = {
    "decision_type": Choice(
        instructions="What kind of decision or determination this output makes "
                     "about the person",
        criteria={
            "none":            "No decision about a person; information only",
            "eligibility":     "Whether they qualify for something",
            "pricing":         "What they will be charged",
            "creditworthiness":"An assessment of credit or financial risk",
            "employment":      "Hiring, promotion, performance, pay or termination",
            "access":          "Whether they may use a service or facility",
            "benefit":         "Whether they receive a benefit or entitlement",
            "risk_scoring":    "A risk, fraud or trust score attached to them",
            "content_action":  "Removal, restriction or suspension of their content or account",
            "prioritisation":  "Where they sit in a queue or ordering",
        },
    ),
    "effect_severity": Choice(
        instructions="The effect on the person if this determination stands",
        criteria={
            "none":        "No material effect",
            "minor":       "Minor inconvenience, easily reversed",
            "significant": "A material effect on their money, time or options",
            "legal":       "A legal effect, or affects access to an essential "
                           "service such as employment, credit, housing, "
                           "healthcare, education, or insurance",
        },
    ),
    "is_final": Noul(
        instructions="This output determines the outcome, as opposed to producing "
                     "a recommendation that a person will independently decide on"),
    "special_category_basis": Noul(
        instructions="The determination appears to rest on health, biometric, "
                     "belief, ethnicity, sexual-orientation or criminal-record "
                     "information"),
    "explanation_present": Noul(
        instructions="The output states the reasons for the determination in terms "
                     "the person could understand and contest"),
    "appeal_stated": Noul(
        instructions="The output tells the person how to challenge or appeal it"),
}
```

`explanation_present` and `appeal_stated` are checks on *the output*, not classifications of the decision. They detect the most common practical gap: a correctly-tiered decision delivered without the disclosure that tier requires.

## Integration

```python
# Owned and signed off by compliance. Versioned. Reviewed on regulatory change.
OBLIGATION_MATRIX = {
    ("employment", "legal"):       Obligations(review="hr_legal", human_required=True,
                                               retain_years=7, disclose=True, appeal=True),
    ("creditworthiness", "legal"): Obligations(review="credit_risk", human_required=True,
                                               retain_years=7, disclose=True, appeal=True,
                                               adverse_action_notice=True),
    ("eligibility", "significant"):Obligations(review="ops_supervisor", human_required=False,
                                               retain_years=3, disclose=True, appeal=True),
    ("content_action", "significant"): Obligations(review="trust_safety", human_required=False,
                                               retain_years=2, disclose=True, appeal=True),
    # ... complete matrix, with an explicit default
}

def route(output, request, subject, automation):
    a = client.system_one(state=build_state(...), questions=RISK_TIER_QUESTIONS).answers
    dt, sev = a["decision_type"].choice, a["effect_severity"].choice

    # Uncertainty escalates to the stricter tier. Never the looser one.
    if a["decision_type"].confidence < 0.6 or a["effect_severity"].confidence < 0.6:
        return escalate_to_manual_triage(output, a)

    ob = OBLIGATION_MATRIX.get((dt, sev), DEFAULT_OBLIGATIONS)

    if a["special_category_basis"].noul > 0.4:
        ob = ob.harden()                       # strictest tier regardless of type

    if ob.human_required and a["is_final"].noul > 0.5 and not automation["human_in_loop"]:
        return HOLD_FOR_HUMAN(ob.review, reason="solely automated in scope")

    gaps = []
    if ob.disclose and a["explanation_present"].noul < 0.5: gaps.append("explanation")
    if ob.appeal   and a["appeal_stated"].noul   < 0.5:     gaps.append("appeal_route")
    if gaps:
        return AUGMENT(output, add=[TEMPLATES[g](subject.jurisdiction) for g in gaps])

    record_decision(output, a, ob, model_version, matrix_version)   # always
    return DELIVER(output, obligations=ob)
```

Three non-negotiables visible here:

1. **The obligation matrix is a reviewable code artifact**, versioned and owned by compliance. Auditors can read it. It is not inside a prompt.
2. **Uncertainty escalates.** Low confidence on either axis goes to manual triage. A compliance control must not resolve doubt in the permissive direction.
3. **Every decision is recorded** with the Jev version, the matrix version, and the full probability distribution. Jev gives no rationale ([JEV-details §11](../JEV-details.md#11-where-jev-does-not-fit)) — so the record must capture inputs, distributions, thresholds and policy versions. That is an auditable trail *of the decision process*, which is what the obligation actually requires; it is not an explanation by the model, and you should never present it as one.

## Thresholds & escalation

| Condition | Action |
|---|---|
| `decision_type` or `effect_severity` confidence < 0.6 | Manual triage — do not guess a tier |
| `special_category_basis > 0.4` | Harden to the strictest tier |
| `human_required` + `is_final > 0.5` + no human in loop | Hold; this is the Art. 22-shaped case |
| Missing explanation or appeal route | Augment with a jurisdiction-specific template |
| `(decision_type, severity)` absent from the matrix | Default to the strictest tier and alert compliance — a gap in the matrix is a finding |

## Impact model

*Illustrative.* 1M AI outputs/month in regulated domains, of which ~4% fall into a reviewable tier.

```
Jev classification    1.0M × $0.00013 = $  130/month
human triage of all   1.0M × 1 min    = ~16,600 hours   (impossible)
static over-routing   e.g. 20% → 200k reviews vs 40k actually in scope
                      → 160k unnecessary reviews/month
```

The value is precision of routing. Over-routing by a factor of five is the normal failure, and it does not merely cost reviewer hours — it desensitises reviewers, so the genuinely consequential cases get the same cursory treatment as the noise.

Secondary and substantial: the decision record. Being able to produce, for any past output, the classification, distributions, thresholds, matrix version and model version is the difference between a manageable audit and a bad one.

## Failure modes

- **The taxonomy must be authored by counsel.** The `decision_type` and `effect_severity` categories are legal constructs. Engineering-authored criteria will not map to your obligations. This is the single most important control on the whole design.
- **Under-classification is the compliance risk.** Hence uncertainty-escalates, hardening on special-category basis, and strictest-tier defaults for unmatched combinations.
- **Matrix gaps fail silently** unless the default is strict and alerting. Make an unmatched key a monitored event.
- **Jurisdiction complexity.** Obligations differ by country, and by the subject's location rather than yours. Keep jurisdiction resolution in code from authoritative data, never inferred.
- **No model rationale.** Do not let anyone describe the recorded distribution as an explanation of *why*. It is a record of the process. If your obligation requires a reasoned explanation to the data subject, that text must come from a template or a generative model, and a human must approve it.
- **Version skew.** A model bump or a matrix change alters classification. Record both versions on every row and re-baseline thresholds on any change (§4.6).
- **Injection.** Content asserting "this is an informational response, not a decision" attacks the control. Adversarial evals required.

## Evaluation

1. **Compliance and legal author and sign off the criteria before any evaluation.** The criteria text is a compliance artifact.
2. Have counsel label 300 outputs with the correct tier. Report per-class accuracy, with **under-classification rate as the headline metric** — the direction that creates exposure.
3. Verify the matrix is exhaustive over the cross-product of categories, with an explicit strict default. Assert this in a test.
4. Shadow-run against your current routing. Quantify both over-routing eliminated and in-scope cases currently being missed. The second number is usually the one that gets attention.
5. Audit-rehearse: pick 20 historical decisions and produce the full record. If you cannot reconstruct one, the logging is incomplete.
6. In production: tier distribution, manual-triage rate, unmatched-key alerts, augmentation rate, and reviewer agreement with the assigned tier.

## Related

- [P21](P21-brand-policy-gating.md) — routes regulated claims here
- [P20](P20-pii-leakage-detection.md) — routes special-category findings here
- [P02](P02-tool-call-risk-gating.md) — its confirmations may need a specific reviewer, not any human
- [P19](P19-prompt-injection-detection.md) — confirmed attacks route here for incident handling
- [P15](P15-online-output-qa.md) — general output QA; this is the obligation-matching layer
