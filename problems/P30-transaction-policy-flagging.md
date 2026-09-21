# P30 — Transaction & Expense Policy-Violation Flagging

| | |
|---|---|
| **Theme** | F · Back-office, risk & data operations |
| **Primitives** | `noul`, `score`, `choice` |
| **Dominant win** | `📈` coverage — review every transaction instead of sampling — plus `$` |
| **Latency budget** | Batch, or <300 ms inline at submission |
| **Volume profile** | Every expense claim, purchase requisition, invoice line or card transaction |
| **Blast radius if wrong** | Moderate — false positives accuse people of things; false negatives let policy breaches through |

---

## Problem

Expense and procurement policy enforcement is a sampling exercise with a rules engine bolted on. The rules engine catches what is expressible as a threshold — over £75 needs a receipt, over £500 needs approval, no alcohol — and those checks are genuinely useful and genuinely narrow. Everything else relies on a manager glancing at a summary line and an audit team sampling 1–2% after the fact.

What the rules cannot see is most of what matters:

- **Business purpose.** "Client dinner" for four people, no client named, on a Saturday.
- **Splitting.** Two £480 invoices from the same vendor on consecutive days, under a £500 approval threshold.
- **Category misuse.** A £900 "office supplies" claim that describes a piece of personal equipment.
- **Duplicate submission** across different claims, months apart, with reworded descriptions.
- **Personal-use ambiguity.** A hotel stay extended over a weekend, legitimately or not.
- **Vendor anomalies.** A new supplier with an invoice that reads like a template.

Each of these is a **semantic judgement about free-text descriptions in context** — exactly the class of decision that was previously too expensive to make on every transaction, so nobody did.

> **Scope note:** the output here is a review queue, not an accusation. Flags route to a human who decides. Nothing in this design should auto-reject a claim or trigger an accusation of misconduct, and the framing of the language matters: "needs review", never "fraudulent".

## Today's pattern

| Approach | Problem |
|---|---|
| **Threshold rules in the expense system** | Catch the expressible; blind to purpose, splitting, category misuse and duplicates across claims |
| **Manager approval** | Approvers are busy and approve by default; they see a line item, not a pattern |
| **Post-hoc audit sampling (1–2%)** | Finds patterns after the money is spent; deterrence value only |
| **Anomaly detection on amounts** | Statistically finds outliers; cannot judge whether an outlier is *legitimate*, which most are |
| **LLM review of each claim** | Right capability, ~$0.01–0.05 per claim, so it is applied to the top slice only |

## Jev design

Two levels, because the most valuable findings are **patterns across transactions**, not properties of one:

1. **Per-transaction** — purpose, category fit, policy alignment, description quality.
2. **Per-cluster** — splitting and duplicates, judged over a candidate group assembled in code.

Candidate clustering (same vendor + same claimant + near dates, or similar amounts) is deterministic and cheap. Jev judges whether a cluster is what it looks like.

### Level 1 — per transaction

```python
state = {
    "transaction": {
        "amount": t.amount, "currency": t.currency,
        "category": t.claimed_category,
        "description": t.description,
        "vendor": t.vendor_name,
        "date": t.date, "day_of_week": t.weekday,
        "attendees": t.attendees,
        "receipt_text": truncate(t.receipt_ocr, 900),
    },
    "claimant": {"role": c.role, "department": c.dept, "cost_centre": c.cc},
    "policy": POLICY_TEXT_FOR[c.dept, t.claimed_category],
}
```

```python
from typesafe_sdk import Choice, Noul, Score

TXN_QUESTIONS = {
    "purpose_stated": Score(
        instructions="How clearly the description establishes a business purpose",
        criteria=[
            "No business purpose stated",
            "Generic: a category label with no specifics",
            "Some specifics but incomplete",
            "Clear: what, why, and who it was for",
        ],
    ),
    "category_fits": Noul(
        instructions="The claimed category matches what the description and "
                     "receipt actually show was purchased"),
    "policy_consistent": Noul(
        instructions="This transaction is consistent with the policy text provided",
        criteria={"true":  "Consistent with the stated policy",
                  "false": "Inconsistent with, or not permitted by, the stated policy"}),
    "personal_use_signals": Noul(
        instructions="The description, timing or items suggest personal rather "
                     "than business benefit"),
    "receipt_matches": Noul(
        instructions="The receipt text is consistent with the claimed amount, "
                     "vendor and description"),
    "description_evasive": Noul(
        instructions="The description appears deliberately vague or generic given "
                     "the amount involved, as opposed to simply brief"),
    "concern": Choice(
        instructions="The primary concern with this transaction, if any",
        criteria={
            "none":            "No concern",
            "missing_purpose": "Business purpose not established",
            "wrong_category":  "Miscategorised",
            "policy_breach":   "Not permitted by policy",
            "personal_use":    "Appears to be personal benefit",
            "receipt_mismatch":"Receipt does not support the claim",
            "vendor_unusual":  "Vendor is unusual for this category or department",
        },
    ),
}
```

### Level 2 — per cluster

```python
state = {
    "cluster_reason": "same_vendor_same_claimant_within_7d",
    "transactions": [
        {"id": t.id, "amount": t.amount, "date": t.date,
         "description": t.description, "category": t.category} for t in cluster
    ],
    "approval_thresholds": {"requires_approval_above": 500, "currency": "GBP"},
}
```

```python
CLUSTER_QUESTIONS = {
    "relationship": Choice(
        instructions="The relationship between these transactions",
        criteria={
            "unrelated":      "Coincidentally similar; genuinely separate purchases",
            "legitimate_series":"A recurring or instalment arrangement, or a "
                              "genuine series of separate needs",
            "split_purchase": "One purchase divided into parts",
            "duplicate":      "The same expense submitted more than once",
            "cannot_tell":    "Insufficient information to judge",
        },
    ),
    "avoids_threshold": Noul(
        instructions="The amounts appear arranged so that no single transaction "
                     "exceeds the stated approval threshold, where a combined "
                     "transaction would have"),
    "descriptions_inconsistent": Noul(
        instructions="The descriptions differ in ways that would obscure the "
                     "relationship between these transactions"),
}
```

`avoids_threshold` combined with `split_purchase` is the highest-value finding in this whole document. Threshold avoidance is a classic control failure, it is invisible to any per-transaction rule, and it is exactly the pattern that post-hoc audit finds months later.

## Integration

```python
def review(txn):
    a = client.system_one(state=build_state(txn), questions=TXN_QUESTIONS).answers

    # Deterministic rules run FIRST and independently. They are the audit backbone.
    rule_findings = policy_rules_engine(txn)

    score = (0.30 * (1 - a["purpose_stated"].score / 3.0)
           + 0.20 * (1 - a["category_fits"].noul)
           + 0.20 * (1 - a["policy_consistent"].noul)
           + 0.15 * a["personal_use_signals"].noul
           + 0.15 * a["description_evasive"].noul)

    if a["receipt_matches"].noul < 0.4:
        score = max(score, 0.70)                 # receipt mismatch is concrete

    if rule_findings or score > 0.55:
        return Queue("review", reasons=rule_findings + [a["concern"].choice],
                     score=score, evidence=a)
    if score > 0.35:
        return Queue("approver_note", note=advisory_note(a))   # flag to the approver
    return Pass


def review_clusters(claimant, window):
    for cluster in candidate_clusters(claimant, window):        # code, deterministic
        c = client.system_one(state=cluster_state(cluster),
                              questions=CLUSTER_QUESTIONS).answers
        rel = c["relationship"]

        if rel.choice == "duplicate" and rel.confidence > 0.7:
            Queue("duplicate_review", cluster)
        elif rel.choice == "split_purchase" and c["avoids_threshold"].noul > 0.6:
            Queue("controls_review", cluster, priority="high")  # threshold avoidance
        elif rel.choice == "cannot_tell" and total(cluster) > THRESHOLD:
            Queue("review", cluster)                            # uncertainty on a
                                                                # material amount
```

Three design commitments:

- **The rules engine runs independently and first.** It is deterministic, auditable, and defensible to an auditor in a way a model score is not. Jev extends coverage; it does not replace the control.
- **A graduated `approver_note` tier** puts advisory context in front of the approver rather than opening a formal review. Most borderline cases are best resolved by the approver asking one question.
- **Amount arithmetic is code.** Whether transactions sum past a threshold, whether a claim exceeds a per-diem, whether dates fall in a period — all deterministic ([JEV-details §11](../JEV-details.md#11-where-jev-does-not-fit)). Jev judges relationships and purposes; your code does the sums.

## Thresholds & escalation

| Signal | Threshold | Action |
|---|---|---|
| Any deterministic rule finding | exact | Review queue, always |
| `receipt_matches` | < 0.4 | Score floor 0.70 — concrete and checkable |
| Composite concern score | > 0.55 | Review queue |
| Composite concern score | 0.35–0.55 | Advisory note to the approver |
| `duplicate`, conf > 0.7 | — | Duplicate review |
| `split_purchase` + `avoids_threshold > 0.6` | — | Controls review, high priority |
| `cannot_tell` on a material cluster | — | Review — uncertainty on real money gets a human |

Thresholds should be set from your review capacity. A queue larger than the team can work is not a control.

## Impact model

*Illustrative.* 400k transactions/month plus ~40k clusters.

```
Jev, 100%           400k × $0.00009 + 40k × $0.00012 = $  41/month
LLM, 100%           400k × $0.020                    = $8,000/month
audit sampling      1.5% coverage, months after the fact
```

The value is coverage and timing. Moving from 1.5% post-hoc sampling to 100% at submission means:

- Findings arrive **before payment**, when they are correctable rather than recoverable.
- Threshold-avoidance and cross-claim duplicate patterns become detectable at all.
- The deterrence effect is real: employees knowing every claim is reviewed changes submission behaviour, which is a larger effect than the recoveries.

**Do not put a recovery figure in the business case unless you have measured it.** Published benchmarks for expense-fraud rates vary enormously and most are vendor-sourced. Measure your own baseline from an audit sample first, then quantify.

## Failure modes

- **False positives accuse people.** The most important risk, and it is cultural. An employee wrongly flagged for a legitimate claim remembers it. Mitigations: framing language ("needs review"), the advisory tier, human decision always, and a fast, logged appeal path. Track false-flag rate as a headline metric and report it to whoever owns employee experience.
- **Legitimate patterns that look like splitting.** Monthly retainers, instalment purchases, genuinely separate same-vendor needs. `legitimate_series` exists for this; validate it hard, because this is where the queue fills with noise.
- **Policy text quality determines everything.** `policy_consistent` is only as good as the `POLICY_TEXT_FOR` entry. Vague policy produces vague judgements. This project will surface that your written policy is ambiguous — treat that as a useful finding rather than a blocker.
- **No arithmetic.** Repeating because it is the common error: sums, thresholds, per-diems, date ranges are all code.
- **Cultural and regional variation.** Expense norms differ by country and by business unit. Calibrate per region; a single rubric across a multinational will systematically over-flag some populations, which is both unfair and a compliance issue in itself.
- **Gaming.** Claimants learn what descriptions pass. Rotate emphasis, and monitor for description patterns that are suspiciously well-formed.
- **Injection.** Invoice or receipt text containing content aimed at the reviewer — plausible in vendor-submitted documents. Keep rules in code; treat `policy_consistent` as advisory.

## Evaluation

1. **Backtest against audit findings.** Take historical transactions where an audit reached a conclusion. Report recall on confirmed findings and, critically, **false-positive rate on confirmed-legitimate transactions.**
2. Have finance and internal audit review 300 flags blind and rate each as useful / not useful. Usefulness, not accuracy, is the metric that determines whether this survives.
3. Evaluate cluster judgements separately and carefully — `legitimate_series` versus `split_purchase` is the hardest distinction here and the one with the most reputational downside.
4. Have internal audit sign off the criteria text before deployment; it is a control document.
5. Pilot on one department before going organisation-wide, and involve HR on the employee-communication side.
6. In production: flag rate, queue depth against capacity, confirmed-finding rate, false-flag rate, appeal outcomes, and submission-behaviour change over time (the deterrence signal).

## Related

- [P28](P28-document-classification-idp.md) — upstream: extracts the fields this judges
- [P29](P29-soc-alert-triage.md) — same alert-triage economics, security domain
- [P22](P22-regulated-review-routing.md) — where findings with employment consequences must route
- [P27](P27-conversation-qa-scoring.md) — shares the "scoring people's work" governance problem
- [P20](P20-pii-leakage-detection.md) — expense data is personal data; handle accordingly
