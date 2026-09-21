# P26 — Self-Service Deflection Eligibility

| | |
|---|---|
| **Theme** | E · Customer & revenue operations |
| **Primitives** | `noul`, `choice` |
| **Dominant win** | `$` cost — the largest per-unit saving in this catalogue, because the alternative is a human |
| **Latency budget** | <200 ms at intake |
| **Volume profile** | Every inbound contact |
| **Blast radius if wrong** | **High on false positives** — a wrongly deflected customer is a worse outcome than an unnecessary human touch |

---

## Problem

Self-service deflection is the highest-value decision in support economics — a deflected contact costs cents, a handled one costs dollars — and it is the one organisations most consistently get wrong, in the direction that damages customers.

The standard implementation is "show help articles before you let them open a ticket", driven by keyword search. Customers experience it as an obstacle course. They have usually already searched the documentation; being shown the same three articles again, then made to click "none of these helped", is the single most reliable way to convert a mild problem into a complaint.

The result is that deflection is measured as a success metric while being experienced as a failure. Deflection rate goes up, CSAT goes down, and the two numbers are reported to different people.

The real question is harder than "does an article match this topic":

1. Does content exist that **fully** answers this, not merely relates to it?
2. Has this customer **already tried** self-service?
3. Is this customer in a state where being deflected is acceptable?
4. Is there an **account-specific** element that no article can address?

All four are judgements. The fourth is where generic deflection fails most often — "why was I charged $340" cannot be answered by a pricing page.

## Today's pattern

| Approach | Problem |
|---|---|
| **Keyword article search before ticket creation** | Topical, not answer-bearing. Ignores whether the customer already looked |
| **Mandatory chatbot gauntlet** | Universally disliked; converts small problems into escalations |
| **Deflection rate as a target** | Optimises the wrong variable. Gaming it harms customers |
| **Category-based rules** ("billing questions → billing FAQ") | Ignores account-specific content entirely |
| **LLM deflection decision** | Accurate, ~$0.01 and several seconds; acceptable on cost, and still usually deployed without the "already tried" signal |

## Jev design

Reframe the question. Not *"is there an article about this?"* but **"would deflecting this specific customer right now be a good outcome for them?"** That reframing is the design, and it is what the `state` must support.

### State

```python
state = {
    "request": contact_text,
    "kb_candidates": [                       # top retrieval hits, NOT a decision yet
        {"title": a.title, "summary": truncate(a.summary, 400), "url": a.url}
        for a in kb_search(contact_text)[:5]
    ],
    "customer_journey": {
        "kb_pages_viewed_24h": [p.title for p in journey.kb_views],
        "prior_contacts_7d": journey.contact_count,
        "reopened_recently": journey.has_reopen,
    },
    "account": {"tier": acct.tier, "has_account_specific_issue_open": acct.open_issue},
}
```

`customer_journey` is the difference between a deflection system customers tolerate and one they resent. If they viewed the very article you are about to suggest, suggesting it is an insult.

### Questions

```python
from typesafe_sdk import Choice, Noul

DEFLECTION_QUESTIONS = {
    "fully_answered": Noul(
        instructions="One of the candidate articles completely answers this "
                     "request, such that the customer would need nothing further",
        criteria={
            "true":  "A candidate fully resolves the request as stated",
            "false": "Every candidate is partial, topical, or requires the "
                     "customer to work out how it applies to them",
        }),
    "best_candidate": Choice(
        instructions="Which candidate article best addresses the request",
        criteria={},                          # filled at runtime from kb_candidates
    ),
    "account_specific": Noul(
        instructions="Answering requires looking at this customer's own data: "
                     "their charges, their configuration, their account state, "
                     "or their specific records"),
    "already_tried": Noul(
        instructions="The customer indicates they have already consulted "
                     "documentation or attempted the standard fix, or the pages "
                     "they viewed cover what the candidates say"),
    "action_required": Noul(
        instructions="Resolving this requires someone with permissions the "
                     "customer does not have: a refund, an override, a "
                     "configuration change, or access to a restricted system"),
    "emotional_state": Choice(
        instructions="The customer's emotional state",
        criteria={"neutral": "Matter-of-fact",
                  "impatient": "In a hurry",
                  "frustrated": "Clearly frustrated",
                  "distressed": "Upset or anxious"},
    ),
    "complexity": Choice(
        instructions="How involved the resolution is",
        criteria={"single_step": "One action or one fact",
                  "few_steps": "A short procedure",
                  "multi_stage": "A long or conditional procedure",
                  "diagnostic": "Requires investigation to determine the cause"},
    ),
}
```

`already_tried` is the single most important question in this document, and it is the one nobody asks. It is also cheap to answer well, because the journey data is already in `state`.

## Integration

```python
def decide(contact, journey, account):
    a = client.system_one(state=build_state(contact, journey, account),
                          questions=DEFLECTION_QUESTIONS).answers

    # Hard no-deflect conditions, checked first.
    if a["account_specific"].noul > 0.4:   return Human("needs account data")
    if a["action_required"].noul > 0.4:    return Human("needs permissions")
    if a["already_tried"].noul > 0.5:      return Human("self-service exhausted")
    if a["emotional_state"].choice in ("frustrated", "distressed"):
        return Human("emotional state")
    if journey.contact_count >= 2:         return Human("repeat contact")

    # Deflect only on a high bar, and never silently.
    if (a["fully_answered"].noul > 0.80
            and a["best_candidate"].confidence > 0.70
            and a["complexity"].choice in ("single_step", "few_steps")):
        return Deflect(
            article=a["best_candidate"].choice,
            # The escape hatch is mandatory, prominent, and one click.
            escape="This didn't answer my question — talk to someone",
            on_escape=Human("deflection rejected"),
        )

    return Human("no confident self-service answer")
```

Note the structure: **five hard no-deflect conditions checked before any deflection is considered**, and deflection itself requires three conditions to hold simultaneously. The thresholds are deliberately high (0.80, 0.70) and every no-deflect bar is deliberately low (0.4, 0.5).

That asymmetry is the entire design, and it follows from the cost structure: a false negative costs one human touch — a few dollars — while a false positive costs a frustrated customer who now has two problems. Optimise for precision, never for deflection rate.

The escape hatch is not optional. One click, prominently placed, no "are you sure". A deflection you cannot easily reject is a wall.

## Thresholds & escalation

| Condition | Threshold | Action |
|---|---|---|
| `account_specific` | > 0.4 | Human — no article can answer |
| `action_required` | > 0.4 | Human — customer lacks permissions |
| `already_tried` | > 0.5 | Human — never re-show what they read |
| `emotional_state ∈ {frustrated, distressed}` | — | Human, regardless of content match |
| Prior contacts in 7d | ≥ 2 | Human |
| `fully_answered` | > 0.80 | Necessary but not sufficient |
| `best_candidate.confidence` | > 0.70 | Required alongside |
| `complexity` | single/few steps | Required alongside |
| Anything else | — | Human |

## Impact model

*Illustrative.* 500k contacts/month at ~$6 fully-loaded per handled contact.

```
no deflection        500k × $6           = $3,000,000/month
Jev decision         500k × $0.00012     = $        60/month
deflect 18%          90k × $0.05         = $     4,500
handle 82%           410k × $6           = $ 2,460,000
                                           ───────────
                                           $ 2,464,560  → ~$535k/month saved
```

18% is a deliberately conservative figure. The point is not to maximise it: **a 15% deflection rate with high precision is worth more than a 35% rate that generates complaints**, because the complaint path costs more than the original contact and damages the relationship. Report deflection rate and deflection-rejection rate together, always, to the same audience.

Compared with a keyword system, the gain is not volume — keyword deflection often shows a *higher* rate — it is that the deflections are correct. The metric to move is rejection rate down, not rate up.

## Failure modes

- **Optimising the deflection rate.** The central risk, and it is organisational rather than technical. If the metric is deflection rate, thresholds will be loosened until customers suffer. Instrument rejection rate and post-deflection CSAT as co-equal metrics from day one.
- **Stale or missing journey data.** Without `kb_pages_viewed_24h`, `already_tried` degrades to what the customer explicitly says, which is a fraction of reality. Instrument the journey before deploying the deflection.
- **Article quality.** Confident deflection to a badly written article is still a bad outcome. `fully_answered` judges the summary, not the reader's experience of the full page. Audit the articles you deflect to most.
- **Account-specific questions phrased generically.** "How do refunds work" may really mean "where is my refund". Keep the `account_specific` bar low and accept the false positives.
- **Emotional state misread across cultures and languages.** Validate per locale.
- **Gaming by customers** who learn that saying "I already read the docs" reaches a human faster. This is fine — honouring it costs one contact, and the alternative is worse.
- **Content drift.** New product, no articles; deflection rate silently falls. That is the system working, but monitor it so you notice the content gap ([P13](P13-knowledge-conflict-detection.md)).

## Evaluation

1. **Shadow first, and grade the counterfactual.** Log the decision without acting for two weeks. For every would-be deflection, check what the human agent actually did: if they sent the same article, the deflection was correct; if they looked up account data, it would have been a failure. This is a clean, cheap validation.
2. Report **precision on the deflect class** as the headline. Recall is almost irrelevant here.
3. A/B against the incumbent system with **rejection rate, post-deflection CSAT, and re-contact-within-48h** as primary metrics — not deflection rate.
4. Validate `already_tried` specifically against journey logs; it is the newest signal and the one carrying most of the improvement.
5. In production: deflection rate, rejection rate, re-contact rate, post-deflection CSAT, and per-article rejection rate (which identifies bad articles precisely).

## Related

- [P24](P24-ticket-triage.md) — triggers this at intake
- [P08](P08-llm-admission-control.md) — the same idea one layer down, where the alternative is an LLM rather than a human
- [P09](P09-full-recall-reranking.md) — better candidate retrieval raises the ceiling on `fully_answered`
- [P13](P13-knowledge-conflict-detection.md) — content gaps this exposes
- [P25](P25-escalation-churn-detection.md) — catches the customer a bad deflection produced
