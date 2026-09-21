# P23 — User-Generated Content Moderation at Publish Time

| | |
|---|---|
| **Theme** | D · Safety, security & compliance |
| **Primitives** | `noul`, `choice`, `score` |
| **Dominant win** | `⚡` latency — pre-publish moderation is only possible at ~100 ms — plus `$` |
| **Latency budget** | **<200 ms hard.** It is in the publish path; a user is waiting |
| **Volume profile** | Every post, comment, message, review or profile edit |
| **Blast radius if wrong** | High in both directions — false negatives publish harm, false positives suppress legitimate speech |

---

## Problem

Any platform with user-generated content — marketplace, community, review site, internal social, support forum, collaboration tool — faces the same structural choice: moderate **before** publication and make users wait, or moderate **after** and accept that harmful content is live for minutes or hours.

Almost everyone chooses post-publication, because pre-publication moderation with an LLM means a multi-second delay on every post. That is not a product anyone ships. So the standard architecture is: publish immediately, run async classifiers, remove later. Harm happens in the window, and for the worst categories — self-harm content, targeted harassment, scam solicitations, doxxing — the window is where the damage is done.

The secondary problem is the review queue. Hard cases need humans; the queue is always over capacity; and it is filled largely by a keyword classifier's false positives, so reviewers spend their attention on noise while genuinely borderline cases wait.

At ~100 ms, pre-publish moderation becomes viable for the first time. That is a change in what the product can be, not an optimization of what it already does.

## Today's pattern

| Approach | Problem |
|---|---|
| **Keyword/regex lists** | Trivially evaded; heavy false positives on reclaimed language and quotation; no severity notion |
| **Dedicated moderation classifiers** (Perspective, OpenAI moderation, in-house) | Fast and decent on explicit categories; weaker on context, coded language, and platform-specific policy; fixed taxonomy you cannot extend |
| **Async LLM moderation post-publish** | Good accuracy, harm window open, and expensive at volume |
| **Full human pre-moderation** | Only viable at low volume |
| **User reporting** | Reactive by construction; the harm already landed |

## Jev design

One call, one `noul` per policy, plus severity and a recommended action (§4.1). The per-policy decomposition is the critical design choice: **each policy gets its own threshold, its own precision/recall curve, and its own appeal statistics.** A single "is this bad?" score cannot be tuned — you cannot loosen spam detection without loosening harassment detection.

Include context, because most moderation errors are context errors: quoting abuse to report it, reclaimed slurs, fiction, medical discussion, and the difference between describing a scam and running one.

### State

```python
state = {
    "content": post_text,
    "context": {
        "parent": truncate(parent_post, 600) if parent_post else None,
        "surface": "public_review",           # public | group | dm | profile
        "community_norms": NORMS_FOR[surface],
    },
    "author_signals": {                        # behavioural, not identity
        "account_age_days": a.age_days,
        "prior_actions": a.prior_action_count,
        "posts_last_hour": a.recent_rate,
    },
}
```

`author_signals` are behavioural aggregates computed in code. They sharpen the spam and coordinated-behaviour judgements substantially without putting identity into the model.

### Questions

```python
from typesafe_sdk import Choice, Noul, Score

MODERATION_QUESTIONS = {
    "harassment": Noul(
        instructions="This content attacks, demeans or threatens a specific person "
                     "or an identifiable group, as opposed to criticising an idea, "
                     "a product or an organisation"),
    "threat": Noul(
        instructions="This content threatens violence or physical harm against "
                     "someone, stated seriously rather than as hyperbole or fiction"),
    "self_harm_risk": Noul(
        instructions="The author appears to be at risk of harming themselves, or "
                     "is seeking means or encouragement to do so"),
    "scam_or_fraud": Noul(
        instructions="This content solicits money, credentials, gift cards or "
                     "off-platform contact in a way consistent with a scam"),
    "spam": Noul(
        instructions="This content is unsolicited promotion, repetitive flooding, "
                     "or automated posting rather than genuine participation"),
    "doxxing": Noul(
        instructions="This content publishes private information about someone "
                     "without evident consent: address, phone number, workplace, "
                     "or other identifying details"),
    "sexual_minors": Noul(
        instructions="This content sexualises a minor in any way"),
    "illegal_goods": Noul(
        instructions="This content offers or seeks goods or services that are "
                     "illegal to trade"),
    "is_quotation": Noul(
        instructions="Any offensive content here is quoted, reported or described "
                     "in order to criticise, report or discuss it, rather than "
                     "being the author's own expression"),
    "severity": Score(
        instructions="Overall severity of harm this content would cause if published",
        criteria=["None: no harm", "Low: mildly unpleasant",
                  "Medium: would upset or mislead someone",
                  "High: would seriously harm someone"],
    ),
    "recommended_action": Choice(
        instructions="The proportionate action for this content",
        criteria={"allow": "Publish normally",
                  "label": "Publish with a warning or reduced distribution",
                  "hold": "Hold for human review before publishing",
                  "block": "Do not publish",
                  "escalate": "Do not publish and escalate urgently"},
    ),
}
```

`is_quotation` is the highest-value false-positive control in this set. "Someone DMed me saying [slur], is this allowed?" is a report, not an attack, and keyword systems get it wrong every single time.

`self_harm_risk` is deliberately framed around the *author's* wellbeing rather than content policy, because its action path is support resources and urgent routing, not suppression.

## Integration

```python
BLOCK_NOW = {"sexual_minors": 0.35, "threat": 0.55, "doxxing": 0.5}

def moderate(content, ctx, author):
    a = client.system_one(state=build_state(content, ctx, author),
                          questions=MODERATION_QUESTIONS).answers

    # Highest-severity categories: block on low thresholds, no quotation exemption
    # for the most serious category.
    if a["sexual_minors"].noul > BLOCK_NOW["sexual_minors"]:
        return Action("block", escalate="legal_urgent", preserve_evidence=True)

    if a["self_harm_risk"].noul > 0.4:
        # Not a suppression path. Publish or hold per policy, and surface resources.
        return Action(a["recommended_action"].choice,
                      attach=SUPPORT_RESOURCES, notify="wellbeing_team")

    quoted = a["is_quotation"].noul > 0.6
    for policy, bar in BLOCK_NOW.items():
        if policy == "sexual_minors":
            continue
        if a[policy].noul > bar and not quoted:
            return Action("block", reason=policy, appeal=True)

    if a["scam_or_fraud"].noul > 0.5 or a["illegal_goods"].noul > 0.5:
        return Action("hold", reason="fraud_review")

    if a["harassment"].noul > 0.5 and not quoted:
        return Action("label" if a["severity"].score < 2.0 else "hold",
                      reason="harassment", appeal=True)

    if a["spam"].noul > 0.6 and author.account_age_days < 7:
        return Action("hold", reason="spam_new_account")

    rec = a["recommended_action"]
    if rec.confidence < 0.55:
        return Action("hold", reason="low_confidence")     # humans get the hard cases
    return Action(rec.choice)
```

Three structural points:

- **Graduated actions.** `label` and `hold` exist so that borderline content is neither published unmoderated nor blocked outright. Most real content is borderline, and a binary system is wrong on it either way.
- **Low confidence routes to humans** — which means the review queue now contains genuinely hard cases rather than keyword false positives. Same reviewer headcount, far better applied.
- **Every block carries an appeal path**, and every action is logged with its distributions. Under the DSA and comparable regimes, statement-of-reasons and appeal are obligations, not features.

## Thresholds & escalation

| Policy | Threshold | Action | Quotation exemption |
|---|---|---|---|
| `sexual_minors` | > 0.35 | Block + legal escalation + preserve evidence | **No** |
| `threat` | > 0.55 | Block, appeal available | Yes |
| `doxxing` | > 0.50 | Block, appeal available | Yes |
| `self_harm_risk` | > 0.40 | Attach resources, notify wellbeing; do not suppress | n/a |
| `scam_or_fraud` | > 0.50 | Hold for review | Yes |
| `harassment` | > 0.50 | Label if severity < 2, else hold | Yes |
| `spam` | > 0.60 | Hold if account < 7 days, else label | n/a |
| `recommended_action` conf | < 0.55 | Hold | — |

Set each threshold from *its own* precision/recall curve and the asymmetry of its harm. That per-policy tuning is only possible because the policies are separate questions.

## Impact model

*Illustrative.* 20M posts/month at a ~900-token envelope ≈ $0.000038.

```
Jev, 100% pre-publish   20M × $0.000038 = $   760/month, ~100 ms
LLM, 100% pre-publish   20M × $0.004    = $80,000/month, ~2 s   (not shippable)
LLM, async post-publish 20M × $0.004    = $80,000/month + harm window
keyword + async LLM     cheap, heavy false positives, harm window
```

The product change is the point: **the harm window closes.** Content that would have been live for 20 minutes is never published. For the top-severity categories that is the entire objective, and no async architecture can achieve it at any price.

Secondary: the human review queue is now populated by low-confidence and genuinely borderline cases. Reviewer capacity is unchanged and its value goes up substantially.

## Failure modes

- **Both error directions are serious.** False negatives publish harm; false positives suppress legitimate speech and generate appeals. This is why thresholds are per-policy and why `label`/`hold` exist between allow and block.
- **Context and coded language.** Communities develop in-group language that reads as hostile externally, and abusers develop coded language that reads as benign. `community_norms` in `state` helps; expect to refresh it and to retain human review for drift.
- **Quotation exemption is abusable.** "Someone said [slur], can you believe it" as a delivery mechanism. Keep it as a *threshold shift*, never a bypass, and never for the top category.
- **Language coverage.** Validate per language. Do not assume performance transfers; report per-language metrics.
- **Injection.** Content addressing the moderator directly. Because this is a control, treat it adversarially: keep thresholds in code, and red-team it.
- **Appeal volume.** Pre-publish blocking generates appeals synchronously, because the user is right there. Staff and instrument the appeal path *before* launch — overturn rate per policy is also your best live precision estimate.
- **Reviewer welfare.** Routing the hardest content to humans concentrates exposure to the worst material. That is a real operational duty, not a footnote.

## Evaluation

1. Label 2,000 items per major policy against your actual written policy — not a generic taxonomy. Include the hard classes deliberately: quotation, reclaimed language, fiction, medical discussion, satire.
2. Report per-policy precision/recall curves and pick thresholds from them explicitly, with the harm asymmetry documented per policy.
3. Benchmark against your incumbent classifier on the same set. You need to know the incremental value per policy; keep the incumbent where it wins.
4. Measure per-language performance separately.
5. Red-team evasion: obfuscation, coded language, and content aimed at the moderator.
6. In production: per-policy action rates, appeal and overturn rates (overturn rate is your live precision proxy), human-queue size and composition, and time-to-action. Track the harm-window metric explicitly as the before/after for the whole project.

## Related

- [P15](P15-online-output-qa.md) — the same gating shape applied to AI-generated rather than user content
- [P19](P19-prompt-injection-detection.md) — UGC is an injection vector when it later enters a RAG index
- [P20](P20-pii-leakage-detection.md) — `doxxing` overlaps; share definitions
- [P22](P22-regulated-review-routing.md) — statement-of-reasons and appeal obligations
- [P25](P25-escalation-churn-detection.md) — related sentiment/severity mechanics
