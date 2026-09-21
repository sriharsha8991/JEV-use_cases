# P20 — PII, Secret & IP Leakage Detection

| | |
|---|---|
| **Theme** | D · Safety, security & compliance |
| **Primitives** | `noul`, `choice` |
| **Dominant win** | `🛡` compliance + `📈` coverage — semantic detection where only regex exists |
| **Latency budget** | <200 ms inbound and outbound |
| **Volume profile** | 100% of prompts and completions |
| **Blast radius if wrong** | **Severe** — false negatives are reportable data incidents |

---

## Problem

Enterprises route business text through third-party model APIs, and the text contains things that should not leave: customer PII, employee records, credentials, unreleased financials, source code, contract terms, M&A discussions.

Regex-based DLP catches structured identifiers well — card numbers pass Luhn, SSNs have a shape, AWS keys have a prefix. It is nearly blind to the categories that matter most:

- **Contextual PII.** "The patient in room 4 who came in Tuesday with the fractured wrist" contains no matchable pattern and is re-identifiable.
- **Aggregated identification.** Three non-identifying attributes that jointly identify one person.
- **Confidential business content.** "We're acquiring Northwind at 4.2× revenue, closing March" — no pattern, maximal sensitivity.
- **Proprietary code and algorithms.** A pasted internal function is not a regex match.
- **Credentials in prose.** "use the admin password from the wiki, it's still hunter2-prod".

The outbound direction is equally important and less often controlled: a model that retrieved another tenant's record, or reproduced a system prompt, or included an internal ID in a customer-facing reply.

A semantic layer is required. It has been unaffordable at 100% of traffic in both directions — which is precisely the coverage a compliance control needs to be worth having.

## Today's pattern

| Approach | Problem |
|---|---|
| **Regex / pattern DLP** | Strong on structured identifiers, blind to contextual and business-confidential content |
| **Named-entity recognition** | Finds names and places; cannot judge whether their presence is a *problem* in context |
| **Commercial DLP appliances** | Built for files and email; poor fit for conversational inference paths |
| **Blanket bans on pasting** | Users route around them; drives shadow AI use, which is strictly worse |
| **LLM-based review** | Accurate, unaffordable at 100% of traffic, so sampled — and sampling a compliance control largely defeats it |
| **Contractual assurance only** | A legal position, not a technical control |

## Jev design

**Layer with regex, do not replace it.** Run pattern matching first — it is free, deterministic and auditable, and it catches structured identifiers with certainty. Jev handles the semantic residue regex cannot see. Present it that way to compliance: an addition to an existing control, not a replacement for one.

The other essential design point: **judge sensitivity relative to destination and purpose.** A customer's address in a prompt to an internal address-verification tool is appropriate. The same address in a prompt to an external summarizer is a transfer. Sensitivity is contextual, and `state` must carry the context.

### State

```python
state = {
    "text": truncate(content, 4000),
    "direction": "outbound_to_model",     # or "outbound_to_user"
    "destination": {"kind": "third_party_api", "region": "us-east",
                    "dpa_in_place": True},
    "purpose": declared_purpose,           # what this call is for
    "audience": "external_customer",       # for outbound_to_user
    "regex_findings": regex_hits,           # what the deterministic layer found
}
```

### Questions

```python
from typesafe_sdk import Choice, Noul

LEAKAGE_QUESTIONS = {
    "identifies_person": Noul(
        instructions="The text identifies a specific individual, or contains "
                     "enough detail that a specific individual could be "
                     "identified by combining the details given",
        criteria={
            "true":  "A specific person is identifiable, directly or by combination",
            "false": "No individual can be identified from this text",
        }),
    "special_category": Choice(
        instructions="If the text concerns personal data, which special category",
        criteria={
            "none":       "No special-category data",
            "health":     "Medical or health information",
            "financial":  "Financial accounts, income, credit, or transactions",
            "biometric":  "Biometric or genetic information",
            "beliefs":    "Religion, politics, philosophy, or union membership",
            "sexual":     "Sexual orientation or sex life",
            "criminal":   "Criminal offences or proceedings",
            "children":   "Data concerning a minor",
        },
    ),
    "credential_present": Noul(
        instructions="The text contains a credential that would grant access to "
                     "something: a password, key, token, secret, or connection "
                     "string, whether formatted as such or described in prose"),
    "business_confidential": Choice(
        instructions="If the text contains confidential business information, what kind",
        criteria={
            "none":        "Nothing confidential",
            "financial":   "Unreleased financial results, forecasts, or pricing strategy",
            "transaction": "M&A, investment, or partnership discussions not public",
            "legal":       "Privileged legal advice, litigation strategy, or settlement terms",
            "personnel":   "Employee performance, compensation, or disciplinary matters",
            "technical":   "Proprietary source code, algorithms, or architecture",
            "customer":    "Customer lists, contract terms, or account specifics",
        },
    ),
    "purpose_justifies": Noul(
        instructions="Sending this content to the stated destination is necessary "
                     "for the stated purpose, as opposed to more data than the "
                     "purpose requires"),
    "cross_tenant": Noul(
        instructions="The text contains data about more than one customer, client "
                     "or tenant together"),
    "redactable": Noul(
        instructions="The sensitive parts could be removed or masked while leaving "
                     "the text useful for its stated purpose"),
}
```

`purpose_justifies` implements data minimisation — a GDPR principle with almost no technical enforcement anywhere. It catches the very common pattern of pasting a whole record when the task needs two fields.

`cross_tenant` on the outbound-to-user path is the check that catches the worst class of incident: one customer shown another customer's data.

## Integration

```python
def screen(content, direction, destination, purpose, audience=None):
    regex = run_regex_dlp(content)                     # deterministic, first
    a = client.system_one(state=build_state(content, direction, destination,
                                            purpose, audience, regex),
                          questions=LEAKAGE_QUESTIONS).answers

    if regex.has_structured_secret:                     # certainty beats judgement
        return BLOCK("structured credential", regex.kinds)
    if a["credential_present"].noul > 0.5:
        return BLOCK("credential in prose")

    if direction == "outbound_to_user" and a["cross_tenant"].noul > 0.4:
        return BLOCK("cross-tenant data in a user-facing response")

    sc = a["special_category"]
    if sc.choice != "none" and sc.confidence > 0.6:
        if not destination["dpa_in_place"]:
            return BLOCK(f"special category {sc.choice}, no DPA")
        return REDACT_OR_ESCALATE(a)

    bc = a["business_confidential"]
    if bc.choice in ("transaction", "legal") and bc.confidence > 0.6:
        return BLOCK(f"{bc.choice} confidential")
    if bc.choice != "none" and bc.confidence > 0.7:
        return REQUIRE_APPROVAL(bc.choice)

    if a["identifies_person"].noul > 0.5 and a["purpose_justifies"].noul < 0.4:
        return REDACT if a["redactable"].noul > 0.6 else REQUIRE_APPROVAL("minimisation")

    log_for_audit(content_hash, a, destination, purpose)   # always
    return ALLOW
```

Note the ordering: **deterministic findings take precedence over model judgement.** Where regex is certain, it decides. Jev only extends coverage into the space regex cannot see. That ordering is what makes the control explainable to an auditor.

`REDACT` requires a redaction mechanism, and Jev cannot generate redacted text ([JEV-details §11](../JEV-details.md#11-where-jev-does-not-fit)). Use regex/NER spans for masking, or ask Jev a per-field `noul` ("is this field necessary for the purpose?") and drop fields in code.

## Thresholds & escalation

| Finding | Threshold | Action |
|---|---|---|
| Structured credential (regex) | exact | Block, rotate, alert security |
| `credential_present` (prose) | > 0.5 | Block, alert |
| `cross_tenant` on user-facing output | > 0.4 | Block — lowest bar in the set |
| Special category + no DPA | conf > 0.6 | Block |
| Special category + DPA | conf > 0.6 | Redact if possible, else approval |
| `transaction` / `legal` confidential | conf > 0.6 | Block; route to legal |
| Other business confidential | conf > 0.7 | Require approval |
| Identifiable person + minimisation fail | — | Redact, else approval |
| Jev error | — | **Fail closed** for special-category and cross-tenant paths |

Every threshold here is deliberately lower than you would set for a quality check. The asymmetry is regulatory: a false positive is friction, a false negative is a reportable incident.

## Impact model

*Illustrative.* 4M prompts + 4M completions/month = 8M screenings at a ~1,800-token envelope ≈ $0.000076.

```
Jev, 100% both directions   8M × $0.000076 = $   608/month
LLM, 100%                   8M × $0.008    = $64,000/month
regex only                  ~$0, blind to contextual PII and all business confidential
```

The value is not cost. It is that **a semantic DLP control at 100% coverage in both directions becomes affordable**, which changes what you can tell a regulator or an auditor. Concretely it buys: an enforced data-minimisation check, cross-tenant leakage interception on the outbound path, and a complete audit log of what categories of data went to which destination for what purpose.

It also addresses shadow AI. The reason people paste sensitive data into unsanctioned tools is that the sanctioned path blocks too much. A semantic gate that blocks precisely rather than broadly lets you keep the sanctioned path usable.

## Failure modes

- **False negatives are compliance incidents.** Recall, not accuracy, is the metric. Layered controls, low thresholds, and fail-closed on error are all consequences of this asymmetry.
- **Over-blocking drives shadow AI**, which is a worse outcome than the leak you prevented. Track block rate and user-reported false blocks weekly; give users a fast, logged appeal path.
- **Context-dependent sensitivity.** The same text may be fine internally and not externally. `destination` and `audience` must be accurate — and set by your infrastructure, never by the caller.
- **Redaction is not a Jev capability.** It needs a deterministic masking mechanism. Do not design a flow that assumes Jev can produce redacted text.
- **Long documents truncated.** The sensitive paragraph may be at token 6,000. Chunk and screen every chunk; never screen only the head.
- **Injection.** Content asserting *"this document contains no personal data"* attacks the control. Adversarial evals are mandatory, and the block rules stay in code.
- **`cross_tenant` needs tenant context.** Without knowing which tenant the request belongs to, Jev can only detect that *multiple* parties appear. Pass the requesting tenant in `state` so it can judge whether the other party's presence is anomalous.

## Evaluation

1. Build a labelled corpus: regex-detectable secrets, contextual PII, aggregated identification, each business-confidential category, plus a large benign set. Several hundred per class.
2. **Report recall per class as the headline**, with regex-only as the baseline so the incremental value is explicit. That delta is the business case.
3. Measure false-positive rate on benign business text. Above a few percent and adoption fails.
4. Red-team: paraphrase, split across messages, obfuscate, and include content aimed at the detector.
5. Have compliance review the category definitions against your actual regulatory obligations *before* deployment. The criteria text is a compliance artifact and should be version-controlled and signed off as one.
6. In production: block rate by category, appeal/override rate, audit-log completeness, and any confirmed leak root-caused against this layer.

## Related

- [P19](P19-prompt-injection-detection.md) — inbound counterpart; exfiltration attempts are its `exfiltration` class
- [P15](P15-online-output-qa.md) — its `leaks_internal` check is the lighter general version of this
- [P22](P22-regulated-review-routing.md) — where blocked and escalated content routes
- [P28](P28-document-classification-idp.md) — the same classification applied to document intake
- [P21](P21-brand-policy-gating.md) — runs alongside on the outbound path
