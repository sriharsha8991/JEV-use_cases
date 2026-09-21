# P29 — Security Alert & Incident Triage in the SOC

| | |
|---|---|
| **Theme** | F · Back-office, risk & data operations |
| **Primitives** | `choice`, `score`, `noul` |
| **Dominant win** | `⚡` latency + `$` — triage at alert-arrival rate rather than analyst rate |
| **Latency budget** | <300 ms per alert |
| **Volume profile** | Every alert from every detection source; tens of thousands per day is normal |
| **Blast radius if wrong** | **Severe** — a suppressed true positive is a missed breach |

---

## Problem

Security operations centres are drowning. A mid-sized enterprise generates tens of thousands of alerts a day across EDR, SIEM, cloud audit logs, identity providers, email security and network sensors. The overwhelming majority are benign. Analysts cannot look at all of them, so they look at the ones the tooling ranked highest, and tooling ranks by static severity written by the vendor — not by whether *this* alert, in *this* environment, on *this* asset, matters.

The consequences are well documented and expensive:

- **Alert fatigue.** Analysts habituate to a category and stop reading it, which is where breaches hide.
- **Static severity is context-free.** "Powershell encoded command" is critical on a finance workstation and routine on a build server.
- **Correlation is manual.** Five alerts that are one intrusion get triaged as five unrelated events by five analysts.
- **Tier-1 attrition.** The job is largely repetitive triage, and burnout is a hiring problem as much as a security one.

SOAR playbooks automate *response* but still need someone to decide which playbook applies. LLM-based triage is accurate and both too slow and too expensive at alert volume — and at thirty thousand alerts a day, a 4-second, 2-cent decision is $18,000 a month and a permanent backlog.

> **Scope note:** this is defensive security tooling — triage and prioritisation of alerts in an environment you operate. Jev classifies and ranks; it does not contain, block or remediate. Keep containment actions behind human authorisation or an explicitly-approved playbook ([P02](P02-tool-call-risk-gating.md)).

## Today's pattern

| Approach | Problem |
|---|---|
| **Static severity from the detection vendor** | Environment-blind. The same rule fires identically on a domain controller and a test VM |
| **Hand-written SIEM correlation rules** | Enormous maintenance burden; brittle; only catch anticipated patterns |
| **Risk-based alerting / UEBA scoring** | Better, and opaque — analysts cannot tell why a score is what it is, so they distrust it |
| **Tier-1 analyst manual triage** | Does not scale; high attrition; inconsistent between shifts |
| **LLM triage per alert** | Right capability, wrong economics and latency at alert volume |

## Jev design

The decisive design choice, exactly as in [P02](P02-tool-call-risk-gating.md): **judge the alert in its environmental context, not the alert in isolation.** Asset criticality, identity privilege, recent related activity and known change windows are what turn a generic detection into a prioritised finding. They must be in `state`, pulled from your CMDB, IdP and change calendar.

### State

```python
state = {
    "alert": {
        "rule": a.rule_name,
        "description": a.rule_description,
        "raw": truncate(a.raw_event, 1800),
        "vendor_severity": a.severity,
        "source": a.product,
    },
    "asset": {
        "role": asset.role,                     # domain_controller | workstation | build_agent
        "criticality": asset.criticality,       # crown_jewel | high | standard
        "internet_facing": asset.exposed,
        "data_classification": asset.data_class,
    },
    "identity": {
        "privilege": ident.privilege_level,     # admin | service | standard
        "is_service_account": ident.is_service,
        "unusual_hours": ident.outside_normal_window,
    },
    "recent_context": {
        "related_alerts_1h": [x.rule_name for x in related][:10],
        "change_window_active": change_cal.is_active(asset),
        "asset_alert_rate_baseline": asset.alerts_per_day_p50,
    },
}
```

`change_window_active` alone removes a large slice of routine false positives — most "suspicious" administrative activity is scheduled maintenance, and nothing in the detection stack knows that.

### Questions

```python
from typesafe_sdk import Choice, Noul, Score

SOC_QUESTIONS = {
    "disposition": Choice(
        instructions="The most likely explanation for this alert",
        criteria={
            "true_positive":     "Genuine malicious or unauthorised activity",
            "suspicious":        "Cannot be explained benignly; needs investigation",
            "benign_expected":   "Legitimate activity consistent with the asset's "
                                 "normal role and the context given",
            "benign_change":     "Legitimate activity explained by the active "
                                 "change window",
            "misconfiguration":  "Caused by a system or tooling misconfiguration "
                                 "rather than by an actor",
            "tuning_needed":     "The detection rule is firing on activity that is "
                                 "normal in this environment",
        },
    ),
    "kill_chain_stage": Choice(
        instructions="If this represents attacker activity, what stage it suggests",
        criteria={
            "none": "Not attacker activity",
            "recon": "Reconnaissance or discovery",
            "initial_access": "Gaining a foothold",
            "execution": "Running code",
            "persistence": "Establishing continued access",
            "privilege_escalation": "Gaining higher privileges",
            "lateral_movement": "Moving to other systems",
            "collection": "Gathering data",
            "exfiltration": "Moving data out",
            "impact": "Destruction, encryption or disruption",
        },
    ),
    "part_of_sequence": Noul(
        instructions="This alert appears related to the other recent alerts listed, "
                     "as part of one activity rather than independently"),
    "asset_impact": Score(
        instructions="How serious the consequences would be if this activity is "
                     "malicious, given the asset and data involved",
        criteria=["Negligible", "Limited to one system",
                  "Significant: sensitive data or privileged access at risk",
                  "Critical: crown-jewel asset or organisation-wide exposure"],
    ),
    "requires_containment": Noul(
        instructions="If malicious, this activity is of a kind where delay "
                     "materially worsens the outcome: active encryption, "
                     "ongoing exfiltration, or spreading access"),
    "evidence_sufficient": Noul(
        instructions="The information provided is enough to reach a confident "
                     "disposition, as opposed to needing additional data"),
}
```

`part_of_sequence` is the correlation signal that hand-written SIEM rules struggle with, and it is the difference between five triaged alerts and one investigated incident.

`tuning_needed` is the quiet long-term win: it produces a ranked list of rules that are wrong *for your environment*, which is how you reduce alert volume at source rather than filtering it forever.

## Integration

```python
def triage(alert, ctx):
    a = client.system_one(state=build_state(alert, ctx), questions=SOC_QUESTIONS).answers
    d = a["disposition"]

    # Suppression requires BOTH a benign disposition and high confidence,
    # and is never permitted on crown-jewel assets.
    suppressible = (d.choice in ("benign_expected", "benign_change")
                    and d.confidence > 0.85
                    and ctx.asset.criticality != "crown_jewel"
                    and a["asset_impact"].score < 2.0)

    if suppressible:
        return Close("auto_benign", evidence=a, reviewable=True)

    if d.choice == "tuning_needed" and d.confidence > 0.75:
        return Close("tuning_candidate", queue_rule_review=alert.rule)

    # Priority is arithmetic you own.
    pri = (0.35 * a["asset_impact"].score / 3.0
         + 0.30 * DISPOSITION_WEIGHT[d.choice]
         + 0.20 * KILL_CHAIN_WEIGHT[a["kill_chain_stage"].choice]
         + 0.15 * a["part_of_sequence"].noul)

    if a["requires_containment"].noul > 0.5:
        pri = max(pri, 0.85)
    if not a["evidence_sufficient"].noul > 0.5:
        enrich_and_requeue(alert)               # fetch more data, re-triage

    if a["part_of_sequence"].noul > 0.6:
        incident = correlate_into_incident(alert, ctx.related)   # one case, not five
        return Attach(incident, priority=bucket(pri))

    return Queue(tier=tier_for(pri), priority=bucket(pri), rationale=a)
```

Three commitments that a security team will insist on, correctly:

1. **Suppression is narrow and reviewable.** Four conditions must hold, crown-jewel assets are never auto-closed, and every auto-closure retains full evidence and is sampled by humans. Auto-closing alerts is the single most dangerous thing this system does.
2. **Jev never contains.** It prioritises and correlates. Containment stays with humans or approved playbooks.
3. **Every queued alert carries its rationale** — the distributions, the context used, the threshold. Analysts will not trust a score they cannot interrogate, and the distribution is what makes it interrogable.

## Thresholds & escalation

| Condition | Action |
|---|---|
| Benign disposition, conf > 0.85, non-crown-jewel, impact < 2 | Auto-close, reviewable, sampled |
| `tuning_needed`, conf > 0.75 | Close and queue the rule for review |
| `requires_containment > 0.5` | Priority floor 0.85; page on-call |
| `part_of_sequence > 0.6` | Correlate into one incident |
| `evidence_sufficient < 0.5` | Enrich and re-triage before queuing |
| `disposition.confidence < 0.6` | Queue to a human — never suppress on uncertainty |
| Crown-jewel asset, any non-benign disposition | Always queue, regardless of score |

The asymmetry is the point: **suppression demands high confidence; escalation does not.** A false escalation costs analyst minutes; a false suppression can cost a breach.

## Impact model

*Illustrative.* 30,000 alerts/day = 900k/month, ~2,500-token envelope ≈ $0.00011.

```
Jev, 100%          900k × $0.00011 = $    99/month
LLM, 100%          900k × $0.020   = $18,000/month + 4 s each (backlog)
tier-1 analysts    triage capacity ~150–250 alerts/analyst/day
```

The operational effect, assuming a conservative 60% high-confidence benign rate:

- Tier-1 volume drops by roughly half to two-thirds, with the remainder arriving **pre-prioritised, pre-correlated and with rationale attached**.
- Correlation collapses multi-alert intrusions into single incidents, which shortens investigation directly.
- `tuning_needed` gives a ranked backlog for reducing alert volume at source — the only durable fix.
- Analyst attention moves from volume to judgement, which is also the retention argument.

Do not promise a reduction in mean time to detect without measuring it. The mechanism is plausible — prioritisation and correlation should surface real incidents sooner — but it depends on your alert mix, and it must be demonstrated on your data, not assumed.

## Failure modes

- **False suppression is the catastrophic failure.** Everything in the design is shaped by it: four-condition gating, crown-jewel exemption, retained evidence, mandatory human sampling of auto-closures. Sample at a rate your security leadership signs off on, and never let the sampling lapse.
- **Adversarial evasion.** An attacker who knows the triage logic can shape activity to read as `benign_change`. Do not publish the criteria; layer with detections that do not route through this; treat the criteria as sensitive configuration.
- **Stale asset and identity context.** A wrong criticality rating mis-prioritises everything on that asset. CMDB accuracy becomes a security control — and in most organisations it is poor. Audit it, and fail toward higher criticality when data is missing.
- **Novel attack techniques** land in an existing category or `suspicious`. That is acceptable — `suspicious` queues rather than suppresses — but review the taxonomy against current threat intelligence regularly.
- **Injection via alert content.** Log fields frequently contain attacker-controlled strings (filenames, user agents, commit messages). Content engineered to read as benign is a real vector here. Adversarial evals are mandatory, and suppression logic must stay in code.
- **Over-tuning.** Acting on `tuning_needed` too aggressively disables detections that would have mattered. Rule changes need human review and a documented decision.
- **Fail-closed on error.** A Jev timeout or 429 must queue the alert, never suppress it.

## Evaluation

1. **Backtest against historical dispositions.** SOCs have labelled data: closed alerts with analyst dispositions. Take 5,000 spanning all sources. Report the confusion matrix, with **false-suppression rate on confirmed true positives as the single gating metric.**
2. Replay past confirmed incidents and check that (a) the constituent alerts were not suppressed and (b) correlation grouped them. This is the strongest validation available and it is worth doing exhaustively.
3. Red-team: craft alerts whose content argues for benign disposition.
4. Run in shadow for a month with suppression logged but not applied; have analysts review every would-be suppression. Only enable suppression when the false-suppression rate on that review is zero, and keep sampling afterwards.
5. In production: suppression rate and sampled review agreement, tier-1 queue depth, correlation rate, escalation precision, time-to-triage, and every incident root-caused against this layer.

## Related

- [P02](P02-tool-call-risk-gating.md) — same risk-classification pattern; keep containment behind it
- [P19](P19-prompt-injection-detection.md) — AI-specific detections feed this queue
- [P06](P06-failure-recovery-decision.md) — structurally the same triage design, different domain
- [P24](P24-ticket-triage.md) — the support-side analogue
- [P30](P30-transaction-policy-flagging.md) — the same alert-triage economics applied to financial controls
