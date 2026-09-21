# P18 — Semantic Drift & Distribution-Shift Detection

| | |
|---|---|
| **Theme** | C · Evaluation & observability |
| **Primitives** | `choice`, `score`, `noul` |
| **Dominant win** | `📈` coverage — semantic monitoring where only volumetric monitoring exists |
| **Latency budget** | Batch, hourly or daily |
| **Volume profile** | 100% of traffic, or a large stratified sample |
| **Blast radius if wrong** | Low — output is alerts and dashboards |

---

## Problem

Your eval suite says the system is fine. Your dashboards are green. Users are unhappy. All three can be true at once, because **the traffic has changed and nothing measures that.**

Concrete shapes this takes:

- A new product launches; a class of question appears that the corpus does not cover. Retrieval returns nothing relevant; the model answers anyway. No error, no metric moves.
- A competitor changes pricing; a wave of comparison questions arrives that the prompt never anticipated.
- A marketing campaign brings a less technical user segment whose phrasing degrades intent classification. Latency and error rates are unchanged.
- A model version bump shifts output distribution slightly — a bit more hedging, a bit more length. Every eval passes; user satisfaction drifts down.
- Your eval suite, built in January, now represents 60% of production traffic. It is passing tests for a distribution that no longer exists.

Standard monitoring is volumetric: requests, latency, errors, tokens, cost. Drift is **semantic**, and semantics have been too expensive to monitor continuously. Embedding-based drift detectors exist and tell you *that* the distribution moved without telling you *into what* — which is not actionable.

## Today's pattern

| Approach | Problem |
|---|---|
| **Volumetric dashboards** | Blind to semantic change by construction |
| **Embedding distribution distance** (PSI, MMD, KL on clusters) | Detects movement; cannot name it. "Cluster 7 grew 40%" is not a ticket |
| **User satisfaction surveys / CSAT** | Lagging by weeks, sparse, and non-diagnostic |
| **Periodic manual traffic review** | Quarterly at best, immediately stale |
| **LLM traffic analysis** | Right capability, prohibitive at 100% of traffic, so it is sampled thinly |

## Jev design

Classify every request and response along a **fixed, stable taxonomy** and monitor the composition of that taxonomy over time. The taxonomy must stay fixed for the time series to be meaningful — which is a strength of the `choice` primitive, because your categories are explicit, versioned artifacts rather than emergent clusters that renumber on every re-run.

### State

```python
state = {
    "request": request_text,
    "response": truncate(response_text, 1200),
    "outcome": {"had_context": bool(used_chunks),
                "refused": was_refusal,
                "escalated": was_escalated},
}
```

### Questions

```python
from typesafe_sdk import Choice, Noul, Score

DRIFT_QUESTIONS = {
    "topic": Choice(
        instructions="Which subject area this request is about",
        criteria=TOPIC_TAXONOMY,          # 15–40 stable, versioned categories
    ),
    "intent": Choice(
        instructions="What the user is trying to accomplish",
        criteria={
            "learn":     "Understand how something works",
            "do":        "Accomplish a specific task",
            "diagnose":  "Find out why something is wrong",
            "decide":    "Choose between options",
            "complain":  "Express dissatisfaction",
            "verify":    "Confirm something they already believe",
        },
    ),
    "sophistication": Score(
        instructions="The apparent expertise of the user, judged from their phrasing",
        criteria=["Novice: no domain vocabulary",
                  "Intermediate: some correct domain terms",
                  "Expert: precise, correct, specific terminology"],
    ),
    "coverage": Choice(
        instructions="How well the system handled this request",
        criteria={
            "answered":        "Answered substantively",
            "partial":         "Answered part of it",
            "no_information":  "Said it did not have the information",
            "refused":         "Declined on policy grounds",
            "misunderstood":   "Answered a different question than the one asked",
        },
    ),
    "novel_need": Noul(
        instructions="This request concerns something the response indicates the "
                     "system has no information about"),
    "frustration": Score(
        instructions="How frustrated the user appears",
        criteria=["Neutral", "Mildly impatient", "Clearly frustrated", "Angry"],
    ),
}
```

`novel_need` is the leading indicator. It rises *before* CSAT falls and before anyone files a ticket, because it fires the first time users ask about something the corpus does not cover. It is the earliest available signal of a knowledge or product gap.

## Integration

```python
def daily_drift_report(day):
    rows = fan_out([classify(r) for r in sample_traffic(day)])
    now = composition(rows)                    # share per category, per dimension
    base = baseline_composition()              # trailing 28-day, per weekday

    alerts = []
    for dim in ("topic", "intent", "coverage"):
        for cat, share in now[dim].items():
            delta = share - base[dim].get(cat, 0.0)
            if abs(delta) > threshold(dim, cat):
                alerts.append(Drift(dim, cat, share, delta))

    # Ratio metrics, computed in code — never asked of the model.
    if now["novel_need_rate"] > base["novel_need_rate"] * 1.5:
        alerts.append(KnowledgeGap(top_topics=top_novel_topics(rows)))
    if now["coverage"]["misunderstood"] > base["coverage"]["misunderstood"] + 0.02:
        alerts.append(IntentRegression(examples=sample(rows, "misunderstood")))
    if now["mean_frustration"] > base["mean_frustration"] + 0.3:
        alerts.append(SentimentDrift())

    # Eval-suite representativeness — the check nobody runs.
    gap = js_divergence(now["topic"], eval_suite_composition())
    if gap > 0.15:
        alerts.append(EvalSuiteStale(gap=gap, underrepresented=missing_topics(now)))

    return Report(now, alerts, examples_by_category=rows)
```

Two design points:

- **All arithmetic — shares, ratios, divergences, trends — is in code.** Jev labels; your code counts. This is not a preference, it is a constraint ([JEV-details §11](../JEV-details.md#11-where-jev-does-not-fit)).
- **`EvalSuiteStale` closes the loop.** Comparing production topic composition against your eval suite's composition tells you when [P14](P14-llm-judge-replacement.md) and [P16](P16-regression-gating-ci.md) have stopped testing reality. This is the check that most directly prevents "all tests green, users unhappy".

## Thresholds & escalation

| Signal | Threshold | Action |
|---|---|---|
| Topic share change | >3pp week-over-week | Investigate; may be seasonal |
| `novel_need` rate | >1.5× baseline | Knowledge-gap alert with top topics → content backlog |
| `misunderstood` share | +2pp | Intent-classification regression → [P10](P10-query-intent-routing.md)/[P05](P05-specialist-dispatch.md) |
| `no_information` share | +3pp | Corpus gap → [P13](P13-knowledge-conflict-detection.md) |
| Mean `frustration` | +0.3 | Sentiment drift; correlate with other alerts before acting |
| `sophistication` distribution shift | — | Audience change; prompts may need re-tuning |
| Eval/production JS divergence | >0.15 | Refresh the eval suite |

Compare against a **weekday-matched trailing baseline**. Enterprise traffic is strongly weekly-periodic, and a naive day-over-day comparison will alert every Monday.

## Impact model

*Illustrative.* 4M requests/month, 100% classification at a ~1,800-token envelope ≈ $0.000076.

```
Jev, 100%        4M × $0.000076 = $  304/month
LLM, 100%        4M × $0.010    = $40,000/month  (so: sampled at 1%, ~$400, low power)
embedding drift  ~$40/month, detects movement, cannot name it
```

At 100% coverage you get statistical power on small categories, which is where drift actually starts. A 1% sample cannot detect a shift in a category that is 0.5% of traffic — and emerging needs always begin small. That is the difference between a leading and a lagging indicator.

Value realised: knowledge gaps surface days-to-weeks before they appear in CSAT; eval-suite staleness becomes measurable rather than assumed; and model-version bumps get a semantic before/after comparison rather than a hope.

## Failure modes

- **Taxonomy drift defeats the purpose.** If `TOPIC_TAXONOMY` changes, the time series breaks. Version it, keep old categories for historical comparability, and add rather than rename. An `other` bucket growing past ~10% is the signal to *add* a category — and the transition must be dated in the dashboard.
- **Seasonality read as drift.** Weekday-matched baselines, and hold alerts for two consecutive periods before escalating.
- **Alert fatigue.** Six dimensions × 40 categories is a lot of surface. Alert on a curated set of composite indicators, not on every category delta.
- **Correlation mistaken for cause.** A frustration rise plus a topic shift may be one event or two. Always attach example requests to every alert; humans need the specifics to diagnose.
- **Sampling bias if you do not run 100%.** If you must sample, stratify by surface and time of day; naive sampling under-represents exactly the low-volume emerging categories you are trying to detect.
- **Response text in `state`** means a model-version change shifts response-derived dimensions (`coverage`, `novel_need`) even with identical traffic. Record both the Jev version and the generation model version on every row, and treat a generation-model change as a re-baseline event too.

## Evaluation

1. **Backtest against known events.** Take three past incidents — a launch, a campaign, a model bump — and confirm the detector fires, and how early relative to when you actually noticed.
2. Validate the taxonomy on 300 hand-labelled requests; report per-category accuracy. Categories below ~70% accuracy are badly defined and will generate noise.
3. Verify the `novel_need` leading-indicator claim by correlating its rate against subsequently-filed content requests and CSAT, with a time lag.
4. Measure alert precision over the first month: what fraction of alerts led to a real action? Below ~40% and the dashboard will be ignored.
5. Track: alert precision, mean lead time versus the previous detection route, `other`-bucket share, and eval/production divergence trend.

## Related

- [P14](P14-llm-judge-replacement.md) / [P16](P16-regression-gating-ci.md) — consumers of the eval-staleness signal
- [P15](P15-online-output-qa.md) — its per-check rates are additional drift series
- [P13](P13-knowledge-conflict-detection.md) — acts on the corpus gaps this finds
- [P10](P10-query-intent-routing.md) — its knowledge-gap log is the query-time counterpart
- [P17](P17-agent-trace-triage.md) — cause-distribution shift is drift on the failure side
