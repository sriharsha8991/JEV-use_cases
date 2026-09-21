# P16 — Prompt & Model Regression Gating in CI/CD

| | |
|---|---|
| **Theme** | C · Evaluation & observability |
| **Primitives** | `score`, `noul`, `choice` |
| **Dominant win** | `⚡` latency (developer feedback loop) + `$` |
| **Latency budget** | The whole point — target under 3 minutes for a full suite |
| **Volume profile** | Per commit / per pull request |
| **Blast radius if wrong** | Moderate — a bad gate blocks good changes or ships regressions |

---

## Problem

Prompts, tool definitions, retrieval settings and model versions are production configuration with no test gate. A one-word prompt edit can degrade a whole class of behaviour, and the change looks trivial in review: a diff of one line, approved in thirty seconds, deployed, and discovered three days later in a support queue.

The reason there is no gate is cadence. Running an LLM-judged eval suite takes long enough and costs enough that it cannot sit in a pull-request check. So teams run it nightly at best, which means regressions are caught after merge, by which time three other changes have landed and attribution is guesswork.

There is a subtler problem that pure aggregate scoring hides. **An aggregate score can hold steady while a specific behaviour breaks.** Composite quality stays at 0.81; meanwhile the system stopped following the JSON format on German-language inputs. Aggregates are the wrong instrument for regression detection — you need per-behaviour assertions.

## Today's pattern

| Approach | Problem |
|---|---|
| **No gate; review the diff** | The default. A prompt diff does not reveal its behavioural effect |
| **Nightly LLM eval suite** | Post-merge, slow attribution, tens of dollars per run |
| **Exact-match / snapshot tests on outputs** | Break on every harmless rewording; teams delete them within a month |
| **Manual spot-checking before merge** | Inconsistent, unrecorded, and skipped under deadline |
| **Aggregate score threshold only** | Misses per-behaviour regressions entirely |

## Jev design

Two layers, and the second is the one most teams lack:

1. **Aggregate quality**, reusing the [P14](P14-llm-judge-replacement.md) rubric, with a tolerance band.
2. **Behavioural assertions** — per-case `noul` checks that must hold, expressed as properties rather than expected strings. These are the actual unit tests of a prompt.

### Behavioural assertions as a fixture

```yaml
# evals/cases/refund_policy.yaml
input: "Can I get a refund after 40 days?"
context: ["Refunds are available within 30 days of purchase."]
assertions:
  states_no: "The response makes clear that a refund is not available in this case"
  cites_window: "The response mentions the 30-day window"
  no_exception: "The response does not offer or imply an exception to the policy"
  no_escalation_promise: "The response does not promise that someone will override the policy"
```

Each assertion becomes a `noul`, and all assertions for a case go in one call (§4.1):

```python
from typesafe_sdk import Noul

def build_questions(case):
    qs = {f"assert_{k}": Noul(instructions=v) for k, v in case.assertions.items()}
    qs.update(JUDGE_QUESTIONS)          # the P14 rubric, same call
    return qs
```

This is the key design point. **An assertion is a property of the response, not a string.** `no_exception` keeps passing through arbitrary rewordings and fails the moment the behaviour changes — which is exactly what snapshot tests fail to do and what makes them get deleted.

## Integration

```python
def gate(pr_outputs, baseline):
    results = fan_out([judge(c, o) for c, o in pr_outputs])

    hard, soft = [], []
    for case, a in results:
        for key in case.assertions:
            ans = a[f"assert_{key}"]
            if ans.noul < 0.5:
                hard.append((case.id, key, ans.noul))       # assertion failed
            elif ans.noul < 0.7:
                soft.append((case.id, key, ans.noul))       # weak pass, warn

    agg = mean(composite(a) for _, a in results)
    delta = agg - baseline.aggregate

    # Per-failure-class regression, which the aggregate can hide.
    class_deltas = {
        k: histogram_now[k] - baseline.histogram[k]
        for k in FAILURE_CLASSES
    }
    class_regressions = {k: d for k, d in class_deltas.items() if d > 0.03}

    if hard:
        return Fail(f"{len(hard)} behavioural assertions failed", detail=hard)
    if delta < -0.02:
        return Fail(f"aggregate quality {delta:+.3f}")
    if class_regressions:
        return Fail("failure-class regression", detail=class_regressions)
    if soft or delta < -0.005:
        return Warn(detail={"weak": soft, "delta": delta})
    return Pass(aggregate=agg)
```

Three gates, in priority order: **assertions are hard failures**, aggregate movement is a band, and per-class movement catches what the aggregate hides. The third check is why a suite can fail with an unchanged aggregate score — and that is a feature, not a false positive.

## Thresholds & escalation

| Check | Threshold | Result |
|---|---|---|
| Behavioural assertion | `noul < 0.5` | **Fail** — blocks merge |
| Behavioural assertion | `0.5 ≤ noul < 0.7` | Warn — weak pass, worth a look |
| Aggregate quality | delta < −0.02 | Fail |
| Aggregate quality | −0.02 ≤ delta < −0.005 | Warn |
| Failure-class share | any class up >3pp | Fail |
| New case with no baseline | — | Record baseline, do not gate on the first run |
| Pinned model version changed | — | Force a full re-baseline; never compare across versions |

That last row is essential (§4.6). Comparing a suite run under `jev-1.13.0` against a baseline from `jev-1.12.0` produces meaningless deltas and a mystery failure. Make the model version part of the baseline key and fail loudly on a mismatch.

## Impact model

*Illustrative.* 600 cases (rubric + ~4 assertions each), 60 PRs/month.

```
frontier judge   600 × 10 dims × $0.008 = $48/run → $2,880/month, ~18 min parallelised
Jev              600 × 1 call × $0.0002 = $0.12/run → $7/month, ~90 s
```

At 90 seconds and 12 cents, this fits inside a pull-request check. That is the entire value: regressions are caught **before merge, attributed to one diff, by the author, while the change is still in their head.** Post-merge nightly detection is a different and much worse product.

Secondary effect: because it is cheap, engineers run it locally before pushing. Suites that are cheap get used; suites that are expensive get bypassed with `[skip ci]`.

## Failure modes

- **Flaky assertions** erode trust faster than anything else. An assertion that fails 1 run in 20 will be commented out. Test each new assertion by running it 10× against a known-good output before admitting it to the suite; reject any that is not deterministic.
- **Assertions that encode current behaviour rather than required behaviour.** These block legitimate improvements. Review assertions like API contracts: each one should express a requirement someone would defend.
- **Suite rot.** Cases accumulate, nobody prunes, runtime creeps up. Track per-case discriminative power — a case that has never failed in 200 runs is not testing anything.
- **Gaming.** Prompt edits that satisfy the assertion literally while degrading real behaviour. Refresh cases from production traffic ([P18](P18-drift-detection.md)) so the suite tracks reality.
- **Judge drift on a pinned version is still possible** if you change question wording. Treat question text as part of the baseline: any edit forces a re-baseline.
- **Non-determinism from the system under test**, not the judge. Fix temperature and seeds where you can; otherwise run N=3 and gate on the median.

## Evaluation

1. **Backtest against history.** Take 20 known past regressions from your incident log, reconstruct the diffs, and check the gate catches them. Catch rate on real historical regressions is the only credible validation.
2. Measure the false-failure rate on 50 known-good merged PRs. Anything above ~2% and engineers will route around the gate.
3. Measure wall-clock and cost per run; both must stay inside PR-check tolerance as the suite grows.
4. Track over time: gate failure rate, override rate (how often someone merges past a failure — persistently high means the gate is miscalibrated or distrusted), and post-merge incidents that the gate should have caught. That last metric is the gate's own regression test.

## Related

- [P14](P14-llm-judge-replacement.md) — supplies the rubric and the calibration discipline
- [P15](P15-online-output-qa.md) — production counterpart; its findings become new CI cases
- [P18](P18-drift-detection.md) — keeps the suite representative of live traffic
- [P17](P17-agent-trace-triage.md) — for agent systems, trace-level assertions extend this pattern
