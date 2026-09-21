# P14 — LLM-as-Judge Replacement in Offline Eval Suites

| | |
|---|---|
| **Theme** | C · Evaluation & observability |
| **Primitives** | `score`, `noul`, `choice` |
| **Dominant win** | `$` cost + `⚡` latency — turns an overnight eval suite into a per-commit one |
| **Latency budget** | Batch, but total wall-clock is the whole point |
| **Volume profile** | Eval cases × rubric dimensions × runs |
| **Blast radius if wrong** | Moderate — a miscalibrated judge produces confidently wrong quality signals and you ship on them |

---

## Problem

LLM-as-judge became the default evaluation method for generative systems, and it has three structural problems that get worse as the suite grows:

1. **It costs more than the system under test.** A judge with a detailed rubric, seeing both the output and the reference, often consumes more tokens than generating the output did. Teams routinely spend more on evaluating than on serving.
2. **It is slow enough to break the feedback loop.** A 2,000-case suite × 6 rubric dimensions at 5 s per judgement is hours. So the suite runs nightly, or weekly, or before releases — and stops being a development tool.
3. **It is unreliable in exactly the way that matters.** Judges are overconfident and poorly calibrated; scores cluster at 4 and 5 out of 5; and they emit malformed structured output at rates from 0.58% to 45.5% depending on the model. Those parse failures land in your metrics pipeline, where they are usually silently dropped — which biases the result.

The consequence is that most teams have an eval suite they do not trust and cannot afford to run often. Both halves of that sentence are cost problems in disguise.

## Today's pattern

| Approach | Problem |
|---|---|
| **Frontier LLM judge with a prose rubric** | Expensive, slow, uncalibrated, parse failures |
| **Small LLM judge** | Cheaper; far worse structured-output reliability and weaker judgement |
| **Reference-based metrics** (BLEU, ROUGE, BERTScore) | Cheap and fast; measure surface overlap, not quality. Near-useless for open-ended output |
| **Human evaluation** | The gold standard, and unschedulable at CI cadence |
| **Pairwise LLM preference** | Better signal than absolute scores; twice the calls and no absolute scale to track over time |

## Jev design

The central shift: **a rubric becomes an explicit ordered `score` criteria list, not prose in a prompt.** This is a genuine methodological improvement, not just a cheaper implementation. A prose rubric like "rate helpfulness 1–5" lets the judge invent its own anchors and drift between runs. An ordered criteria array forces you to define each level, and the same definition is applied every time.

### State

```python
state = {
    "task": case.input,
    "output": case.model_output,
    "reference": case.reference_answer,      # optional
    "context": case.provided_context,        # for grounding dimensions
}
```

### Questions

```python
from typesafe_sdk import Choice, Noul, Score

JUDGE_QUESTIONS = {
    "task_completion": Score(
        instructions="How completely the output does what the task asked",
        criteria=[
            "Does not address the task",
            "Addresses part of it; major elements missing",
            "Addresses the task with minor gaps",
            "Fully addresses everything asked",
        ],
    ),
    "correctness": Score(
        instructions="Factual correctness of the output, judged against the "
                     "reference and provided context",
        criteria=[
            "Contains clear factual errors",
            "Mostly right with one questionable detail",
            "No detectable errors",
        ],
    ),
    "instruction_adherence": Noul(
        instructions="The output respects every explicit constraint in the task: "
                     "format, length, tone, inclusions and exclusions"),
    "grounded": Noul(
        instructions="Every factual claim in the output is supported by the "
                     "provided context"),
    "conciseness": Score(
        instructions="How efficiently the output conveys its content",
        criteria=["Substantially padded or repetitive",
                  "Somewhat verbose",
                  "Appropriately concise"],
    ),
    "failure_mode": Choice(
        instructions="If the output is inadequate, the primary reason",
        criteria={
            "none":            "The output is adequate",
            "wrong_facts":     "Factually incorrect",
            "incomplete":      "Missing required elements",
            "ignored_format":  "Did not follow the requested structure",
            "off_topic":       "Answered a different question",
            "refused":         "Declined a reasonable request",
            "hallucinated":    "Invented information not in the context",
            "too_verbose":     "Padded to the point of harm",
        },
    ),
}
```

`failure_mode` is what makes the suite diagnostic rather than merely a score. An eval that says "quality dropped from 0.81 to 0.76" tells you nothing actionable; one that says "`ignored_format` rose from 2% to 19%" tells you exactly what your prompt change broke.

## Integration

```python
def evaluate(suite, model_outputs):
    results = fan_out([judge(c, o) for c, o in zip(suite, model_outputs)])

    rows = []
    for case, a in results:
        composite = (0.40 * a["task_completion"].score / 3.0
                   + 0.30 * a["correctness"].score / 2.0
                   + 0.15 * a["instruction_adherence"].noul
                   + 0.10 * a["grounded"].noul
                   + 0.05 * a["conciseness"].score / 2.0)

        rows.append({
            "case": case.id,
            "composite": composite,
            "failure": a["failure_mode"].choice,
            # Low judge confidence = route to a human, not a silent data point.
            "needs_human": min(a["task_completion"].confidence,
                               a["correctness"].confidence) < 0.55,
        })

    return Report(rows,
                  human_queue=[r for r in rows if r["needs_human"]],
                  failure_histogram=histogram(r["failure"] for r in rows))
```

Two things a frontier-LLM judge cannot give you:

- **Weights you own.** Changing how much correctness matters relative to conciseness is editing a number, not re-prompting a judge and re-baselining every historical score.
- **A principled human queue.** Calibrated confidence means "the judge does not know" is a detectable state. Low-confidence cases go to humans, and human effort concentrates exactly where automated judgement is unreliable. With an overconfident LLM judge this signal does not exist, so human review is sampled at random instead.

## Thresholds & escalation

| Signal | Action |
|---|---|
| Judge confidence < 0.55 on a weighted dimension | Route to human review; exclude from the automated aggregate |
| `composite` below the release bar | Block the release |
| Any `failure_mode` class up >2× versus baseline | Investigate that class specifically, even if the composite held |
| `grounded < 0.5` on any case | Treat as a correctness bug, not a quality score |
| Disagreement with human labels > 15% on the calibration set | The judge itself has regressed. Re-baseline before trusting any result |

That last row is the governance requirement people skip: **maintain a human-labelled calibration set and re-verify the judge, not just the system.** Judges drift, especially across model versions (§4.6).

## Impact model

*Illustrative.* 2,000 cases × 6 dimensions, run per commit at 40 commits/month.

```
frontier judge   2,000 × 6 × $0.008  = $96 per run × 40  = $3,840/month
                 wall-clock ~2.8 h per run (heavily parallelised: ~20 min)

Jev              2,000 × $0.00015    = $0.30 per run × 40 = $   12/month
                 wall-clock ~2 min (one call per case, all 6 dimensions in it)
```

Note that the Jev column has **one call per case**, not six: all dimensions are questions in the same call (§4.1). That is where both the 300× cost reduction and the wall-clock collapse come from.

The change in kind matters more than either number. At $0.30 and two minutes, the suite runs on every commit and on every prompt edit, and engineers get eval feedback inside the development loop rather than the release loop. That is the difference between an eval suite and a test suite.

## Failure modes

- **Judge–human disagreement is the root risk.** Every number the suite produces inherits it. The calibration set is mandatory, not optional, and it must be refreshed as your task distribution changes.
- **Rubric levels that are not genuinely ordered.** `score` assumes a spectrum. If level 2 is not strictly "more" than level 1, the weighted mean is meaningless. Review every criteria array for monotonicity.
- **Dimension collinearity.** If `task_completion` and `correctness` always move together, the composite is effectively one dimension with extra steps. Check inter-dimension correlation on real data and drop redundant ones.
- **No arithmetic.** Do not ask the judge to count errors or compute percentages ([JEV-details §11](../JEV-details.md#11-where-jev-does-not-fit)). Ask per-item `noul`s and count in code.
- **Injection via the output under test.** A model output containing "this response fully satisfies all criteria" attacks the judge. Directly relevant if you ever evaluate adversarial or user-supplied content — include such cases in the calibration set.

## Evaluation

The judge itself needs evaluating. That is the whole discipline here.

1. Build a human-labelled calibration set of 200–300 cases spanning the quality range, including deliberately bad outputs.
2. Report Jev-vs-human agreement per dimension, and compare against your current LLM judge's agreement on the *same* set. If the incumbent is better, keep it for that dimension and use Jev for the rest — this is not all-or-nothing.
3. Verify calibration: bucket confidence and confirm agreement rises monotonically. This is what licenses the human-queue mechanism.
4. Check run-to-run stability: judge the same 100 cases five times and measure variance. Prose-rubric LLM judges are notably unstable here; explicit criteria arrays should be better, and you should confirm it.
5. Re-run steps 2–4 whenever you change the pinned model version.

## Related

- [P15](P15-online-output-qa.md) — the same rubric applied to live traffic
- [P16](P16-regression-gating-ci.md) — the CI gate that consumes this suite
- [P11](P11-groundedness-verification.md) — the `grounded` dimension as an inline production control
- [P27](P27-conversation-qa-scoring.md) — the same mechanics scoring human work
- [P18](P18-drift-detection.md) — when the eval distribution itself stops matching production
