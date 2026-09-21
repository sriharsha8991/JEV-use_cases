# Jev Enterprise Playbook — Master Index

**A catalogue of 30 enterprise problem statements that TypeSafe AI's Jev can solve, with emphasis on making existing LLM and agent workloads faster, cheaper and more reliable.**

> Companion to [JEV-details.md](JEV-details.md) — the technical reference on what Jev is, its API surface, primitives, limits and economics. Read that first if you have not.
> Compiled 2026-09-22. Jev is in early access; all vendor performance figures are self-reported and unreproduced. Every impact number in this catalogue is an **illustrative model**, not a measurement.

---

## 1. The thesis

Most enterprise AI stacks built in 2024–2026 made the same architectural choice: **one model does everything.** A frontier LLM classifies the ticket, decides the route, checks the guardrail, judges the output, reranks the chunks, compacts the context, *and* writes the answer. It is the only tool available, so it becomes the tool for every job.

That collapses three very different workloads into one price and one latency:

| Workload | What it needs | What a frontier LLM gives it |
|---|---|---|
| **Generation** — write the reply, the code, the summary | Open-ended language, reasoning | Correctly matched |
| **Judgement** — route it, score it, gate it, verify it | A bounded decision, fast | 100–1000× overpriced, 30–300× too slow |
| **Computation** — count, sum, compare, sort | Determinism | Actively unreliable |

Jev exists for the middle row, and the middle row is **where most of the call volume actually is**. In a typical agent loop, generation is a minority of model invocations. Routing, gating, relevance checks, completion checks, retry decisions and output QA are the majority — and every one of them is a bounded decision being paid for at generation prices.

**The architectural move this catalogue describes: split your stack into a decision plane and a generation plane.**

```
                        ┌──────────────────────────────────────┐
   request ──────────►  │  DECISION PLANE  (Jev, ~100 ms)      │
                        │  admit · route · gate · score · rank │
                        │  verify · compact · terminate · QA   │
                        └───────────┬──────────────────────────┘
                                    │  only what genuinely needs prose
                                    ▼
                        ┌──────────────────────────────────────┐
                        │  GENERATION PLANE  (LLM, seconds)    │
                        │  write · reason · synthesize · code  │
                        └──────────────────────────────────────┘
                                    │
                                    ▼
                        ┌──────────────────────────────────────┐
                        │  DECISION PLANE  (Jev, ~100 ms)      │
                        │  ground · comply · redact · audit    │
                        └──────────────────────────────────────┘
```

Jev brackets the LLM on both sides. It decides what reaches the expensive model, and it checks what comes back.

---

## 2. Reference unit economics

Used consistently across all 30 docs so the numbers are comparable.

| Quantity | Value |
|---|---|
| Jev input price | **$0.042 / M tokens** |
| Jev output price | **free / unmetered** |
| Jev latency | **70–500 ms, typically ~100 ms** |
| Reference decision envelope | 2,000-token `state` + ~600 tokens of questions = **2,600 tokens** |
| **Cost per decision call** | **≈ $0.00011** |
| **Cost per 10M decisions / month** | **≈ $1,100** |
| Vendor benchmark, per case | $0.000081 (Jev) vs $0.013880 (GPT-5.6 Terra) |
| Vendor benchmark, per 1,000 workflows | $0.39 (Jev) vs $3.31 (GPT-5.6 Luna) vs $19.49 (Claude Haiku 4.5) |

**The two facts that drive every design in this catalogue:**

1. **Output is free and questions parallelize.** Adding the 12th question to a call costs a few hundred input tokens and roughly zero extra latency. So you stop rationing questions and start asking everything you might need in one shot (§4.1).
2. **A decision costs ~1/100th to ~1/400th of the generation it might avoid.** So a routing or admission decision is essentially free insurance against an expensive call. This inverts the usual "don't add another model hop" instinct.

---

## 3. The 30 problems

Themes A–D are about **optimizing LLM workflows you already run**. Themes E–F are about **decisions you currently make with rules, heuristics, humans, or not at all**.

Legend — **Win**: the dominant benefit. `$` cost · `⚡` latency · `🛡` reliability/safety · `📈` coverage (doing something you could not previously afford to do at all).

### A. Agent & inference-loop optimization

| # | Problem | Primitives | Win |
|---|---|---|---|
| [P01](problems/P01-tiered-model-routing.md) | Tiered model routing — stop sending easy requests to frontier models | `choice` `score` | `$` |
| [P02](problems/P02-tool-call-risk-gating.md) | Tool-call risk gating for autonomous agents | `choice` `noul` | `🛡` `⚡` |
| [P03](problems/P03-context-compaction.md) | Context compaction — prune agent history by judged relevance | `noul` `score` | `$` `⚡` |
| [P04](problems/P04-task-completion-detection.md) | Task-completion detection — end the loop instead of burning turns | `noul` `choice` | `$` |
| [P05](problems/P05-specialist-dispatch.md) | Specialist agent, skill and tool dispatch | `choice` | `$` `🛡` |
| [P06](problems/P06-failure-recovery-decision.md) | Failure recovery — retry, reroute, or escalate | `choice` `noul` | `$` `🛡` |
| [P07](problems/P07-clarify-vs-proceed.md) | Clarify vs. proceed — ask the user only when it matters | `noul` `score` | `🛡` `$` |
| [P08](problems/P08-llm-admission-control.md) | LLM admission control — deflect calls that need no LLM at all | `noul` `choice` | `$` `⚡` |

### B. RAG & grounding

| # | Problem | Primitives | Win |
|---|---|---|---|
| [P09](problems/P09-full-recall-reranking.md) | Full-recall reranking — judge every candidate, not the top 5 | `score` `noul` | `📈` `$` |
| [P10](problems/P10-query-intent-routing.md) | Query intent & retrieval strategy selection | `choice` `noul` | `⚡` `🛡` |
| [P11](problems/P11-groundedness-verification.md) | Groundedness verification — catch unsupported claims before delivery | `noul` `score` | `🛡` `📈` |
| [P12](problems/P12-citation-verification.md) | Citation relevance & sufficiency checking | `noul` `choice` | `🛡` |
| [P13](problems/P13-knowledge-conflict-detection.md) | Knowledge-base conflict & staleness detection | `noul` `choice` | `🛡` `📈` |

### C. Evaluation & observability

| # | Problem | Primitives | Win |
|---|---|---|---|
| [P14](problems/P14-llm-judge-replacement.md) | LLM-as-judge replacement in offline eval suites | `score` `noul` `choice` | `$` `⚡` |
| [P15](problems/P15-online-output-qa.md) | Online output QA at 100% coverage instead of 1% sampling | `noul` `score` | `📈` `🛡` |
| [P16](problems/P16-regression-gating-ci.md) | Prompt & model regression gating in CI/CD | `score` `noul` | `⚡` `$` |
| [P17](problems/P17-agent-trace-triage.md) | Agent trace failure triage at scale | `choice` `score` | `📈` `$` |
| [P18](problems/P18-drift-detection.md) | Semantic drift & distribution-shift detection | `choice` `score` | `📈` |

### D. Safety, security & compliance

| # | Problem | Primitives | Win |
|---|---|---|---|
| [P19](problems/P19-prompt-injection-detection.md) | Prompt injection & jailbreak detection at the AI gateway | `noul` `choice` | `🛡` `⚡` |
| [P20](problems/P20-pii-leakage-detection.md) | PII, secret & IP leakage detection in prompts and completions | `noul` `choice` | `🛡` `📈` |
| [P21](problems/P21-brand-policy-gating.md) | Brand, tone & policy compliance gating on generated content | `score` `noul` | `🛡` `📈` |
| [P22](problems/P22-regulated-review-routing.md) | Regulated-review routing & AI risk tiering | `choice` `noul` | `🛡` |
| [P23](problems/P23-ugc-moderation.md) | User-generated content moderation at publish time | `noul` `choice` `score` | `⚡` `$` |

### E. Customer & revenue operations

| # | Problem | Primitives | Win |
|---|---|---|---|
| [P24](problems/P24-ticket-triage.md) | Ticket triage, routing & priority scoring | `choice` `score` `noul` | `$` `⚡` |
| [P25](problems/P25-escalation-churn-detection.md) | Live escalation & churn-risk detection mid-conversation | `score` `noul` | `⚡` `📈` |
| [P26](problems/P26-deflection-eligibility.md) | Self-service deflection eligibility | `noul` `choice` | `$` |
| [P27](problems/P27-conversation-qa-scoring.md) | Conversation QA scoring for human and AI agents | `score` `noul` | `📈` `$` |

### F. Back-office, risk & data operations

| # | Problem | Primitives | Win |
|---|---|---|---|
| [P28](problems/P28-document-classification-idp.md) | Document classification & field-candidate selection in IDP | `choice` `noul` | `$` `🛡` |
| [P29](problems/P29-soc-alert-triage.md) | Security alert & incident triage in the SOC | `choice` `score` `noul` | `⚡` `$` |
| [P30](problems/P30-transaction-policy-flagging.md) | Transaction & expense policy-violation flagging | `noul` `score` `choice` | `📈` `$` |

---

## 4. Cross-cutting patterns

Every doc below leans on some subset of these six. They are described once here and referenced by name rather than re-explained 30 times.

### 4.1 Speculative fan-out
Ask every question you might need in a single call; decide in code which answers to use. Because questions evaluate in parallel against one encoded state and output is unmetered, the marginal question is nearly free. Vendor-reported **12.2× cheaper and 10× faster** than the sequential equivalent. Practical effect: a decision that would have been three chained LLM calls becomes one Jev call.

### 4.2 Atomic decomposition
Never ask one broad question that hides several judgements. Ask each factor separately and combine with arithmetic **you own**. This is the difference between a system you can tune per-dimension and a black box you can only re-prompt. It is also the only way to get per-dimension precision/recall curves.

### 4.3 Confidence-gated action
Thresholds belong to *consequences*, not to systems. One global 0.8 cutoff is a design smell.

```python
def gate(answer, *, auto, review):
    if answer.confidence >= auto:   return "act"
    if answer.confidence >= review: return "review"
    return "escalate"

gate(ans, auto=0.60, review=0.40)   # read-only lookup
gate(ans, auto=0.95, review=0.80)   # irreversible / moves money / sends mail
```

### 4.4 The cascade
Jev triages at the front door → trivial cases resolve in deterministic code → hard cases reach a frontier model → low-confidence cases reach a human. The whole economic argument of this catalogue is that in most enterprise traffic the first two buckets are the majority, and they are currently being billed as if they were the third.

### 4.5 Retrieve, then judge
Accuracy degrades with irrelevant state — "context rot" is explicitly documented for Jev. Project and filter in code first; send the minimum fields the question needs. This also keeps you inside the ~32k state budget.

### 4.6 Pin, log, replay
Pin `jev-1.13.0` rather than `jev-latest` once thresholds are tuned. Log the returned `model` field on every call. Replay a golden set whenever a question, a criterion or the version changes. Without this, a silent model bump invalidates every threshold in production at once.

---

## 5. What every problem doc contains

A fixed template, so the 30 docs are comparable and skimmable:

| Section | Purpose |
|---|---|
| **Header table** | Theme, primitives, dominant win, latency budget, volume profile, blast radius if wrong |
| **Problem** | The business and engineering pain, stated concretely |
| **Today's pattern** | How this is currently done with an LLM, rules or humans — and specifically why it hurts |
| **Jev design** | The `state` shape and the actual question schema, as runnable code |
| **Integration** | Where this sits in the request path, with wiring code |
| **Thresholds & escalation** | Confidence policy tied to consequence |
| **Impact model** | Illustrative arithmetic from §2 unit economics — clearly labelled as a model |
| **Failure modes** | What breaks, including the adversarial case |
| **Evaluation** | How to prove it works on your traffic before you trust it |
| **Related** | Links to adjacent problems |

---

## 6. Adoption roadmap

Do not start with the highest-value problem. Start with the one where you already have labels and a rollback.

**Phase 0 — Instrument (before any Jev call).**
Log every LLM call in your stack with: purpose, prompt tokens, completion tokens, latency, cost, and whether the output was *parsed as structured data*. That last flag is the shortlist: **every call whose output you parse into a bounded value is a Jev candidate.** Most teams are surprised by how large this set is.

**Phase 1 — Shadow (weeks 1–2).**
Pick one problem with existing ground truth — usually [P24](problems/P24-ticket-triage.md) or [P14](problems/P14-llm-judge-replacement.md). Run Jev in parallel with the incumbent, act on neither. Measure agreement, per-class precision/recall, and **calibration** (bucket by confidence, check that accuracy rises monotonically). If calibration does not hold on your traffic, the confidence-gating patterns do not apply and you should stop.

**Phase 2 — Gate (weeks 3–6).**
Promote to production behind confidence gates (§4.3) with the incumbent as the fallback path. Tune thresholds against Phase 1 curves. Target: a large auto-act fraction with measured error below your existing human error rate — not below zero.

**Phase 3 — Displace (months 2–3).**
Attack the inner loop: [P01](problems/P01-tiered-model-routing.md), [P02](problems/P02-tool-call-risk-gating.md), [P03](problems/P03-context-compaction.md), [P08](problems/P08-llm-admission-control.md). These deliver the largest cost deltas because they eliminate frontier calls rather than replacing cheap ones. They also carry the most blast radius, which is why they come third.

**Phase 4 — Expand coverage (month 3+).**
Turn on the `📈` problems — the ones you never did because you could not afford to: [P09](problems/P09-full-recall-reranking.md), [P15](problems/P15-online-output-qa.md), [P17](problems/P17-agent-trace-triage.md), [P20](problems/P20-pii-leakage-detection.md), [P27](problems/P27-conversation-qa-scoring.md), [P30](problems/P30-transaction-policy-flagging.md). This phase usually produces more business value than the cost savings did, and it is invisible on a cost dashboard.

---

## 7. Enterprise risk register

Read this before you socialize any of it internally. These are the objections your platform, security and risk teams will raise, and they are legitimate.

| Risk | Reality | Mitigation |
|---|---|---|
| **Vendor maturity** | Company out of stealth 2026-09-15, single model, early-access waitlist | Keep a fallback path (cheap LLM or conservative default) behind a feature flag on every Jev call path |
| **No self-hosting** | Hosted API only; no weights, no on-prem, no air-gap | Blocks regulated/sovereign deployments outright. Confirm data-residency and retention terms in writing before any pilot |
| **Unreproduced benchmarks** | Vendor designed the workflows, built the harness, ran the eval | Ignore their numbers. Measure on your traffic in Phase 1 |
| **"0% hallucination" misread** | It means schema conformity. Jev returns wrong *valid* values | Never let this phrase into a control document. Track semantic accuracy separately from type validity |
| **No rationale** | Returns probabilities, never reasons | For any decision needing an explainable trail, log inputs + full probability distribution + threshold + policy version. That is an audit trail *of the decision*, not an explanation *by the model* |
| **Prompt injection** | `state` carries user content, and Jev *is* the guardrail in P02/P19/P23 | A successful shift is a control bypass, not a bad answer. Adversarial evals are mandatory, not optional. See [P19](problems/P19-prompt-injection-detection.md) |
| **Silent model bumps** | `jev-latest` advances without notice | §4.6. Pin, log, replay |
| **Calibration assumed** | Vendor claim, your distribution | Phase 1 gate. If confidence buckets do not stratify accuracy on your data, do not ship confidence gating |
| **Concentration risk** | A decision plane in the request path of everything | Circuit-break to conservative defaults on 429/529; define the degraded mode before launch, not after the incident |

---

## 8. How to size this for your own stack

A back-of-envelope worth doing before reading all 30 docs:

```
1. From Phase 0 logs, count monthly LLM calls whose output you parse
   into a bounded value (enum, score, boolean, route, verdict).   → N
2. Average current cost of those calls.                           → C
3. Jev replacement cost ≈ N × $0.00011   (§2 reference envelope)
4. Gross saving ≈ N × (C − $0.00011)
5. Discount by the fraction you will keep routing to the LLM
   because confidence lands below your auto-act threshold.        → ×(1 − e)
6. Add the latency delta: for each call in a user-facing path,
   seconds saved × calls, which is usually the number that
   actually gets the project funded.
```

Step 6 tends to matter more than steps 1–5. Cost savings are a line item; removing 6 seconds from an interactive path is a product change. And the `📈` problems in Phase 4 do not appear anywhere in this arithmetic, because their baseline is zero.

---

## 9. Contents

- **README.md** — this file: thesis, catalogue, cross-cutting patterns, roadmap, risk register
- [JEV-details.md](JEV-details.md) — Jev technical reference: primitives, API, limits, architecture, economics
- [problems/](problems/) — the 30 problem statements, P01–P30

Sources for all vendor figures are listed in [JEV-details.md §14](JEV-details.md#14-sources).
