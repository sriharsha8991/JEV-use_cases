# Jev — TypeSafe AI's "System One" Decision Model

> Technical reference on what Jev is, how its API works, and where it fits.
> **For the enterprise catalogue of 30 problem statements and adoption roadmap, see [README.md](README.md).**
> Compiled 2026-09-22 from vendor documentation and third-party analysis. Jev is ~1 week old at time of writing (public launch 2026-09-15) and in **early access behind a waitlist**. Treat all performance numbers as vendor-reported unless stated otherwise.

---

## 1. TL;DR

**Jev is not a chatbot and not an LLM in the usual sense.** It takes your application state plus a set of *typed questions*, and returns *typed answers with calibrated probabilities* — never free-form text.

```
                   LLM                                    Jev
   ┌────────────────────────────────┐    ┌────────────────────────────────────┐
   │ prompt → tokens → tokens → …   │    │ state ──┐                          │
   │        → string → JSON.parse() │    │         ├─ encode once             │
   │        → validate → hope       │    │ questions ─ evaluated in parallel  │
   └────────────────────────────────┘    │         └→ typed values + p(·)     │
                                         └────────────────────────────────────┘
     3–330 s, $0.03–0.18 / case            70–500 ms, ~$0.0004 / case
     0.58–45.5% invalid structured output   0% invalid output (by construction)
```

The single-sentence pitch: **Jev replaces the "ask an LLM to classify this and return JSON" call** — routing, scoring, guardrails, verification, triage — with something roughly two orders of magnitude faster and cheaper, whose output is structurally impossible to malform. It replaces nothing else.

---

## 2. Origin and status

| | |
|---|---|
| Vendor | TypeSafe AI (San Francisco, founded 2024) |
| Out of stealth | 2026-09-15 |
| Funding | $40M seed led by DCVC; ~$200M post-money valuation (per Forbes) |
| Founders | Diogo Almeida (CEO) — ex-OpenAI, InstructGPT author, RLHF / ChatGPT / GPT-4 contributor; Erik Gafni; Sasha Sheng |
| First model | `jev-latest` (currently resolves to `jev-1.13.0`) |
| Availability | Hosted API, early-access waitlist. **No open weights, no published parameter count, no self-hosting.** |
| Modality | **Text only.** No image, audio or video input. |

The name comes from Kahneman's *Thinking, Fast and Slow*: LLM chain-of-thought is "System 2" — slow, deliberate, verbal. Jev is positioned as **"System 1"** — fast, intuitive, pre-verbal judgement. TypeSafe frames this as a new *model class*, not a smaller LLM.

Almeida's framing of the gap: the industry has *"been optimizing for humans, and we're superhuman at pleasing humans."* Jev optimizes for being consumed by **software**.

---

## 3. The core mental model

An LLM-as-classifier call is a **string round-trip**. You serialize intent into a prompt, the model serializes a decision into text, and you deserialize it back into a program value. Every step is lossy and every step can fail: the model can emit invalid JSON, invent an enum member that isn't in your schema, call a tool that doesn't exist, or bury the answer in prose.

Jev removes the round-trip. The *answer space is part of the request*. You declare "this question has exactly these three outcomes", and the response is a probability distribution over exactly those three outcomes.

Two consequences follow, and they are the whole value proposition:

1. **Structural validity is guaranteed, not hoped for.** There is no parsing step, so there is no parse failure, no out-of-schema enum, no hallucinated tool name. TypeSafe reports a 0% structured-output error rate and 0% tool-call error rate — a claim about *type safety*, not about *being right*.
2. **Questions are independent, so they parallelize.** Jev encodes the state once and evaluates every question against it simultaneously. Adding a 10th question is nearly free in both latency and cost. This inverts normal prompt economics and enables the fan-out pattern (§8.1).

The crucial corollary, stated plainly in TypeSafe's own docs and every serious review: **Jev cannot hallucinate a value, but it can absolutely return the wrong valid value.** "0% hallucination" means schema conformity. It is not an accuracy claim.

---

## 4. Architecture — what is known vs. inferred

Let's be honest about the boundary here, because the vendor has disclosed very little.

**Disclosed:**

- A "new architecture" built around a **parallel sampler** that produces all outputs in a single query, rather than autoregressive token-by-token emission.
- Trained with **RLCD — Reinforcement Learning for Calibrated Decisions.** Where RLHF optimizes for human preference and RLVR for verifiable rewards, RLCD is described as optimizing for *epistemically honest probabilities on decision tasks*. This is the stated mechanism behind the calibration claim.
- Questions are "evaluated independently and in parallel" against a shared state.

**Inferred but unconfirmed** — do not present these as fact:

- A single encoder pass over the state feeding per-question classification heads. This is *consistent* with flat latency in question count, but TypeSafe has not confirmed a dedicated encoder or an exact forward-pass count.
- KV-cache sharing across questions; mixture-of-experts routing; base-model lineage; parameter count; training data. All undisclosed.

**Observable behaviour you can rely on:**

- Latency is roughly flat in the number of questions; cost grows only with the incremental question tokens.
- Questions cannot depend on each other's answers. Dependent reasoning requires a second request, with your code in between.

---

## 5. API surface

### 5.1 Endpoint

```
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

One endpoint. That is the entire API.

### 5.2 Request body

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | `string` | yes | `"jev-latest"`, or a pinned version such as `"jev-1.13.0"` |
| `state` | `string \| object \| array` | yes | The thing being judged: raw text, or structured data (a ticket, a chat log, a DB row, current app state) |
| `questions` | `map<string, Question>` | yes | Keys are yours; answers come back under the same keys |

### 5.3 The three primitives

Every `Question` has a `type` and `instructions`. Each type then defines `criteria` differently.

#### `noul` — yes/no as a probability

The odd name is TypeSafe's coinage for a boolean-shaped judgement that returns belief strength rather than a hard boolean.

```json
{
  "type": "noul",
  "instructions": "The customer is explicitly asking for a refund",
  "criteria": { "true": "what yes means", "false": "what no means" }
}
```

`criteria` is **optional**. Returns `{ "type": "noul", "noul": 0.93 }` — a single probability in `[0,1]`. **No separate `confidence` field**; the number *is* the belief.

#### `choice` — one of N

```json
{
  "type": "choice",
  "instructions": "Which team should handle this",
  "criteria": {
    "billing":   "Payment or subscription issues",
    "technical": "Bugs or integration problems",
    "sales":     "Pricing or account questions",
    "other":     null
  }
}
```

`criteria` is **required**; maximum **255 options**. A `null` value is a legal option with no elaboration. Returns:

```json
{
  "type": "choice",
  "choice": "billing",
  "probabilities": { "billing": 0.84, "technical": 0.11, "sales": 0.03, "other": 0.02 },
  "confidence": 0.84
}
```

#### `score` — position on an ordered rubric

```json
{
  "type": "score",
  "instructions": "How frustrated the customer appears",
  "criteria": [
    "Calm, just stating facts",
    "Frustrated but civil",
    "Very angry, strong language"
  ]
}
```

`criteria` is **required**: an *ordered* array of **2–10 levels**, described in words, not numbers. Returns:

```json
{
  "type": "score",
  "score": 1.035,
  "legend": { "0": "Calm, just stating facts", "1": "Frustrated but civil", "2": "Very angry, strong language" },
  "probabilities": { "0": 0.12, "1": 0.74, "2": 0.14 },
  "confidence": 0.74
}
```

`score` is the **probability-weighted mean across levels**, so it lands *between* levels — that fractional part is signal, not noise.

Note the design choice: rubric levels are prose, not a 0–100 scale. A bare numeric range invites the model to invent its own rubric; enumerated levels force you to define it.

### 5.4 Response body

| Field | Type | Description |
|---|---|---|
| `model` | `string` | Resolved version that served the request, e.g. `jev-1.13.0` — **log this** |
| `answers` | `map<string, Answer>` | Keyed by your question ids |
| `usage` | `object` | `input_tokens`, `output_tokens` |

### 5.5 Errors

| Status | Meaning |
|---|---|
| `401` | Missing or invalid API key |
| `422` | Request validation failed (bad schema, too many options, too few levels) |
| `429` | Rate limit exceeded |
| `529` | Overloaded |

For `429` / `529`, retry with **exponential backoff**, not immediately.

### 5.6 Limits and economics

| | |
|---|---|
| Context | ~64k tokens total across state + all questions; ~32k for state + the single longest question (~150k chars of English). Sources differ slightly on how the budget splits — measure it. |
| Choice cardinality | 255 options max |
| Score levels | 2–10 |
| Rate limits | ~250,000 tokens/sec, ~1,200 req/min (dynamic during early access) |
| Price | **$0.042 per million input tokens. Output tokens are free / unmetered.** |
| Latency | 70–500 ms end-to-end, most around **~100 ms** (measured from US West Coast) |

Output being free is not a rounding detail — it means the fan-out pattern in §8.1 is economically almost unconstrained, which changes how you design against this model.

---

## 6. SDKs and ecosystem

| Target | Package / id |
|---|---|
| Python 3.10+ | `pip install typesafe-sdk` → `TypeSafeClient`, `Choice`, `Score`, `Noul` |
| Node 20+ | `npm install @typesafe-ai/sdk` → `TypeSafeClient`, `choice()`, `score()`, `noul()` |
| Vercel AI SDK (Node 22+) | `@ai-sdk/typesafe-ai` + `experimental_evaluate()` |
| Vercel AI Gateway | model id `typesafe-ai/jev` |
| LangChain | `langchain_typesafe` → `TypeSafeClassifier` |
| LiteLLM | pass-through route for TypeSafe |

Both first-party SDKs read `TYPESAFE_API_KEY` from the environment and default to `jev-latest`. The TS SDK infers answer types from the question schema, so `response.answers.category.choice` is typed as your literal union — the "type safe" in the company name is meant literally.

### Python

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient()

response = client.system_one(
    state={
        "ticket": {
            "subject": "Duplicate charge",
            "messages": [
                {"from": "customer",
                 "text": "I was charged twice for order A-104. Please refund the duplicate."},
            ],
        },
        "order": {"id": "A-104", "charges": [
            {"amount_usd": 49, "status": "captured"},
            {"amount_usd": 49, "status": "captured"},
        ]},
        "refund_policy": "Duplicate charges are eligible for a refund.",
    },
    questions={
        "department": Choice(
            instructions="Which team should handle this",
            criteria={
                "billing": "Payment or subscription issues",
                "technical": "Bugs or integration problems",
                "sales": "Pricing or account questions",
            },
        ),
        "frustration": Score(
            instructions="How frustrated the customer appears",
            criteria=[
                "Calm, just stating facts",
                "Frustrated but civil",
                "Very angry, strong language",
            ],
        ),
        "refund_requested": Noul(
            instructions="The customer is explicitly asking for a refund"),
        "policy_supports": Noul(
            instructions="The stated refund policy covers this situation"),
    },
)

dept = response.answers["department"]
print(dept.choice, dept.confidence)               # billing 0.84
print(response.answers["frustration"].score)      # 1.035
print(response.answers["refund_requested"].noul)  # 0.93
```

### TypeScript

```typescript
import { choice, noul, TypeSafeClient } from "@typesafe-ai/sdk";

const client = new TypeSafeClient();

const response = await client.systemOne({
  state: { document: "I was charged twice. Please fix this ASAP." },
  questions: {
    category: choice("What is this ticket about?", {
      billing:   "Payment or subscription issues",
      technical: "Bugs or integration problems",
      other:     "Anything else",
    }),
    urgent: noul("The message conveys urgency"),
  },
});

response.answers.category.choice; // "billing" | "technical" | "other"
```

---

## 7. Calibration — the feature that makes this usable

Frontier LLMs are notoriously overconfident: ask one for a probability and you get a number that looks like a probability and behaves like a vibe. RLCD is aimed specifically at this, and the claim is that **Jev's confidence is calibrated — higher confidence genuinely corresponds to higher accuracy.**

That claim, *if it holds on your traffic*, is what lets you write this instead of a prompt:

```python
a = response.answers["action"]
if a.confidence < 0.5:
    escalate_to_human()      # model doesn't know
elif a.confidence < 0.9:
    queue_for_review()       # probably right, verify
else:
    execute(a.choice)        # act
```

Where `confidence` comes from: it is derived from the **shape of the probability distribution**. A flat distribution over options means low confidence; a peaked one means high. `noul` has no `confidence` field because its single probability already encodes exactly that.

**Threshold guidance that actually matters:** do not set one global threshold. Set it per *consequence*. A read-only lookup can act at 0.6; anything that moves money, sends a message, or deletes data should demand 0.9+ and otherwise fall back to explicit confirmation. Then **verify those thresholds against labelled data from your own traffic** — the vendor's calibration curve is not yours.

---

## 8. Design patterns

### 8.1 Speculative fan-out

Because questions parallelize and output is free, ask **everything you might need up front** in one call and let code decide what to use. Don't chain calls to decide what to ask next. TypeSafe reports **12.2× cheaper and 10× faster** than the sequential equivalent.

### 8.2 Atomic decomposition

The single most important authoring rule, straight from the docs: *ask each factor as a separate question, then combine the results with logic in your code.* One broad "is this ticket high priority?" hides five judgements and gives you nothing to debug. Five `noul`s plus arithmetic you own is inspectable, tunable and testable.

### 8.3 Composite scoring

Score independent dimensions separately, normalize each by its level count, weight by an importance coefficient you control, sum. You own the weights, so you can retune priority without retraining or re-prompting anything.

### 8.4 The cascade

Jev triages at the front door → trivial cases go straight to deterministic code → genuinely hard cases go to a frontier LLM → low-confidence cases go to a human. Most production traffic is not hard; this is where the cost curve bends.

### 8.5 Retrieve, then judge

**Context rot is real here.** Accuracy degrades with irrelevant state. Filter and project in code first, send only the fields the question needs. Do not dump the whole record and hope.

### 8.6 Pin the version

`jev-latest` advances without notice. If you have tuned confidence thresholds — and you should have — pin `jev-1.13.0`, log the returned `model` field for auditability, and replay a golden set before moving.

---

## 9. Performance and economics

TypeSafe's own evaluation: **4 workflows** — security incident response, agent-trace observability, invoice processing, customer service — scored against the averaged predictions of GPT-6 Astra and Fable as reference.

| Model | Accuracy vs. reference | Invalid structured output |
|---|---|---|
| **Jev** | **67.8%** | **0%** (structural guarantee) |
| GPT-5.6 Terra | 67.9% | 0.58% |
| GPT-5.6 Sol | 74.1% | 17.0% (on tool calls) |
| Claude Opus 5 | 73.1% | 5.73% |
| Claude Haiku 4.5 | — | 45.5% |

| Metric | Jev | GPT-5.6 Terra |
|---|---|---|
| Latency / case | 0.114 s | 8.566 s (**193.6×**) |
| Cost / case | $0.000081 | $0.013880 (**444.6×**) |

Per 1,000 workflows: **$0.39 (Jev)** vs $3.31 (GPT-5.6 Luna) vs $19.49 (Claude Haiku 4.5).

**Read this table correctly.** Jev is *statistically tied with Terra on accuracy* while being roughly 25–190× faster and 76–445× cheaper, and it gives up a few points to the strongest reasoning models. That is the actual trade: **a small accuracy concession for a large latency, cost and reliability win**, plus elimination of an entire class of parse/validation failure.

**Caveats, which are not small:**

- TypeSafe designed the workflows, built the harness and ran the evaluation. **No independent reproduction exists.**
- The company itself concedes these figures "sit at the high end of real use."
- Early-access pricing may be subsidized; it is unconfirmed whether $0.042/M is sustainable.
- Accuracy is measured as *agreement with other models*, not against human ground truth.

---

## 10. What problems Jev can solve

The shape of a good Jev problem: **high volume × bounded answer space × latency or cost pressure × no prose required.** If all four hold, Jev is likely the right tool. If any fails, reconsider.

### 10.1 Content moderation and safety gating

Pre-publish filtering of messages, comments and posts — a place where LLM latency is simply disqualifying, because you cannot make a user wait 8 seconds to post. One ~100 ms call carries the whole policy:

```
noul   "Is this harassment or a threat against a person or group?"
noul   "Does this contain a scam, phishing, or gift-card solicitation?"
noul   "Is this spam or repetitive flooding?"
noul   "Does this indicate self-harm risk for the sender?"
choice "Recommended action: allow / hide / timeout / escalate"
score  "Overall severity, 0–3"
```

Then threshold in code (`hide if harassment >= 0.5`). Separate `noul`s per policy rather than one "is this bad?" means each policy gets its own tunable threshold and its own precision/recall curve.

### 10.2 Routing and classification

Support tickets to teams, emails to queues, documents to pipelines, intents to handlers, requests to models. The canonical `choice` workload. **Model routing** is the reflexive one: use Jev to decide whether a request needs a frontier model at all — the routing decision costs a tiny fraction of the call it might avoid.

### 10.3 Agent guardrails and tool-call gating

Before an agent executes a tool call, judge it: is this destructive, does it need confirmation, is it in scope, does it touch production? This is the "AutoMode" / risk-gating pattern already used by coding harnesses, and it has to be fast because it sits in the inner loop — an 8-second safety check makes the agent unusable; a ~100 ms check does not.

Related: **context compaction** — score each past tool result for continued relevance and drop the rest, instead of paying an LLM to summarize its own history.

### 10.4 Scoring, prioritization and ranking

Lead quality, churn risk, urgency, frustration, relevance, content quality, review helpfulness. The `score` primitive with prose rubrics, combined via §8.3 into weighted composites. Replaces both hand-tuned heuristics and the bespoke classifier you were going to train.

### 10.5 Verification, fact-checking and LLM-output QA

*"Does this evidence support this claim?"* · *"Is this citation actually relevant?"* · *"Does this answer address the question asked?"* · *"Did the model follow the system prompt?"* — evaluated in parallel. This is the **LLM-as-judge replacement** case, and it is compelling because judging is exactly a bounded decision, while grading with a frontier model is one of the most expensive habits in production AI today.

### 10.6 RAG and retrieval quality

Post-retrieval reranking and relevance filtering, at a price point where you can afford to judge *every* chunk rather than trusting the embedding score, and a latency that fits inside the retrieval step.

### 10.7 Real-time control loops

Game AI, robotics, browser automation, interactive agents — anything whose loop budget is tens to hundreds of milliseconds. Tetris bots, Minecraft agents, next-browser-action selection at fractions of a cent per step. This category was **not previously possible at all** with frontier LLMs; it is a capability unlock rather than a cost optimization.

### 10.8 Bulk labelling, screening and structured extraction

Résumé screening, PII detection, fraud flagging, invoice field classification, dataset annotation, feature generation for downstream ML. Corpus-scale work that LLM pricing made prohibitive.

Note the constraint on extraction: Jev **cannot emit novel text**, so "extraction" means *selecting among candidates you supply as `choice` options*. Pull candidates with code or a regex, let Jev pick the right one.

### 10.9 Branching business logic that resists rules

Any `if` statement whose condition is a judgement rather than a computation. This is arguably the most general framing of what Jev is for: it makes semantic judgement cheap enough to put **inside ordinary control flow**, at ordinary function-call latency.

---

## 11. Where Jev does not fit

| Limitation | Detail |
|---|---|
| **No generation** | Will not write prose, code, summaries, replies or explanations. Not a partial limitation — it is the design. |
| **No rationale** | Returns probabilities, never reasons. If a regulated domain needs an audit trail *explaining* a decision, Jev alone cannot provide one. |
| **Not a calculator** | Counting and arithmetic are unreliable. Count in code; ask one `noul` per item. |
| **Dates and ordering** | Dates are treated as text, not ordered quantities. Extract via `choice` over enumerated options, then order in code. |
| **No dependent reasoning** | Questions cannot see each other's answers. Multi-step chains need multiple requests with your code in between. |
| **Text only** | Multimodal input requires a separate vision/audio front-end. |
| **Finite context** | ~32–64k tokens. Not for whole-codebase or long-document analysis. |
| **You author the schema** | Jev does not invent answer spaces. If you cannot enumerate the outcomes, you cannot ask the question. |
| **High cardinality** | Beyond 255 options, cascade through multi-stage scoring. |
| **Closed and hosted** | No weights, no self-hosting, waitlist access, one vendor. A real availability and lock-in consideration. |
| **Contradictions** | Instructions that conflict with criteria degrade answers rather than erroring. |

**Keep a generative model for:** open-ended generation, dependency-heavy multi-step reasoning, and high-stakes finance/legal/medical decisions that require an explainable trail.

---

## 12. Reliability, security and operations

- **Authoring discipline.** Jev reads instructions **literally**. Negations and implicit conditions are taken at face value. When an answer looks wrong, reread the question first — usually the wording is underspecified, not the model. Never bury several judgements in one question, and never ask for something code can compute exactly.
- **Prompt injection is a live threat.** `state` frequently contains user-controlled content, and hostile content can shift answers. Because Jev *is* the guardrail in several patterns above, a successful shift is a security bypass rather than merely a bad answer. Treat state as untrusted input and include adversarial cases in your evals.
- **Evaluate before replacing anything.** Build a labelled set from *your* traffic and measure accuracy, calibration and threshold behaviour. Vendor benchmarks are directional at best and self-reported at worst.
- **A calibrated probability is not a safety guarantee.** Validate thresholds, abstention rules, exception handling and human escalation on representative data.
- **Version discipline.** Pin the model, log the returned `model` field, replay a golden set whenever you change a question, a criterion or a version.
- **Single-vendor dependency.** Hosted-only, early access, no fallback. If Jev sits in a critical path, design the degraded mode — a cheap LLM fallback or a conservative default — before you need it.

---

## 13. Quick decision checklist

```
Is the answer space enumerable in advance?           no  → use an LLM
Do you need prose, code, or an explanation?          yes → use an LLM
Does step N depend on step N-1's answer?             yes → LLM, or split into calls
Is volume high, or latency/cost the bottleneck?      no  → an LLM is fine; don't add a vendor
Can you build a labelled eval set from real traffic? no  → build one first
                                                     otherwise → Jev fits
```

---

## 14. Sources

Primary:

- [TypeSafe AI — API reference](https://docs.typesafe.ai/api)
- [TypeSafe AI — documentation](https://docs.typesafe.ai/)

Technical deep dives:

- [Flavio Copes — A deep dive into Jev](https://flaviocopes.com/jev/)
- [DataCamp — Jev: TypeSafe's System One Model That Never Hallucinates](https://www.datacamp.com/blog/system-one-models-jev)
- [DEV — How to Use Jev: a practical guide](https://dev.to/valyuai/how-to-use-jev-a-practical-guide-to-typesafes-system-one-model-g5e)
- [SmartScope — Public specifications, inferred architecture and adoption boundaries](https://smartscope.blog/en/blog/jev-black-box-architecture-2026/)
- [MarkTechPost — TypeSafe AI releases Jev](https://www.marktechpost.com/2026/09/19/typesafe-ai-releases-jev/)

Use cases and integration:

- [CloudRaft — Top use cases of Jev](https://www.cloudraft.io/blog/top-use-cases-of-jev-typesafe-ai-model)
- [LangChain — Building a harness with Jev](https://www.langchain.com/blog/building-a-harness-with-jev)
- [LiteLLM — TypeSafe AI (Jev) pass-through](https://docs.litellm.ai/docs/pass_through/typesafe)

Company and funding:

- [SiliconANGLE — TypeSafe AI exits stealth with $40M](https://siliconangle.com/2026/09/16/typesafe-ai-exits-stealth-with-40m-to-build-ai-for-use-by-software/)
- [AI News — ChatGPT pioneer launches Jev model for programmatic logic](https://www.artificialintelligence-news.com/news/chatgpt-pioneer-launches-jev-model-for-programmatic-logic/)
- [Dealroom — TypeSafe exits stealth with $40M seed](https://dealroom.co/news/151032-typesafe-exits-stealth-with-40m-seed-to-build-ai-for-software-not-people/)
