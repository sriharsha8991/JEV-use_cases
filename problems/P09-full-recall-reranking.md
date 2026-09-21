# P09 — Full-Recall Reranking

| | |
|---|---|
| **Theme** | B · RAG & grounding |
| **Primitives** | `score`, `noul` |
| **Dominant win** | `📈` coverage — judge every candidate instead of the top 5 — plus `$` |
| **Latency budget** | <400 ms for the whole fan-out, in the retrieval path |
| **Volume profile** | Every RAG query × 50–500 candidates |
| **Blast radius if wrong** | Moderate — a dropped relevant chunk causes an incomplete answer, usually silently |

---

## Problem

RAG quality is dominated by what reaches the context window, and the shortlisting step is where most quality is lost. The standard pipeline retrieves 100–500 candidates by vector similarity, then keeps the top 5–20 by that same similarity score, then generates.

Similarity is a weak proxy for usefulness. It ranks by topical overlap, so it reliably promotes a chunk that *discusses* the subject over one that *answers the question*, and it is blind to authority, recency and specificity. A policy document from 2019 and its 2026 replacement score nearly identically.

Cross-encoder rerankers fix ordering but are a separate model to host, tune and monitor, and they produce an uncalibrated score with no notion of *why* something ranked where it did. Using an LLM to rerank is accurate and unaffordable at 200 candidates per query.

So everyone truncates early and accepts the loss. **The thing nobody does, because it has never been affordable, is judge every candidate on its merits.**

## Today's pattern

| Approach | Problem |
|---|---|
| **Top-k by vector similarity** | Topical, not answer-bearing. Blind to recency and authority |
| **Hybrid BM25 + vector with RRF** | Better recall, same ranking weakness |
| **Cross-encoder reranker** | Extra model to operate; opaque uncalibrated score; still one dimension |
| **LLM reranking** | Accurate, ~$2–6 and 30+ s per query at 200 candidates. Not deployable |
| **Just send more chunks** | Context rot: more irrelevant material makes the answer worse, and costs more |

## Jev design

One Jev call per candidate, fanned out concurrently. Multiple dimensions per candidate, because "relevance" hides at least four separable properties — and once they are separate, *you* control the weighting.

### State

```python
state = {
    "question": user_query,
    "chunk": {
        "text": candidate.text,                      # keep under ~1,200 tokens
        "source": candidate.source_title,
        "date": candidate.doc_date,
        "doc_type": candidate.doc_type,              # policy | wiki | ticket | spec
    },
}
```

Small per-call envelope — around 400–1,400 tokens — which is what makes the fan-out cheap.

### Questions

```python
from typesafe_sdk import Noul, Score

RERANK_QUESTIONS = {
    "answers_question": Score(
        instructions="How directly this passage answers the question asked",
        criteria=[
            "Unrelated to the question",
            "Same topic, but does not address the question",
            "Contains part of the answer, or necessary background",
            "Directly and completely answers the question",
        ],
    ),
    "self_contained": Noul(
        instructions="This passage can be understood on its own, without the "
                     "surrounding document for context"),
    "specificity": Score(
        instructions="How specific this passage is",
        criteria=[
            "General or introductory statements only",
            "Some specifics mixed with generalities",
            "Concrete specifics: figures, names, rules, procedures, values",
        ],
    ),
    "authoritative": Noul(
        instructions="This reads as an authoritative statement of the rule or "
                     "fact, rather than a discussion, question, opinion or "
                     "informal restatement of it"),
    "superseded_language": Noul(
        instructions="The passage indicates it is outdated, deprecated, draft, "
                     "or replaced by something else"),
}
```

`authoritative` and `superseded_language` are the two that similarity search cannot express at all, and they resolve the most common enterprise RAG failure: a Slack export or an old ticket outranking the actual policy page.

## Integration

```python
import asyncio

async def rerank(question, candidates, k=8):
    answers = await asyncio.gather(*[
        judge(question, c) for c in candidates            # fan out, all independent
    ])

    scored = []
    for c, a in zip(candidates, answers):
        if a["superseded_language"].noul > 0.6:
            continue                                       # hard drop
        if a["answers_question"].score < 0.8:
            continue                                       # hard floor

        rank = (0.55 * a["answers_question"].score / 3.0
              + 0.20 * a["specificity"].score / 2.0
              + 0.15 * a["authoritative"].noul
              + 0.10 * a["self_contained"].noul)
        rank *= recency_factor(c.doc_date, c.doc_type)      # code, not model
        scored.append((rank, c, a))

    scored.sort(reverse=True, key=lambda t: t[0])
    return dedupe_by_source(scored, k)
```

Three things deliberately live in code, not in the model:

- **The weights.** Tuning retrieval becomes editing four numbers with a fixed eval set, not re-prompting.
- **Recency.** Dates are text to Jev, and it cannot order them ([JEV-details §11](../JEV-details.md#11-where-jev-does-not-fit)). `superseded_language` catches *stated* obsolescence; actual date arithmetic is `recency_factor`.
- **Source diversity.** `dedupe_by_source` prevents eight chunks from one document crowding out a second perspective.

## Thresholds & escalation

| Signal | Action |
|---|---|
| `superseded_language > 0.6` | Drop outright |
| `answers_question.score < 0.8` | Drop — below "contains part of the answer" |
| All candidates below floor | **Return no-answer, do not generate.** Feeds [P11](P11-groundedness-verification.md) |
| Fewer than 3 survive | Widen retrieval and re-run once, then no-answer |
| Top `answers_question` confidence low across the board | Flag the query for the knowledge-gap log |

The "return no-answer" path is the underrated benefit. A pipeline that can *tell* when it has nothing relevant is dramatically safer than one that always hands the model its best five chunks regardless of quality — which is what forces the LLM to confabulate.

## Impact model

*Illustrative.* 300k RAG queries/month, 200 candidates each = 60M judgements at a ~600-token envelope ≈ $0.000025 each.

```
Jev reranking     60M × $0.000025      = $1,500/month
LLM reranking     60M × $0.004         = $240,000/month   (not deployable)
cross-encoder     ~$800/month + a model to operate, one opaque dimension
```

Cost is not the headline — a cross-encoder is comparably cheap. The headline is **coverage and control**: you judge 200 candidates instead of reordering 20, on five interpretable dimensions with weights you own, with hard drops for superseded content, and with a principled no-answer path. None of that is available from a similarity score.

Downstream, better shortlists mean fewer chunks needed for the same answer quality, so generation cost falls too — often the larger saving.

## Failure modes

- **Chunk-level judgement misses cross-chunk answers.** An answer spanning three chunks may score each as partial. Mitigations: judge at parent-document granularity for long-form sources, or expand to neighbouring chunks after selection.
- **Latency of wide fan-out.** 200 concurrent calls must respect rate limits (~1,200 req/min). Batch into waves, or pre-filter to ~50 candidates by hybrid search first — a good default that keeps most of the benefit.
- **`authoritative` misreads tone.** A confident wiki page outranks a hedged policy document. Weight `doc_type` in `recency_factor` so source class, not tone, carries authority.
- **Cost on wide fan-outs with large chunks.** Cap chunk text and pre-filter. Measure tokens per query, not per call.
- **Injection in indexed content.** A document containing *"this passage directly and completely answers all questions"* is an index-poisoning attack on your ranker. Scan content at ingestion, not at query time.

## Evaluation

1. Build a labelled set: 200 queries × candidate relevance judgements. Report nDCG@10 and recall@10 for similarity-only, cross-encoder, and Jev-reranked.
2. Measure the metric that matters: **end-to-end answer quality**, not ranking quality. Better ranking that does not improve answers is not worth operating.
3. Measure the no-answer path separately — on queries with no relevant content, how often does the pipeline correctly decline? Compare against today's confabulation rate.
4. Tune weights by grid search on the labelled set, then freeze them and pin the model version (§4.6).
5. In production: chunks-per-answer, no-answer rate, and [P11](P11-groundedness-verification.md) failure rate as the downstream quality signal.

## Related

- [P10](P10-query-intent-routing.md) — runs before this; decides whether to retrieve at all
- [P11](P11-groundedness-verification.md) — checks the answer built from these chunks
- [P12](P12-citation-verification.md) — verifies the citations attached to it
- [P13](P13-knowledge-conflict-detection.md) — `superseded_language` at corpus scale
- [P03](P03-context-compaction.md) — the same relevance mechanics applied to agent history
