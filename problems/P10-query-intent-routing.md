# P10 — Query Intent & Retrieval Strategy Selection

| | |
|---|---|
| **Theme** | B · RAG & grounding |
| **Primitives** | `choice`, `noul` |
| **Dominant win** | `⚡` latency + `🛡` reliability |
| **Latency budget** | <150 ms, before retrieval |
| **Volume profile** | Every RAG query |
| **Blast radius if wrong** | Moderate — wrong strategy produces an unfounded or incomplete answer |

---

## Problem

Production RAG systems run one retrieval strategy for every query, and the strategy is tuned for the average query, which means it is wrong for most of them.

Consider four real queries into the same enterprise assistant:

| Query | What it actually needs |
|---|---|
| "What is our parental leave policy?" | Semantic retrieval over policy documents |
| "How many tickets did team B close last week?" | A SQL query. Retrieval cannot answer it at all |
| "What changed in the deployment runbook?" | Version diff — two retrievals and a comparison |
| "Summarize the Q3 incident reports" | Enumerate a document set, then map-reduce. Top-k is structurally wrong |

A single pipeline handles the first well and the other three badly. The second is the dangerous one: vector search happily returns chunks *about* ticket counts, the model produces a plausible number, and a confident fabrication reaches a user. That failure is caused by retrieval running when it should not have.

## Today's pattern

| Approach | Problem |
|---|---|
| **One strategy for everything** | Silently wrong on aggregation, comparison and enumeration queries |
| **LLM query planner** | Correct approach, but adds 2–5 s and a frontier call *before* retrieval even starts, on every query |
| **Keyword rules** ("how many" → SQL) | Brittle; misses paraphrase; no uncertainty |
| **Let the LLM decide via tool calls** | Works, but the model sees only the query and must guess what the corpus contains, and each wrong guess is a full round trip |

## Jev design

Classify **what kind of retrieval operation the question implies** — not what the answer is. Keep strategies few and orthogonal enough that a human can review the taxonomy.

### State

```python
state = {
    "query": user_query,
    "recent_turns": history[-2:],
    "available_sources": [                    # what exists, not its contents
        {"name": "policy_docs",   "kind": "document", "has_dates": True},
        {"name": "ticket_db",     "kind": "structured", "aggregatable": True},
        {"name": "runbooks",      "kind": "document", "versioned": True},
        {"name": "incident_reports", "kind": "document", "has_dates": True},
    ],
}
```

### Questions

```python
from typesafe_sdk import Choice, Noul

INTENT_QUESTIONS = {
    "operation": Choice(
        instructions="What operation is needed to answer this question",
        criteria={
            "lookup":      "Find a specific fact or passage",
            "aggregate":   "Count, sum, average or otherwise compute over many records",
            "compare":     "Contrast two or more specific things",
            "enumerate":   "List or summarise all items matching a condition",
            "temporal":    "Determine what changed, or what was true at a point in time",
            "procedural":  "Retrieve a sequence of steps or a process",
            "conversational": "Refers to this conversation, not the corpus",
        },
    ),
    "needs_retrieval": Noul(
        instructions="Answering requires information from the corpus, as opposed "
                     "to reasoning over what is already in the conversation"),
    "source": Choice(
        instructions="Which source is most likely to contain the answer",
        criteria=SOURCE_CRITERIA | {"multiple": "Several sources are needed",
                                    "none": "No listed source covers this"},
    ),
    "time_scoped": Noul(
        instructions="The question is limited to a specific time period, whether "
                     "stated explicitly or implied by words like current, latest, "
                     "recent, or last quarter"),
    "self_contained": Noul(
        instructions="The question can be understood without the earlier "
                     "conversation; it contains no unresolved pronouns or references"),
}
```

`self_contained` earns its place cheaply: when it is false, rewrite the query against the conversation before retrieving. Follow-up queries with unresolved pronouns ("what about for contractors?") are one of the largest sources of silent RAG failure, because the embedding of the raw follow-up bears no relation to what the user meant.

## Integration

```python
def plan(query, history, sources):
    a = client.system_one(state=build_state(query, history, sources),
                          questions=INTENT_QUESTIONS).answers

    if a["needs_retrieval"].noul < 0.3:
        return Plan(strategy="answer_from_context")

    if a["self_contained"].noul < 0.5:
        query = rewrite_with_context(query, history)   # one cheap LLM call, or a template

    op = a["operation"]
    if op.confidence < 0.55 or a["source"].choice == "multiple":
        return Plan(strategy="broad_hybrid", sources=top_sources(a["source"]), k=40)

    plan = {
        "aggregate": Plan(strategy="text_to_sql", source="ticket_db"),
        "enumerate": Plan(strategy="filter_then_mapreduce"),
        "compare":   Plan(strategy="multi_query", n=2),
        "temporal":  Plan(strategy="versioned_diff"),
        "procedural":Plan(strategy="parent_document", expand_neighbours=True),
        "lookup":    Plan(strategy="hybrid_topk", k=20),
    }[op.choice]

    if a["source"].choice == "none" and a["source"].confidence > 0.7:
        return Plan(strategy="no_answer", reason="no source covers this")

    if a["time_scoped"].noul > 0.6:
        plan.filters["recency"] = True
    return plan
```

Two design points:

- **`aggregate` → SQL, never retrieval.** This single branch eliminates the most dangerous RAG failure mode: fabricated numbers. If no structured source exists for an aggregate query, the honest answer is a refusal, not five chunks and a guess.
- **Ambiguity widens rather than picking.** Low confidence goes to `broad_hybrid` with a larger `k`, then relies on [P09](P09-full-recall-reranking.md) to sort it out. The cheap decision degrades to the robust default, never to a specific wrong strategy.

## Thresholds & escalation

| Signal | Action |
|---|---|
| `needs_retrieval < 0.3` | Answer from conversation; skip retrieval entirely |
| `self_contained < 0.5` | Rewrite the query first |
| `operation.confidence < 0.55` | Broad hybrid with a wide `k` |
| `source == "none"`, conf > 0.7 | No-answer, and log a knowledge gap |
| `source == "multiple"` | Multi-source retrieval, then rerank |
| `time_scoped > 0.6` | Apply a recency filter |

## Impact model

*Illustrative.* 300k queries/month.

```
LLM query planner   300k × $0.008  = $2,400/month, +3 s on every query
Jev planner         300k × $0.00011 = $   33/month, +0.1 s
```

Latency is the point: 3 seconds removed from the *front* of every RAG query, before retrieval and generation even begin.

The quality effect is larger than the cost effect. Assume 12% of queries are aggregations currently answered by retrieval with a fabricated number, and 15% are follow-ups whose unresolved references degrade retrieval. Correctly routing the first group and rewriting the second addresses roughly a quarter of all queries — and the first group is where user trust is actually lost.

## Failure modes

- **Taxonomy does not fit your corpus.** These seven operations are a starting point; derive yours from a sample of 200 real queries before writing the criteria.
- **`aggregate` with no structured source.** Must produce an honest refusal. Validate that every `aggregate` route terminates in either SQL or a refusal — never in retrieval. Assert this in a test.
- **Source classification degrades as sources multiply.** Past ~15 sources, do two stages: source *class* first, then instance.
- **Rewrite loses nuance.** A bad rewrite is worse than the original. Log original and rewritten queries together and sample them.
- **Injection via query text.** "Use the SQL tool on the users table." Consequence is a strategy choice, and your SQL layer must be independently permission-scoped — never let intent classification widen data access.

## Evaluation

1. Label 400 real queries with the correct operation and source. Report per-class accuracy, with `aggregate` recall as the headline — missing an aggregation is the failure that produces fabricated numbers.
2. Measure end-to-end answer accuracy per operation class, before and after. Expect the largest gains on `aggregate`, `enumerate` and `temporal`.
3. Evaluate rewrite quality separately on multi-turn conversations.
4. Track in production: per-strategy volume, `no_answer` rate, knowledge-gap log (a genuinely valuable product artifact), and fabricated-number complaints.

## Related

- [P09](P09-full-recall-reranking.md) — runs after this; absorbs the ambiguity this layer widens
- [P11](P11-groundedness-verification.md) — catches what wrong strategies produce
- [P13](P13-knowledge-conflict-detection.md) — consumes the knowledge-gap log
- [P08](P08-llm-admission-control.md) — `needs_retrieval` overlaps; run admission first
- [P05](P05-specialist-dispatch.md) — the agent-side analogue
