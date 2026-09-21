# P13 — Knowledge-Base Conflict & Staleness Detection

| | |
|---|---|
| **Theme** | B · RAG & grounding |
| **Primitives** | `noul`, `choice` |
| **Dominant win** | `📈` coverage — corpus-wide hygiene nobody currently performs — plus `🛡` |
| **Latency budget** | Batch/offline; not in a request path |
| **Volume profile** | Corpus-scale: documents × documents in candidate pairs, run on a schedule |
| **Blast radius if wrong** | Low — output is a review queue for humans, not an automated edit |

---

## Problem

RAG systems are only as good as the corpus, and enterprise corpora rot in predictable ways nobody monitors:

- **Duplicated policy** across a wiki, a PDF and an intranet page, drifting apart over years.
- **Superseded documents** still indexed because nothing deletes anything.
- **Direct contradictions** between sources — the wiki says 30 days' notice, the policy PDF says 60.
- **Orphaned drafts** indistinguishable from approved content.
- **Regional variants** with no jurisdiction markers, so the wrong one is retrieved.

The user-visible symptom is maddening: the assistant is *inconsistent*. It gives 30 days on Monday and 60 on Tuesday, depending on which chunk ranked higher. Reranking cannot fix it, because both sources look authoritative and only one is current. The defect is in the corpus, and no amount of retrieval tuning will repair it.

Nobody audits this, because auditing an N-document corpus for contradictions is an O(N²) comparison problem. At 50,000 documents that is prohibitive with humans and prohibitive with LLMs. At ~$0.0001 per comparison, against a candidate set narrowed by embedding similarity, it becomes a scheduled job.

## Today's pattern

| Approach | Problem |
|---|---|
| **Nothing** | The overwhelming default. Conflicts are discovered by users, one at a time |
| **Manual content audits** | Annual at best, immediately stale, and they never complete |
| **Near-duplicate detection by hash or embedding** | Finds duplicates, cannot tell you which is current or whether they actually disagree |
| **Metadata-driven lifecycle** | Correct answer, but requires discipline no organisation sustains; `last_reviewed` fields go stale too |
| **LLM corpus audit** | Right capability, wrong economics at corpus scale |

## Jev design

Two passes. Pass 1 judges each document alone for staleness signals. Pass 2 judges candidate *pairs* — narrowed by embedding similarity — for genuine contradiction.

### Pass 1 — per-document staleness

```python
state = {
    "doc": {"title": d.title, "text": truncate(d.text, 4000),
            "last_modified": d.mtime, "path": d.path},
}
```

```python
from typesafe_sdk import Choice, Noul

STALENESS_QUESTIONS = {
    "lifecycle": Choice(
        instructions="What lifecycle stage this document's own content indicates",
        criteria={
            "current":     "Presents itself as current and in force",
            "draft":       "Marked draft, proposed, WIP, or under review",
            "superseded":  "States it is replaced, retired, deprecated or archived",
            "historical":  "Describes a past state of affairs, e.g. a postmortem or old release note",
            "unclear":     "Gives no indication either way",
        },
    ),
    "has_expiry_language": Noul(
        instructions="The document contains dates, periods or events after which "
                     "its content is expected to no longer apply"),
    "references_retired": Noul(
        instructions="It refers to systems, teams, products or roles as current "
                     "that an organisation would plausibly have retired"),
    "authority": Choice(
        instructions="What kind of source this is",
        criteria={
            "policy":    "An official policy, standard or contract",
            "procedure": "An operational runbook or how-to",
            "reference":  "Reference material or documentation",
            "discussion":"A thread, ticket, chat export or email",
            "personal":  "Personal notes or a scratch document",
        },
    ),
    "jurisdiction_scoped": Noul(
        instructions="The content applies only to a specific country, region, "
                     "entity or business unit rather than universally"),
}
```

### Pass 2 — pairwise conflict

```python
state = {
    "topic": cluster_label,
    "a": {"title": a.title, "source": a.path, "text": truncate(a.text, 2000)},
    "b": {"title": b.title, "source": b.path, "text": truncate(b.text, 2000)},
}
```

```python
CONFLICT_QUESTIONS = {
    "relationship": Choice(
        instructions="The relationship between these two documents on their shared topic",
        criteria={
            "agree":        "They state the same thing",
            "contradict":   "They state incompatible things about the same subject",
            "different_scope": "They differ because they apply to different regions, "
                              "entities, populations or time periods — not a real conflict",
            "one_supersedes":"One is plainly a later revision of the other",
            "complementary": "They cover different aspects and do not overlap",
            "unrelated":    "They are not actually about the same topic",
        },
    ),
    "conflict_is_material": Noul(
        instructions="If someone relied on the wrong one of these two, they would "
                     "reach a wrong decision or take a wrong action"),
    "conflict_locus": Choice(
        instructions="What kind of detail they disagree about",
        criteria={
            "number":     "An amount, period, threshold or percentage",
            "eligibility":"Who or what qualifies",
            "process":    "The steps or sequence required",
            "ownership":  "Who is responsible or must approve",
            "definition": "What a term means",
            "none":       "No specific disagreement",
        },
    ),
    "newer_is_a": Noul(
        instructions="Document A's content reads as the later or more current "
                     "version, judging from its content rather than any timestamp"),
}
```

`different_scope` is the essential category. Most apparent contradictions in a real corpus are legitimate regional or entity variants, and a detector without this category produces a queue of false positives and gets switched off in week two.

## Integration

```python
def audit(corpus):
    docs = [(d, judge_staleness(d)) for d in corpus]              # pass 1, fan out

    for d, a in docs:
        if a["lifecycle"].choice in ("superseded", "draft") and a["lifecycle"].confidence > 0.7:
            deindex(d, reason=a["lifecycle"].choice)               # remove from RAG index
        if a["references_retired"].noul > 0.6:
            queue_review(d, "references possibly retired entities")
        set_metadata(d, authority=a["authority"].choice,
                        jurisdiction_scoped=a["jurisdiction_scoped"].noul > 0.5)

    # pass 2 over similarity-narrowed candidate pairs only
    for a_doc, b_doc in candidate_pairs(corpus, similarity_threshold=0.82):
        c = judge_conflict(a_doc, b_doc)
        rel = c["relationship"].choice

        if rel == "contradict" and c["conflict_is_material"].noul > 0.6:
            open_ticket(a_doc, b_doc, locus=c["conflict_locus"].choice,
                        owner=content_owner(a_doc, b_doc), priority="high")
        elif rel == "one_supersedes" and c["relationship"].confidence > 0.7:
            older = b_doc if c["newer_is_a"].noul > 0.6 else a_doc
            queue_review(older, "appears superseded by sibling")
        elif rel == "different_scope":
            queue_metadata_fix(a_doc, b_doc, "add jurisdiction/scope markers")
```

Nothing here edits content. The outputs are **de-indexing, metadata enrichment, and tickets to content owners** — all reversible, all reviewable. That is what makes a model-driven corpus audit acceptable to a governance team.

The metadata side effect is quietly the most valuable part: `authority` and `jurisdiction_scoped` become filters and ranking features for [P09](P09-full-recall-reranking.md), so the audit improves retrieval even where it finds no conflicts.

## Thresholds & escalation

| Signal | Action |
|---|---|
| `lifecycle ∈ {superseded, draft}`, conf > 0.7 | De-index automatically; notify the owner |
| `contradict` + `material > 0.6` | High-priority ticket to both owners |
| `contradict` + `material < 0.6` | Low-priority backlog |
| `one_supersedes`, conf > 0.7 | Review queue, not automatic de-index — get this wrong and you delete the current version |
| `different_scope` | Metadata fix: add jurisdiction markers |
| `references_retired > 0.6` | Human review |

`one_supersedes` deliberately does not auto-act. `newer_is_a` is a content judgement without access to reliable dates, and the cost of de-indexing the *current* policy is far higher than the cost of a review step.

## Impact model

*Illustrative.* 50k documents, quarterly audit. Pass 1: 50k × ~5,000 tokens ≈ $0.00021 each. Pass 2: similarity narrowing yields ~120k pairs at ~4,500 tokens ≈ $0.00019 each.

```
pass 1      50k × $0.00021   = $  11
pass 2     120k × $0.00019   = $  23
                               ─────
per audit                      $  34      → ~$136/year
```

An LLM equivalent runs into five figures per audit and would not be scheduled. A human audit of 120k document pairs does not happen at any budget.

The return is on the user side: assistant inconsistency is one of the fastest ways to lose trust in an enterprise deployment, and it is a *corpus* defect that retrieval tuning cannot address. Add the de-indexing of drafts and superseded documents — usually a meaningful fraction of an unmanaged corpus — and retrieval precision improves measurably as a side effect.

## Failure modes

- **False-positive conflicts drown the queue.** The first-order risk. `different_scope` plus the `material` gate exist for this; tune thresholds until the queue is credible before expanding coverage. A queue nobody trusts is worse than no queue.
- **De-indexing the current version.** Only auto-de-index on explicit self-declared lifecycle language, never on inferred supersession. Keep a reinstatement path and log every de-index.
- **Candidate pairing misses cross-domain conflicts.** Similarity narrowing will not pair a finance policy with an HR page that contradicts it. Accept the limit, or add a topic-cluster pass with looser thresholds.
- **Long documents exceed the state budget.** Truncation may cut the conflicting section. Audit at section granularity for long documents, not document granularity.
- **No content owner.** Tickets with no assignee die. Resolve ownership from repository metadata; where none exists, that absence is itself a governance finding worth reporting.

## Evaluation

1. Have content owners label 200 document pairs: real conflict / scope difference / supersession / no conflict. Report per-class accuracy, with **`different_scope` precision as the headline** — that class determines whether the queue is trusted.
2. Seed known conflicts deliberately and confirm detection.
3. Measure the downstream effect: retrieval precision and assistant answer consistency (ask the same question 20 times across a week and measure variance) before and after de-indexing.
4. Track over time: conflicts found per audit, time-to-resolution, repeat offenders by source system, and the de-index reinstatement rate — a rising reinstatement rate means the lifecycle threshold is too loose.

## Related

- [P09](P09-full-recall-reranking.md) — consumes the metadata this produces; `superseded_language` is the query-time version of this check
- [P12](P12-citation-verification.md) — its stale-citation warnings are the live signal that feeds this audit
- [P10](P10-query-intent-routing.md) — its knowledge-gap log identifies where the corpus is missing, not just wrong
- [P18](P18-drift-detection.md) — the same batch-audit mechanics applied to production traffic
