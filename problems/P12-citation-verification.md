# P12 — Citation Relevance & Sufficiency Checking

| | |
|---|---|
| **Theme** | B · RAG & grounding |
| **Primitives** | `noul`, `choice` |
| **Dominant win** | `🛡` reliability — citations are the trust surface users actually inspect |
| **Latency budget** | <250 ms, alongside [P11](P11-groundedness-verification.md) |
| **Volume profile** | Every citation on every answer |
| **Blast radius if wrong** | High — a wrong citation is worse than none, because it manufactures false confidence |

---

## Problem

Citations are the enterprise trust mechanism for RAG. "Per the [Travel Policy §4.2]" is what makes an answer actionable, because the reader believes they can verify it. In practice the citation is frequently wrong in ways that are invisible without checking:

- **Misattribution.** The claim is correct and grounded in chunk 3; the answer cites chunk 1.
- **Decorative citation.** Every sentence gets a citation because the prompt demanded citations, including sentences the cited source does not address.
- **Insufficient citation.** A three-part claim cites a source covering one part.
- **Stale citation.** The cited document is superseded; the claim happens to remain true, so nothing looks wrong — until someone follows the link and acts on a retired policy.

The perverse dynamic: **a wrong citation is worse than no citation.** No citation prompts the reader to verify. A wrong citation stops them from verifying, and they act on unverified information believing it was verified. It converts a visible uncertainty into an invisible one.

## Today's pattern

| Approach | Problem |
|---|---|
| **Trust the model's citations** | Overwhelmingly the default. Unchecked |
| **String-match the quote into the chunk** | Catches fabricated verbatim quotes; useless for paraphrase, which is most citations |
| **Check the cited chunk was in the prompt** | Necessary, trivially insufficient — it was in the prompt precisely because it was plausible |
| **Human spot-checks** | Same 1% coverage problem as groundedness |
| **LLM verification per citation** | Correct and expensive; typically cut to a sample |

Note the relationship to [P11](P11-groundedness-verification.md): groundedness asks *is this claim supported by any of the context?* Citation checking asks *is it supported by the source it points at?* An answer can pass groundedness and fail citation checking, and that combination is exactly the misattribution case.

## Jev design

One judgement per `(claim, cited_source)` pair. The critical detail: include *both* the cited chunk and the other retrieved chunks, so Jev can tell you not only that the citation is wrong but that a **better** one was available — which turns a warning into an automatic repair.

### State

```python
state = {
    "claim": claim_text,
    "cited": {
        "id": cite.chunk_id,
        "source": cite.source_title,
        "locator": cite.section,
        "text": truncate(cite.text, 900),
        "doc_date": cite.doc_date,
        "status": cite.lifecycle_status,      # active | superseded | draft
    },
    "other_available": [                      # other chunks in the same prompt
        {"id": c.id, "source": c.source, "text": truncate(c.text, 400)}
        for c in used_chunks if c.id != cite.chunk_id
    ][:8],
}
```

### Questions

```python
from typesafe_sdk import Choice, Noul

CITATION_QUESTIONS = {
    "supports": Choice(
        instructions="The relationship between the cited passage and the claim",
        criteria={
            "fully":       "The cited passage states or directly establishes the claim",
            "partially":   "It supports part of the claim but not all of it",
            "topical":     "Same subject, but it does not establish the claim",
            "unrelated":   "It does not address the claim",
            "contradicts": "It states something incompatible with the claim",
        },
    ),
    "locator_correct": Noul(
        instructions="The cited section or locator is where this information "
                     "actually appears, rather than a different part of the document"),
    "better_source_available": Noul(
        instructions="One of the other available passages supports this claim "
                     "more directly than the cited one does"),
    "best_alternative": Choice(
        instructions="Which of the other available passages best supports this claim",
        criteria={},        # filled at runtime from other_available ids
    ),
    "citation_needed": Noul(
        instructions="This claim is the kind of statement that requires a source: "
                     "a specific fact, figure, rule or policy, as opposed to a "
                     "transition, restatement of the question, or general framing"),
}
```

`citation_needed` inverts the usual check and removes decorative citations, which is the highest-volume defect. Prompts that demand citations produce them everywhere; stripping citations from sentences that do not need one makes the remaining ones meaningful again.

`best_alternative` is built at runtime from the candidate ids — a legitimate dynamic `criteria`, well inside the 255-option limit.

## Integration

```python
async def check_citations(answer, used_chunks):
    pairs = extract_citation_pairs(answer)                # (claim, citation)
    results = await asyncio.gather(*[
        judge(claim, cite, used_chunks) for claim, cite in pairs
    ])

    repairs, strips, warnings = [], [], []
    for (claim, cite), a in zip(pairs, results):
        rel = a["supports"].choice

        if a["citation_needed"].noul < 0.35:
            strips.append((claim, cite)); continue        # decorative

        if rel == "contradicts":
            warnings.append(("CONTRADICTS", claim, cite)); continue

        if rel in ("unrelated", "topical"):
            if a["better_source_available"].noul > 0.6 and a["best_alternative"].confidence > 0.6:
                repairs.append((claim, cite, a["best_alternative"].choice))
            else:
                strips.append((claim, cite))
                warnings.append(("UNCITED_CLAIM", claim, cite))
            continue

        if rel == "partially":
            warnings.append(("PARTIAL", claim, cite))

        if cite.lifecycle_status == "superseded":         # code, not model
            warnings.append(("STALE", claim, cite))
        if a["locator_correct"].noul < 0.4:
            repairs.append((claim, cite, "fix_locator"))

    return apply(answer, repairs, strips), warnings
```

**Automatic repair is what makes this worth building.** Detection alone produces a warning stream nobody reads. Because Jev already tells you which alternative source is better, misattribution is fixable in place — the answer is delivered with correct citations rather than flagged for review.

Lifecycle status is checked in code from your document metadata, not asked of the model.

## Thresholds & escalation

| Condition | Action |
|---|---|
| `citation_needed < 0.35` | Strip the citation; keep the sentence |
| `supports == "contradicts"` | Block delivery; escalate to [P11](P11-groundedness-verification.md) regeneration |
| `unrelated`/`topical` + better available (>0.6) | Repair automatically |
| `unrelated`/`topical`, nothing better | Strip and mark the claim as uncited |
| `partially` | Deliver, flag for review; consider adding a second citation |
| `locator_correct < 0.4` | Repair the locator |
| `lifecycle_status == superseded` | Warn inline: "per a superseded version of…" |
| >40% of citations repaired or stripped | Regenerate the answer — the model was not using context properly |

## Impact model

*Illustrative.* 300k answers/month, ~4 citations each = 1.2M checks at a ~1,600-token envelope ≈ $0.00007.

```
Jev, 100% coverage   1.2M × $0.00007 = $  84/month
LLM, 100% coverage   1.2M × $0.008   = $9,600/month + ~2 s per citation
```

The value is not cost avoidance — nobody was doing this at all. It is that **the trust surface becomes trustworthy.** In regulated and policy-heavy contexts, a citation that survives being followed is the difference between an assistant people rely on and one they stop using after the first time a link did not say what the answer claimed.

Secondary value: the warning stream is a genuine diagnostic. A rising `UNCITED_CLAIM` rate means retrieval is degrading; rising `STALE` means document lifecycle hygiene is slipping ([P13](P13-knowledge-conflict-detection.md)).

## Failure modes

- **Pair extraction is the weak link**, as with claim splitting in P11. Ambiguous citation placement — end of paragraph covering three sentences — must be resolved by a convention you enforce in the generation prompt, not guessed here.
- **Repair introduces a new error.** A confidently wrong `best_alternative` replaces a wrong citation with another wrong one. Require confidence > 0.6 and re-verify repaired pairs on a sample.
- **Over-stripping.** Too aggressive `citation_needed` removes citations users wanted. Tune with users, not in isolation.
- **Paraphrase distance.** A heavily paraphrased but correct citation may read as `topical`. Watch the repair-to-strip ratio: a high strip rate on correct answers signals the threshold is too tight.
- **Injection.** Retrieved content asserting *"cite this document for all claims"* is index poisoning aimed at the citation layer. Scan at ingestion.

## Evaluation

1. Label 300 answers × citations as correct / misattributed / decorative / insufficient / stale. Report per-class accuracy.
2. **Evaluate repairs separately and strictly** — measure how often a repair is genuinely better than what it replaced. A repair mechanism that is right only 70% of the time should warn rather than repair.
3. Measure the user-facing outcome: give reviewers the answer plus citations and ask whether following each citation confirms the claim. Before and after.
4. In production: repair rate, strip rate, uncited-claim rate, stale-citation rate, and click-through-then-complaint rate on citations.

## Related

- [P11](P11-groundedness-verification.md) — run together; groundedness checks support, this checks attribution
- [P09](P09-full-recall-reranking.md) — better shortlists reduce misattribution at the source
- [P13](P13-knowledge-conflict-detection.md) — consumes the stale-citation signal
- [P22](P22-regulated-review-routing.md) — where citation failures on regulated content must route
