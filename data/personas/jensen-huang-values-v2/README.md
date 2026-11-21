# Optional Jensen conditional-values corpus

This corpus preserves all 32 baseline records and nine source records unchanged
from PR29 head `6631af9`. It adds 28 AI-reviewed editorial paraphrases:
24 from newly acquired sources and four from re-reading Acquired/Stripe.
There are 60 records, 16 source resources and 13 distinct event families.
These are development sources, not independent evaluation events.

| New source | Event / publication date | Capture |
| --- | --- | --- |
| Stanford GSB View From The Top | Published 2024-04-25; recording date unknown | Full publisher transcript |
| No Priors, Sarah Guo / Elad Gil | Published 2023-04-20; recording date unknown | Full publisher transcript |
| Milken Institute | Recorded 2026-05-04; publication date unknown | Full 20-page publisher transcript |
| Stanford ETL / STVP | Recorded 2009-04-08 | Four complete selected clip transcripts, one event |

Where only an event date is known, the legacy `published_at` field uses the
conservative observed-availability bound 2026-11-21, explicitly marked by
`publication_date_verified: false` and `published_at_basis`. It does not claim
that the transcript was first published on that date. `recorded_at` separately
retains the event date. This prevents assuming earlier public availability from
a recording date. Milken's PDF creation metadata is May 7, not publication proof.

Every new summary retains speaker, question context (explicitly paraphrased),
source locator, raw capture hash, review-text location and episode identity.
Only Milken has publisher time cues; no other timestamps or acoustic boundaries
were invented. All new records are factual-context eligible, not original
speech-style training, aligned speech or observed motion data. Review is by AI
agents; independent human source review remains a deployment acceptance step.

The graph contains 13 values/beliefs/style claims, 12 conditional strategies and
24 evidence anchors. Conditions and exceptions are editorial applications, not
claims that Huang actually advised a particular student. `supports`, `qualifies`
and `contradicts` are supported relationship types; this curated candidate uses
support and qualification links, with no invented contradictory testimony.
Multiple accounts of the same episode do not count as independent experiences.

Full transcripts are not redistributed in public Git. A verified private source
capture and the public source URLs/hashes are recorded in
[the execution record](../../../docs/persona-values-v2-execution.md) and
[acquisition manifest](acquisition-manifest.json). Repeated event IDs also ensure
the four Stanford clips cannot be treated as four unseen events in evaluation.

To try this candidate with the existing configured LLM:

```bash
export BIAS_CORPUS_DIR=data/personas/jensen-huang-values-v2
export BIAS_PERSONA_SNAPSHOT=data/snapshots/jensen-values-2026-11-21.json
export BIAS_PERSONA_GRAPH=data/personas/jensen-huang-values-v2/persona_graph.v2.json
export BIAS_PERSONA_MEMORY_FORMAT=graph
```

The graph is off by default. `flat` uses the same semantic contents in prose
relations. Restore the baseline by unsetting these four variables. The graph
adds no online extraction, model training, search service or additional LLM call.
It cannot supply missing project facts, live news or private emotional states.
