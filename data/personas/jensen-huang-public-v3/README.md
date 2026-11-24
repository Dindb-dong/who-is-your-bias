# Jensen Huang public-source RAG corpus v3

This corpus supports a clearly labeled Jensen Huang AI simulation that answers directly in the first person when public evidence supports the experience or stated reasoning. It expands factual and decision coverage while preserving the difference between original wording, editorial interpretation, and a newly generated application of a criterion.

## Contents and current scope

| Item | Count or status |
|---|---:|
| Registered sources | 46 |
| Distinct source-event families | 39 |
| Evidence records in the built corpus | 155 |
| Retained records from values-v2 | 54 |
| Newly reviewed records | 101 |
| New records containing a short original excerpt | 26 |
| Records eligible for the recorded snapshot's general historical query | 152 |
| Model fine-tuning or new weights | None |
| Speech, gesture, or visual likeness qualification | Not performed |

The counts come from [build-report.json](build-report.json), [corpus-manifest.json](corpus-manifest.json), and the [retrieval report](../../../reports/jensen-public-rag-2026-11-24/retrieval.json). The 46 source pages are not 46 independent interviews: clips and reports of the same event share an event family in [split-manifest.json](split-manifest.json).

The initial values-v2 corpus contains 60 records. Six inherited No Priors 2023 records belong to an event reserved by the older acquisition register and are excluded from this new corpus; values-v2 remains unchanged. Prior exposure means No Priors 2023 must not be presented as a fresh independent holdout.

## What the material covers

| Coverage | Representative sources |
|---|---|
| Public life experiences, early jobs, founding, education, philanthropy | Stripe, NVIDIA's Denny's account, Oregon State, Oneida, Stanford, NTU, Caltech, Hoover |
| Decisions under uncertainty, missing markets, changed technical assumptions, failure, resource allocation | Stripe, Stanford GSB/STVP, Acquired, Sequoia, Strange Loop, Milken |
| Programmability, system constraints, compute economics, supply chains, AI applications | Dwarkesh, Lex Fridman, CSIS, Citadel Securities, CNBC, Y Combinator |
| Dated company announcements and public technical explanations | NVIDIA GTC, COMPUTEX, CES, SIGGRAPH, World Governments Summit |
| Presentation history, synthetic-speaker attribution, verified social-channel identity | NVIDIA production reports and its SEC filing |

See the [complete source catalog](../../acquisition/jensen-public-2026-11-24/README.md#complete-built-source-catalog) for URLs, source IDs, dates, provenance, and event families. Coverage is selective; a late snapshot date does not imply complete contemporary coverage.

## Evidence and first-person presentation

New records contain bounded editorial paraphrases. Their `retrieval_aliases` are authored English and Korean search prompts, not original interview questions. The `persona_memory` object records question context, decision criteria, conditions for revising a judgment, response structure, any short verified excerpt, and explicit `first_person_eligible` attribution. An empty annotation means the reviewed material did not justify that field.

Only `verbatim_excerpt` is represented as original wording. A source may be an original publisher transcript while its runtime evidence remains an editorial summary. This release does not add a complete original Q&A training dataset, aligned speech, or an acoustic style corpus. The 26 excerpt-bearing records do not constitute a measured style-replication result.

The [profile](profile.json) and [production perspective contract](../../../src/bias/persona_voice.py) use `jensen-public-first-person.v1`:

- Use first-person experience only when speaker attribution supports it. An interviewer question, a company report, or a third-party biography is not automatically a personal memory.
- Apply a supported decision criterion to a new situation and explain the condition that would change the decision. Do not present the new application as an authentic historical quotation.
- Keep source markers and source cards available without repeatedly narrating the answer as a news report.
- Treat a retrieval miss as missing evidence. It cannot establish that an experience never occurred.
- Preserve dated company-claim attribution, the interface's simulation notice, honest identity answers, and attribution to the student team for project implementation.

These are implementation contracts and provenance controls. Their presence does not establish that every generated answer obeys them.

## Retrieval and temporal eligibility

The `jensen-public` backend implements deterministic bilingual BM25 with reviewed query aliases and decision annotations. It uses a finite English/Korean concept vocabulary and event-aware result ordering; it does not require an embedding model download or API key. It is not a semantic entailment checker.

The raw corpus contains 155 records. The [registered snapshot](../../snapshots/jensen-public-2026-11-24.json) makes 152 eligible for the report's general historical query. The three excluded records, `JHJ-044` through `JHJ-046`, refer to `stanford-etl-2003-time`: its exact PDF URL returned HTTP 404 during raw capture, its original publication date remains unknown, and its provisional same-day date has no verified acquisition timestamp. The event's 2003 recording date does not prove when this document became available. Exclusion is conservative temporal handling, not a finding that the interview content is false.

Eight other documents with unknown publication dates have exact original bytes preserved in a separately stored archive. The [capture manifest and persistence receipt](../../acquisition/jensen-public-2026-11-24/README.md#original-document-capture-and-dates) support observed availability with hashes and acquisition timestamps. They do not invent publication dates. Production routing and current-versus-historical question handling may further restrict records.

## Rebuild and assess

Run from the repository root:

```bash
PYTHONPATH=src python tools/build_jensen_public_corpus.py
PYTHONPATH=src python tools/evaluate_jensen_public_rag.py
```

For an exact cutoff, pass the timezone-aware `built_as_of` value from `corpus-manifest.json` to the builder's `--as-of` option. The builder validates references, explicit first-person metadata, bounded excerpts, source-event partitions, and snapshot consistency. The manifest records acquisition-shard hashes and available original-source hashes.

The [development evaluation](../../../reports/jensen-public-rag-2026-11-24/README.md) measures source-family retrieval on authored cases. A separate NBA Technology Summit 2025 heldout protocol is prepared but has not been executed. No heldout model-quality, human-likeness, or audio/video improvement score is claimed.
