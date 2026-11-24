# Jensen public RAG development evaluation — 2026-11-24

The expanded bilingual retriever recovered an expected source family in its top four results for **54 of 54 authored positive development cases** and returned no hits for **10 of 10 abstention controls**. This is a retrieval regression result after development fixes. It is not a heldout model-answer accuracy, persona likeness, or semantic correctness score.

## Compared configurations

The [machine-readable report](retrieval.json) evaluates the same 64 questions with four configurations:

| Configuration | Corpus records | Positive source-family hits at 4 | Controls without hits |
|---|---:|---:|---:|
| Original corpus + legacy lexical retrieval | 32 | 5 / 54 | 10 / 10 |
| Values-v2 corpus + legacy lexical retrieval | 60 | 5 / 54 | 10 / 10 |
| Public-v3 corpus + legacy lexical retrieval | 155 | 30 / 54 | 10 / 10 |
| Public-v3 corpus + public bilingual BM25 | 155 | 54 / 54 | 10 / 10 |

The expanded corpus improves this development test even with the old retriever. The final configuration also changes bilingual token handling and indexed metadata. These comparisons do not isolate a single causal factor, and the small, authored question set does not estimate open-world reliability.

## Cases and scoring

The [case file](../../data/evaluation/jensen-public-rag-2026-11-24.json) contains **27 English and 27 Korean positive questions**, plus **6 English and 4 Korean controls**. It covers early jobs and founding, CUDA persistence, absent markets versus invalid measurements, programmability, system bottlenecks, leadership, public technical explanations, synthetic presentation, and social-channel attribution.

Questions were authored and frozen before the first comparison, then used during development. Five Korean alias/token-coverage issues were corrected after inspecting failures. The final 54/54 result is therefore a development/regression result on seen cases; it must not be described as unseen-interview generalization or an unbiased multilingual benchmark.

A positive passes when at least one retrieved source ID is in the case's accepted source set. This source-family target lets old and new corpora receive credit for relevant material from the same source even if their evidence IDs differ. Each row additionally reports `specific_new_turn_hit`; that separate diagnostic is not the headline metric. A match does not check whether the selected passage entails every detail of the question or whether the model generates a faithful answer.

A control passes when retrieval returns no hits. This measures these ten specified questions only. It does not establish universal abstention, privacy protection, or hallucination prevention. `retrieval_ms` is a local diagnostic and is not an end-to-end latency benchmark.

## Raw retrieval versus production temporal selection

The comparison calls each raw retriever with `limit=4` and `behavior=True`. It does not invoke the production conversational engine, project/news/advice routing, full prompt construction, or a language-model provider. It searches the complete 155-record public-v3 corpus.

The same report separately checks the registered snapshot using the general query `public interview experiences`. That temporal selection admits **152 records** and excludes:

| Evidence IDs | Source | Reason |
|---|---|---|
| `JHJ-044`, `JHJ-045`, `JHJ-046` | `stanford-etl-2003-time` | Unknown publication date, provisional same-day availability bound, and no verified raw capture; exact PDF URL returned HTTP 404 |

The machine reason is `publication_or_event_after_snapshot`. Here it reflects the conservative end-of-day bound on an unknown publication date, not a claim that the 2003 event occurred after the snapshot. Eight other undated sources have genuine captured bytes and verified acquisition timestamps, supported by an uploaded archive with an independently matched hash. See the [acquisition provenance](../../data/acquisition/jensen-public-2026-11-24/README.md#original-document-capture-and-dates).

Production questions can impose different temporal or routing restrictions. Do not report the raw 54/54 result as proof that every production route retrieves the same records or produces a correct answer.

## Dataset isolation and next evaluation

The built corpus contains **46 sources across 39 event families**, with **54 retained plus 101 newly reviewed records**. Its manifest excludes six inherited No Priors 2023 records because the older acquisition register reserved that event. Previous development exposure prevents treating No Priors 2023 as fresh independent evaluation. Original baseline files are preserved for reproducible comparison.

The separate [NBA Technology Summit 2025 protocol](../../data/acquisition/jensen-public-2026-11-24/judgment-heldout.json) is **prepared, not executed**. Its source event is not loaded into active retrieval or answer templates. A future comparison should keep the model revision, decoding, language, history, and output budget constant; randomize anonymous condition labels; and rate factual support, decision criteria, and discourse/style separately. For voice, synthesize the same answer text before attributing a difference to the speech system.

No model fine-tuning, live-provider answer-quality evaluation, blind human persona scoring, listening comparison, gesture timing assessment, or video likeness test was performed by this retrieval evaluation. Runtime tests and deployment/activation checks are separate evidence and are not counted here.

## Reproduction and artifacts

From the repository root, after building the public-v3 corpus:

```bash
PYTHONPATH=src python tools/evaluate_jensen_public_rag.py
```

The [evaluator](../../tools/evaluate_jensen_public_rag.py) writes `retrieval.json`. Its `corpus_sha256` must match the [corpus build report](../../data/personas/jensen-huang-public-v3/build-report.json) and the [snapshot](../../data/snapshots/jensen-public-2026-11-24.json). Source provenance, event grouping, shard hashes, and temporal eligibility are documented in the [corpus README](../../data/personas/jensen-huang-public-v3/README.md) and [acquisition README](../../data/acquisition/jensen-public-2026-11-24/README.md).
