# Reserved source references for adaptation evaluation

The 2026 roadmap requires fresh events for any new efficacy claim. Its frozen
training acquisition command rejects reserved test families. This workflow
provides the complementary **evaluation-only** path without changing that guard
or the historical A/B/C experiment.

`tools.persona_data.heldout` acquires reserved publisher documents, extracts
complete attributed original exchanges, checks them against explicitly supplied
exposed corpora, and freezes private reference cases. It produces neither a
retrieval corpus nor text, speech or visual training projections.

## Actual acquisition result

The [executed acquisition record](../reports/heldout/2026-v1/acquisition.json)
records successful ordinary HTTPS downloads of both reserved Lex publisher
pages. Extraction correctly produced **zero reference cases**:

| Registered event | Acquired primary page | Result |
| --- | --- | --- |
| Lex #18 | [Publisher episode page](https://lexfridman.com/elon-musk/) | Episode metadata and media links; no complete labeled publisher transcript |
| Lex #252 | [Publisher episode page](https://lexfridman.com/elon-musk-3/) | Episode metadata, topic outline and media links; no complete labeled publisher transcript |

The official pages expose publication dates, but this change leaves the
original frozen source registration untouched. Its unknown dates remain null.
The separately available Lex #49 publisher PDF belongs to an existing training
family; it was not substituted for a reserved event. Neither page outlines nor
media links were converted into invented reference answers. An attributable
publisher transcript, or a separately implemented and reviewed original-media
transcription path, is still required for these reservations. Tesla Q3 2026
remains a separate reserved source; its announced event alone supplies no
accepted speaker turns.

A [second bounded publisher-link investigation](../reports/heldout/2026-v2/publisher-investigation.json)
rechecked all three episode pages, the official podcast index, publisher search
and the actual linked PDF. The roadmap's **“interview 2 with transcript PDF”**
links to `/elon-musk-2/`, which is **Lex #49 (`lex-049-2019`, already train)**,
not reserved Lex #18. The downloaded 14-page PDF has explicit speaker labels;
its first page identifies the second conversation and a November 12, 2019
export. Its SHA-256 is
`999374c2f86694b31824dcc41bcc9f7b9c076a83a17327c6284af348416be6d2`.
This is a valid publisher transcript for an already exposed training event,
so it cannot resolve the fresh evaluation-data gap. No PDF or transcript link
for reserved #18 or #252 was found in the inspected official pages/search;
this bounded result is not a claim that no such document could exist elsewhere.
The v1 records and frozen assignments remain unchanged, and **fresh references
remain zero**. No parser/runtime changes or new human review were needed.

The successful materialization tests use clearly synthetic publisher-format
fixtures. They establish artifact integrity and split isolation, not newly
acquired real evaluation data or persona quality.

## Commands

Run from the repository root, choosing fresh directories each time:

```bash
uv run python -m tools.persona_data acquire-heldout \
  --source lex-018-2019 --source lex-252-2021 \
  --output data/local/heldout-acquisition-new

# Supply the full actual runtime/training corpus and all earlier exposed
# references. A preview alone cannot check an unprovided full corpus.
uv run python -m tools.persona_data build-heldout \
  --acquisition data/local/heldout-acquisition-new \
  --comparison-corpus data/local/persona-2026/corpus \
  --comparison-corpus data/qna/elon-musk-v1 \
  --comparison-corpus data/personas/elon-musk \
  --output data/local/heldout-references-new \
  --report reports/local/heldout-references-new

uv run python -m tools.persona_data verify-heldout \
  data/local/heldout-references-new
```

With the currently registered landing pages, `build-heldout` fails explicitly
with the source-specific missing-transcript reasons and creates no reference
dataset. Failed acquisitions remain in `heldout-acquisition.json`; no alternate
source is silently substituted.

The ordinary `acquire` and `build` commands continue to reject these same test
families. Renaming the held-out acquisition log into the training log format
does not bypass the split check.

### Register a newly discovered publisher transcript

If an original transcript URL is subsequently found and its event identity is
verified, record a separate document registration. Do not mutate the old split
assignment to make acquisition succeed. `acquire-heldout --documents FILE`
accepts the following structure:

```json
{
  "schema_version": "persona-heldout-documents.v1",
  "registered_at": "ACTUAL_TIME_WITH_TIMEZONE",
  "source_registry_sha256": "CANONICAL_REGISTRY_DIGEST",
  "split_registration_sha256": "CANONICAL_SPLIT_DIGEST",
  "documents": [
    {
      "source_id": "lex-018-2019",
      "acquisition_url": "VERIFIED_HTTPS_LEXFRIDMAN_PUBLISHER_DOCUMENT",
      "discovery_url": "VERIFIED_HTTPS_LEXFRIDMAN_DISCOVERY_PAGE",
      "adapter": "lex_publisher_html",
      "basis": "Record the actual evidence linking this original document to the reserved event."
    }
  ]
}
```

These are placeholders, not existing document URLs or a claimed registration.
Compute the two canonical digests with
`tools.persona_data.parsers.digest` after `load_registry`. Supported explicit
document formats are Lex's labeled/timestamped transcript HTML and its
speaker-labeled PDF (`lex_publisher_pdf`, requiring `pdftotext`). Both the
discovery and document URL must be on the original Lex publisher domain. The
document supplement is copied and hash-bound in the acquisition. An unrecognized
format still fails; no generic prose-to-dialogue inference is performed.

## Artifact and Python contract

```python
from pathlib import Path
from tools.persona_data.heldout import load_heldout

manifest, cases = load_heldout(Path("data/local/heldout-references-new"))
```

The artifact has `schema_version: persona-heldout-dataset.v1` and
`purpose: evaluation_only`. `manifest.json` binds the exact bytes of:

- `source-families.json` and `split-registration.json`;
- `cases.jsonl`, containing the private original references;
- `recipe.json`, containing exact source/transcript/selection identities;
- `exposure-index.json`, containing hashes of compared original wording and
  the file hashes of every comparison corpus.

Each `persona-heldout-case.v1` row includes:

| Fields | Meaning |
| --- | --- |
| `id`, `source_id`, `source_event_id`, `event_family_id`, `split` | Stable source lineage; split is always `test` |
| `question`, `question_sha256` | Original interviewer question and its exact UTF-8 digest |
| `context`, `context_before`, `context_sha256` | Zero or one preceding original turn, retained as a list of `{role, content}` messages plus the original string |
| `reference_answer`, `answer_sha256` | Private original answer used only for evaluation |
| `source_url`, `source_artifact_sha256`, `transcript_locator` | Publisher identity, exact downloaded document and canonical source-cue hashes |
| `published_at`, `event_at`, `ingested_at` | Distinct date roles; unknown publication/recording dates stay null |
| `review`, `reference_review_status` | Structural publisher-label review, explicitly `publisher_structure_only` |
| `projection_eligibility` | Factual, text-style, aligned-speech and visible-behavior flags are all false |

No runtime `profile.json`, `evidence.jsonl`, `corpus/`, `projections/` or training
JSONL is emitted. `Corpus.load` cannot interpret this artifact as a runtime
corpus. The held-out loader also rejects these names if they are subsequently
introduced into its directory. Consumers must pin the hash of `manifest.json`
itself in their experiment registration. File hashes detect changes relative
to that frozen manifest; they are not a third-party signature.

The adaptation evaluator must keep `reference_answer` outside all inference
prompts. It should freeze retrieved facts/style once per case using its actual
eligible training corpus, and require that corpus's file hashes to appear in
`manifest.leakage_policy.checked_corpora`. Comparing only a preview does not
establish absence of leakage from a different training corpus or base model.

## Leakage checks and their limits

Build and load both enforce reserved family/split identities and fail when a
reference source ID, event family, event ID or canonical source URL already
appears in a supplied comparison corpus. All records in those corpora count as
exposed, including historical test records already seen during earlier work.

Text checks cover normalized whole question/answer duplicates, exact answer
matches of ten or more words, and shared consecutive spans of twelve words
between a reference answer/context and previously exposed questions, answers
or context. Short acknowledgements are not treated as leaked substantive
answers. Exact duplicate held-out exchanges are deduplicated. An answer with a
twelve-word span already present in its own supplied question/context is
excluded, keeping that reference wording out of the generation prompt.

These checks are deliberately conservative. A matching span fails the build
for review; it is not silently removed until the score improves. They cannot
establish semantic independence, detect every paraphrase, or reveal a base
model's pretraining exposure. A different registry/corpus requires a new
artifact. Preserve the earlier rejected or failed run.

Reference review remains structural and automated. Human semantic review,
speaker/acoustic verification, fresh-case independence and blinded outcome
assessment remain separate requirements. `human_review_completed` is false;
changing that flag does not convert this artifact into a reviewed experiment.

## Verification

```bash
uv run pytest tests/persona_data/test_heldout.py -q
```

The regressions exercise successful isolated materialization, null dates and
original context, training-path refusal, comparison requirements, frozen
registry changes, downloaded byte/source identity, source-family overlap,
exact/embedded wording leakage, landing-page rejection, case/recipe tampering,
symlinks, runtime/training file refusal, private-wording exclusion from public
reports and immutable outputs.
