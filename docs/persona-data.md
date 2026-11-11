# 2026 persona data and adaptation workflow

This implementation advances WP0, WP1's data preparation and WP3's reproducible training interface in the [2026 roadmap](2026-persona-roadmap.md). It preserves the historical pilot and its negative retrieval result. Source text, trained weights and complete downloaded interviews remain in ignored local directories.

## What actually ran

On 2026-11-11, ordinary HTTPS acquisition succeeded for ten primary publisher documents. Nine supported original-dialogue formats yielded **977 original question/answer exchanges from nine independent training events**. The build contains **922 complete published exchanges and 55 explicitly marked excerpts** whose next source block has uncertain speaker attribution. “Complete” describes the selected publisher transcript exchange, not an independently verified complete acoustic utterance.

The TIME Person of the Year excerpt article was acquired but rejected as Q&A: topic headings and edited quotations are not original interviewer questions. The pipeline never manufactures questions for those quotations. No synthetic dialogue augmentation was used.

| Event | Accepted Q&A | Publisher format |
| --- | ---: | --- |
| [Dwarkesh, 2026](https://www.dwarkesh.com/p/elon-musk) | 281 | Explicit speaker-name headings and intervening paragraphs |
| [Recode, 2018](https://www.vox.com/2018/11/2/18053428/recode-decode-full-podcast-transcript-elon-musk-tesla-spacex-boring-company-kara-swisher) | 346 | Full-name anchors and publisher's bold-question/plain-answer transcript convention |
| [The Henry Ford, 2008 event](https://thf-prod-blobs.s3.amazonaws.com/docs/thehenryfordlibraries/innovator-transcripts/transcript_musk_full-length.pdf?sfvrsn=747231ce_1) | 72 | Explicit speaker labels in the publisher PDF |
| [Lex Fridman #49, 2019](https://lexfridman.com/wordpress/wp-content/uploads/2019/11/elon_musk_lex_fridman_2_transcript.pdf) | 41 | Explicit names and publisher timestamps in the PDF |
| [TED2013](https://www.ted.com/talks/elon_musk_the_mind_behind_tesla_spacex_solarcity?view=transcript) | 32 | Pinned publisher caption and speaker structure |
| [TED2017](https://www.ted.com/talks/elon_musk_the_future_we_re_building_and_boring?view=transcript) | 66 | Pinned publisher caption and speaker structure |
| [TED2022 live](https://www.ted.com/talks/elon_musk_elon_musk_talks_twitter_tesla_and_how_his_brain_works_live_at_ted2022?view=transcript) | 59 | Distinct live event, separate from TED2022 recorded |
| [Y Combinator, 2016](https://www.ycombinator.com/blog/elon-musk-on-how-to-build-the-future) | 42 | Explicit names and publisher timestamps |
| [WIRED, 2012](https://www.wired.com/2012/10/ff-elon-musk-qa/) | 38 | Full-name anchors followed by explicit abbreviated speaker labels |
| **Total** | **977** | **Nine independent events** |

The measured source inventory, acquisition hashes, accepted/rejected counts and materialization recipes are in [the expansion report](../reports/data-expansion/2026-v1/coverage.json). These are nine events, not 977 independent interviews. Recode and Dwarkesh dominate the raw exchange count. The data remains below the roadmap's 20–30-event target and has not produced the planned 2–5 hours of acoustically accepted speech. Only one newly materialized event is from 2026; the manifest date does not make historical positions current.

## Frozen families, exposure and holdouts

[Source families](../data/acquisition/elon-musk-2026/source-families.json) reconcile all 31 existing summary sources with the original-Q&A recipes and register three additional reserved events. The [split registration](../data/acquisition/elon-musk-2026/split-registration.json) was written before the new acquisition and extraction. It contains 28 training families, three development families and three reserved test families.

The already exposed Lex #400, Lex #438 and TED2022 recorded events are **development only in this new dataset**. Their historical files and old experiment assignments are untouched. Fresh test reservations cover Lex #18, Lex #252 and the [officially announced Tesla Q3 2026 Q&A webcast](https://ir.tesla.com/press-release/tesla-releases-third-quarter-2026-financial-results). Their transcripts and reference answers were not acquired for this build. The Lex reservation dates remain null until separately verified; an event name is not a verified publication timestamp. A reservation is not a completed test set or evidence of efficacy.

Family identity is explicit. The loader rejects duplicate assignments, recording families spanning splits and promotion of the exposed pilot evaluation events to training. Builds reject reserved test events. Exact normalized exchange duplicates are removed; long exact answer matches across different splits fail the build. This does not claim exhaustive detection of paraphrased, quoted or semantically duplicated passages.

## Four independent projections

Every projection retains the same segment ID, event family, source URL, original text hash and transcript hash. Eligibility is independent:

| Projection | Actual eligible records | Gate |
| --- | ---: | --- |
| Factual/position evidence | 745 | Training split, accepted publisher attribution, at least eight answer words; incomplete excerpts retain their qualification |
| Text style | 922 | Training split, original exact text, attributed original question and complete published exchange |
| Aligned speech | 0 | Separate accepted acoustic review, exact transcript/source mapping, real WAV hash, speaker identity, bounded interval, no uncleared overlap/cut/noise |
| Visible behavior | 0 | Separate accepted continuous-track review, source mapping, real media hash/duration and explicit visible channels |

The runtime corpus retains all 977 attributed QAs, including short conversational replies, with explicit `projection_eligibility` booleans on every record. Factual retrieval uses the 745 accepted factual records; the separate text-style path can use all 922 complete exchanges. A short backchannel may be a style example while remaining ineligible as factual evidence. Text eligibility does not confer factual support for an arbitrary generated claim. The retriever and per-claim grounding evaluation must establish relevance separately.

Audio and visual eligibility require a ledger keyed by record ID, with the original answer/transcript/source hashes, recording ID, reviewed source-clock mapping, reviewer identity/type/date and modality-specific quality checks. Unknown timing remains null. Caption onsets, PDF timestamps and the Henry Ford SMPTE labels do not establish acoustic end boundaries, clean speech hours or a learned behavior policy. Empty projections are written explicitly.

The current source review is by an AI agent: publisher structure, attribution patterns and extraction rules were inspected. It is not independent human review, factual-entailment adjudication or a listening/likeness judgment. New records use only the generic `conversation` situation label; topic tags are deterministic lexical hints. No situation-specific labeling advantage is claimed from this expansion.

## Reproduce the local runtime corpus

Run from the repository root with the existing Python environment and `pdftotext` available. The PDF adapter uses an argument vector without a shell. Choose fresh output directories; existing snapshots are never overwritten.

```bash
uv run python -m tools.persona_data acquire \
  --source ted-2013 --source ted-2017 --source ted-2022-live \
  --source dwarkesh-2026 --source recode-2018 --source yc-2016 \
  --source wired-2012 --source lex-049-2019 --source henry-ford-2008 \
  --source time-2021-boca-chica \
  --output data/local/persona-2026-acquisition-replay

uv run python -m tools.persona_data build \
  --acquisition data/local/persona-2026-acquisition-replay \
  --output data/local/persona-2026-fresh \
  --report reports/local/persona-2026-fresh \
  --recipe reports/data-expansion/2026-v1/materialization-recipes.json

PERSONA_BUILD_CUTOFF="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
uv run bias --corpus data/local/persona-2026-fresh/corpus snapshot \
  --id elon-musk-original-replay.v1 \
  --as-of "$PERSONA_BUILD_CUTOFF" --ingested-at "$PERSONA_BUILD_CUTOFF" \
  --current-since 2026-01-01 \
  --output data/local/persona-2026-fresh/snapshot.json

BIAS_CORPUS_DIR=data/local/persona-2026-fresh/corpus \
BIAS_PERSONA_SNAPSHOT=data/local/persona-2026-fresh/snapshot.json uv run bias serve
```

A changed publisher transcript fails pinned materialization. Review the changed source and register a new recipe version instead of ignoring a digest mismatch. Page-byte hashes can change independently of the selected canonical source content. The exact hash definitions are recorded in the recipe. Full original source wording is emitted only under the chosen local build directory.

The runtime snapshot binds the exact corpus and a recorded UTC build cutoff. Per-source acquisition timestamps from the original download ledger take precedence over the CLI fallback timestamp. The local source metadata explicitly records original text inclusion and keeps the old summary inventory's curation label as lineage metadata. Dwarkesh's primary page establishes publication on 2026-02-05; it does not establish a recording date or a present-day assertion period. That source can support an attributed discussion of the published 2026 interview, while its recording date remains unknown and cannot establish a current position by itself.

The actual build is `data/local/persona-2026/corpus`; the accompanying `segments.jsonl`, `canonical/`, `projection-decisions.json` and `projections/` preserve the local audit trail. Ten corpus/projection files were rebuilt from the acquired cache and matched byte for byte; [the replay check](../reports/data-expansion/2026-v1/replay-verification.json) records their hashes.

## Text adaptation

The actual export produced **595 original complete training exchanges across nine events**, with a cap of 150 per event and seed 42. Excerpts, held-out sources and generated augmentation are excluded. The dataset includes historical dates, original interviewer context and a persistent simulation instruction. Original answer text remains the supervised target. [The dataset manifest](../reports/data-expansion/2026-v1/text-adaptation-dataset.json) records the exact bytes and event counts.

```bash
uv run python -m tools.adaptation export-text \
  --projection data/local/persona-2026/projections/text_style.jsonl \
  --output data/local/text-adaptation-new --seed 42 --max-per-event 150

# Install torch, transformers, peft and huggingface-hub in the training environment.
# Replace MODEL with a Hugging Face repository ID and the revision with its exact commit.
python -m tools.adaptation train-text \
  --dataset data/local/text-adaptation-new --output data/local/text-adapter-run \
  --model MODEL --revision IMMUTABLE_40_CHARACTER_REVISION \
  --device cuda:0 --epochs 1 --rank 8 --max-length 1024
```

The launcher records actual model/tokenizer revisions, downloaded checkpoint hashes, adapter settings, optimizer settings, seed, precision, hardware, package versions, loss values, elapsed time, failures and output hashes. It masks system/user tokens from training loss, checks that the model's chat template preserves the assistant prefix, and drops an over-budget exchange whole. It does not silently truncate the answer. A `--max-steps` bound supports a declared execution experiment.

No text adaptation was executed in this data-delivery step: the application environment lacked Torch/Transformers/PEFT, and a model-specific quality comparison has not been run. A runnable trainer and a real dataset are not a trained or validated Musk persona. Historical facts can still be memorized despite the intended separation between style adaptation and updateable factual retrieval. Compare base + fixed factual retrieval, the same base + style examples, and the same base + adapter under identical evidence and fresh evaluation policy before making a quality claim.

## Qwen3-TTS adaptation

The exporter follows the upstream [single-speaker Base fine-tuning format](https://github.com/QwenLM/Qwen3-TTS/blob/main/finetuning/README.md): each JSONL row contains `audio`, `text` and `ref_audio`, and all rows share one explicitly selected accepted reference. It crops actual hash-checked WAV intervals and records the resulting clip hashes. The [actual voice export](../reports/data-expansion/2026-v1/voice-adaptation-dataset.json) is unready with zero samples; training rejects an empty or changed dataset.

```bash
uv run python -m tools.adaptation export-voice \
  --projection data/local/persona-2026/projections/aligned_speech.jsonl \
  --output data/local/voice-adaptation-new --asset-root .

# After accepted speech exists, select --reference-id during export, then:
python -m tools.adaptation train-voice \
  --dataset data/local/voice-adaptation-new --output data/local/voice-adapter-run \
  --upstream /path/to/Qwen3-TTS --upstream-revision IMMUTABLE_40_CHARACTER_REVISION \
  --base-model /path/to/Qwen3-TTS-Base --tokenizer-model /path/to/Qwen3-TTS-Tokenizer-12Hz \
  --device cuda:0 --batch-size 2 --epochs 3 --execute
```

Without `--execute`, a valid nonempty dataset produces a recorded command plan only. With it, the launcher runs upstream `prepare_data.py` and `sft_12hz.py`, preserves logs and failures, and hashes checkpoints. It verifies the specified clean upstream revision and local base/tokenizer files. The `--device` option controls audio-token preparation; the pinned upstream Accelerate configuration selects SFT hardware. The inspected upstream CLI does not expose a seed option, so the report explicitly leaves seed control unknown. These commands do not claim that Qwen3-TTS Base supports arbitrary prosody instructions or that a successful training loss establishes English/Korean intelligibility, timbre or natural prosody.

## Verification

```bash
uv run pytest tests/persona_data -q
```

The 19 focused regressions cover original speaker anchors, typography boundaries, chapter attribution reset, PDF headers, unresolved clocks, whole-turn preservation, exposed/fresh split isolation, independent modality eligibility and runtime role flags, real-media bounds/hash checks, event caps, dataset tampering, constant voice references and assistant-only loss masking. Actual ten-source acquisition, nine-source extraction, `Corpus.load`, byte-identical pinned replay and both dataset exports were exercised. Human audiovisual evaluation and new efficacy-test acquisition remain unperformed.
