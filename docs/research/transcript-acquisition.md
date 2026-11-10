# Publisher transcript acquisition

The TED extractor converts the public publisher page's caption data into complete, local review drafts. It preserves interviewer questions, Musk's answers, unlabeled captions, stage directions, paragraph order, and source timestamps. It does not add these drafts to the reviewed persona corpus.

The measured run on November 10, 2026 extracted **3,451 caption cues across four original TED interviews**, with **27,411 whitespace-delimited words** and a combined publisher-declared duration of **3 hours, 45 seconds**. These are substantially more source material than the short, reviewed paraphrases in the retrieval corpus. They are still draft material: speaker identity, caption accuracy, and alignment with a particular audio/video asset require review.

Implementation: `tools/transcripts/ted.py` and `tools/transcripts/__main__.py`. Machine-readable measured counts and hashes: `tools/transcripts/measured-coverage.json`. The extractor uses the Python standard library; tests and optional evidence import use the repository's existing development environment.

## Reproduce acquisition

Run commands from the checkout. Each output directory must be new: the tool refuses to overwrite existing files or follow an existing output-directory symlink.

Fetch one ordinary public publisher page and extract its embedded `__NEXT_DATA__` JSON:

```bash
.venv/bin/python -m tools.transcripts ted \
  --url 'https://www.ted.com/talks/elon_musk_the_mind_behind_tesla_spacex_solarcity?view=transcript' \
  --source-id ted-2013 \
  --output data/local/transcripts/reproduce/ted-2013
```

Or read a cached JSON document or saved public HTML page:

```bash
.venv/bin/python -m tools.transcripts ted \
  --input data/raw/expansion-a/elon_musk_the_mind_behind_tesla_spacex_solarcity.json \
  --source-id ted-2013 \
  --output data/local/transcripts/reproduce-cached/ted-2013
```

To reproduce all four sources from their public pages, choose a fresh output parent and run:

```bash
while read -r transcript_source transcript_slug
do
  .venv/bin/python -m tools.transcripts ted \
    --url "https://www.ted.com/talks/${transcript_slug}?view=transcript" \
    --source-id "$transcript_source" \
    --output "data/local/transcripts/reproduce-all/${transcript_source}"
done <<'TED_SOURCES'
ted-2013 elon_musk_the_mind_behind_tesla_spacex_solarcity
ted-2017 elon_musk_the_future_we_re_building_and_boring
ted-2022-recorded elon_musk_a_future_worth_getting_excited_about
ted-2022-live elon_musk_elon_musk_talks_twitter_tesla_and_how_his_brain_works_live_at_ted2022
TED_SOURCES
```

The command prints only source metadata and counts. It accepts HTTPS talk-page URLs on `ted.com` or `www.ted.com`, checks redirects, limits input to 16 MiB, and rejects malformed or nonfinite timestamps. It uses ordinary public access without login, cookies, or media downloading. A publisher format change produces an error instead of substituting a third-party transcript. Public pages can change, so compare the caption-content hash before treating a new run as the same material.

## Output contract

| File | Purpose |
|---|---|
| `transcript.json` | Complete canonical draft: original cue text, source order, paragraph links, textual speaker spans, conservative caption groups, timestamps, and provenance. |
| `whisperx.json` | Compatible `segments` with `start`, `end`, `text`, `speaker`, and extra review/context fields. It can be read by the existing importer. This format name does not mean WhisperX was run. |
| `manifest.json` | `publisher_media.v1` inventory with the source clock, observed TED HLS/MP4 and YouTube links, and output artifact hashes. It explicitly says no media was downloaded or aligned. It is not a prepared `media.v1` clip. |
| `summary.json` | Counts, source identifiers, input/content hashes, declared duration, and review status without transcript text. |

The measured outputs are in `data/local/transcripts/publisher-ted-v1/<source-id>/`. Complete copyrighted source material remains in ignored `data/raw` and `data/local` directories. Only code, synthetic fixtures, source metadata, counts, and hashes are committed.

`input_sha256` identifies the exact input bytes. `publisher_caption_sha256` hashes the ordered language/paragraph/cue text and millisecond onsets, excluding mutable page metadata such as view counts. `manifest.json` also identifies each emitted file by SHA-256 and byte size. The fresh public HTTPS smoke run for TED2013 had different page bytes but the same caption-content hash as the cached input.

## Speaker attribution and conversation context

The extractor recognizes a speaker identity only after that full name occurs as an explicit text label in the particular transcript. For this TED interview adapter, the recognized full names are Chris Anderson and Elon Musk. Abbreviations `CA` and `EM` resolve only when the corresponding full-name label is present. The page title, featured speaker, voice characteristics, turn alternation, and subject matter do not establish identity.

An explicit label can carry through later cues **within the same publisher paragraph**. Attribution resets at every paragraph. An unknown label clears the previous identity. This deliberately leaves some likely answer continuations unknown; review can resolve them from the source instead of inheriting a guess.

When one caption contains text from multiple speakers, the caption's `speaker` remains null. `speaker_spans` retains the label and character range of each textual part; the extractor does not invent an audio timestamp between the parts. Standalone parenthetical stage directions also have no speaker identity. Their presence does not identify who laughed, nodded, or moved.

Every cue retains its original text and previous/next cue identifiers. Paragraphs retain their complete text and neighboring paragraph identifiers. Consecutive captions sharing a paragraph and label context are grouped as `utterances`, but these groups are not acoustically verified utterances. No question/answer roles or intention labels are inferred. Consumers can recover the actual preceding interviewer context through these links; selecting Musk-only records must not replace the canonical conversation record.

The schema field `known_speakers` therefore means **attributed by publisher text labels**, not independently verified audio identity. Every extracted segment remains `review_status: draft`; the canonical document has `confidence: unverified_transcript`. `behavioral_observations` is empty. Text acquisition is separate from any later gesture, facial-landmark, voice, or personality analysis.

## Time semantics

The named clock is `ted_transcript_cue`, with `source_start_seconds: 0`. A cue's observed `time` is in milliseconds and becomes its onset in seconds. TED's caption JSON does not supply a precise end for each cue.

For compatibility with the existing timestamped importer, the next distinct cue onset supplies an explicitly labeled **end proxy**. The last cue can use the declared publisher duration when that duration is later than its onset. If neither is available, the canonical cue and text remain present with a missing end; the WhisperX-compatible file omits that cue and lists its identifier in `omitted_cue_ids`. All four measured sources had exportable end proxies for every cue. No word-level alignment is fabricated (`words` is empty).

Per-speaker interval totals sum these onset-to-next-onset intervals. They include pauses and may include stage-direction text; they are **not measured speaking time**. Shared onsets can make totals overlap and are explicitly flagged. None of the four measured sources had shared onsets. A fresh media review must establish alignment before cue times drive an avatar, an audio clip, or a training segment.

The publisher's TED and YouTube versions have different durations, as documented in [the source expansion notes](expansion-a.md). The extractor inventories both media links but does not treat their clocks as interchangeable. Raw `publisher_recorded_on` metadata is preserved without resolving inconsistencies with the written event descriptions; the reviewed source metadata explains those discrepancies.

## Measured coverage

These counts come from successful extraction of the four cached publisher documents. A separate ordinary-HTTPS extraction of the 2013 page also succeeded and matched its caption-content hash; it is the same source and is not counted twice.

| Source | Publisher duration | Paragraphs | Cues | Caption groups | Whitespace words | Musk-labeled cues | Anderson-labeled cues | Unassigned cues |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [TED2013](https://www.ted.com/talks/elon_musk_the_mind_behind_tesla_spacex_solarcity?view=transcript) | 20:44 | 75 | 430 | 88 | 3,575 | 293 | 124 | 13 |
| [TED2017](https://www.ted.com/talks/elon_musk_the_future_we_re_building_and_boring?view=transcript) | 40:37 | 193 | 789 | 195 | 6,073 | 436 | 245 | 108 |
| [TED2022 recorded](https://www.ted.com/talks/elon_musk_a_future_worth_getting_excited_about?view=transcript) | 1:06:23 | 216 | 1,242 | 224 | 10,197 | 658 | 418 | 166 |
| [TED2022 live](https://www.ted.com/talks/elon_musk_elon_musk_talks_twitter_tesla_and_how_his_brain_works_live_at_ted2022?view=transcript) | 53:01 | 223 | 990 | 230 | 7,566 | 435 | 338 | 217 |
| **Total** | **3:00:45** | **707** | **3,451** | **737** | **27,411** | **1,822** | **1,125** | **504** |

Word counts include text labels and stage directions. The 504 unassigned cues consist of 11 mixed-speaker cues, 95 standalone stage-direction cues, and 398 other unlabeled/unknown cues. Among the 2,947 labeled cues, 550 contain explicit publisher labels and 2,397 inherit a label within the same paragraph.

| Source | Musk-labeled interval proxies (seconds) | Anderson-labeled interval proxies (seconds) | Unassigned interval proxies (seconds) |
|---|---:|---:|---:|
| TED2013 | 821.694 | 373.519 | 48.433 |
| TED2017 | 1,414.700 | 701.400 | 320.380 |
| TED2022 recorded | 2,178.454 | 1,263.132 | 541.126 |
| TED2022 live | 1,470.382 | 1,009.219 | 700.152 |
| **Total** | **5,885.230** | **3,347.270** | **1,610.091** |

These are caption-interval proxies, not usable clean-voice duration. The totals exclude the small gaps before each source's first cue.

## Import as draft evidence

The existing importer was run against all four emitted `whisperx.json` files with their manifests. It produced **3,451 draft evidence records**, all `kind: transcript`, `confidence: unverified_transcript`, `review_status: draft`, and no behavioral observation. The local `evidence-drafts.jsonl` files were not added to reviewed retrieval data.

Reproduce one import using a new output filename:

```bash
.venv/bin/python -m bias.cli ingest \
  data/local/transcripts/publisher-ted-v1/ted-2013/whisperx.json \
  --source-id ted-2013 \
  --manifest data/local/transcripts/publisher-ted-v1/ted-2013/manifest.json \
  --output data/local/transcripts/publisher-ted-v1/ted-2013/reimport-drafts.jsonl
```

The generic evidence importer preserves the speaker label and source timeline but has a simpler segment schema. Keep `transcript.json` and `whisperx.json` alongside imported evidence: they retain the original paragraph/question context, label basis, and end-proxy distinctions needed for review. Do not interpret importing a segment as checking its contents or timing.

## Lex18 and Lex252 status

The [episode18 page](https://lexfridman.com/elon-musk/), [episode252 page](https://lexfridman.com/elon-musk-3/), [podcast index](https://lexfridman.com/podcast/), and [older AI index](https://lexfridman.com/ai/) were inspected for actual publisher links. The pages expose original audio and the indexes identify the videos, but no publisher transcript HTML or PDF link for either episode was found in those pages. This is a scoped acquisition result, not a claim that no transcript exists anywhere.

| Episode | Observed original audio | Observed video | Transcript status |
|---|---|---|---|
| 18 | [Host-linked MP3](https://media.blubrry.com/takeituneasy/content.blubrry.com/takeituneasy/mit_ai_elon_musk.mp3) | [dEv99vxKjVI](https://www.youtube.com/watch?v=dEv99vxKjVI) | Media links observed; transcript unreviewed. |
| 252 | [Host-linked MP3](https://media.blubrry.com/takeituneasy/content.blubrry.com/takeituneasy/lex_ai_elon_musk_3.mp3) | [DxREm3s1scA](https://www.youtube.com/watch?v=DxREm3s1scA) | Media links observed; transcript unreviewed. |

The host's discoverable Part II PDF belongs to episode49; it was not substituted for episode18. Third-party transcripts and chapter headings were not promoted into reviewed answers. These two candidates remain outside the additional reviewed source/evidence totals. Their observed media can feed the separate local ASR workflow, after which speaker and text review would still be necessary.

## Verification and limits

```bash
.venv/bin/python -m pytest tests/transcripts -q
```

The 33 synthetic-fixture tests cover context retention, ambiguous and unknown speakers, abbreviation resolution, stage directions, timestamps and proxy ends, malicious/unsupported URLs, redirect restrictions, malformed JSON, deterministic caption hashes, output preservation, and draft compatibility with the real importer. Actual extraction and import of all four cached TED documents succeeded. A fresh public-page acquisition also succeeded as described above.

No full interview video was downloaded in this step. No ASR, diarization, forced alignment, training, voice generation, gesture inference, or persona-fidelity evaluation was run. The deliverable is reproducible access to substantially larger, contextual source material with explicit provenance and review boundaries.
