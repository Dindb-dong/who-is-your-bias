# Observed voice and motion linked to actual Q&A

The R8 pipeline links **original Q&A cue IDs** to measurements from the same
recorded interview. It computes observable source features and prepares review
records. Its outputs do not train a gesture policy or establish that a generated
voice sounds like the respondent.

**Status:** the measurement pilot and review tooling have run; human observation
review, perceptual voice judgments and a validated personal movement policy have
not been completed. The [artifact and version audit](research/observation-pilot-status.md)
records the verified counts, immutable hashes and compatibility with the expanded
92-exchange text corpus. More text records do not imply more measured media.

The implementation is in [`tools/observations`](../tools/observations). Its
[frozen pilot plan](../data/observations/ted-pilot-plan.v1.json) uses the TED 2013
and TED 2017 recordings already acquired and measured by the project. It retains
TED 2017 as a whole-event **observation validation** set. Both events may belong
to text training; the report records the two split assignments separately.
Calling a motion-validation event an unseen text-test event would be incorrect.

## Executed pilot: 2026-11-11

The [preserved report](../data/observations/ted-pilot.v1.json) was actually built
against the hydrated original-Q&A corpus. It joins four exchanges from two
independently recorded events to **19 contiguous matched speech-rate windows**
and **10 distinct visual windows**. Those windows contain 512 sampled frames;
eight frames are excluded by known camera-cut tails. Two bilateral-smile
candidate intervals survive the minimum-duration check. No acoustic pause,
audible laughter, semantic gesture or human voice rating is marked reviewed.

| Exchange | Situation | Observation split | Matched speech-rate windows | Visual windows |
| --- | --- | --- | ---: | ---: |
| `qa-ted-2013-energy` | Explanation | Train | 3 | 1 |
| `qa-ted-2013-efficiency` | Challenge | Train | 7 | 4 |
| `qa-ted-2017-tunnels` | Humor | Validation | 4 | 2 |
| `qa-ted-2017-capacity` | Explanation | Validation | 5 | 3 |

The linked Q&A locators refer to complete selected source cue spans, even when
the checked-in distribution contains short excerpts. The report records that
scope explicitly. All four whole-span speech-rate values remain unknown because
the ASR-to-publisher matching has gaps; the 19 measured subwindows preserve those
gaps instead of silently filling them. The energy answer's first sound is
outside the cached ASR window. Candidate turn gaps for the efficiency and tunnel
answers are 0.86 s and 1.10 s respectively, with validated pauses left **null**
pending actual acoustic-boundary and continuity review.

The extraction took 0.813 s wall time and 0.805 s process CPU time on the shared
Linux x86-64 host, with 160.6 MiB process high-water RSS. This is lightweight
feature recomputation; the already-completed ASR and MediaPipe runs are separate
costs. Exact hashes and original model settings are retained in the report.
The four exchange-to-ASR links reuse three distinct ASR runs. The report verifies
13 cached visual windows in its provenance; only the ten windows overlapping the
four selected exchanges enter the stated frame and feature counts. All measured
source speech in this pilot is English. Korean unit handling is implemented and
tested, but no Korean source-speaking-rate measurement is claimed.

The [review page](../data/observations/ted-pilot-review/index.html) and
[empty review template](../data/observations/ted-pilot-review/review-template.json)
are ready to use after checkout. Importing that untouched template produced the
[ungraded summary](../data/observations/ted-pilot-review/ungraded-summary.json):
four unreviewed exchanges and an empty runtime index. A separate
[agent visual-screening record](../data/observations/visual-screening.v1.json)
documents actual contact-sheet inspection without claiming full tracking,
speaker association, listening, or human review.

TED 2022 and both Lex test events have no observations in this pilot and are
listed as unmeasured, rather than disappearing from the coverage report. The
small observed-event inventory does not establish gesture generalization or
condition-dependent voice quality.

## What the measurements mean

| Requirement | Implemented measurement | What remains unknown without review |
| --- | --- | --- |
| Link a reaction to the actual exchange | Q&A ID, content digest, source event, original question/answer cue IDs, situations and topics | Whether a visible image track belongs to the audio respondent |
| Speech rate | Exact publisher/ASR matched words over a contiguous source-clock interval; includes internal pauses | Full-answer rate when the clip or token alignment is incomplete; articulation rate |
| Pause before answering | Candidate question-final and answer-initial token times in the same acquired recording clock | Actual acoustic boundaries and whether the original recording contains an edit |
| Visible smile | Bilateral `mouthSmileLeft` / `mouthSmileRight` score crossing over a declared minimum interval | Human visual confirmation, identity and any emotional interpretation |
| Audible laughter | A distinct annotation type, requiring actual listening and respondent attribution | All laughter remains ungraded until that review occurs; subtitles and smiles do not supply it |
| Head movement | Relative quaternion rotation path over adjacent frames | Nod/tilt meaning and the contribution of camera motion |
| Arm/hand movement | Wrist movement relative to the shoulder when shoulder, elbow and wrist are visible; explicit off-screen, low-confidence and missing states | A semantic wave/emphasis label, finger articulation, and camera-motion correction |
| Listening or speaking | Source conversation-phase candidate based on matched question/answer windows | Whether the detected image track is the speaker; ambiguous or crossing intervals remain `unknown` |

English rates use **orthographic words per second**. Korean rates use Hangul
syllable blocks per second, with the unit explicitly recorded. A Korean string
containing Latin letters or digits is rejected until a reviewed spoken
transcription is provided. A number written as `200` does not establish how
many syllables were spoken. English words and Korean blocks are not directly
comparable, and neither is presented as a phonetic syllable count.

If an answer contains unmatched ASR words, the whole-span rate stays unknown.
The report retains rates for each contiguous matched interval of at least three
ASR words instead. Intervening interviewer words or an unobserved gap are never
silently included in a respondent's rate. These are descriptive small windows,
not independent samples of the respondent's general speaking style.

## Source-clock and attribution safeguards

The publisher transcript uses `ted_transcript_cue`; the acquired media uses
`ted_h264_zero_start_capture`. They are distinct clocks. The builder recomputes
the original token alignment from the exact publisher and ASR artifacts,
checks their hashes, and joins by cue IDs. It never fits a global caption offset
and never substitutes a caption cue end for the last audible word.

The first or last answer timestamp is only emitted as a **candidate** when the
corresponding original boundary token was matched. A clip starting in the middle
of an answer cannot establish its onset. Even a complete token match leaves
the validated pause null until acoustic-boundary and recording-continuity review.
Overlapping speech is not converted into a positive pause.

Visual input is verified against the preparation manifest, input-video hash,
frame observations and converted geometry. The existing cut-tail exclusions
are applied before descriptor computation. Features do not bridge a missing
frame run or excluded camera interval. A frame-local detection index does not
become a person identity. The report keeps image-to-speaker association and
track continuity unreviewed even where the publisher's text identifies the
respondent.

The fixed smile threshold is `0.5` on **both** mouth-corner scores for at least
`0.25 s`; pose visibility and presence thresholds are `0.7`. These are declared
engineering screens, not learned thresholds or calibrated behavioral accuracy.
They were not optimized on the held-out event. Passing a screen produces a
candidate for review, not a completed observation.

## Reproduce the extraction

The builder needs the Q&A corpus created by the proposal experiment and the
original local acquisition cache. Source audio/video and full transcript text
are not copied into committed observation reports.

The expected cache layout is:

```text
motion-acquisition/ted-2013-transcript.json
motion-acquisition/ted-2017-transcript.json
motion-acquisition/offline-v021/<window-id>/manifest.json
motion-acquisition/offline-v021/<window-id>/observations-manifest.json
motion-acquisition/offline-v021/<window-id>/observations.jsonl
motion-acquisition/offline-v021/<window-id>/video.mp4
motion-context-asr/ted-2013-30-120/{manifest,asr,alignment}.json
motion-context-asr/ted-2017-20-120/{manifest,asr,alignment}.json
motion-context-asr/ted-2017-120-179/{manifest,asr,alignment}.json
motion-context-asr/<run-id>/audio.wav
```

See [motion measurement](motion-library.md),
[publisher transcripts](research/transcript-acquisition.md), and
[reference alignment](research/voice-reference.md) for the existing acquisition
and model-execution steps. The observation report records the exact input hashes,
MediaPipe model hashes and version, ASR revision/settings, Q&A corpus digest,
split manifest digest, thresholds, and implementation hashes.

```bash
PYTHONPATH=src:. python -m tools.observations build \
  --plan data/observations/ted-pilot-plan.v1.json \
  --corpus data/local/qna/elon-musk-v1 \
  --cache-root /absolute/path/to/acquisition-cache \
  --repository . \
  --output reports/local/exchange-observations.json
```

Existing output files are preserved unless `--overwrite` is supplied. Runtime
timing in this report measures **feature recomputation from existing model
outputs**. It excludes original acquisition and ASR/MediaPipe inference and is
not a live application latency measurement. The separate live-turn/presentation
benchmarks own those measurements.

## Review and accepted observations

Generate a review page and an empty annotation form:

```bash
PYTHONPATH=src:. python -m tools.observations review-bundle \
  --report reports/local/exchange-observations.json \
  --output-directory reports/local/exchange-observation-review
```

The form is bound to the immutable report's SHA-256 and each Q&A's content
digest. It starts with no reviewer, false checkboxes, `not_reviewed` decisions,
null acoustic boundaries and empty annotations. The generated page does not
claim that someone watched or listened simply because it was opened.

A reviewer explicitly supplies:

- The reviewer name, **human or agent** attribution, review date and evidence
  note.
- Which source-clock, listening, track, speaker-association, camera and
  recording-continuity checks were actually completed.
- Source-clock intervals for visible smile, audible laughter, head nod/tilt,
  hand wave/emphasis or unobservable hands, plus listening/speaking/unknown
  phase and hand side where relevant.
- Actual acoustic question-end, answer-start and answer-end boundaries only
  after listening and continuity verification.

Import the completed form:

```bash
PYTHONPATH=src:. python -m tools.observations review \
  --report reports/local/exchange-observations.json \
  --annotations reports/local/exchange-observation-review/completed-review.json \
  --output reports/local/reviewed-exchange-observations.json
```

The importer rejects wrong report hashes, duplicate exchanges, changed event
identity, annotations beyond the observed modality's intervals, gestures across
camera exclusions, laughter without listening, and head/hand gesture labels
without camera review. All three reviewed acoustic boundaries must lie inside
one audio run that covers the complete question-end, answer-onset and answer-end
boundaries. Video coverage, a second audio run or an unrecorded gap cannot justify
a pause measurement. Review metadata records the supplied judgments; a checkbox
itself is not independent proof that review occurred.

Accepted agent and human judgments are reported separately. Only accepted
**human-reviewed training-event** observations can enter the exported
`runtime_index`. Validation and test events remain excluded even after favorable
review. Candidate observations never change the studio's voice or motion policy.
This preserves the distinction between source-observation retrieval and a
trained model of how someone would react to a new question.

## Voice listening remains a separate outcome

Use the existing [generated-speech listening bundle](voice-evaluation.md), which
contains actual Qwen-generated WAV files and supports naturalness and
intelligibility review. The local version with the source reference additionally
supports reference-similarity review. The public repository copy deliberately
does not offer a similarity score when no reference is present.

No source geometry or text-retrieval test can fill those scores. A four-sample
listening form with null scores remains an uncompleted listening study. Keep
human ratings null until a person supplies them and report new demonstration
hardware timing separately from the preserved CPU baseline.
