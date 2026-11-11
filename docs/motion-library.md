# Observed motion library experiment

The project now has a reproducible library of measured geometry windows and an optional source-context selector. It does **not** have a validated personal gesture policy. Unsupervised geometry prototypes, manually written source paraphrases, ASR timing estimates, and runtime retrieval are recorded as separate operations.

## What was actually measured

Two distinct publisher events were acquired from the H.264 URLs exposed by the official TED pages: [TED2013](https://www.ted.com/talks/elon_musk_the_mind_behind_tesla_spacex_solarcity) and [TED2017](https://www.ted.com/talks/elon_musk_the_future_we_re_building_and_boring). Each acquisition started at the beginning of the publisher file and was bounded to 180 seconds. There was no remote input seek. Raw video and full captions remain local; the repository contains derived geometry, original short paraphrases, hashes, review metadata, and numeric timing evidence.

| Measurement | TED2013 | TED2017 | Total |
| --- | ---: | ---: | ---: |
| Prepared windows | 7 | 6 | 13 |
| Sampled duration | 31.5 s | 52.5 s | 84 s |
| Sampled frames at requested 8 fps | 252 | 420 | 672 |
| Accepted geometry after camera-cut exclusions | 29.25 s | 51.75 s | 81 s |
| Accepted frames | 234 | 414 | 648 |
| Mined face/body primitives | 12 | 28 | 40 |
| Motion split | train | validation | 2 events |

The 40 primitives cover **29 distinct temporal windows**. Face and body primitives may describe the same time interval; they are not 40 independent examples. Every primitive is 2.5 seconds long. Incomplete tails are excluded from primitive mining. All 13 clips remain `geometry_preview`: image-to-audio-speaker association and full track-continuity review have not been completed. Naming the publisher event is not face identification.

The acquisition clock is explicitly `ted_h264_zero_start_capture`. Each bounded remux has audio beginning at 0 and its first video PTS at 0.023 seconds. Acquisition hashes, local trim positions, spatial crop coordinates, observation hashes, review hashes, model hashes, and reference-pose timestamps are preserved. No direct equivalence with the publisher caption clock is assumed.

## Camera, visibility, and calibration QA

Initial contact sheets at one-second intervals missed four camera transitions near the ends of windows. Sequential decoded samples and large adjacent image-difference peaks exposed them. Conservative tail exclusions were applied **before** choosing calibration frames or fitting descriptors:

| Clip | Original local end | Accepted local end |
| --- | ---: | ---: |
| `ted-2013-close-083` | 5,000 ms | 4,375 ms |
| `ted-2013-close-104` | 7,500 ms | 5,875 ms |
| `ted-2017-body-065` | 7,500 ms | 7,125 ms |
| `ted-2017-close-025` | 17,500 ms | 17,125 ms |

The review files retain these exclusions and the exact observation SHA-256. This screening does not promote track continuity or speaker association to reviewed status.

Only a single detection in the selected image crop is retained. Multiple detections are omitted rather than assigned a person identity. Out-of-frame landmarks and pose visibility/presence below 0.65 remain unobserved. Per-clip and per-channel coverage is included in the catalog. For example, the first cropped 2013 body window has 20 pose frames and zero detected faces; face data is not invented for it. Several closeups have usable faces but insufficient hips/arms for comparable body descriptors.

Rotations are right-handed, Y-up, local XYZW quaternion deltas. A reference frame with observed joint support provides the source pose; it is not an automatically detected anatomical neutral pose. Body rotations are direction-only swing estimates, bounded to 70 degrees from that reference. Axial twist and finger articulation are not measured. The [MediaPipe Pose Landmarker documentation](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python) describes the underlying landmark outputs and VIDEO timestamp API; those outputs are estimates, not motion-capture ground truth.

A primitive slice retains the parent calibration. Its `neutral_frame_ms` may be negative or greater than the slice duration. `calibration.parent_reference` records the parent asset hash, original reference timestamp, window start, and `retained_parent_local_delta` operation. This avoids silently inventing a new neutral pose when a window is served. `source_start_seconds` and `observation_offset_ms` are rebased together.

## What is fitted, and what is authored

The mining implementation is `tools/motion/mining.py`:

1. It validates clip bounds, source provenance, coordinate conventions, frame ordering, finite coefficients, and normalized quaternions.
2. It mines contiguous 2.5-second windows without bridging missing-frame gaps. A channel needs at least 80% coverage and enough valid adjacent pairs.
3. Face descriptors contain coefficient range and mean absolute speed. Body descriptors contain reference-relative quaternion angular range and mean angular speed. Quaternion sign changes do not count as movement. Angular distance divides out the small norm error allowed by clip validation, so a stationary serialized rotation cannot acquire artificial velocity.
4. Features must be present in every training window. Missing validation features remain missing. Fixed variation floors prevent nearly constant detector noise from receiving enormous standardized weight; these are engineering settings, not measured detector accuracy.
5. Feature means, scales, and deterministic k-medoids are fitted using **TED2013 only**. The medoids are actual source windows. Face and body families are fitted separately.
6. TED2017 is evaluated after fitting. It cannot alter feature selection, normalization, prototypes, distance envelopes, or runtime retrieval.

Source-event split consistency is checked even when several source IDs or clips refer to one event. A corpus validation/test source cannot be promoted into motion training. Both TED events belong to the broader persona corpus's training partition; the separate **motion** experiment reserves TED2017 for validation. This does not change the persona corpus's existing held-out event.

Cluster IDs such as `face-01` and `upper-body-02` have no emotion or communicative-intent meaning. No authored “confident,” “thoughtful,” or “emphasis” label is presented as learned. The source paraphrases are authored bilingual summaries of publisher context. Runtime selection uses BM25 over those summaries, with English stopwords removed and literal Hangul bigrams for Korean particles. A fixed 20% query-token overlap gate helps the selector abstain on unsupported queries. Geometry distance only breaks ties between relevant source contexts.

## Actual held-out results

| Family | Training windows | Fitted features | Medoids | Validation windows | Comparable | Missing required features | Mean normalized prototype distance | Inside training envelope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Face | 8 | 84 | 3 | 15 | 15 | 0 | 2.740852 | 0 / 15 |
| Upper body | 4 | 4 | 3 | 13 | 5 | 8 | 0.591145 | 0 / 5 |

The four common variable body features describe chest and right-upper-arm range/speed. Other observed joints remain in the source clips but cannot be treated as shared fitted features across every training window.

These results **do not establish cross-event generalization**. There are only eight face training windows for 84 features and four body windows. Some medoid clusters contain one training example and therefore have a zero training radius. A training envelope is an observed distance range, not a calibrated confidence interval. Camera framing, crop, detector error, and pose-reference differences can affect the descriptors. The catalog reports semantic-gesture accuracy, person-identification accuracy, and conditional-personal-policy accuracy as `null`; there is no labeled ground truth for those claims.

## Source text and media clocks

Three bounded ASR contexts cover the geometry windows: TED2013 capture seconds 30–120, and TED2017 seconds 20–120 and 120–179. The voice workflow used local `Systran/faster-whisper-small`, pinned revision `536b0662742c02347bc0e980a01041f333bce120`, with word timestamps, no publisher prompt, and two CPU threads. Full-context publisher token agreement was 95.60%, 95.22%, and 93.71%, respectively.

`tools/motion/contexts.py` verifies the original audio/ASR/publisher hashes, recomputes the token-sequence alignment, and verifies that geometry and audio came from the same acquired video hash and clock. All 13 geometry contexts pass the declared within-window gates: at least four completely matched ASR words, at least 80% word coverage, and at least 80% agreement with the declared context's cue IDs. Their context alignment is labeled **`asr_word_matches`**, not human-reviewed forced alignment. The evidence report contains cue IDs, numeric word anchors, hashes, and coverage; it does not include full transcript text.

Binding also checks that the ASR and prepared-audio source offsets agree within
one millisecond, and that the declared publisher clock, cue order, and interval
match the exact publisher artifact. Each evidence entry hashes the authored
context declaration, including its paraphrase, cue IDs, and interval. Editing
that declaration requires a new binding run; a catalog cannot keep an old
`asr_word_matches` label by retaining only the clip ID and evidence-file hash.
This hash preserves the declaration used for alignment and does not establish
that its paraphrase is semantically accurate. The recovered artifacts were
rebound with these checks; all 13 passed, and the raw geometry, primitive
descriptors, fitted values, and original evaluation results were unchanged.

Cue onset and ASR word onset have different uncertainty. No global offset is applied. Human listening is incomplete, and the caption's speaker label is never assigned to an image track. Context timing support therefore changes the selector's context provenance while all geometry stays preview-only.

## Runtime contract

`src/bias/motion.py` provides:

```python
from bias.motion import MotionLibrary, MotionSelection, MotionError, clip_digest

library = MotionLibrary.load(
    "data/motion/elon-musk-ted-v1.json",
    "apps/web/public",
)
selection = library.select("Why are electric cars efficient?", allow_geometry_preview=True)
clip = library.clip(selection.primitive_id, allow_geometry_preview=True) if selection else None
```

Both calls default to reviewed-only. The current catalog returns no selection under that default. The application keeps motion disabled until an explicit mode is selected. Its geometry-preview option permits the 12 TED2013 training primitives; the 28 held-out TED2017 primitives are unavailable to automatic selection and the clip-serving helper. The existing authored conversation cues remain a separate baseline.

Selection records include library version, source/event, source clock and interval, feature family, review status, context alignment, cluster ID, and `/api/motion/clip/{primitive_id}`. The response contains only the selected family: face/head or upper-body channels. `parent_clip_sha256` identifies the checked-in source asset bytes. `clip_sha256` identifies canonical response JSON: UTF-8, sorted keys, compact separators, and unescaped Unicode. The API can serve those exact bytes and verify their SHA-256.

Catalog loading confines paths to the asset root, verifies asset/evidence hashes and source identities, rejects conflicting context identities and split leakage, and checks clip bounds. An unknown ID, held-out ID, or insufficient review gate fails closed. Source-context retrieval is not a probability that the person would make that gesture in a new conversation.

Malformed review/calibration objects and evidence-entry containers raise
`MotionError`. The optional-library startup handler can therefore report an
invalid catalog while keeping ordinary chat available.

## Reproduction

From the repository root, rebuilding the checked-in derived library requires only the normal project environment:

```bash
PYTHONPATH=src:. python -m tools.motion build \
  data/motion/elon-musk-ted-input.v1.json apps/web/public \
  data/motion/elon-musk-ted-v1.json

PYTHONPATH=src:. python -m tools.motion check \
  data/motion/elon-musk-ted-v1.json apps/web/public \
  --query 'electric cars efficient' --allow-geometry-preview

python -m pytest tests/motion -q
```

To repeat measurement, create a **separate** environment with the project base dependencies and `tools/motion/requirements-vision-offline.txt`; do not install the project's NumPy-2 development extra into the NumPy-1 legacy vision environment. Supply the two acquired videos and official model files with the hashes in `data/motion/ted-window-plan.v1.json`:

```bash
PYTHONPATH=src:. python -m tools.motion measure \
  data/motion/ted-window-plan.v1.json data/local/motion-media \
  data/local/motion-measurements-new \
  --face-model models/face_landmarker.task \
  --pose-model models/pose_landmarker_lite.task
```

This command downloads nothing, requires the audited runtime version, verifies input/model hashes, and refuses to overwrite bundles. Export geometry with `python -m tools.animation`, using the corresponding review file, accepted end and reference timestamp from the plan. New observation bytes require matching review hashes; do not copy a review assertion to different observations. `bind-contexts` similarly accepts explicit local publisher files and alignment directories, rechecks their evidence, and updates only source-context metadata. Rebuild the catalog after changing the input manifest.

### Runtime change after an actual approval rejection

The first MediaPipe 0.10.35 attempt completed one 40-frame window, then automatic approval review rejected an unexpected request to `play.googleapis.com` because its exact metadata/payload had not been established. That runtime was not retried, patched, redirected, or used for the final library. The [official privacy notice](https://developers.google.com/edge/mediapipe/solutions/privacy) describes SDK metrics collection, and [upstream discussion #6306](https://github.com/google-ai-edge/mediapipe/issues/6306) did not provide a supported opt-out.

A read-only comparison established a materially different implementation: the official [0.10.21 Tasks runner](https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/mediapipe/tasks/cc/core/task_runner.cc) has no TasksLogger path, whereas [0.10.35](https://github.com/google-ai-edge/mediapipe/blob/v0.10.35/mediapipe/tasks/cc/core/task_runner.cc) adds a logger factory and session/invocation calls. The official CPython-3.12 manylinux-2.28 wheel was hash-checked against PyPI metadata (`956eb1ebc275c629e61b085b2cab89c3a5b9e93bad1bb107348d98dafb5a4bb5`), installed separately, and API compatibility checked before a normal reviewed execution. All 13 windows were measured again with 0.10.21; no runtime versions are mixed in the library.

The normal 0.10.21 run completed without another outbound approval rejection. Observation time summed to 37.75 seconds, with a 784.3 MiB process RSS peak and two-CPU affinity. Concurrent language-model evaluation shared the machine, so these are not isolated performance results. Static inspection and this execution are not a universal network-silence certification. Exact runtime/source hashes, the rejected action, and accepted measurements are recorded in `data/motion/measurement-run.v1.json`.

## Remaining evidence needed

The next quality improvement is more independently reviewed source events with stable camera framing, comparable face/body coverage, explicit audio-to-image speaker association, and repeated conversational conditions. Human-reviewed gesture labels and held-out conditional-selection evaluation are needed before a trained personal motion policy can be claimed. Better avatar geometry or more rig controls cannot establish that missing behavioral evidence.
