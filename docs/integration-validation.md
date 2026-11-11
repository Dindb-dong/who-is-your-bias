# Measured integration results

These follow-up runs were performed on 2026-11-10 UTC / 2026-11-11 KST. They extend the [initial direct-engine smoke](inference-validation.md) with actual local services and media. A completed execution, a recognized sentence, a valid citation identifier, and a convincing replica are separate claims.

The machine has CPU inference only. Model processes were run sequentially to avoid simultaneous language, speech, ASR, and video-model memory pressure. Results are single-machine development observations, not a latency distribution or production capacity benchmark. The repository retains code, metadata, and concise findings; original recordings, full transcripts, weights, and generated audio remain in ignored local paths.

## Language model and application HTTP

The optional local language server loaded the pinned Qwen3-0.6B checkpoint and returned real generated text through its OpenAI-compatible endpoint. The main application's `/api/chat/stream` also completed a real provider-backed turn. An initial ordinary supported question completed in about 13.33 seconds, with `provider: openai`; the model omitted citation markers, so the application correctly preserved `grounding.status: unverified`.

An initial second smoke prompt appended a long citation-format instruction. Retrieval abstained before contacting the model, and that run's requested second model turn was therefore not counted as successful model execution. This remains a recorded retrieval limitation. No citation was appended to make the report pass. The separate [generation evaluation report and protocol](evaluation.md) describe the A/B/C cases, measured token usage, completed/error conditions, and ungraded semantic review.

The language checkpoint revision is `c1899de289a04d12100db370d81485cdf75e47ca`. The service was configured for CPU bfloat16, four threads, and a 160-token request cap. A healthy configured server is distinct from a loaded model; health reports both. The server keeps its single-worker admission lock until model work actually stops, including after HTTP cancellation.

## Publisher-linked reference and real ASR

The [TED2013 publisher page](https://www.ted.com/talks/elon_musk_the_mind_behind_tesla_spacex_solarcity?view=transcript) supplied the native video URL and explicitly labeled caption text. The media opening was acquired through normal public HTTPS. The reference workflow compares those captions to actual ASR without supplying the caption text as a recognition prompt.

| Observation | Result |
| --- | --- |
| Context input | 49.049188 seconds of native recording |
| Context ASR | 14.438 seconds on CPU int8, four threads |
| ASR model | `Systran/faster-whisper-small.en`, revision `d1d751a5f8271d482d14ca55d9e2deeebbae577f` |
| Canonical reference | 9.830042 seconds, 24 kHz, mono PCM16 |
| Native source interval | 21.45–31.280042 seconds |
| Text comparison | 32 normalized words agree with publisher cues and two decodings of the same content |
| Signal levels | Peak −1.613 dBFS; RMS −22.895 dBFS; no clipped samples |
| Review state | **Candidate**; listening, acoustic overlap, and subjective quality remain unreviewed |

The second decoding used an 11.75-second parent clip. A disagreement in its final phrase was resolved by excluding that phrase from the shorter candidate. The retained PCM prefix matches exactly; this is not a claim of a third independent reference decoding. Native video and caption clocks differed by approximately 2.974–4.560 seconds over matched cue-leading words, so no constant subtitle offset was assumed for the entire interview.

See [reference preparation](research/voice-reference.md) and `tools/references/validation.json` for source hashes, local clocks, software versions, the review ledger, and reproducible commands. Attribution comes from publisher labels and matched words; the workflow does not infer identity from a face or voice embedding.

## Transcript-conditioned voice through the actual API

The real Qwen3-TTS voice service received the canonical candidate WAV and its text through the normal **audio-and-transcript** conditioning mode. This run did not use speaker-embedding-only mode. The main application's `/api/speech/prepare` invoked that local HTTP service, ran actual Rhubarb recognition on the generated waveform, and returned a clip URL. The smoke downloaded the clip and verified both its SHA-256 and the mouth-cue duration against the audio bytes.

The requested sentence was newly authored simulation disclosure text: `This is a research simulation.` It was not attributed to the source speaker.

| Measured item | Result |
| --- | --- |
| Voice model | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |
| Immutable checkpoint | `5d83992436eae1d760afd27aff78a71d676296fc` |
| Device / precision / threads | CPU / bfloat16 / 4 |
| Output codec-token cap | 96 |
| Complete preparation and HTTP retrieval | **44.563 seconds**, including lazy model load, reference encoding, generation, and mouth analysis |
| Generated waveform | **1.52 seconds**, 24 kHz, mono PCM16; 73,004 WAV bytes |
| Mouth timing | **10 segments**, `audio_recognition`, PocketSphinx via Rhubarb1.14.0 |
| Voice process observed RSS high-water mark | 3,165,236 KiB, sampled from `/proc`; approximately 3.02 GiB |
| Main API process observed RSS high-water mark | 47,988 KiB |
| Voice service after request | Model loaded, prompt cached, no active generation |

Generated audio SHA-256: `1a78aa966e2bde3045d7a2d37fc9f1264b1e1e55c187e7460f5891862344f219`.

Reference audio SHA-256: `f20f897fde67d14d2fb77d42d98b0890bf3c1d1fd121b41879dfd448f0d47e4c`.

The observed memory figures are process measurements from this run, not hardware requirements for every checkpoint or a test of both model servers resident at once. Preparation is slower than real time on this CPU and is completed before playback. The upstream waveform API does not expose EOS completion, so this execution alone does not certify natural stopping or absence of truncation for other inputs.

The [committed voice measurement summary](../tools/inference/voice-http-validation.json) includes waveform measurements, execution metadata, provenance, and raw-report hashes. The complete local report is `reports/local/voice-http.json`; the waveform and mouth cues are `voice-http.wav` and `voice-http.timing.json` in the same directory. [Speech timing](speech-timing.md) documents reproduction against a running API and the bounded cache/fallback behavior.

## Independent ASR screen of the generated waveform

A separate faster-whisper process transcribed the actual 1.52-second generated WAV, without an initial prompt or the requested sentence supplied as a hint. It recognized the five normalized words of the requested sentence with zero substitutions, deletions, or insertions. The normalized token word-error rate is **0/5 = 0.0** for this single short sample; recognition took **6.719 seconds**.

This supports intelligibility for that one sentence under that recognizer. It is not human listening evaluation, target-voice similarity, naturalness, or evidence that arbitrary long or Korean outputs have the same quality. Full measurements are in `reports/local/voice-http-asr-screening.json`; committed metadata is included in `tools/references/validation.json`.

## Actual video observations and replayable samples

MediaPipe0.10.35 ran real face/pose inference on two bounded publisher-video inputs. The first pass observed 64 frames over eight seconds. A camera transition in its opening was excluded, leaving a six-second face/head preview with 48 frames and 51 named face coefficients plus head rotations. The pose model detected a body, but the cropped camera view did not provide sufficiently visible elbows and wrists, so that preview contains no invented arm tracks.

A single additional 20-second publisher-video acquisition exposed a wider view. A fixed 280 × 350 pixel crop at `(170, 50)` within the 854 × 480 source excluded the second seated person while keeping shoulders, elbows, and wrists. The second inference pass processed 40 frames over five seconds. Chest, upper-arm, forearm, and hand-direction rotations passed the in-frame and visibility/presence threshold of 0.65 in all 40 frames. Facial detection was present in only seven frames, so the body sample intentionally uses pose data only.

| Replay sample | Exported measurements | Clock and review |
| --- | --- | --- |
| `ted-2013-geometry-preview.json` | 48 frames / six seconds; face coefficients and head orientation | Native source window 14–20 seconds; geometry preview, attribution unreviewed |
| `ted-2013-body-geometry-preview.json` | 40 frames / five seconds; chest and both arm/hand-direction tracks | Requested native source window 65–70 seconds; exact first-frame placement unverified; geometry preview |

Both small measurement files are included under `apps/web/public/samples/`; the full recordings are not. The Research lab loads either sample on request and requires an explicit geometry-preview action before replay. Recorded input, preparation, video, observation, and review hashes, together with crop parameters, make the transform chain inspectable. The measured body path does not relabel designed facial animation as observed.

The left and right forearm reference-relative maxima in the body sample were 55.56° and 27.83°; neither reached the exporter's 70° bound. Direction-based limb rotations do not recover axial twist or individual finger articulation. Neutral-frame selection is a rotation reference, not automatic anatomical calibration. These two short clips establish that the observation/export/rig path executes; they do not establish recurring personal mannerisms or a learned gesture policy. [Avatar calibration](avatar-calibration.md) records the implementation, model assets, reproduction commands, and observed limits.

## Runtime telemetry interruption and resolution

An earlier voice HTTP attempt was stopped by automatic approval review when a transitive dependency attempted an unrelated Microsoft telemetry request. No completed audio/HTTP result was observed for that attempt, and its report remains marked blocked. The subsequent run used the official [ONNX Runtime process-start opt-out](https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md), `ORT_DISABLE_TELEMETRY=1`, before Python and optional model imports. The installed ONNX Runtime1.31.1 binary exposed that setting. The model was loaded from an existing local checkpoint with Hub offline mode enabled.

The completed retry did not authorize or route around the rejected telemetry request. It disabled the uploader using the library's supported configuration. Optional inference entry points and environment examples now apply this setting before model dependencies initialize.

## Interpretation and remaining quality work

The implemented speech and animation transports can bind measured data to the correct clocks. The reference remains a candidate, the procedural avatar remains a stylized rig, and conversation gestures remain authored rules. A learned situation-to-gesture policy, a high-likeness production asset, multi-sentence and Korean listening evaluation, independent persona evaluation, and factual entailment review require additional evidence and work. Mechanically valid citation identifiers do not substitute for those evaluations.
