# Voice-reference preparation and review

The repository now has a reproducible path from public publisher media and explicitly labeled captions to a short reference WAV, its transcript, source-clock provenance, and a hash-bound review ledger. Preparation creates a **candidate**. It does not silently approve an unknown speaker, declare clean audio from signal levels, or claim a listening review occurred.

The actual run produced a **9.830042-second, 24 kHz mono PCM reference candidate** from TED2013. Its 32 normalized words agree with the publisher-labeled passage and two unprompted ASR decodings of the same audio content. The canonical candidate remains pending listening/overlap/quality review. It is suitable for explicitly labeled engineering experiments, with that status preserved; it is not a certified likeness or a voice-similarity result.

Implementation is under `tools/references`; synthetic tests are under `tests/references`. Measured metadata, including full SHA-256 values and UTC timestamps, is in `tools/references/validation.json`. Full source material, reference text, recordings, and model weights are not committed.

## Actual source and results

The original [TED2013 interview page](https://www.ted.com/talks/elon_musk_the_mind_behind_tesla_spacex_solarcity?view=transcript) exposes a native MP4 link in its public `videoPlayerData.resources.h264.file` metadata. Ordinary HTTPS access retrieved the opening of that specific asset. The acquisition record is in `data/raw/voice-reference/ted-2013-native/acquisition.json`; the local MP4 is `publisher-opening.mp4` in the same directory.

| Measurement | Observed result |
|---|---|
| Acquired native opening | 50.114756 seconds; 7,320,515 bytes; 854 × 480 video with stereo audio |
| ASR context | 49.049188 seconds, converted to 16 kHz mono PCM |
| ASR implementation | faster-whisper 1.2.1, CTranslate2 4.8.5, CPU int8, 4 threads, 1 worker |
| Model | `Systran/faster-whisper-small.en`, revision `d1d751a5f8271d482d14ca55d9e2deeebbae577f` |
| Downloaded model files | 486,098,798 bytes |
| First actual ASR execution | 14.438 seconds; 133 normalized tokens, 128 matching publisher context |
| Independent parent-clip ASR | 11.75 seconds of input, decoded in 7.807 seconds |
| Canonical reference | Native MP4 interval 21.45–31.280042 seconds; measured WAV duration 9.830042 seconds |
| Canonical reference text | 32 normalized tokens, agreeing with publisher cues `cue-00007` through `cue-00010` and both decodings |
| Candidate status | Publisher attribution, text comparison, and native-clock preparation reviewed; acoustic overlap and listening quality remain unknown |

The parent candidate, `ted-2013-reference-v1`, lasted 11.750042 seconds. Its last phrase had a one-token disagreement between the two ASR results. The whole phrase was removed from canonical `ted-2013-reference-v2`. The shorter WAV's PCM samples were verified to be identical to the corresponding parent prefix, and the second decoding's last included word ends at native 31.21 seconds. This preserves the useful independent check without pretending that a third ASR run was performed.

The canonical local files are:

```text
data/local/references/ted-2013-reference-v2/audio.wav
data/local/references/ted-2013-reference-v2/reference.txt
data/local/references/ted-2013-reference-v2/reference.json
data/local/references/ted-2013-reference-v2/review-ledger.jsonl
```

Audio SHA-256: `f20f897fde67d14d2fb77d42d98b0890bf3c1d1fd121b41879dfd448f0d47e4c`.

Reference-text SHA-256: `22cbc93befad6c0d493433c4b392a2ad8c1357b04575d9e0055609ca68aa9a87`.

The matching transcript and ASR text stay in ignored local files; this document does not reproduce the passage. The recorded run uses UTC timestamps, on November10 UTC / November11 KST.

## Attribution and timing grounds

The publisher explicitly labels the first answer as Elon Musk at `cue-00004`. The selected cues continue that label within the same publisher paragraph. Matching actual ASR words to the labeled conversation provides the attribution basis. The recognizer did not receive the publisher text as a prompt. Neither facial appearance nor voice resemblance was used to identify a person.

The native recording and caption clocks differ. In the measured context, the publisher's answer begins at caption time 9.000 seconds while the first corresponding ASR word begins near native 13.56 seconds. Across matched cue-leading words, observed differences range from 2.974 to 4.560 seconds, with median 3.6155 seconds. Caption cue onsets and ASR word boundaries have different uncertainty; these observations do not establish a constant offset for the whole interview. Reference selection uses the actual native audio word times, with the source clock named `ted_native_mp4`.

Unknown and mixed publisher cues cannot establish the requested speaker in `prepare`. Selected cues must exist, be consecutive, and carry the same explicit or paragraph-supported label. The ASR input hash must match the prepared context audio; the original media hash must match the context's source input. A proposed boundary crossing an observed ASR word is rejected. These checks support review but do not replace listening or phoneme alignment.

## Reproduce source preparation

Run these commands from the checkout. Use new output directories; the tools refuse to overwrite candidates or transcription output.

Acquire the public captions with the existing publisher extractor:

```bash
.venv/bin/python -m tools.transcripts ted \
  --url 'https://www.ted.com/talks/elon_musk_the_mind_behind_tesla_spacex_solarcity?view=transcript' \
  --source-id ted-2013 \
  --output data/local/references/reproduce-publisher
```

Acquire only the native opening and prepare ASR context:

```bash
mkdir -p data/raw/voice-reference/reproduce-native

ffmpeg -nostdin -hide_banner -loglevel error -n -rw_timeout 20000000 \
  -i 'https://py.tedcdn.com/consus/projects/00/03/56/011/products/2013-elon-musk-011-fallback-ac0208b9ee7c6d6d53feec1fec06c734-1200k.mp4' \
  -t 50 -map 0:v:0 -map 0:a:0 -c copy -movflags +faststart \
  data/raw/voice-reference/reproduce-native/publisher-opening.mp4

.venv/bin/python -m tools.media prepare \
  data/raw/voice-reference/reproduce-native/publisher-opening.mp4 \
  data/local/references/reproduce-context \
  --source-id ted-2013 \
  --source-url 'https://www.ted.com/talks/elon_musk_the_mind_behind_tesla_spacex_solarcity?view=transcript' \
  --source-timeline ted_native_mp4 --start 0 --duration 49 \
  --sample-rate 16000 --audio-only
```

The MP4 URL above was observed in the publisher page; it was not inferred from a YouTube identifier. If the publisher changes the asset or captions, inspect the new matching report before reusing measured clip boundaries. The source SHA-256 in the new manifest makes such changes visible.

## Reproduce actual CPU ASR

Create an isolated environment. The model files stay outside the repository in `../work/asr-hf-cache`.

```bash
uv venv ../work/asr-venv
uv pip install --python ../work/asr-venv/bin/python \
  -r tools/references/requirements-asr.txt

HF_HUB_DISABLE_XET=1 ../work/asr-venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="Systran/faster-whisper-small.en",
    revision="d1d751a5f8271d482d14ca55d9e2deeebbae577f",
    cache_dir="../work/asr-hf-cache",
    allow_patterns=["config.json", "model.bin", "tokenizer.json", "vocabulary.*", "preprocessor_config.json"],
    max_workers=2,
)
print(path)
PY
```

Use the returned snapshot path as `--model-directory`:

```bash
ORT_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 timeout 180s \
  ../work/asr-venv/bin/python -m tools.references asr \
  data/local/references/reproduce-context/manifest.json \
  --model-directory ../work/asr-hf-cache/models--Systran--faster-whisper-small.en/snapshots/d1d751a5f8271d482d14ca55d9e2deeebbae577f \
  --model-repo Systran/faster-whisper-small.en \
  --revision d1d751a5f8271d482d14ca55d9e2deeebbae577f \
  --cpu-threads 4 \
  --output data/local/references/reproduce-context/transcript.json

.venv/bin/python -m tools.references match \
  --publisher data/local/references/reproduce-publisher/transcript.json \
  --asr data/local/references/reproduce-context/transcript.json \
  --output data/local/references/reproduce-context/alignment.json
```

The runner loads only a local model, rejects model files totaling over 1GB, allows at most four CPU threads, and accepts at most 120 seconds of verified PCM context. It records model revision, weight hash, dependency versions, settings, and timing. ASR stays draft and has no speaker identity.

The runner and existing media ASR adapter default `ORT_DISABLE_TELEMETRY=1` before optional imports. Reproduction commands also set it before Python starts. This follows [ONNX Runtime's official runtime opt-out](https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md#disabling-telemetry), which prevents the 1DS uploader, events, and persistent identifier from being initialized. The initial two measured ASR runs preceded this code change; their timing/results are not relabeled as reruns with the new setting.

The [official faster-whisper API](https://github.com/SYSTRAN/faster-whisper#usage) supports CPU int8 inference, word timestamps, and NumPy waveform input. In this environment, the initial file-decoding attempt failed because PyAV19 rejected `metadata_errors`. The successful runner reads the already-prepared mono16kHz PCM16 WAV with Python's `wave` module and passes a float32 waveform through the documented API. It does not silently alter the source transcript to make the test pass.

## Prepare the canonical interval

After inspecting the matching report, the measured source supports this interval and cue selection:

```bash
.venv/bin/python -m tools.references prepare \
  --media-manifest data/local/references/reproduce-context/manifest.json \
  --publisher data/local/references/reproduce-publisher/transcript.json \
  --asr data/local/references/reproduce-context/transcript.json \
  --source-media data/raw/voice-reference/reproduce-native/publisher-opening.mp4 \
  --speaker 'Elon Musk' \
  --cue-id cue-00007 --cue-id cue-00008 --cue-id cue-00009 --cue-id cue-00010 \
  --start 21.45 --duration 9.83 \
  --output data/local/references/reproduce-reference
```

`reference.txt` comes from the selected observed ASR words. The tool compares its normalized tokens with the selected publisher text and records agreement, matched-word counts, and minimum ASR confidence. Different punctuation does not count as an audible-word difference. The tool does not interpret lexical agreement as human approval.

To repeat the separate parent-clip decoding used in the measured run, first prepare the 11.75-second interval with the same command but include `--cue-id cue-00011`, set `--duration 11.75`, and use a distinct output directory. Convert that candidate's `audio.wav` to a new16kHz context with `tools.media prepare`, setting `--input-source-offset 21.45 --duration 11.75`. Run the ASR command on that context. The measured second-pass result is retained in `data/local/references/ted-2013-reference-asr/transcript.json`.

## Review contract and ledger

`reference.json` uses `voice_reference.v1`. It includes the source ID/URL/clock, native interval, original input hash, audio properties, publisher-label grounds, automatic check results, and relative artifact paths with hashes and byte sizes. Required artifacts include the WAV, reference text, selected publisher context, ASR context, alignment report, and the ordinary `media.v1` preparation manifest. Thus existing media tools can still read `manifest.json` while voice consumers use `reference.json`.

An explicit assessment contains a named reviewer, a `listening_review_completed` boolean, and five checks: `publisher_attribution`, `transcript_text`, `clip_timing`, `single_speaker`, and `audio_quality`. Each check supplies `status: verified|unknown|rejected` and concrete written `grounds`. Every verified check plus completed listening review is required for `reviewed`; an unknown check retains `candidate`; a rejected check makes the status `rejected`.

The measured v2 assessment is in `data/local/references/ted-2013-reference-v2/review-assessment.json`. Its reviewer is explicitly an AI source-context review, not an implied human listener. Three checks are verified with stated limits; the two acoustic suitability checks remain unknown and `listening_review_completed` remains false.

```bash
.venv/bin/python -m tools.references review \
  data/local/references/reproduce-reference/reference.json \
  --assessment /path/to/explicit-review-assessment.json

.venv/bin/python -m tools.references verify \
  data/local/references/reproduce-reference/reference.json
```

Review appends to `review-ledger.jsonl`. Every entry binds the artifact/metadata fingerprint, reviewer, checks, listening declaration, decision, UTC timestamp, and preceding entry hash. Verification rejects changed artifacts, inconsistent review status, altered source timing, path escapes, or broken ledger links. This is a consistency and provenance mechanism, not a cryptographic signature proving who performed a review; retaining a trusted manifest/head hash is necessary to detect complete deliberate rewrites.

Voice integrations should preserve the supplied candidate status. The actual service reference paths are the v2 `audio.wav` and `reference.txt`; the provenance sidecar is v2 `reference.json`. Synthesis from these files does not itself approve a reference or measure similarity to the named speaker.

## Native video for observed motion work

`data/local/references/ted-2013-vision-sample` contains a `media.v1` manifest, `video.mp4`, and `audio.wav`. It spans native12–20 seconds and is eight seconds long. It was passed to the separate MediaPipe/animation work with its hashes and source clock. The first answer begins around native13.56 seconds, so the sample includes an earlier pause/tail. A later native25-second frame is an audience cut; a speaker turn does not imply that every video frame shows that speaker.

Face/pose detection indexes are not identities or persistent tracks. Track continuity and speaker association must be reviewed separately before geometry is treated as person-specific motion. This reference workflow does not make that association from facial appearance.

## Verification performed

```bash
.venv/bin/python -m pytest tests/references -q
```

**22 tests passed**, covering explicit-label attribution, unknown/mixed speakers, source/audio hashes, word-boundary rejection, bounded duration, real FFmpeg extraction of a synthetic tone, output preservation, reference-text handling, review transitions, ledger binding, path containment, optional dependency errors, and metadata-only CLI output.

Two real reference-material ASR executions succeeded as recorded above. Artifact hashes and the v2 ledger were checked against the actual files. A third, independent ASR run screened the separately generated 1.52-second HTTP speech sample with no text prompt and the telemetry opt-out set before Python started: all five requested normalized words were recognized, giving a single-sentence token WER of0.0. Its detailed local report is `reports/local/voice-http-asr-screening.json`. That narrow result measures an ASR-based intelligibility screen, not human listening, speaker identity, or voice similarity.

No voice training or voice-similarity evaluation occurred in this workflow. Generated speech and motion measurements belong to their separate execution reports; they should retain the reference and tracking review statuses described here.
