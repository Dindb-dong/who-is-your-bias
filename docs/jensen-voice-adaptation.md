# Jensen speech adaptation experiment

This is the M4 execution record for the [Jensen recovery handoff](jensen-replication-handoff.md).
The reproducible preparation is measured in
[preparation.json](../reports/voice/jensen-2026-11-13/preparation.json).
The bounded speech-weight experiment completed **16 actual optimizer updates**.
The final adapter, optimizer/random state and progress counters were uploaded,
retrieved in a fresh operation, and verified by SHA-256. See
[training-completed.json](../reports/voice/jensen-2026-11-13/training-completed.json)
for the exact contract and durable receipts. Both conditions now have four actual
English/Korean waveforms, with independent ASR and WavLM diagnostics for all eight.
The [measured evaluation](../reports/voice/jensen-2026-11-13/evaluation.json) records
the outputs and limitations. The adapter remains a candidate; no quality promotion
has been made.

The [generated listening page](../reports/voice/jensen-2026-11-13/listening/README.md)
includes all eight exact output WAVs in the repository. Download that directory
and open `index.html` locally for the labeled comparison. Original source/reference
audio is not included in that public directory.

The [codec receipt](../reports/voice/jensen-2026-11-13/codecs.json) records all eight
encodings (1,151 frames, 16 codebooks per frame). The encoding loop took 6.13
seconds, excluding model loading. All source clips still have **candidate**
status and zero human listening reviews.

## Source and subject binding

The dataset contains eight original speech intervals, totaling **91.84 seconds**,
from four registered training event families:

| Event family | Clips | Media clock |
| --- | ---: | --- |
| Goldman Sachs 2024 | 1 | Publisher's edited video package, elapsed seconds |
| Acquired 2023 | 1 | Acquired podcast, elapsed seconds |
| Lex #494 | 3 | Lex podcast, elapsed seconds |
| Stanford View From The Top 2024 | 3 | Edited Stanford podcast, elapsed seconds |

Each interval has the publisher's explicit Jensen Huang speaker label, contiguous
word agreement with unprompted ASR, source and audio hashes, and word boundaries.
The untimed Goldman, Acquired, and Stanford transcripts are matched by words;
no transcript clock is invented. Lex's publisher video timestamps are preserved
as original provenance but ignored for the independently decoded podcast clock.
No fixed advertisement offset is assumed.

Nine contexts totaling 999 seconds were decoded with pinned multilingual
`Systran/faster-whisper-small` at revision
`536b0662742c02347bc0e980a01041f333bce120`, on CPU with int8 computation.
The initial exact publisher matching produced 19 candidates. A fixed second-stage
screen required minimum word confidence 0.5, exclusion of context-edge windows,
and exact normalized transcription in a fresh unprompted decoding of the short
clip. Eight passed; seven failed confidence, two touched a context edge, and two
disagreed in the second decoding.

This screen uses the same ASR model on a different input window. It is **not** an
independent speaker identification model or a listening review. Zero clips have
human listening review, and overlapping speakers, music, and perceived fidelity
have not been independently graded. The dataset remains an explicit experiment.
It does not restore or claim the interrupted session's 60 clips / 6.4 minutes.

`tools.adaptation.voice_data` validates source labels, exact aligned text,
audio bytes, and each source-to-event mapping. Dataset v2 includes an exact copy
and hash of `data/acquisition/jensen-huang/split-registration.json`; all 14
registered event families are preserved, with WEF Davos 2026 and No Priors 2023
remaining test events. No audio from those events enters this dataset. The draft
v1's two unused obsolete test labels were corrected before any adaptation.

Named serving also requires the hash-bound reference manifest. The Jensen and
historical Musk bindings are distinct; a configured name cannot silently relabel
the other speaker's data. Candidate references require an explicit environment
setting and retain that status in health/evaluation records.

## Actual optimization contract

The CPU experiment is implemented in `tools.adaptation.voice_lora`. It follows
the original speech-code objective in Qwen's
[pinned upstream trainer](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/sft_12hz.py)
and imports its
[pinned dataset implementation](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/dataset.py).
The upstream checkout must be clean at that exact revision.

The selected base is `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, revision
`5d83992436eae1d760afd27aff78a71d676296fc`. Its complete downloaded snapshot is
2,516,100,892 bytes including the acoustic tokenizer. Every file has a saved
hash receipt. This is a download measurement; it is not a training result.

The objective is first-codebook cross entropy plus 0.3 times the remaining
codebook loss, using actual codec targets extracted from the selected speech.
Low-rank matrices are attached to the talker's attention `q_proj` and `v_proj`
modules. The base, acoustic tokenizer, speaker encoder, text projection, and
other weights remain frozen. The 0.6B text projection is applied before combining
text and speech embeddings, consistent with its inference path.

The planned pilot uses CPU float32, SDPA attention, rank 4 / alpha 8, LoRA dropout
0.05, learning rate 0.00002, AdamW, seed 42, batch size one, four threads, and
16 planned optimizer updates; all **16 completed**, covering two shuffled passes
through eight source clips. The 573,440 trainable parameters include 344,064 LoRA-B
elements, all nonzero in the final checkpoint. Their L2 norm is 0.103470; the
maximum absolute weight is 0.000322710. PEFT initializes those matrices to zero.
These are actual changed model weights, separate from reference conditioning.

The mean training objective was 12.004980 in the first pass and 11.860619 in the
second pass. These are training-exposed objective values with dropout, not
independent intelligibility or voice-likeness results. All quality and promotion
gates remain separate from the optimizer completion.

A representative frozen-linear forward/backward probe measured float32 at
15.95 ms versus bfloat16 at 21.74 ms with four CPU threads. The actual six completed
training blocks, including model loads, totaled 223.20 seconds. That total excludes
artifact transfers, tool scheduling and an interrupted unfinished block. The first
actual update took 21.22 seconds; later update times varied by speech example and
warm-up. The maximum measured process RSS was 6,869,979,136 bytes. The measured
full-model run establishes these resource values; the microbenchmark alone does not.

Every successful update atomically saves learned adapter tensors, optimizer
state, Python and Torch random state, cursor, actual losses, constant learning
rate policy, model/data/tokenizer/code identities, and package versions. Previous
checkpoints are retained. The bounded process pauses for remote upload and fresh
readback before the next process resumes. The tiny-neural regression verifies
matching next-update weights after reconstructing the model and optimizer; it is
a checkpoint implementation test, not evidence of Qwen speech adaptation.

## Execution and resume

The preparation and initial one-update optimization commands below completed
successfully. A fresh process loaded the remotely retrieved checkpoint and
completed step 2. Further bounded blocks completed at steps 6, 10, 14 and 16; every
block was uploaded and read back before the next began. A tool-runtime reset left
no valid checkpoint after step 14. The run recovered from the verified step 14
state and completed the remaining two updates. Set `ART` to the private artifact root and run from the repository with
the pinned inference environment. The selected dataset archive must first be
uploaded, retrieved in a fresh operation, and verified by SHA-256.

```bash
python -m tools.adaptation.voice_lora prepare \
  --dataset "$ART/voice/dataset-v2" --output "$ART/voice/codes-v2" \
  --base-model "$ART/models/qwen-tts06-base" \
  --persona-id jensen-huang --allow-candidates --threads 4
```

Persist and verify the completed codec directory before starting optimization.
A first bounded optimization process is:

```bash
python -m tools.adaptation.voice_lora train \
  --dataset "$ART/voice/dataset-v2" --codes "$ART/voice/codes-v2" \
  --output "$ART/voice/lora-v2" \
  --base-model "$ART/models/qwen-tts06-base" \
  --base-revision 5d83992436eae1d760afd27aff78a71d676296fc \
  --upstream "$ART/qwen-upstream" \
  --upstream-revision 022e286b98fbec7e1e916cb940cdf532cd9f488e \
  --persona-id jensen-huang --allow-candidates --threads 4 \
  --precision float32 --rank 4 --learning-rate 0.00002 \
  --max-updates 16 --stop-after 1
```

The step 1 archive was retrieved and extracted under
`$ART/voice/remote-step1-restore/`; the actual next process added
`--resume "$ART/voice/remote-step1-restore/voice/lora-v2/step-00000001"` with
`--stop-after 1`. It restored optimizer/RNG state and produced step 2. For the
remaining run, use the identical contract, the last verified checkpoint path, and
`--stop-after 4` (a count of additional updates, capped at 16 total). The driver rejects changed input, model, code, or optimizer
settings. Do not edit the driver halfway through a run and bypass that check. This pilot is
now complete at its declared 16-update ceiling; the resume examples document the
verified interrupted-run procedure, not a request to duplicate completed updates.

## Output evaluation gates

The [frozen evaluation protocol](../reports/voice/jensen-2026-11-13/evaluation-protocol.json)
records model and reference hashes before any synthesis. Use the same source
reference, fixed cases, generation settings, and seed for
both the unadapted and adapted model. The existing `tools.voice_evaluation`
workflow records actual HTTP behavior, output WAV hashes, latency, and health
metadata. The health snapshot now retains subject, reference identity, actual
adapter identity, and synthesis kind.

The concrete [runtime binding](../reports/voice/jensen-2026-11-13/runtime-binding.json)
pins the frozen protocol and completed training receipt. It selects the exact
Goldman reference and step 16 adapter, verifies the base model and artifact bytes,
and requires explicit candidate opt-in. Run the check without loading a model:

```bash
python -m tools.voice_evaluation.serve_candidate \
  --binding reports/voice/jensen-2026-11-13/runtime-binding.json \
  --artifacts "$ART" --condition adapted --allow-candidates --verify-only
```

Remove `--verify-only` to start the loopback service on port 8092. Use
`--condition reference` for the matched base model, and stop that service before
starting the adapted condition. The launcher uses the frozen float32 CPU settings
and discards stale voice environment defaults. It preserves an optional
`BIAS_VOICE_API_KEY`, exposes the simulation header, and does not change application
defaults. A passed byte/subject preflight establishes the selected artifacts;
actual model loading and waveform generation remain separately measured stages.

Some execution hosts isolate loopback networking between separate commands. The
first separate server/client invocation here failed immediately with `ConnectError`
before any model load; its failure receipt is preserved. Run the server and real
HTTP client in one supervised invocation on those hosts:

```bash
python -m tools.voice_evaluation.local \
  --binding reports/voice/jensen-2026-11-13/runtime-binding.json \
  --artifacts "$ART" --condition reference --allow-candidates \
  --output "$ART/voice/eval-reference-v2"
```

Repeat with `--condition adapted` and a new output path after saving the first
condition. This runner validates the frozen case bytes, checkpoints each waveform,
keeps server logs and a process receipt, and stops its server before returning.

The frozen English and Korean cases were generated, then each TTS process was
stopped to release memory before independent ASR and speaker diagnostics ran.
English WER, Korean CER, cold/warm wall time, failures, and raw outputs are saved.
The separately pinned `microsoft/wavlm-base-plus-sv` diagnostic also compares
complete generated clips with all eight source clips, reporting raw cosines and
event-balanced averages. The verification model is independent of both Qwen and
the ASR model; the references themselves are training-exposed. No calibrated
identity threshold, unseen-event result, or Korean language invariance is claimed.
These measures do not replace perceived Jensen similarity. Human listening and
perceived speaker likeness remain unmeasured until actual reviews are supplied;
the adapter must not be promoted based on training loss or uncalibrated cosines.

## Actual matched speech results

All eight requests produced verified mono WAV files, totaling **57.76 seconds**.
The [reference-condition receipt](../reports/voice/jensen-2026-11-13/synthesis-reference.json)
and [adapted-condition receipt](../reports/voice/jensen-2026-11-13/synthesis-adapted.json)
retain HTTP timing, signal checks, source binding, and actual loaded adapter
identity. The adapted service reported the verified step 16 checkpoint and
`weight_adaptation: true` after each real waveform response.

The independent recognizer received only the generated audio and declared language,
with no transcript prompt, prefix, or hotwords. The scores below keep the frozen
normalization; no numeric rewriting was introduced after observing the results.

| Case | Reference ASR error | Adapted ASR error | Reference WavLM cosine | Adapted WavLM cosine |
| --- | ---: | ---: | ---: | ---: |
| Short English | WER 0/5, 0% | WER 0/5, 0% | 0.936892 | 0.944122 |
| Multiple English sentences | WER 0/33, 0% | WER 0/33, 0% | 0.956202 | 0.948573 |
| Multiple Korean sentences | CER 1/58, 1.724% | CER 0/58, 0% | 0.793490 | 0.800503 |
| English numbers | WER 5/21, 23.810% | WER 5/21, 23.810% | 0.944494 | 0.956157 |

For the number case, both recognized transcripts used `9.30`, `12`, and `24`
where the input used words. Those formatting differences account for the literal
word-error score. The reference-condition Korean transcript omitted the initial
`이`; the adapted transcript contained all 58 normalized characters. None of the
eight ASR suffix, repetition, or substantial-deletion screens fired. ASR remains
an imperfect recognizer; these observations do not prove backend EOS, natural
prosody, or a general intelligibility improvement.

The WavLM value is an event-balanced mean of raw cosines against eight source
clips: average within each source event, then weight all four events equally.
It increased in three cases and decreased in the longer English case. There is
one output per condition and case, no impostor calibration, no held-out event,
and no human rating. These are descriptive scores, not resemblance percentages
or an overall improvement result. The source-reference pair cosines ranged from
0.935389 to 0.984556 across 28 pairs; they do not independently establish identity.

| Case | Reference audio / HTTP wall time | Adapted audio / HTTP wall time |
| --- | ---: | ---: |
| Short English, cold | 2.16 s / 56.15 s | 1.92 s / 35.25 s |
| Multiple English sentences | 10.56 s / 81.61 s | 9.92 s / 71.09 s |
| Multiple Korean sentences | 10.16 s / 85.14 s | 8.40 s / 57.04 s |
| English numbers | 7.36 s / 76.26 s | 7.28 s / 56.52 s |

The condition runs took 305.45 and 230.02 seconds including startup; there was no
warm-up request. These timings describe an ordered base-then-adapted run with
different waveform lengths and cache state, so they do not establish a speedup.
The service returned complete WAVs and did not stream audio. TTS peak process RSS
was 6,341,668,864 and 6,113,681,408 bytes. In the adapted invocation, available
memory fell from 9,249,016 KiB before model loading to 4,695,104 KiB before shutdown.
Full text and voice models were serialized on the approximately 10 GB host;
simultaneous full-model operation was not verified.

The two ASR runs took 12.16 and 11.58 seconds. The independent WavLM run took
21.74 seconds and completed all eight output comparisons and 28 source pairs.
An eight-sample condition-blind listening bundle was created at
`voice/listening-v1/listener/index.html`, with a source comparison recording.
Its manifest hash is `1f3e3ffe0076b5334ba67b2ad423374a318496f843d6f8fbfe8ad72ad9b996ec`.
All listening fields remain **unrated**; bundle creation is not a human review.

## Saved evidence and remaining work

`voice-dataset-pilot-v2.tar.gz` contains the portable audio/text training rows,
source-bound reference bundles, screen results, model receipts, and frozen split
registration. It is 9,881,794 bytes; its SHA-256 is
`30a907a9493f0c1a0c87091fad4c49da9d4300c0990c1ca0f2744a8f1f473510`.
The dataset manifest hash is
`843d4798449eee4b8d52cb170d05face601642666ec9674832721acc94c60b12`.
The coordinator records durable object identifiers and readback receipts in the
run artifact manifest. The archive's creation alone is not a persistence claim.

The final [delivery receipt](../reports/voice/jensen-2026-11-13/delivery.json)
identifies `jensen-voice-candidate-evaluation-v1.tar.gz`: **18,933,760 bytes**, SHA-256
`017656a776a2af2b2d1c3ca83a9eb58a88533e903a2f4a95766fb465f96fe568`.
It contains the eight selected source-reference bundles, portable dataset,
acoustic tokens, final learned adapter and optimizer state, generated waveforms,
complete ASR/WavLM outputs, and the unrated blind listening bundle. The large
pretrained dependencies are listed by immutable upstream revision and file hashes.
Their weights are separate from this archive; the actual learned weights are
included.

The first materialization was truncated and failed both SHA-256 and gzip integrity.
A fresh retry of the same exact stored object recovered the full matching bytes.
After extracting into a new directory, all **134 payload file hashes** passed;
the eight-row dataset, all eight source-reference manifests, and the final step 16
checkpoint passed their real loaders. The failed attempt is retained explicitly
in the receipt. The repository's generated listening WAVs were also verified
byte-for-byte against the measured outputs; no source/reference recording was
added to that public directory.

Completed gates: dataset upload and fresh readback, acoustic tokenization and
codec persistence, 16 actual optimizer updates, bounded checkpoint persistence,
fresh-process remote-checkpoint resume, and recovery after a real tool-runtime
interruption. The final archive is `voice-lora-v2-step16.tar.gz`, 6,475,745 bytes,
SHA-256 `d87c8ac3a6c00d7f35023bb2cc048ebb76984f2288d62ec27c19f7d22951cf7e`.

Completed output gates: eight matched English/Korean waveforms, successful
independent ASR on all eight, independent WavLM diagnostics on all eight, and an
unrated blind listening bundle. Each synthesis condition was uploaded and freshly
retrieved with a matching SHA-256 before continuing. Remaining quality gates are
human listening, calibrated speaker comparisons, and unseen-event generalization.
The candidate pilot remains separate from application promotion and from a
qualified speech likeness.
