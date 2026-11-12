# Jensen speech adaptation experiment

This is the M4 execution record for the [Jensen recovery handoff](jensen-replication-handoff.md).
The reproducible preparation is measured in
[preparation.json](../reports/voice/jensen-2026-11-13/preparation.json).
The current checkpoint has a persisted **candidate** dataset, completed acoustic
tokenization, and **one actual speech optimizer update**. Its saved tensors and
state are measured in [pilot-step1.json](../reports/voice/jensen-2026-11-13/pilot-step1.json).
Fresh-process resume, generated speech, independent output ASR, and human
listening remain separate completion gates. The [codec receipt](../reports/voice/jensen-2026-11-13/codecs.json)
records all eight encodings (1,151 frames, 16 codebooks per frame). The encoding
loop took 6.13 seconds, excluding model loading.

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
16 planned optimizer updates. **One update has completed** with codec objective
loss 12.889015; the remaining 15 updates have not run. The first update took
21.22 seconds, including its input preparation and checkpoint save; the complete
process block including model load took 54.21 seconds. Peak process RSS was
6,675,206,144 bytes. All 344,064 learned LoRA-B elements are nonzero after that
step (their PEFT initialization is zero); their L2 norm is 0.0117255. This proves
actual changed weights, not improved perceptual quality.
A representative frozen-linear forward/backward probe measured float32 at
15.95 ms versus bfloat16 at 21.74 ms (four CPU threads). The first actual speech
update measured the full model time and memory above; this microbenchmark alone
does not do so. A speech LoRA checkpoint counts as weight adaptation only after an actual
optimizer update and a valid saved adapter. Reference prompting remains a
separate conditioning mechanism in both the base and adapted conditions.

Every successful update atomically saves learned adapter tensors, optimizer
state, Python and Torch random state, cursor, actual losses, constant learning
rate policy, model/data/tokenizer/code identities, and package versions. Previous
checkpoints are retained. The bounded process pauses for remote upload and fresh
readback before the next process resumes. The tiny-neural regression verifies
matching next-update weights after reconstructing the model and optimizer; it is
a checkpoint implementation test, not evidence of Qwen speech adaptation.

## Execution and resume

The preparation and one-update optimization commands below completed successfully.
The fresh-process resume command remains proposed until its actual run receipt exists. Set `ART` to the private artifact root and run from the repository with
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

After that actual checkpoint is saved and read back, use identical arguments,
add `--resume "$ART/voice/lora-v2/step-00000001"`, and choose the next bounded
`--stop-after` block. The driver rejects changed input, model, code, or optimizer
settings. Do not edit the driver halfway through a run and bypass that check.

## Output evaluation gates

The [frozen evaluation protocol](../reports/voice/jensen-2026-11-13/evaluation-protocol.json)
records model and reference hashes before any synthesis. Use the same source
reference, fixed cases, generation settings, and seed for
both the unadapted and adapted model. The existing `tools.voice_evaluation`
workflow records actual HTTP behavior, output WAV hashes, latency, and health
metadata. The health snapshot now retains subject, reference identity, actual
adapter identity, and synthesis kind.

Generate the frozen English and Korean cases, stop the TTS service to release
memory, and transcribe the generated waveforms with the independent ASR model.
Report English WER, Korean CER, cold/warm wall time, failures, and raw outputs.
The separately pinned `microsoft/wavlm-base-plus-sv` diagnostic also compares
complete generated clips with all eight source clips, reporting raw cosines and
event-balanced averages. The verification model is independent of both Qwen and
the ASR model; the references themselves are training-exposed. No calibrated
identity threshold, unseen-event result, or Korean language invariance is claimed.
These measures do not replace perceived Jensen similarity. Human
listening and speaker similarity remain unmeasured until their actual reviews
are supplied; the adapter must not be promoted based on training loss alone.

## Saved evidence and remaining work

`voice-dataset-pilot-v2.tar.gz` contains the portable audio/text training rows,
source-bound reference bundles, screen results, model receipts, and frozen split
registration. It is 9,881,794 bytes; its SHA-256 is
`30a907a9493f0c1a0c87091fad4c49da9d4300c0990c1ca0f2744a8f1f473510`.
The dataset manifest hash is
`843d4798449eee4b8d52cb170d05face601642666ec9674832721acc94c60b12`.
The coordinator records durable object identifiers and readback receipts in the
run artifact manifest. The archive's creation alone is not a persistence claim.

Completed gates: dataset upload and fresh readback, acoustic tokenization, codec
upload and fresh readback, one actual optimizer update and local resumable checkpoint.
Next gates: checkpoint persistence and fresh-process resume; remaining bounded updates; English/Korean
synthesis for both conditions; independent output ASR; and a measured report. The candidate pilot remains separate from application
promotion and from a qualified speech likeness.
