# Calibrating the existing Gaussian head's expressions

This workflow fits a small supervised ARKit-to-FLAME regression from **paired,
reviewed source frames**. It does not retrain or replace the Gaussian identity,
shaped mesh, SH colours, skinning, or expression basis. FLAME coefficients are
dense latent coordinates: no coefficient is manually labelled “eyebrow”.

## Inputs and preparation

Keep all datasets, checkpoints, learned sidecars, and licensed bundle bytes in
ignored local paths or the authorized private artifact destination. Never commit
them to the public repository. Read `docs/local-only-assets.md` first.

1. Use the exact source sequence from the head tracker, including its original
   source clock. Collect visibly raised/lowered brows, blinks, eyes wide/squint,
   smiles, closed mouth and open mouth frames. A face-only head bundle is not a
   source of ARKit labels; the source video or existing observations are needed.
   Verify which source timestamp corresponds to bundle frame 0 and whether
   consecutive bundle frames follow that video's sampling interval without
   dropped frames or edits. The exporter currently writes `fps=15` as a
   constant; this field alone does not prove the source-clock mapping. The
   current preparation assumes `source_time = frame_index / fps + offset`.
   Do not apply it to a nonlinear or unverified mapping. Observations from a
   different interview, including recovered Goldman 2024 footage, cannot be
   paired with the existing `jensen_a` FLAME targets.
2. Run the existing `tools.media observe` workflow on bounded source windows.
   Its `observations.v1` manifest and `observation.v1` JSONL are the inputs here.
   Reuse the existing hash-bound `motion-review.v1` selection review. Training
   accepts only `speaker_reviewed`, with continuity, source clock and speaker
   association reviewed. Do not convert unreviewed geometry into reviewed data.
3. Write `data/local/jensen-expression/recipe.json` with explicit, disjoint
   half-open frame ranges. Paths are relative to this recipe or absolute local
   paths. The example below is a **recipe template**, not executed training.
   Change source/event IDs, clock offset, and paths to verified values.

```json
{
  "schema": "gaussian-expression-preparation.v1",
  "subject_id": "jensen-huang",
  "bundle_stem": "../../../apps/web/public/models/gaussian-head/jensen_a_c1c3_240k_sh",
  "source_id": "REPLACE_WITH_VERIFIED_SOURCE_ID",
  "source_timeline": "REPLACE_WITH_VERIFIED_SOURCE_CLOCK",
  "event_id": "REPLACE_WITH_VERIFIED_EVENT_ID",
  "sequence_id": "jensen_a",
  "bundle_source_start_seconds": 0,
  "max_alignment_error_ms": 30,
  "gaussian_train_ranges": [[0, 2324]],
  "fit_train_ranges": [[0, 1800]],
  "validation_ranges": [[1800, 2324]],
  "observation_runs": [
    {"manifest": "observations/segment-01/observations-manifest.json", "review": "observations/segment-01/review.json"},
    {"manifest": "observations/segment-02/observations-manifest.json", "review": "observations/segment-02/review.json"}
  ]
}
```

The documented bundle has 3,319 frames, with Gaussian training frames 0–2323.
The recipe above reserves some of those frames for calibration validation and
keeps Gaussian test frames 2324–3318 untouched. This measures within-sequence
interpolation, **not** unseen-event generalization. An event-independent study
needs additional source sequences and their own fixed partitions.

### Generate observations when only the matching source video is available

Use the isolated vision environment pinned by
[`tools/motion/requirements-vision-offline.txt`](../../motion/requirements-vision-offline.txt),
with FFmpeg, FFprobe and a verified local `face_landmarker.task`. Keep this
NumPy 1 environment separate from the NumPy 2 development environment used for
calibration. Replace every uppercase placeholder below with verified local
paths and source metadata. These are command templates, not executed runs.

```bash
/PATH/TO/VISION_ENV/bin/python -m tools.media prepare \
  /PRIVATE/PATH/jensen_a.mp4 \
  data/local/jensen-expression/observations/segment-01 \
  --source-id VERIFIED_SOURCE_ID \
  --source-url https://VERIFIED_PROVENANCE_URL \
  --source-timeline VERIFIED_SOURCE_CLOCK \
  --start 0 --duration 120 \
  --input-source-offset VERIFIED_SOURCE_OFFSET_SECONDS

/PATH/TO/VISION_ENV/bin/python -m tools.media observe \
  data/local/jensen-expression/observations/segment-01/manifest.json \
  --face-model /PRIVATE/PATH/face_landmarker.task \
  --sample-fps 15
```

`prepare` accepts at most 120 seconds per segment; use new output directories
and the correct `--start` for subsequent windows. Keep every window within the
verified Gaussian training portion. The `--sample-fps 15` example matches the
documented bundle only after its frame/time mapping has been checked.

The existing media preparation command requires an audio stream. A silent VHAP
video or image sequence cannot pass through it unchanged; use an existing
verified media manifest or add an explicit video-only preparation path before
running these commands. Do not invent audio provenance to bypass that check.

The observer creates `observations.jsonl` and `observations-manifest.json`.
The expression recipe takes the latter, while `observe` takes the original
media `manifest.json`. Add a hash-matching `motion-review.v1` selection document
after reviewing the actual source clock, continuity and selected speaker; the
observer's default `unreviewed` output is not a training acceptance record.

### Build the paired calibration dataset

```bash
uv run python -m tools.gaussian_head.expression prepare \
  --recipe data/local/jensen-expression/recipe.json \
  --output data/local/jensen-expression/paired.npz
```

Preparation verifies observation hashes, reviewer selections, matching source
clocks, finite weights, and the exact .bin hash. Both calibration training and
calibration validation must remain inside the declared Gaussian training split.
It rejects training/validation overlap, either calibration partition using
Gaussian held-out frames, duplicate frame ranges and
unreviewed speakers. Missing detections and out-of-tolerance alignment are
excluded and counted; they are never filled with invented neutral observations.

## Bounded fitting and recovery

Before fitting real data, verify a private artifact upload/readback and save the
code/recipe checkpoint according to the repository instructions. The commands
below must run in a workspace with NumPy installed (`uv sync --locked --extra dev`).

```bash
uv run python -m tools.gaussian_head.expression fit \
  --dataset data/local/jensen-expression/paired.npz \
  --checkpoint data/local/jensen-expression/fit.npz \
  --output data/local/jensen-expression/jensen_a_c1c3_240k_sh.expression-driver.json \
  --batch-size 256 --max-batches 1

uv run python -m tools.gaussian_head.expression fit \
  --dataset data/local/jensen-expression/paired.npz \
  --checkpoint data/local/jensen-expression/fit.npz \
  --output data/local/jensen-expression/jensen_a_c1c3_240k_sh.expression-driver.json \
  --batch-size 256 --resume
```

The first command exercises an atomic bounded checkpoint. Preserve it privately
before continuing. A checkpoint records the exact dataset digest, processed
example count, fit configuration, normalization and sufficient statistics. It
also retains the previous checkpoint. Resume rejects a changed dataset or
configuration. The final solve is deterministic ridge regression; **optimizer
updates are zero**, because this algorithm does not use an iterative optimizer.
Do not describe a numerical fixture as Jensen training or a fitted mapping as a
new likeness model.

The fit requires actual brow variation and closed/open mouth examples, removes
constant input channels, clips predictions to measured expression support, and
compares disjoint validation error with the training-mean baseline. A mapping
that fails to improve that baseline is not exported for runtime use. The report
is still retained. Numerical improvement does not establish visual naturalness;
review eyebrow direction, blinks, mouth closure and speech sync on the actual
head before calling the result accepted.

## Applying a qualified calibration

Copy the accepted `*.expression-driver.json` alongside the exact head's `.json`
and `.bin` in `apps/web/public/models/gaussian-head/`. The runtime validates the
subject, bundle hash, dimensions and disjoint calibration partitions. Named
face controls come from the shared performance and audio clock; silent mouth
articulation is zero while brows, eyes, cheeks and smiles remain active.

The recorded-motion fallback is explicitly labelled observed replay. It removes
the former global 0.25 idle damping and interpolates only verified training
frames. Without a calibrated mapping, recorded FLAME lip-expression residuals
remain, even though the jaw follows current speech. Unknown bundles with no
declared training split use one static reference expression. This is an honest
fallback, not a semantic expression-training result.
