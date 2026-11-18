# Jensen gesture data expansion: executed result

The 2026-11-17 follow-up recovered the previously saved Goldman source video and
pinned vision models, extracted **38 additional seconds / 380 frames**, and fitted
a separate gesture candidate. Each arm now has **14 accepted training windows**,
up from five. The initial candidate remains available and the expanded candidate
requires the same explicit geometry-preview opt-in.

The source remains unreviewed monocular geometry. This work establishes actual
source recovery, additional observations, fitting, checkpoint resumption and
runtime compatibility. Natural conversational movement has not passed human or
fresh-event evaluation.

## Recovered inputs and frozen extraction

The [recovery receipt](../reports/gesture/jensen-expansion-recovery-2026-11-17.json)
records the exact archive IDs, byte counts, SHA-256 digests and independently
verified download/readback results. The recovered video is the first 180 seconds
of the Goldman recording, 1278×720 at approximately 59.94 fps. The recovered
MediaPipe face, lite-pose and hand task hashes match the existing measurement
pipeline. No full recording or new model download was needed.

Before new extraction, the [expansion recipe](../data/motion/jensen-gesture-expansion.v1.json)
froze the source intervals, one fixed crop, selection rules, confidence threshold,
rotation-reference rule, event partition and fitting settings. Contact-sheet
inspection and source cut detection placed all four intervals inside continuous
shots and outside the existing three Goldman clips.

| Source interval, seconds | Frames | Complete left arm | Complete right arm | Chest observed |
| --- | ---: | ---: | ---: | ---: |
| 100–114 | 140 | 80 | 81 | 0 |
| 116–124, fixed right-side crop | 80 | 71 | 60 | 78 |
| 136–146 | 100 | 72 | 63 | 0 |
| 172–178 | 60 | 46 | 60 | 0 |

A complete arm means upper arm, forearm and hand controls are all observed in
that frame. These counts precede the stricter temporal-window acceptance check.
The wide-shot crop uses input pixels `[760, 20, 510, 560]`, without mirroring; it
retains the visible upper body while excluding the other participant and audience.
The other three views do not contain sufficiently visible hips to recover chest
orientation. The current model still trains arms and fingers only: there is no
matching torso validation series.

Extraction ran with the unchanged MediaPipe 0.10.21 implementation in an isolated
Python environment, on two CPU cores at 10 observations per second. Ambiguous or
multiple-person detections are omitted by the existing explicit preview selector.
Hands are associated through selected-pose wrist proximity; handedness labels do
not establish person identity. Visibility remains 0.65 and missing joints remain
missing.

The new [preview exporter](../tools/motion/preview_export.py) chooses the reference
frame with the most observable arm controls, then finger controls, then the
earliest timestamp. This preserves channels that were genuinely observed; it
does not declare an anatomical neutral pose. All speaker, continuity and source
clock review flags remain false. No emotion or gesture-meaning labels were added.

## Prepared dataset and executed fitting

The [expanded source manifest](../data/motion/jensen-gesture-expanded-input.v1.json)
pins the four new private clips and all six unchanged original clips. Preparation
accepts an explicit private `--asset-root`, confines members to that root, and
checks every content hash. The prepared dataset contains ten clips / 960 frames.
The added data comes from the existing Goldman training event; event diversity
has not increased.

| Group | Initial training windows | Expanded training windows | WIRED validation windows |
| --- | ---: | ---: | ---: |
| Left arm | 5 | 14 | 1 |
| Right arm | 5 | 14 | 1 |
| Left fingers | 17 | 24 | 5 |
| Right fingers | 13 | 24 | 5 |

There are 12 shared source-window IDs across all four groups, compared with five
initially. The runtime uses a shared window ID to coordinate both arms and hands.

The [expanded fitting plan](../data/motion/jensen-gesture-training-expanded.v1.json)
retains 2-second windows, 20 samples, 60% minimum coverage, rank two, ridge 0.01
and 60 alternating updates. All four groups actually completed 60 updates.
Reloading their saved checkpoints and exporting again produced identical bytes.
The previous 50-update checkpoints are also retained. No hyperparameter or window
tuning followed the new validation result.

The expanded model is **144,793 bytes**, SHA-256
`ede0a38935d22aa980c622214a6d8a77a2c80b5241856f6d0fcbf46b757ec0b4`.
The initial model remains byte-identical to its original receipt.

## Reconstruction comparison and limits

The [measured report](../reports/gesture/jensen-expanded-2026-11-17.json) contains
the complete training and validation results. The WIRED window values and masks
are identical to the first fit. That event was already exposed in earlier
repository work and in the first fit, so it is not fresh independent evaluation.

The expanded left-arm fit supports six coordinates that the initial model could
not support. To avoid comparing different denominators, the table below evaluates
both models on the intersection of their supported coordinates. Evaluation latent
codes are inferred from each validation window. The metric measures representation
reconstruction, not causal audio prediction or perceived naturalness.

| Group | Neutral RMSE, rad | Initial basis RMSE, rad | Expanded basis RMSE, rad |
| --- | ---: | ---: | ---: |
| Left arm, 174 coordinates | 0.090916 | 0.111606 | 0.067559 |
| Right arm, 180 coordinates | 0.283679 | 0.307779 | 0.276436 |
| Left fingers, 4,230 coordinates | 0.364177 | 0.349228 | 0.349874 |
| Right fingers, 4,140 coordinates | 0.307825 | 0.310778 | 0.303785 |

All expanded groups are below the neutral baseline on this reused comparison.
Left-finger reconstruction is slightly worse than the initial candidate. The arm
validation population remains one two-second window per side, and the source
geometry has not been manually qualified. These results justify retaining an
inspectable candidate, not accepting a learned model of Jensen's behavior.

The actual expanded model also passed the application parser and deterministic
sampler: 480 active and 120 rest samples across 30 seconds, up to 36 channels,
quaternion norm error at most `2.22e-16`, and the configured 0.8-radian maximum.
This checks numeric runtime compatibility; it is not a visual naturalness score.

## Install and reproduce

The private gesture package contains both model files, both runs' prepared
datasets and checkpoints, the private expanded clips and review receipts, frozen
recipes, comparisons and a checksum manifest. Its README describes installation.
Place the two files from `models/gesture/` in
`apps/web/public/models/gesture/` before building the web app.

- Initial candidate: `/?head=gaussian&gesture=learned`
- Expanded candidate: `/?head=gaussian&gesture=learned&gestureBundle=jensen-gesture-basis-expanded`

Both routes preserve the existing learned-geometry opt-in, subject binding and
reduced-motion behavior. The existing Gaussian head bundle is a separate local
asset. The expanded candidate does not replace the initial default file.

With the corresponding repository branch and the extracted private package:

```bash
uv run python -m tools.motion.gesture_training prepare \
  data/motion/jensen-gesture-training-expanded.v1.json \
  data/local/jensen-gesture-reproduction \
  --asset-root /path/to/extracted-package/assets

uv run python -m tools.motion.gesture_training train \
  data/local/jensen-gesture-reproduction \
  --output data/local/jensen-gesture-reproduction/model.json
```

For checkpoint re-export, pass the package's `runs/expanded` directory to `train`
with `--resume` and a new output filename. The supplied dataset and fitting
configuration are hash-bound; changing source bytes requires a new run.

The next data requirement is continuous, uncropped views with visible shoulders
(pose landmarks 11/12), elbows (13/14), wrists (15/16), and finger anchors (19/20),
plus the selected 21-point hand tracks. Torso training additionally needs hips
(23/24) and corresponding validation coverage. Another event must be reserved
before fitting or inspection if the goal is a fresh generalization result.
