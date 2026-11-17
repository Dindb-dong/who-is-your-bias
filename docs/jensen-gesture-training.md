# Jensen upper-body gesture fitting

This workflow fits a compact temporal motion basis from the **existing measured
Jensen source geometry**. It adds a learnable, reproducible body-motion artifact
without changing the Gaussian face weights. It is a bounded CPU experiment, not
an audio-conditioned gesture network or a qualified model of Jensen's behavior.

The source observations remain `geometry_preview`. Fitting their trajectories
does not establish speaker association, gesture meaning, or human motion quality.
The application must require the geometry-preview opt-in before using the model.

## Why the Gaussian preview had no gestures

At the PR #23 baseline, the `head=gaussian` branch of `App.tsx` rendered only
`GaussianHeadStage`. It passed speech activity and amplitude, while bypassing the
body renderer's timeline, measured performance clip, framing and attention inputs.
The head bundle contains FLAME face/neck geometry, not a torso/arm/hand rig.
Adding more facial training cannot create those missing joints.

The existing motion catalog also has no automatic Jensen clip selection: its six
contexts are unset and its source geometry has not passed speaker-review gates.
Those gates are preserved. Learned motion below has its own explicit preview
label; it is not a promotion of the catalog or a source-context match.

## Verified preparation data

The frozen [training plan](../data/motion/jensen-gesture-training.v1.json) pins the
hash of [the existing source manifest](../data/motion/jensen-multievent-input.v1.json).
Preparation verifies every source clip's bytes, identity, quaternion conventions,
review record and event assignment before producing any training window.

| Source event | Input clips | Observed frames | Role in this fit |
| --- | ---: | ---: | --- |
| Goldman Sachs 2024 | 3 | 240 | Training |
| Stripe Sessions 2024 | 2 | 240 | Training, primarily fingers |
| WIRED Big Interview 2024 | 1 | 100 | Validation only |

There are 580 observed source frames across approximately 58 seconds. The 2-second
windows are correlated observations, not independent people or independent
events. All three events were already exposed in earlier repository development;
the WIRED partition is disjoint for this new fit, not globally unseen material.

| Group | Accepted training windows | Accepted validation windows | Excluded windows |
| --- | ---: | ---: | ---: |
| Left arm: upper arm, forearm, hand | 5 | 1 | 23 |
| Right arm: upper arm, forearm, hand | 5 | 1 | 23 |
| Left fingers: 15 joints | 17 | 5 | 7 |
| Right fingers: 15 joints | 13 | 5 | 11 |

These counts come from the frozen plan's actual preparation. The preparation
receipt records per-window exclusion reasons. Cropped arms in the Stripe clips
are not converted into neutral observations. There is no sufficiently observed
chest/spine series in this input, so this model does not claim learned torso motion.

## Preparation and fitting

The implementation is [gesture_training.py](../tools/motion/gesture_training.py).
Its normal requirements are the project development environment with NumPy;
training performs no network requests and needs neither CUDA nor licensed model
downloads.

1. Sample each 2-second window at 20 fixed timestamps. Interpolate only adjacent,
   present quaternions within the source clip's declared maximum gap.
2. Reject a group if any joint has under 60% coverage or its first observation
   occurs after that permitted onset gap. Keep remaining missing coordinates
   masked throughout fitting and evaluation.
3. Rebase every joint on its first observed local rotation. Convert relative
   rotations to shortest axis-angle vectors. Quaternion sign changes do not
   become artificial motion.
4. Fit rank-two temporal factors with alternating ridge least squares. Only
   observed training coordinates contribute to any update. Per-coordinate means,
   basis vectors, latent examples, support counts and amplitude limits use the
   training events exclusively.
5. Evaluate representation reconstruction separately on WIRED windows. Their
   latent codes are inferred using the frozen basis; no held-out observation
   changes the means, factors, runtime examples or limits.

The validation numbers therefore measure **masked reconstruction**, not gesture
prediction. They should be compared with both neutral and training-mean baselines.
They do not measure audio synchronization, gesture semantics, perceptual quality,
or similarity of a novel conversational performance.

## Executed result: 2026-11-17

The [measurement receipt](../reports/gesture/jensen-2026-11-17.json) records the
actual run: four groups completed 60 alternating updates each on CPU. The exported
model is 141,273 bytes, SHA-256
`f5ac3321fdf34525f1eaff21286e4aca34dfe252e9505a0eb2c0549d74a0f324`.
Loading all four saved checkpoints and exporting again produced identical bytes.

| WIRED validation group | Neutral coordinate RMSE, rad | Training-mean coordinate RMSE, rad | Fitted-basis coordinate RMSE, rad |
| --- | ---: | ---: | ---: |
| Left arm, 1 window | 0.09092 | 0.12478 | 0.11161 |
| Right arm, 1 window | 0.28368 | 0.31893 | 0.30778 |
| Left fingers, 5 windows | 0.36418 | 0.35817 | 0.34923 |
| Right fingers, 5 windows | 0.30782 | 0.32094 | 0.31078 |

Only the left-finger group beats the neutral baseline. All groups improve on the
training-mean baseline, which alone is insufficient evidence for promotion.
RMSE is over the observed XYZ axis-angle coordinates; it is not a quaternion
geodesic angle, pixel error or perceptual score.
**The artifact remains an opt-in trained geometry preview.** It is not the default
gesture policy, and this run does not establish generalization or naturalness.
No further hyperparameter tuning was performed after inspecting validation.

The remaining limitation is concrete: both arm groups have five accepted windows
from one training event and only one usable validation window. More continuous,
visible arm data is needed before a stronger cross-event conclusion is possible.

## Reproduction and resumption

Read [AGENTS.md](../AGENTS.md) first. Establish and verify private durable artifact
storage before an expensive run. All paths below are ignored local paths, and are
not a substitute for preserving the resulting bytes privately.

```bash
uv sync --locked --extra dev
uv run python -m tools.motion.gesture_training prepare \
  data/motion/jensen-gesture-training.v1.json \
  data/local/jensen-performance-2026-11-17/gesture

uv run python -m tools.motion.gesture_training train \
  data/local/jensen-performance-2026-11-17/gesture \
  --output apps/web/public/models/gesture/jensen-gesture-basis.json
```

To resume, repeat the training command with `--resume`. It verifies the dataset,
fitting configuration and existing progress counters. Each group writes an atomic
checkpoint every ten iterations and on completion; the preceding checkpoint is
retained. The checkpoint contains factors, latent state, completed iteration,
loss history and exact input/configuration binding. Updates are deterministic;
there is no stochastic optimizer, scheduler or unrecorded random state.

Before full training, the test suite verifies that a two-update save/load followed
by three updates equals an uninterrupted five-update run, and that changed input
bytes cannot resume a checkpoint. The prepared dataset is likewise hash-bound.

```bash
uv run pytest tests/motion/test_gesture_training.py -q
```

## Runtime artifact contract

The ignored output is `gesture-basis.v1`, with `persona_id: jensen-huang` and
`review_status: geometry_preview`. It contains four groups, not a FLAME expression
basis and not an `avatar-clip.v1` recording.

| Field | Meaning |
| --- | --- |
| `sample_count`, `duration_ms` | Fixed temporal grid, currently 20 samples over 2,000 ms |
| `groups[name].channels` | Ordered rig bone controls |
| `mean` | Flattened mean trajectory, ordered time → channel → XYZ |
| `basis` | Rank-by-coordinate temporal factor matrix |
| `latent_examples` | Fitted codes for training windows only |
| `latent_ids` | Exact source clip/window IDs corresponding to those codes |
| `support_counts` | Training observation counts per coordinate |
| `max_angle_rad` | Training-derived amplitude limit capped by an engineering bound |
| `iterations_completed` | Actual completed alternating-update count |

Reconstruct `mean + latent × basis`, interpolate time, bound each joint's vector
length, and exponentiate axis-angle to an XYZW quaternion. Apply a smooth onset/
offset envelope and return to neutral when speech stops or the preview is
disabled. Missing/unsupported groups must remain absent. Respect reduced-motion
preferences. These body rotations must not replace speech timing or the Gaussian
face identity.

Choose the same `latent_id` across left/right arm and finger groups to preserve
observed bilateral coordination. All five accepted arm windows are present in
both finger groups. Independently selecting unrelated left/right gestures is not
equivalent to reproducing their observed coordination.

## What is still required for a larger gesture model

The pipeline can be expanded using the existing local-video measurement workflow
in [motion-library.md](motion-library.md), followed by a newly frozen source
manifest and event partition. Collect continuous upper-body views with visible
shoulders, elbows, wrists and fingers, and preserve exact source/video/audio clock
mapping. Keep camera cuts and occlusions explicit. New event families must remain
outside training for a meaningful future validation set.

Training a causal audio/text-conditioned generator additionally requires aligned
speech features, more observed motion, a declared conditioning task, a held-out
evaluation protocol and an accessible compute environment. A fitted geometry
basis or improved head rendering does not establish those missing capabilities.
