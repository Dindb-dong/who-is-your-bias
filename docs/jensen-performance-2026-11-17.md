# Jensen Gaussian face, expression and upper-body performance

## Task and checkpoint

- Repository: `Dindb-dong/who-is-your-bias`.
- Task branch: `feat/jensen-gesture-expression-20261117`.
- Starting `main` / verified remote head: `cdf3c522e02a788f143fff4cabb375d172d46bfb` (PR #23).
- Requested outcome: preserve the current Gaussian face, prepare and train gesture/expression drivers, and connect upper-body, hand and facial motion to studio playback.
- Status: implementation in progress. This is an execution record, not an acceptance report.

## Verified diagnosis

1. `App.tsx` replaces `AvatarStage` with `GaussianHeadStage` when `?head=gaussian` is selected. The replacement accepts only speaking/amplitude and renders a head. It bypasses the existing body rig, performance clips, behavior plan, audio timing, attention and framing.
2. `GaussianHeadStage.tsx` replays integer tracked frames against a render clock. It multiplies every FLAME expression coefficient by `0.25 + 0.75 * speakingMix`, attenuating upper-face motion at rest. It does not receive named expression cues.
3. FLAME PCA coefficients are not named ARKit expression controls. Mapping a brow channel to an arbitrary coefficient would be incorrect; a calibrated mapping must use aligned observations and retain the existing identity/appearance assets.
4. The six committed Jensen motion clips contain actual extracted geometry, but retain `geometry_preview` review status. Their contexts are unset. They must not silently become reviewed or semantically labelled Jensen habits.
5. PR #23 lists the current face as `jensen_a_c1c3_240k_sh`; the head-only component still defaults to `jensen_a`.

## Work packages

- Integrate Gaussian splats and the existing clothed body in the same camera/render path, retain hand occlusion, shared playback and attention, and dispose resources correctly.
- Add a tested expression controller and reproducible, split-aware CPU calibration tooling. Restore recorded expression strength when no calibrated map is available and label that fallback accurately.
- Prepare masked gesture windows from the real committed observations, fit a compact temporal model with event-separated validation, and integrate its optional local artifact without bypassing the geometry-preview review gate.
- Preserve small implementation checkpoints remotely, then run focused Python/web/browser checks and record measured outcomes separately from missing-asset visual checks.

## Available inputs and current limitations

- Source code, generated clothed GLB and bounded extracted motion clips are present at the starting revision.
- The promised Gaussian ZIP has not yet appeared among the accessible attachments. Licensed FLAME/SMPL-X files and cluster source/checkpoint paths are not present in this environment.
- This host has nine logical CPUs, about 10 GB RAM and no visible CUDA GPU. Small CPU fits are feasible; a full GaussianAvatars appearance retrain is not being represented as feasible here.
- Existing face `.json`/`.bin` files will remain ignored and unchanged. Private derived artifacts must have a verified durable destination before expensive work.

## Acceptance and handoff

Acceptance requires source/dataset hashes, actual executed fitting steps and saved outputs; event-separated metrics; body and face on the same playback clock; stop/seek/reduced-motion/attention behavior; desktop/mobile framing; and rendered review with the real face bundle. Synthetic fixtures validate contracts and math only, never likeness or learned Jensen quality.

This document will receive subsequent verified checkpoints, exact commands, artifact receipts, test results and remaining gates. Do not merge solely because this checkpoint exists.
