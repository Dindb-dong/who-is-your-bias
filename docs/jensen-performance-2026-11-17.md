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

## Verified implementation checkpoint

The follow-up is preserved in [draft PR #25](https://github.com/Dindb-dong/who-is-your-bias/pull/25).
The last verified remote checkpoint before this status update is
`5c783c12162c08b69cf0b34847f43a0dc77b4270`; every preceding implementation batch was
committed separately, published, fetched again, and verified against its complete Git tree.

Implemented:

- Shared-camera/depth Gaussian face and clothed GLB body, with original-head masking that preserves raised hands.
- Collar-to-neck transform blending, shared speech/body clocks, eyes-first attention, stop/reduced-motion handling, restoration on missing assets, and resource cleanup.
- Exact binary-hash binding, structural bundle validation and explicit expression replay/static fallbacks. The documented current SH-baked face remains the default.
- Supervised semantic-expression preparation and resumable calibration, with both fitting and calibration validation kept out of the Gaussian held-out frames. This tooling is tested; it has not fitted the user's absent face data.
- Real gesture fitting and optional audio-clock runtime application. The UI requires an explicit experimental geometry-preview opt-in. A Jensen model is synchronously excluded for another persona.
- The legacy GLB audition protocol refuses the new composite/learned modes because it does not record their asset identities; it does not produce misleading comparison receipts.

## Executed gesture experiment and preserved outputs

See [methods and commands](jensen-gesture-training.md) and
[the measured receipt](../reports/gesture/jensen-2026-11-17.json).
Six hash-verified existing clips supplied 580 observed frames. Four groups completed
60 deterministic alternating ridge updates each. The model is 141,273 bytes; its
SHA-256 is `f5ac3321fdf34525f1eaff21286e4aca34dfe252e9505a0eb2c0549d74a0f324`.
Actual saved checkpoints reloaded and re-exported byte-identical model bytes.

The private `jensen-performance-training-2026-11-17.zip` preserves prepared data,
current/previous checkpoints, the runtime model and measured results. Its receipt
and per-member hashes are recorded in the measured report. Independent materialization
verified all archive bytes, ZIP CRCs, and the runtime model digest.

Only the left-finger group beat the neutral reconstruction baseline on WIRED.
The arms have five training windows and one validation window each. The model is
therefore retained as an optional trained geometry preview, without default promotion,
naturalness claims, learned torso motion or audio/text-conditioned generation claims.

## Bounded source expansion

The earlier private Goldman media and reference/model bundles were recovered and
hash-verified; see [the recovery receipt](../reports/gesture/jensen-expansion-recovery-2026-11-17.json).
This is recovered footage, not a claim of new event acquisition.
The [frozen expansion recipe](../data/motion/jensen-gesture-expansion.v1.json) selects
38 seconds with visible arm coverage, excludes shot boundaries, and fixes the crop,
source clocks, review status, event partition and unchanged fitting settings before
new extraction/evaluation. Derived data remains private through an explicitly
bounded `--asset-root`. The expansion is in progress, not yet an improved-model result.

## Validation state at this checkpoint

- Local full web suite: 261 passed before the subsequent focused attention/loader/seam additions.
- Local relevant Python suite: 40 passed before the subsequent holdout/private-root additions.
- Subsequent focused tests cover those additions; final counts will be recorded after integration settles.
- Web typecheck and production build passed.
- Real-WebGL browser fixtures cover pixel displacement/occlusion, production audio/body playback,
  failure recovery, remounting, learned-preview gates and persona isolation. All fixtures are synthetic.
- Local Chromium was blocked before page launch by the environment's Unix-socket restriction.
  Browser execution is being verified through the repository's CI; no local pixel pass is claimed.
- Actual Jensen face/neck appearance, expression naturalness, target-machine performance and real
  semantic expression calibration remain unverified until the private face bundle is supplied.
