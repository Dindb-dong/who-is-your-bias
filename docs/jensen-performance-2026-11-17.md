# Jensen Gaussian face, expression and upper-body performance

## Task and checkpoint

- Repository: `Dindb-dong/who-is-your-bias`.
- Task branch: `feat/jensen-gesture-expression-20261117`.
- Starting `main` / verified remote head: `cdf3c522e02a788f143fff4cabb375d172d46bfb` (PR #23).
- Requested outcome: preserve the current Gaussian face, prepare and train gesture/expression drivers, and connect upper-body, hand and facial motion to studio playback.
- Status: code integrated and two gesture candidates fitted and privately preserved. Real-face visual acceptance and semantic expression fitting remain blocked on the absent paired face inputs.

## Verified diagnosis at the PR #23 baseline

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
- The promised Gaussian ZIP has not yet appeared among the accessible attachments. Semantic calibration also requires the matching `jensen_a` source sequence or its reviewed, clock-aligned face observations. Recovered Goldman gesture footage cannot supply paired labels for unrelated FLAME frames.
- This host has nine logical CPUs, about 10 GB RAM and no visible CUDA GPU. Small CPU fits are feasible; a full GaussianAvatars appearance retrain is not being represented as feasible here.
- Existing face `.json`/`.bin` files will remain ignored and unchanged. Private derived artifacts must have a verified durable destination before expensive work.

## Acceptance and handoff

Acceptance requires source/dataset hashes, actual executed fitting steps and saved outputs; event-separated metrics; body and face on the same playback clock; stop/seek/reduced-motion/attention behavior; desktop/mobile framing; and rendered review with the real face bundle. Synthetic fixtures validate contracts and math only, never likeness or learned Jensen quality.

Detailed commands, artifact receipts and remaining gates follow. Do not merge solely because a checkpoint or a synthetic test exists.

## Verified implementation checkpoint

The follow-up is preserved in [draft PR #25](https://github.com/Dindb-dong/who-is-your-bias/pull/25).
The last verified remote checkpoint before this status update is
`bc3f215015c302a4d353f2e75b311aeb0a23601f`; every preceding implementation batch was
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

## Completed bounded source expansion

The earlier private Goldman media and reference/model bundles were recovered and
hash-verified; see [the recovery receipt](../reports/gesture/jensen-expansion-recovery-2026-11-17.json).
This is recovered footage, not a claim of new event acquisition.
The [frozen expansion recipe](../data/motion/jensen-gesture-expansion.v1.json) selects
38 seconds with visible arm coverage, excludes shot boundaries, and fixes the crop,
source clocks, review status, event partition and unchanged fitting settings before
new extraction/evaluation. All four raw observation batches and the complete
prepared dataset were preserved and independently read back before fitting.
Derived data remains private through an explicitly bounded `--asset-root`.

The expanded dataset contains ten clips / 960 frames. Training support increased
from 5 to 14 windows per arm and to 24 per finger group, with 12 coordinated window
IDs shared across all four groups. A separate candidate completed 60 updates per
group and re-exported identical bytes after checkpoint reload. The initial model
remains unchanged. The expanded model is 144,793 bytes, SHA-256
`ede0a38935d22aa980c622214a6d8a77a2c80b5241856f6d0fcbf46b757ec0b4`.

On identical WIRED windows and common supported coordinates, both arms and right
fingers improve over the initial fit; left fingers are slightly worse. All four
expanded groups beat the neutral reconstruction baseline. This is reused,
small-sample reconstruction evidence, not prediction, a fresh event evaluation
or visual acceptance. See [the executed expansion report](jensen-gesture-expansion.md)
and [measured results](../reports/gesture/jensen-expanded-2026-11-17.json).

The final private `jensen-gesture-candidates-2026-11-17.zip` contains both models,
both datasets and checkpoint sets, private clips, recipes, measured comparisons,
runtime checks and installation instructions. Its 2,871,526 bytes were separately
materialized and verified byte-identical, with all ZIP CRCs and both model hashes
checked. Persistent IDs and member hashes are in the expanded report.

## Applying the preserved candidates

Place the two files in the package's `models/gesture/` directory into
`apps/web/public/models/gesture/` before building. Keep the PR #23 face `.json`
and `.bin` under their existing ignored Gaussian-head directory.

- Combined face and designed body movement: `/?head=gaussian`.
- Initial learned candidate: `/?head=gaussian&gesture=learned`.
- Expanded learned candidate: `/?head=gaussian&gesture=learned&gestureBundle=jensen-gesture-basis-expanded`.

Both learned routes require the explicit geometry-preview mode. The optional
basename is path-bounded and the model remains Jensen-bound. Neither fit becomes
the default motion policy. Source observations stay unreviewed; there is no
trained audio/text conditioning or torso model.

## Validation state at this checkpoint

- Full web suite: 266 passed. Relevant Python suite: 181 passed.
- Web typecheck and production build passed locally and on CI at `28e9349`.
- Real-WebGL browser fixtures cover pixel displacement/occlusion, production audio/body playback,
  failure recovery, remounting, learned-preview gates and persona isolation. All fixtures are synthetic.
- Local Chromium was blocked before page launch by the environment's Unix-socket restriction.
  Browser execution uses repository CI; no local pixel pass is claimed.
- CI at `28e9349` passed seven of eight Gaussian/performance scenarios, including
  desktop/mobile real-WebGL pixels, both learned-playback scenarios, missing-asset
  recovery, and mobile unmount/remount. The sole failure used mobile menu labels
  in the desktop remount test; `91fecd8` corrects those selectors without removing
  assertions. The final run and its exact revision are recorded in PR #25's
  validation section and linked checks.
- Actual Jensen face/neck appearance, expression naturalness, target-machine performance and real
  semantic expression calibration remain unverified until the private bundle and
  its correctly paired source sequence are supplied.

## Next input and resumption boundary

Supply `jensen_a_c1c3_240k_sh.json` and `.bin` from the promised Gaussian ZIP.
For actual semantic expression fitting, also supply the original `jensen_a`
tracking video with its frame order/time mapping, or its observation JSONL,
`observations-manifest.json` and matching reviewed selection. The bundle's fixed
15-FPS header alone is not proof of original-video alignment.

First verify the recorded bundle hashes and inspect the actual combined face,
neck seam, hands and source-expression variation. Then follow
[the paired calibration workflow](../tools/gaussian_head/expression/README.md),
keeping both calibration partitions inside Gaussian training frames. No real
Jensen expression weights have been produced by the synthetic tests in this PR.
Do not repeat the completed gesture fits or overwrite either preserved candidate
as part of this input recovery.
