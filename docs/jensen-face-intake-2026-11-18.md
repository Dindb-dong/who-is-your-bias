# Jensen private-face intake and verification

Status date: **2026-11-18, Asia/Seoul**. For installation and complete preview URLs,
use the [studio quickstart](jensen-studio-quickstart.md).

## Scope and checkpoint

- Repository: `Dindb-dong/who-is-your-bias`.
- Task branch: `feat/jensen-face-verification-20261118`.
- Starting `main` and verified remote base: `973627da0dc668c8544d60fd092da2e88bdb2307`,
  the merged [PR #25](https://github.com/Dindb-dong/who-is-your-bias/pull/25).
- The usage guide was published and read back at `eb4d8e348a120612962c45a5c881f0ab9a9f65b5`.
  The latest verified checkpoint before this record is
  `5026031d360d6d33d0fd8fb4c5bfab6feb6ba14a`, containing both diagnostic scripts
  and their actual-input reports. Each batch was committed, pushed and fetched
  again, with the complete tree and changed-file contents compared.
- This follow-up installs and checks the supplied private face, documents the
  existing expanded gesture candidate and preserves reproducible diagnostics.
  It makes no production-runtime or model-weight change.

## Received artifact

| Field | Verified value |
| --- | --- |
| Artifact | User-supplied `gaussian-head.zip` |
| Role | Private PR #23 identity/appearance bundles and recorded FLAME motion |
| Persistent attachment ID | `libfile_fde3_f-ea48424140a08191983880bdc535925e_a18ece08681081918d81df44f69c4b1f` |
| Bytes | 30,727,476 |
| SHA-256 | `8f33e6b37da63674c43331bd87d899f92b3f6b39e9e111713618c7fc62a24793` |
| Verification | Supplied workspace copy hashed; ZIP CRCs pass; all six member hashes match the PR #23 inventory |
| Installed location | Ignored `apps/web/public/models/gaussian-head/` |
| Contents | `.json`/`.bin` pairs for `jensen_a`, `jensen_a_sh` and `jensen_a_c1c3_240k_sh` |
| Absent from this ZIP | Original source video, named expression observations, reviewed source-clock mapping and an expression-driver sidecar |

The supplied attachment already has the persistent ID above; it was read from
its provided workspace copy and was not uploaded again or independently fetched
from that destination during this follow-up. The original ZIP and extracted
model bytes remain unchanged. Individual hashes and dimensions are recorded in
[the body-attachment report](../reports/gaussian-head/jensen-body-attachment-2026-11-18.json)
and match [the existing private asset inventory](local-only-assets.md).

The separately preserved gesture package remains
`jensen-gesture-candidates-2026-11-17.zip`. Its two runtime files and recorded
training runs are unchanged; see [the expanded gesture receipt](../reports/gesture/jensen-expanded-2026-11-17.json).
Receiving the face ZIP does not require repeating those fits.

## Findings with the actual face

The current c1c3 bundle contains 5,143 vertices, 10,144 faces, 80,014 Gaussians,
100 expression dimensions and 3,319 recorded frames. Its five joints cover the
root, neck, jaw and eyes. Arms, hands and torso come from the separate body rig.

### Expression data is present

[The motion report](../reports/gaussian-head/jensen-bundle-2026-11-18.json) uses
only training frames `[0,2324)` for motion statistics and forward comparisons.
The other 995 frames are excluded from those calculations and from fitting.

| Measurement | Result |
| --- | ---: |
| Expression dimensions with measured variation | 100 / 100 |
| Median coefficient standard deviation | 0.681692 |
| Whole-mesh expression-only RMS deviation from the training mean | 1.208317 mm |
| Same algebraic measure with the former idle multiplier of 0.25 | 0.302079 mm |
| Maximum production/NumPy vertex component discrepancy | 0.000137 mm |
| Maximum sampled Gaussian-center component discrepancy | 0.000806 mm |

Jaw and both eye tracks also vary. This establishes that the bundle has motion
data. The RMS calculation spans the whole mesh before pose correctives and
skinning; it is not a brow-region measurement or a fourfold rendered improvement.
Latent FLAME coefficients cannot be assigned named eyebrow meanings by index.

The production parser and training-slice validator pass. Five uniformly selected
training frames compare every vertex and 128 evenly spaced Gaussian centers
against NumPy; all 80,014 centers are finite in each check. An independent repeat
matched every aggregate metric. Wrong input hashes, an empty training range and
a range reaching the held-out tail are rejected.

### Body attachment and masking pass numerical checks

[The attachment report](../reports/gaussian-head/jensen-body-attachment-2026-11-18.json)
decodes the actual compressed `jensen-source-fitted-v3.glb` and uses the production
rig, FLAME, attachment, masking and performance-controller implementations.

| Check | Result |
| --- | --- |
| Mapped body bones / missing controls | 46 / 0 |
| Head pivot error | `1.4e-17` render-space units |
| Neutral and returned collar-transform error | `2.2e-16` |
| Gaussians retained by the default lower cut | 49,033 / 80,014 |
| Retained Gaussians receiving neck blending | 12,283 |
| Identified hand vertices hidden by the head mask | 0 / 3,586 |
| Five training-frame checks, head turns, nod and designed gesture | Finite geometry and transforms |
| Body cleanup | Exact original geometry/material references restored |
| Current Gaussian binary | Unchanged in memory and on disk |

All three pairs pass full structural validation, including finite stored values;
no held-out frame is selected for a forward pose or motion-quality analysis.
Image/material loading is substituted for CPU inspection. The script performs
no GPU draw, and the body is normalized in render space. These results do not
measure the subject's height, visible collar quality, hand-occlusion pixels,
likeness, expression naturalness or target-machine frame rate.

## Reproduce the bounded checks

Run from the repository root after the private ZIP is installed. The expression
inspector loads TypeScript directly, so use Node **22.18+** (verified here with
Node **24.22.1**) and the development Python dependencies:

```bash
uv sync --locked --extra dev
npm ci

uv run python -m tools.gaussian_head.analysis.inspect_bundle_motion \
  --bundle apps/web/public/models/gaussian-head/jensen_a_c1c3_240k_sh \
  --train-end 2324 \
  --expected-binary-sha256 32d5bc15901a8673901c6234ae7af946d2bfcf2c0aa5e7ed9c816d1c3aff599c \
  --output data/local/jensen-expression-check/motion.json

node tools/gaussian_head/analysis/inspect_body_attachment.mjs \
  --output data/local/jensen-expression-check/body.json
```

Both commands ran successfully against the supplied bytes. Outputs contain
aggregates and hashes; they do not export raw coefficients, vertices or images.
Input-overwrite guards were checked. Source and dependency hashes are preserved
in the reports. These are read-only diagnostics, not training commands.

## Remaining acceptance and next input

The connected browser was unavailable. A subsequent cloud-browser request to
open the private localhost preview was rejected by automatic approval review
because it could expose private face data to that external browser destination.
No alternate browser route or asset upload was used. The temporary local preview
servers were stopped. Actual-face pixels remain unverified in this environment;
the quickstart gives the local comparison procedure.

The runtime and gesture work already passed PR #25's
[synthetic browser and application checks](jensen-performance-2026-11-17.md#validation-state-at-this-checkpoint).
Those results remain synthetic; this follow-up does not relabel them as real-face
visual acceptance. No full application suite was rerun for these documentation
and standalone-diagnostic changes.

Semantic expression fitting still requires the **matching `jensen_a` source
sequence**, supplied as either the original tracking video/image sequence or
existing `observations.jsonl`, `observations-manifest.json` and a hash-bound,
reviewed selection. Include the timestamp corresponding to bundle frame 0,
the verified sampling interval, and any edits or dropped frames. The exporter's
constant `fps=15` is insufficient evidence of that mapping.

Once those inputs are available, follow [the paired preparation and fitting
workflow](../tools/gaussian_head/expression/README.md). Review brow, blink and
closed/open-mouth coverage before fitting; preserve a private prepared-data
checkpoint; keep both calibration partitions inside Gaussian training frames;
and verify a bounded fit/save/resume cycle before applying a qualified sidecar.
A silent video needs the documented video-only preparation prerequisite.
No real semantic expression weights were fitted in this follow-up.
