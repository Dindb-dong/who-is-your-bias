# Avatar qualification and same-audio audition

This implements the asset inspection, conversion, mapping, and comparison workflow
in WP2 of the [2026 persona roadmap](2026-persona-roadmap.md). It does **not** deliver
a newly acquired or qualified Musk likeness. The inspected file is the existing
authored research human, retained as the control asset. Human appearance, articulation,
and naturalness ratings remain ungraded.

## Observed acquisition status

On 2026-11-11, the three free candidates in the
[shortlist](research/musk-3d-avatar-shortlist.md) returned HTTP 403 when their
listing pages were requested. The official
[Sketchfab download procedure](https://sketchfab.com/developers/download-api/downloading-models)
requires authenticated account access before it issues a temporary archive link.
This execution had no connected Sketchfab credentials. No production file was
obtained from those listings. Paid candidates were not purchased and no seller
was contacted.

The [acquisition record](../data/avatar-auditions/acquisition-attempts.v1.json)
preserves exact candidate URLs and blockers. It does not turn indexed listing
metadata into inspected rig capabilities. A downloaded, licensed local package
can now proceed through the executable workflow below. A title, thumbnail,
nominal face target, or successful import cannot qualify likeness on its own.

## Actual control-asset result

[Qualification report](../data/avatar-auditions/research-human-v2.qualification.json)
was generated from the exact bundled GLB by decoding its Meshopt geometry with
the existing pinned glTF-Transform and meshoptimizer dependencies. It binds the
asset, inspector code, dependency lockfile, provenance record and chosen audio.
The report envelope's `reportSha256` covers canonical JSON of its entire report.

| Property | Actual measurement |
| --- | --- |
| File | `apps/web/public/models/research-human-v2.glb` |
| Bytes | 2,752,352 |
| SHA-256 | `aee550c9cb5b20c6afccfb038c42c2eb8d7aab2c8fb74ca977aefed56f633519` |
| Meshes / skin bindings | 7 / 7 |
| Unique skeleton joint names | 163 |
| Vertices / triangles | 28,933 / 50,178 |
| Materials / embedded textures | 7 / 7 |
| Named targets / targets with nonzero position displacement | 67 / 66 |
| Canonical face controls with position displacement | 52 |
| Native visemes | 15, including deliberately neutral `sil` |
| Embedded animation clips | 0 |
| Selection decision | `inspection_only` |
| Musk 2026 likeness | Ungraded; no likeness claim |

The [provenance record](../data/avatar-auditions/research-human-v2.provenance.json)
points to the existing CC0 graphical-input manifest and retained license text.
An operator-supplied permission record is not an automatic legal determination.
The zero-animation result is expected: the application supplies motion at runtime.

The comparison audio is the existing generated narration, SHA-256
`f3e3fb41fca380073cb9d4fdb0209e4062b1e5d60da988e2f58d419c09635e0a`.
This records which bytes to reuse; the offline inspection does not claim to have
rendered, listened to, or rated them with a new candidate.

## 1. Register and convert a local package

Keep source packages and licensed commercial files under `data/local/` or another
untracked location. Create a small provenance JSON with the exact creator, SKU or
source URL, acquisition date when known, selected license and its retained-text
SHA-256, and one declared distribution path: `bundled`, `local_only`,
`separate_license`, or `unknown`. Use `status: "unverified"` until the record is
complete. The bundled proxy's provenance is a schema example, not a license for
another vendor's model.

For an FBX, Blend, glTF, or GLB package:

```bash
uv run python tools/avatar_convert.py \
  data/local/avatar-candidate/source.fbx \
  data/local/avatar-candidate/candidate.glb \
  --report data/local/avatar-candidate/conversion.json
```

Blender must be installed; `--blender` accepts its executable path. The recorded
local integration run used Blender 4.3.2. The converter:

- Uses a clean background process with Python auto-execution disabled.
- Exports the active scene to embedded GLB with skins, morphs and animations.
- Keeps armature rest-pose export enabled and avoids applying modifiers that can
  discard shape keys.
- Records the source/output hashes, imported image hashes, actual export settings,
  Blender version, geometry/skin counts and target-name retention.
- Hashes external glTF package buffers and images before and after conversion;
  remote dependencies are not downloaded.
- Rejects lost shape-key names, a lost source armature/skin, files over 64 MiB,
  external output resources, and existing output/report paths.

The source file is preserved. A material export is not proof that a DCC shader
matches browser PBR appearance. The [Blender glTF documentation](https://docs.blender.org/manual/en/latest/addons/scene_gltf2.html)
describes the supported shader, skin and shape-key export model. Camera and light
objects are omitted so audition framing and illumination come from the studio.

## 2. Inspect actual GLB contents

From the repository root:

```bash
npm ci --prefix tools/avatar --ignore-scripts

node tools/avatar_audition.mjs inspect \
  data/local/avatar-candidate/candidate.glb \
  data/local/avatar-candidate/qualification.json \
  --provenance data/local/avatar-candidate/provenance.json \
  --audio apps/web/public/presentation/narration.wav
```

The inspector supports uncompressed and Meshopt files. It rejects nonembedded
buffers/images before decoding and identifies Draco/KTX2 requirements that the
application does not bundle. No source image or geometry bytes are copied into
the JSON report.

For each mesh/primitive it records vertices, primitive mode, triangle counts,
materials, nonfinite positions, skin-weight defects, morph names, displaced vertex
counts and maximum displacement in mesh-local units. Sparse and compressed
accessors are decoded by the pinned library. It also inventories node rest
transforms, skin joints, embedded texture sizes/hashes, PBR material settings and
animation channels. A missing or zero position target remains distinguishable from
a deforming target; the silence viseme may deliberately be neutral.

Canonical control matches are an initial inspection. External files with custom
names still need explicit calibration in the actual browser. Geometry measurements
do not establish correct lip closure, teeth placement, gaze, expression anatomy,
identity or perceptual quality. Every generated qualification keeps those review
fields ungraded and its decision at `inspection_only`.

## 3. Calibrate and audition in the studio

1. Select the local GLB in **Studio settings**, then open **Research lab**.
2. Open **Rig inspection & calibration**. Inspect individual face, jaw, gaze,
   eyelid and native-viseme controls. Import a mapping for custom names and axes.
3. Open **Asset audition** and export the asset inspection. It includes the exact
   loaded file digest, actual decoded displacements, resolved per-mesh targets,
   calibrated ranges, bone bases/rest transforms, materials and load time.
4. In **Same-audio avatar audition**, load the bundled narration or import a local
   RIFF/WAVE file up to 16 MiB. Optional `speech-timing.v1` imports must match the
   chosen audio's SHA-256. Without timing, articulation is explicitly based on
   measured audio energy.
5. Choose the camera framing, then **Play & record audition**. This starts the
   existing cancellable Web Audio transport using the selected local bytes. It
   does not call the synthesis service. The first playback frames trigger the
   recorder, reset the authored idle phase and fix the camera after clearing
   residual drag damping.
6. Stop the sample or let playback end, then **Export audition** from the rig panel.
   Repeat with another candidate using the same audio, timing, viewport and settings.

The renderer already selected native visemes, face-channel mouth shapes, or
jaw-bone opening as alternatives. That policy is preserved. An incomplete native
viseme blend retains the established face-channel fallback. Reports keep configured
mapping separate from actual target bindings and measured displacement.

The recorded frame interval is a browser observation, not a GPU render-duration
measurement. Geometry and texture counters are renderer counts, not memory bytes.
First observed audio clock is relative to the recording start; it is not acoustic
latency or a complete turn's submission-to-speech latency. Human ratings,
loudspeaker timing, total memory, and simultaneous inference workload identity
remain explicitly unmeasured.

Recordings retain at most three minutes / 36,000 frames with a 10 Hz sample trace.
Changing the model, mapping, audio/timing, framing, viewport or motion settings
stops the recording. Hidden documents, context loss and seeking also stop it.
Local playback completion/stop is recorded as `playback_inactive`; this alone does
not assert that a complete semantic answer was spoken.

## 4. Compare two browser exports

```bash
node tools/avatar_audition.mjs compare \
  data/local/avatar-candidate/audition-a.json \
  data/local/avatar-candidate/audition-b.json \
  data/local/avatar-candidate/comparison.json
```

The comparison verifies both report hashes before examining conditions. Audio,
speech-timing digest, protocol, camera/viewport and motion settings must match.
Both samples must contain speaking frames and their first/last audio offsets must
be within 100 ms. Invalidated, still-active or capped recordings do not pass.
That 100 ms is a recording-window comparability tolerance, not a product latency
budget or a lip-sync-quality threshold.

Matching conditions produce side-by-side measured values with a null quality
winner. They do not automatically select a model. Add blinded human ratings and
dated front/three-quarter/profile appearance evidence, inspect speech deformations,
and verify the exact deployment terms before selecting a candidate.

## Verification and remaining gate

```bash
node --test tools/avatar_audition.test.mjs
uv run --extra dev pytest -q tests/test_avatar_convert.py
npm exec --workspace @bias/web -- vitest run \
  src/avatar/audition.test.ts src/avatar/auditionAudio.test.ts \
  src/avatar/native.integration.test.ts src/avatar/retarget.test.ts
npm run build
npm run test:e2e -- tests/browser/avatar-audition.spec.ts
```

The local run passed the offline decoder/comparison tests, local-audio and recorder
tests, actual compressed-asset deformation tests, and actual Blender import/export
roundtrip. The production build passed. Browser execution was attempted for both
desktop and mobile; this workspace rejected Chromium's process-singleton socket
with `Operation not permitted` before either page loaded. That failed launch is
not rendering evidence. The focused browser cases are committed for the normal
GitHub CI environment and verify actual GLB/WAV bytes, recording and export hashes,
the absence of synthesis requests, and mobile overflow.

WP2's final acceptance remains open: no new Musk-specific file has yet passed an
actual deformation/material/browser/appearance comparison. This work makes the
next lawful package acquisition inspectable and reproducible while retaining the
current proxy and its limitations accurately.
