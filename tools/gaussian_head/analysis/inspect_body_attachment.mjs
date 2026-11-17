#!/usr/bin/env node
/**
 * Inspect the recovered PR #23 bundles against the shipped Jensen body.
 *
 * npm ci
 * node tools/gaussian_head/analysis/inspect_body_attachment.mjs --output /private/path/report.json
 *
 * Inputs default to the ignored local Gaussian directory and the shipped GLB.
 * Optional --bundle-dir and --body paths support a separate private asset root.
 * This performs CPU geometry checks, never starts a browser or writes model data.
 * The aggregate JSON contains hashes, counts and scalar diagnostics only.
 */
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFile, mkdir, writeFile, realpath } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { build } from 'vite';

// Serialized below into an in-memory Vite module so this diagnostic uses the
// production TypeScript implementations without a separate transpiler or copy.
async function inspectRuntime({ bundleDir, bodyPath }) {
  const digest = (data) => createHash('sha256').update(data).digest('hex');
  const vectorBounds = (data) => {
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < data.length; i++) {
      min[i % 3] = Math.min(min[i % 3], data[i]);
      max[i % 3] = Math.max(max[i % 3], data[i]);
    }
    return { min, max };
  };
  const identityError = (matrix) => Math.max(...matrix.elements.map((value, i) => Math.abs(value - (i % 5 === 0 ? 1 : 0))));
  const expected = {
    jensen_a: '727ac52904b7a610b8b3bd59594ed85ef9990fd4b2105e8dc5a57079bc852c0d',
    jensen_a_sh: '1a8c3bed73cdc93553257887bd682b3d87ea5c9dd1d9b2d6f4abed730e0605c7',
    jensen_a_c1c3_240k_sh: '32d5bc15901a8673901c6234ae7af946d2bfcf2c0aa5e7ed9c816d1c3aff599c',
  };
  const accepted = [];
  let header, B, sourceBuffer, binaryHash;
  for (const [name, expectedHash] of Object.entries(expected)) {
    const manifest = await readFile(bundleDir + '/' + name + '.json');
    const h = JSON.parse(manifest.toString('utf8'));
    const raw = await readFile(bundleDir + '/' + name + '.bin');
    const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
    const arrays = parseBundle(h, buffer);
    validateGaussianBundle(h, arrays);
    const hash = digest(raw);
    assert.equal(hash, expectedHash, 'This bounded diagnostic requires the documented PR #23 bundle: ' + name);
    accepted.push({
      bundle: name, manifest_bytes: manifest.length, manifest_sha256: digest(manifest),
      binary_bytes: raw.length, binary_sha256: hash, matches_documented_binary: true, validated: true,
      dimensions: { vertices: h.n_verts, faces: h.n_faces, expressions: h.n_expr, gaussians: h.n_gauss, frames: h.n_frames, fps: h.fps },
    });
    if (name === 'jensen_a_c1c3_240k_sh') { header = h; B = arrays; sourceBuffer = buffer; binaryHash = hash; }
  }

  const bodyBytes = await readFile(bodyPath);
  const bodyHash = digest(bodyBytes);
  assert.equal(bodyHash, 'dea0c921fb7d5668527bd8ee9082f382b14e8d300c43936842e15356dc7d90cd', 'Unexpected fitted body asset.');
  const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
  // Decode real compressed geometry, joints, morphs and skin. Image/material
  // creation is replaced, matching the existing CPU native integration checks.
  loader.register((parser) => ({ name: 'CPU_MATERIALS_ONLY', loadMaterial: async (index) =>
    new MeshStandardMaterial({ name: parser.json.materials[index].name }) }));
  const imported = await loader.parseAsync(bodyBytes.buffer.slice(bodyBytes.byteOffset, bodyBytes.byteOffset + bodyBytes.byteLength), '');
  const mapping = createResearchHumanMapping(imported.scene);
  const rig = createRetargetedRig(imported.scene, mapping);
  const meshes = [];
  rig.root.traverse((node) => { if (node instanceof Mesh) meshes.push(node); });
  const update = () => {
    rig.root.updateMatrixWorld(true);
    for (const mesh of meshes) if (mesh instanceof SkinnedMesh) mesh.skeleton.update();
  };
  rig.apply(neutralPose()); update();
  assert.deepEqual(rig.diagnostics.missing, []);
  const bodyBounds = new Box3().setFromObject(rig.root);
  const attachment = fitGaussianAttachment(rig, B, { bundle: 'jensen_a_c1c3_240k_sh' });
  const world = attachment.head.matrixWorld.clone().multiply(attachment.matrix);
  const pivotError = new Vector3().fromArray(B.j_base, 3).applyMatrix4(world)
    .distanceTo(attachment.head.getWorldPosition(new Vector3()));
  assert.ok(pivotError < 1e-10, 'The mapped head pivot must coincide.');
  const originalRefs = meshes.map((mesh) => ({ mesh, geometry: mesh.geometry, material: mesh.material }));
  const restore = hideOriginalHead(rig, attachment.targetCutY);
  const mask = meshes.map((mesh) => {
    const attr = mesh.geometry.getAttribute('gaussianHeadCut');
    let hidden = 0, handCount = 0, hiddenHands = 0;
    if (attr) for (let i = 0; i < attr.count; i++) {
      if (attr.getX(i) > 0) hidden++;
      if (mesh instanceof SkinnedMesh) {
        const joints = mesh.geometry.getAttribute('skinIndex'), weights = mesh.geometry.getAttribute('skinWeight');
        const isHand = [0, 1, 2, 3].some((slot) => weights.getComponent(i, slot) > .05
          && /^(finger|wrist)/.test(mesh.skeleton.bones[joints.getComponent(i, slot)].name));
        if (isHand) { handCount++; if (attr.getX(i) > 0) hiddenHands++; }
      }
    }
    return { name: mesh.name, vertices: mesh.geometry.getAttribute('position').count,
      mask_applied: !!attr, hidden_vertices: hidden, identified_hand_vertices: handCount, hidden_hand_vertices: hiddenHands };
  });
  assert.ok(mask.some((row) => row.identified_hand_vertices > 100), 'The actual skin must expose identifiable hands.');
  assert.ok(mask.every((row) => row.hidden_hand_vertices === 0), 'The original-head mask must retain the hands.');
  assert.ok(mask.filter((row) => /suit|jacket|shirt|clothes/i.test(row.name)).every((row) => !row.mask_applied), 'Clothing must retain its geometry.');

  const head = new GaussianBodyHead(header, B, attachment);
  attachment.head.add(head.root); update();
  const framer = createUpperBodyFramer(rig.root, rig.diagnostics);
  assert.ok(framer, 'The real body must support upper-body framing.');
  const camera = new PerspectiveCamera(36, 1.4, .01, 100), framed = framer(1.4);
  camera.position.fromArray(framed.position); camera.lookAt(...framed.target); camera.updateMatrixWorld();
  const performance = createGaussianPerformance(header, B, { bundleSha256: binaryHash });
  assert.equal(performance.diagnostics.replayFrameCount, 2324, 'Keep all sampling inside the documented training partition.');
  const rest = performance.sample({ rigPose: neutralPose(), elapsedMs: 0, idleMs: 0, speaking: false });
  head.update(rest, camera, 1400, 1000);
  const correctionError = identityError(head.mesh.material.uniforms.uNeckCorrection.value);
  assert.ok(correctionError < 1e-10, 'Collar blending must be equivalent at neutral rest.');
  const locals = head.mesh.material.uniforms.uLocal.value.image.data;
  const bindings = head.mesh.material.uniforms.uBinding.value.image.data;
  let kept = 0, blended = 0, fullyNeck = 0;
  for (let i = 0; i < header.n_gauss; i++) if (locals[i * 4 + 3] > 0) {
    kept++;
    if (bindings[i * 4 + 1] > 0) blended++;
    if (bindings[i * 4 + 1] >= .999999) fullyNeck++;
  }
  assert.ok(kept > 0 && kept < header.n_gauss);
  const flame = new FlameHead(header, B);
  const baseline = new Float32Array(flame.forward(rest.expr, rest.pose, rest.trans));
  const frameChecks = [];
  // Five training frames only. The delta below includes a fixed live jaw input;
  // it is not an isolated measure of source expression or eyebrow variation.
  for (const index of [0, 143, 599, 1200, 2323]) {
    const elapsedMs = index * 1000 / header.fps;
    const parameters = performance.sample({ rigPose: { ...neutralPose(), jawOpen: .5 }, elapsedMs, idleMs: elapsedMs, speaking: true });
    head.update(parameters, camera, 1400, 1000);
    const verts = flame.forward(parameters.expr, parameters.pose, parameters.trans);
    let maximumDelta = 0;
    for (let i = 0; i < verts.length; i += 3) maximumDelta = Math.max(maximumDelta,
      Math.hypot(verts[i] - baseline[i], verts[i + 1] - baseline[i + 1], verts[i + 2] - baseline[i + 2]));
    const centers = gaussianCenters(B, flame.updateFaceFrames(), new Float32Array(header.n_gauss * 3));
    const result = { training_frame: index, finite_vertices: [...verts].every(Number.isFinite), finite_centers: [...centers].every(Number.isFinite),
      maximum_combined_vertex_delta_source_metres: maximumDelta, expression_source: parameters.expressionSource,
      global_and_neck_rotation_zero: [...parameters.pose.slice(0, 6)].every((value) => value === 0) };
    assert.ok(result.finite_vertices && result.finite_centers && result.global_and_neck_rotation_zero);
    assert.equal(result.expression_source, 'observed_replay');
    frameChecks.push(result);
  }
  const motionChecks = [];
  for (const [name, pose] of [
    ['head_left', { ...neutralPose(), headYaw: .3 }], ['head_right', { ...neutralPose(), headYaw: -.3 }],
    ['nod', { ...neutralPose(), headPitch: .15 }], ['designed_gesture', { ...neutralPose(), openPalm: 1, emphasis: .8, wave: 1 }],
  ]) {
    rig.apply(pose); update();
    const parameters = performance.sample({ rigPose: pose, elapsedMs: 1000, idleMs: 1000, speaking: false });
    head.update(parameters, camera, 1400, 1000);
    const result = { name, finite_attachment: head.root.matrixWorld.elements.every(Number.isFinite),
      finite_neck_correction: head.mesh.material.uniforms.uNeckCorrection.value.elements.every(Number.isFinite) };
    assert.ok(result.finite_attachment && result.finite_neck_correction);
    motionChecks.push(result);
  }
  rig.apply(neutralPose()); update(); head.update(rest, camera, 1400, 1000);
  const returnedCorrectionError = identityError(head.mesh.material.uniforms.uNeckCorrection.value);
  assert.ok(returnedCorrectionError < 1e-10);
  head.dispose(); restore();
  const exactRestoration = originalRefs.every(({ mesh, geometry, material }) => mesh.geometry === geometry && mesh.material === material);
  const unchanged = digest(Buffer.from(sourceBuffer)) === binaryHash
    && digest(await readFile(bundleDir + '/jensen_a_c1c3_240k_sh.bin')) === binaryHash;
  assert.ok(exactRestoration && unchanged, 'Inspection must restore the body and preserve the original Gaussian bytes.');
  const result = {
    inputs: { bundles: accepted, body: { name: 'jensen-source-fitted-v3.glb', bytes: bodyBytes.length, sha256: bodyHash } },
    methods: {
      geometry: 'Production Meshopt GLTF decoding, rig mapping, FLAME forward pass, Gaussian attachment, mask and performance controller.',
      materials: 'Image/material creation replaced by MeshStandardMaterial for CPU inspection; no texture or pixel acceptance.',
      rendering: 'No browser, WebGL context, GPU draw, screenshots or private-asset upload.',
      viewport_for_update: { width: 1400, height: 1000, aspect: 1.4 },
      hand_identification: 'Vertices with weight > 0.05 on a skeleton joint whose name starts with finger or wrist.',
      frame_delta_reference: { training_frame: 0, live_jaw_open: 0, compared_live_jaw_open: .5, includes_jaw_motion: true },
      documented_training_frames: [0, 2323], held_out_frames_sampled: 0,
    },
    body: { mapped_bones: rig.diagnostics.bones.length, missing_controls: rig.diagnostics.missing,
      bounds_world_units: { min: bodyBounds.min.toArray(), max: bodyBounds.max.toArray() } },
    attachment: {
      head_joint: attachment.head.name, neck_joint: attachment.neck?.name, scale: attachment.scale,
      collar_cut_source_metres: .05, source_cut_y_metres: attachment.sourceCutY, target_cut_y_world_units: attachment.targetCutY,
      seam_blend_source_metres: attachment.seamBlendMeters, source_bounds_metres: vectorBounds(B.v_shaped),
      pivot_error_world_units: pivotError, neutral_neck_correction_error: correctionError, returned_neck_correction_error: returnedCorrectionError,
      gaussians_kept: kept, gaussians_masked: header.n_gauss - kept,
      gaussians_partly_neck_blended: blended - fullyNeck, gaussians_fully_neck_blended: fullyNeck,
    },
    original_head_mask: mask, training_frame_checks: frameChecks, motion_checks: motionChecks,
    exact_body_restoration: exactRestoration, gaussian_binary_unchanged: unchanged,
    limitations: [
      'Numerical attachment and finite-geometry results do not establish likeness, collar appearance, hand occlusion pixels or browser frame rate.',
      'Render-space world units follow the body normalization; they are not a measurement of the subject physical height.',
      'Frame displacement includes the fixed live jaw control; no semantic expression driver or eyebrow-quality result is claimed.',
      'No held-out Gaussian frames were sampled, and no training or model modification ran.',
    ],
  };
  rig.dispose();
  return result;
}

const { values } = parseArgs({ options: {
  'bundle-dir': { type: 'string' }, body: { type: 'string' }, output: { type: 'string' }, help: { type: 'boolean' },
} });
if (values.help) {
  console.log('node tools/gaussian_head/analysis/inspect_body_attachment.mjs [--bundle-dir DIR] [--body GLB] [--output JSON]');
  process.exit(0);
}
const repoRoot = fileURLToPath(new URL('../../../', import.meta.url));
const bundleDir = resolve(values['bundle-dir'] ?? resolve(repoRoot, 'apps/web/public/models/gaussian-head'));
const bodyPath = resolve(values.body ?? resolve(repoRoot, 'apps/web/public/models/jensen-source-fitted-v3.glb'));
const output = values.output ? resolve(values.output) : undefined;
if (output) {
  const canonical = async (path) => {
    try { return await realpath(path); }
    catch (error) { if (error.code === 'ENOENT') return resolve(path); throw error; }
  };
  const inputPaths = [bodyPath, ...['jensen_a', 'jensen_a_sh', 'jensen_a_c1c3_240k_sh']
    .flatMap((name) => ['.json', '.bin'].map((extension) => resolve(bundleDir, name + extension)))];
  const protectedPaths = await Promise.all(inputPaths.map(canonical));
  if (protectedPaths.includes(await canonical(output))) throw new Error('The report must not overwrite an input asset.');
}
const modulePath = (name) => JSON.stringify(resolve(repoRoot, 'apps/web/src/avatar', name));
const source = [
  "import assert from 'node:assert/strict';",
  "import { readFile } from 'node:fs/promises';",
  "import { createHash } from 'node:crypto';",
  "import { Box3, Mesh, MeshStandardMaterial, PerspectiveCamera, SkinnedMesh, Vector3 } from 'three';",
  "import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';",
  "import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';",
  'import { createResearchHumanMapping } from ' + modulePath('builtin.ts') + ';',
  'import { createRetargetedRig } from ' + modulePath('retarget.ts') + ';',
  'import { neutralPose } from ' + modulePath('timeline.ts') + ';',
  'import { createUpperBodyFramer } from ' + modulePath('framing.ts') + ';',
  'import { fitGaussianAttachment, GaussianBodyHead, hideOriginalHead } from ' + modulePath('gaussian/gaussianBody.ts') + ';',
  'import { parseBundle, validateGaussianBundle, FlameHead, gaussianCenters } from ' + modulePath('gaussian/gaussianHeadCore.ts') + ';',
  'import { createGaussianPerformance } from ' + modulePath('gaussian/gaussianPerformance.ts') + ';',
  'export ' + inspectRuntime.toString(),
].join('\n');
const buildResult = await build({
  root: repoRoot, configFile: false, logLevel: 'silent',
  plugins: [{ name: 'private-attachment-inspection',
    resolveId(id) { if (id.endsWith('private-attachment-inspection')) return '\0private-attachment-inspection.ts'; },
    load(id) { if (id === '\0private-attachment-inspection.ts') return source; },
  }],
  build: { write: false, lib: { entry: 'private-attachment-inspection', formats: ['cjs'] },
    rollupOptions: { external: ['node:assert/strict', 'node:fs/promises', 'node:crypto'] } },
});
const chunk = (Array.isArray(buildResult) ? buildResult : [buildResult]).flatMap((value) => value.output).find((value) => value.type === 'chunk');
if (!chunk) throw new Error('The production geometry module did not build.');
const module = { exports: {} };
// This executes trusted, locally bundled repository code only. Asset bytes are
// read inside inspectRuntime and are never interpolated into executable source.
new Function('module', 'exports', 'require', chunk.code)(module, module.exports, createRequire(import.meta.url));
const result = await module.exports.inspectRuntime({
  bundleDir, bodyPath,
});
const trackedInputs = [
  'package-lock.json', 'apps/web/src/avatar/builtin.ts', 'apps/web/src/avatar/retarget.ts',
  'apps/web/src/avatar/timeline.ts', 'apps/web/src/avatar/framing.ts', 'apps/web/src/avatar/channels.ts',
  'apps/web/src/avatar/gaussian/gaussianBody.ts', 'apps/web/src/avatar/gaussian/gaussianHeadCore.ts',
  'apps/web/src/avatar/gaussian/gaussianPerformance.ts', 'tools/gaussian_head/analysis/inspect_body_attachment.mjs',
];
const sourceHashes = Object.fromEntries(await Promise.all(trackedInputs.map(async (path) =>
  [path, createHash('sha256').update(await readFile(resolve(repoRoot, path))).digest('hex')])));
const report = {
  schema: 'gaussian-body-attachment-inspection.v1', generated_at: new Date().toISOString(),
  source_commit: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repoRoot, encoding: 'utf8' }).trim(),
  source_sha256: sourceHashes, node_version: process.version, numerical_checks_passed: true, ...result,
};
const text = JSON.stringify(report, null, 2) + '\n';
if (output) {
  await mkdir(dirname(output), { recursive: true }); await writeFile(output, text);
  console.log('Numerical checks passed. Aggregate report written to ' + output);
} else process.stdout.write(text);
