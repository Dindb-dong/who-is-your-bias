/** Local-only GLB qualification. Uses the existing pinned avatar-tool decoder.
 * npm ci --prefix tools/avatar --ignore-scripts
 * node tools/avatar_audition.mjs inspect INPUT.glb REPORT.json [--provenance FILE] [--audio FILE]
 * node tools/avatar_audition.mjs compare BROWSER_RUN_A.json BROWSER_RUN_B.json REPORT.json
 */
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const deps = createRequire(new URL('./avatar/package.json', import.meta.url));
const sha256 = (value) => createHash('sha256').update(value).digest('hex');
const MAX_BYTES = 64 * 1024 * 1024;
const normalize = (name) => name.toLowerCase().replace(/[^a-z0-9]/g, '');
export function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value !== null && typeof value === 'object') return `{${Object.entries(value).filter(([, item]) => item !== undefined)
    .sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, item]) => `${JSON.stringify(key)}:${canonicalJson(item)}`).join(',')}}`;
  if (typeof value === 'number' && !Number.isFinite(value)) throw new Error('Non-finite report value.');
  const result = JSON.stringify(value); if (result === undefined) throw new Error('Unsupported report value.'); return result;
}

export function readGlbJson(bytes) {
  if (bytes.length < 20 || bytes.length > MAX_BYTES || bytes.readUInt32LE(0) !== 0x46546c67
      || bytes.readUInt32LE(4) !== 2 || bytes.readUInt32LE(8) !== bytes.length) throw new Error('Expected a complete binary glTF 2.0 file at most 64 MiB.');
  const length = bytes.readUInt32LE(12);
  if (bytes.readUInt32LE(16) !== 0x4e4f534a || length % 4 || 20 + length > bytes.length) throw new Error('Invalid GLB JSON chunk.');
  const json = JSON.parse(bytes.subarray(20, 20 + length).toString('utf8'));
  if (json.asset?.version !== '2.0') throw new Error('Expected glTF 2.0 asset metadata.');
  for (const resource of [...(json.buffers ?? []), ...(json.images ?? [])]) {
    if (resource.uri && !resource.uri.startsWith('data:')) throw new Error('Only self-contained GLB files are inspected; embed buffers and textures first.');
  }
  for (const extension of json.extensionsRequired ?? []) {
    if (['KHR_draco_mesh_compression', 'KHR_texture_basisu'].includes(extension)) throw new Error(`The application does not bundle the ${extension} decoder; convert to uncompressed or Meshopt GLB.`);
  }
  return json;
}

export async function createIO() {
  const [{ NodeIO }, { ALL_EXTENSIONS }, { MeshoptDecoder }] = await Promise.all([
    import(pathToFileURL(deps.resolve('@gltf-transform/core')).href),
    import(pathToFileURL(deps.resolve('@gltf-transform/extensions')).href),
    import(pathToFileURL(deps.resolve('meshoptimizer')).href),
  ]);
  await MeshoptDecoder.ready;
  return new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.decoder': MeshoptDecoder });
}

export function validateProvenance(value = {}) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) throw new Error('Provenance must be a JSON object.');
  const distribution = value.permitted_distribution ?? 'unknown';
  if (!['bundled', 'local_only', 'separate_license', 'unknown'].includes(distribution)) throw new Error('Invalid permitted_distribution.');
  const status = value.status ?? 'unverified';
  if (!['project_authored', 'license_recorded', 'unverified'].includes(status)) throw new Error('Invalid provenance status.');
  for (const field of ['creator', 'source_url', 'license', 'acquired_at', 'appearance_reference_period']) {
    if (value[field] !== undefined && (typeof value[field] !== 'string' || value[field].length > 4000)) throw new Error(`Invalid provenance ${field}.`);
  }
  if (value.source_url && !/^https:\/\//.test(value.source_url)) throw new Error('The source URL must be HTTPS.');
  if (value.license_sha256 && !/^[a-f0-9]{64}$/.test(value.license_sha256)) throw new Error('Invalid license digest.');
  if (status !== 'unverified' && (!value.creator || !value.source_url || !value.license)) throw new Error('Recorded provenance needs creator, exact source URL and license.');
  // Declared permission is an acquisition record, never a legal determination.
  return { ...value, status, permitted_distribution: distribution, verification_basis: 'operator_supplied_record' };
}

async function vocabulary() {
  const source = await readFile(new URL('../apps/web/src/avatar/channels.ts', import.meta.url), 'utf8');
  const list = (name) => {
    const body = source.match(new RegExp(`export const ${name} = \\[([\\s\\S]*?)\\] as const;`))?.[1];
    if (!body) throw new Error(`Cannot read the runtime's ${name} vocabulary.`);
    return [...body.matchAll(/'([^']+)'/g)].map((match) => match[1]);
  };
  return { face: list('FACE_CHANNELS'), visemes: list('VISEMES'), bones: [...list('BODY_CHANNELS'), ...['left', 'right'].flatMap((side) => list('FINGER_JOINTS').map((joint) => side + joint))] };
}

function countTriangles(mode, count) {
  if (mode === 4) return count / 3;
  if (mode === 5 || mode === 6) return Math.max(0, count - 2);
  return 0;
}

export function inventory(doc) {
  const root = doc.getRoot(); const warnings = []; const deformingNames = new Set(); const allNames = new Set();
  const meshes = root.listMeshes().map((mesh, meshIndex) => {
    const names = mesh.getExtras().targetNames ?? [];
    const primitives = mesh.listPrimitives().map((primitive, primitiveIndex) => {
      const position = primitive.getAttribute('POSITION');
      if (!position) throw new Error(`Mesh ${meshIndex} primitive ${primitiveIndex} has no POSITION accessor.`);
      const counts = { vertices: position.getCount(), triangles: countTriangles(primitive.getMode(), primitive.getIndices()?.getCount() ?? position.getCount()) };
      let invalidPositionVertices = 0;
      for (let i = 0; i < position.getCount(); i++) if (!position.getElement(i, []).every(Number.isFinite)) invalidPositionVertices++;
      const morphs = primitive.listTargets().map((target, index) => {
        const name = target.getName() || names[index] || `unnamed_${index}`; allNames.add(name);
        const attribute = target.getAttribute('POSITION');
        let maxDisplacement = 0; let movedVertices = 0;
        let status = attribute ? 'zero' : 'missing_position';
        if (attribute) {
          if (attribute.getCount() !== position.getCount() || attribute.getElementSize() !== 3) status = 'invalid';
          else for (let i = 0; i < attribute.getCount(); i++) {
            const delta = Math.hypot(...attribute.getElement(i, []));
            if (!Number.isFinite(delta)) { status = 'invalid'; break; }
            maxDisplacement = Math.max(maxDisplacement, delta); if (delta > 1e-7) movedVertices++;
          }
          if (status !== 'invalid') status = movedVertices ? 'deforms' : 'zero';
        }
        if (status === 'deforms') deformingNames.add(name);
        return { name, status, max_displacement_local_units: ['deforms', 'zero'].includes(status) ? maxDisplacement : null,
          moved_vertices: ['deforms', 'zero'].includes(status) ? movedVertices : null };
      });
      const weights = [primitive.getAttribute('WEIGHTS_0'), primitive.getAttribute('WEIGHTS_1')].filter(Boolean);
      let zeroWeightVertices = 0; let nonUnitWeightVertices = 0; let invalidWeightVertices = 0;
      if (weights.length) for (let i = 0; i < position.getCount(); i++) {
        const values = weights.flatMap((weight) => i < weight.getCount() ? weight.getElement(i, []) : [NaN]);
        if (!values.every((v) => Number.isFinite(v) && v >= 0 && v <= 1)) { invalidWeightVertices++; continue; }
        const total = values.reduce((a, b) => a + b, 0);
        if (total < 1e-7) zeroWeightVertices++;
        if (Math.abs(total - 1) > 0.02) nonUnitWeightVertices++;
      }
      if (invalidPositionVertices || invalidWeightVertices || zeroWeightVertices) warnings.push(`Mesh ${meshIndex}/${primitiveIndex} has invalid positions or unusable skin weights.`);
      return { index: primitiveIndex, mode: primitive.getMode(), ...counts, invalid_position_vertices: invalidPositionVertices,
        material: primitive.getMaterial()?.getName() ?? null, morphs,
        skin_weights: weights.length ? { sets: weights.length, zero_weight_vertices: zeroWeightVertices,
          non_unit_weight_vertices: nonUnitWeightVertices, invalid_weight_vertices: invalidWeightVertices } : null };
    });
    return { index: meshIndex, name: mesh.getName(), primitives };
  });
  const skins = root.listSkins().map((skin, index) => ({ index, name: skin.getName(), joints: skin.listJoints().map((joint) => joint.getName()),
    inverse_bind_matrices: skin.getInverseBindMatrices()?.getCount() ?? 0 }));
  const nodes = root.listNodes().map((node, index) => ({ index, name: node.getName(), mesh: node.getMesh()?.getName() ?? null,
    skin: node.getSkin() ? root.listSkins().indexOf(node.getSkin()) : null,
    translation: node.getTranslation(), rotation_xyzw: node.getRotation(), scale: node.getScale() }));
  const materials = root.listMaterials().map((material) => ({ name: material.getName(), alpha_mode: material.getAlphaMode(),
    alpha_cutoff: material.getAlphaCutoff(), double_sided: material.getDoubleSided(), base_color: material.getBaseColorFactor(),
    metallic: material.getMetallicFactor(), roughness: material.getRoughnessFactor(),
    base_color_texture: material.getBaseColorTexture()?.getName() ?? null, normal_texture: material.getNormalTexture()?.getName() ?? null,
    metallic_roughness_texture: material.getMetallicRoughnessTexture()?.getName() ?? null }));
  const textures = root.listTextures().map((texture) => ({ name: texture.getName(), mime: texture.getMimeType(),
    encoded_bytes: texture.getImage()?.length ?? 0, dimensions: texture.getSize(), sha256: texture.getImage() ? sha256(texture.getImage()) : null }));
  const animations = root.listAnimations().map((animation) => ({ name: animation.getName(),
    channels: animation.listChannels().map((channel) => {
      const sampler = channel.getSampler(); const input = sampler?.getInput();
      let end = null;
      if (input?.getCount()) { end = 0; for (const time of input.getArray()) {
        if (!Number.isFinite(time) || time < 0) throw new Error('Animation contains invalid sample times.'); end = Math.max(end, time);
      } }
      return { node: channel.getTargetNode()?.getName() ?? null, path: channel.getTargetPath(), interpolation: sampler?.getInterpolation(),
        samples: input?.getCount() ?? 0, end_seconds: end };
    }) }));
  if (!skins.length) warnings.push('No skin was found. Body bone names alone do not establish deformable skin.');
  if (!deformingNames.size) warnings.push('No nonzero POSITION morph targets were measured.');
  return { meshes, skins, nodes, materials, textures, animations, named_targets: [...allNames].sort(),
    deforming_targets: [...deformingNames].sort(), warnings };
}

export async function inspectGlb(input, { provenance = {}, audio } = {}) {
  const bytes = await readFile(input); const json = readGlbJson(bytes); const io = await createIO();
  const doc = await io.readBinary(bytes); const measured = inventory(doc); const names = await vocabulary();
  const aliases = { jawOpen: ['mouthOpen'], eyeBlinkLeft: ['eyeBlink_L', 'blinkLeft'], eyeBlinkRight: ['eyeBlink_R', 'blinkRight'],
    mouthSmileLeft: ['mouthSmile_L'], mouthSmileRight: ['mouthSmile_R'], browDownLeft: ['browDown_L'], browDownRight: ['browDown_R'] };
  const active = new Set(measured.deforming_targets.map(normalize));
  const supportedFace = names.face.filter((control) => [control, ...(aliases[control] ?? [])].some((name) => active.has(normalize(name))));
  const supportedVisemes = names.visemes.filter((control) => {
    const alternatives = [`viseme_${control}`, `viseme${control}`, ...({ I: ['viseme_ih'], O: ['viseme_oh'], U: ['viseme_ou'] }[control] ?? [])];
    // A silence target is deliberately allowed to be neutral, while speech must deform.
    const targets = control === 'sil' ? new Set(measured.named_targets.map(normalize)) : active;
    return alternatives.some((name) => targets.has(normalize(name)));
  });
  const report = {
    schema_version: 'avatar-qualification.v1', asset: { filename: basename(input), sha256: sha256(bytes), bytes: bytes.length,
      generator: json.asset.generator ?? null, extensions_required: json.extensionsRequired ?? [] },
    provenance: validateProvenance(provenance), measurement: { decoder: 'gltf-transform 4.5.1 / meshoptimizer 1.2.0',
      tool_sha256: sha256(await readFile(fileURLToPath(import.meta.url))),
      dependency_lock_sha256: sha256(await readFile(new URL('./avatar/package-lock.json', import.meta.url))),
      basis: 'actual decoded GLB accessors; morph displacement in mesh-local units', ...measured },
    controls: { basis: 'canonical runtime names and aliases; explicit custom mappings require browser inspection',
      face_with_position_deformation: supportedFace, missing_face: names.face.filter((c) => !supportedFace.includes(c)),
      native_visemes: supportedVisemes, missing_visemes: names.visemes.filter((c) => !supportedVisemes.includes(c)),
      jaw_policy: 'runtime selects native visemes or face morphs or jaw bone, without additive opening' },
    same_audio_reference: audio ? { filename: basename(audio), sha256: sha256(await readFile(audio)) } : null,
    review: { appearance_period: provenance.appearance_reference_period ?? null, likeness: 'ungraded', material_appearance: 'ungraded',
      mouth_interior: 'ungraded', dynamic_articulation: 'ungraded', browser_performance: 'unmeasured', listening: 'ungraded' },
    decision: 'inspection_only',
    remaining: ['Review acquisition and permitted distribution for this exact file.', 'Inspect mouth interior, neutral and speaking likeness against dated references.',
      'Import the file in the browser, verify explicit rig mapping, and export a same-audio audition.', 'Record human appearance and naturalness ratings before candidate selection.'],
  };
  return { report, reportSha256: sha256(canonicalJson(report)) };
}

export function compareAuditions(a, b) {
  for (const [label, envelope] of [['A', a], ['B', b]]) {
    if (envelope?.report?.schema_version !== 'avatar-audition-run.v1' || envelope.reportSha256 !== sha256(canonicalJson(envelope.report))) throw new Error(`${label}: invalid or changed browser audition report.`);
  }
  const left = a.report; const right = b.report; const reasons = [];
  for (const [key, label] of [['protocol', 'protocol'], ['audioSha256', 'audio'], ['speechTimingSha256', 'speech timing'], ['camera', 'camera/viewport'], ['settings', 'motion/settings']]) {
    if (canonicalJson(left[key]) !== canonicalJson(right[key])) reasons.push(`${label} differs`);
  }
  for (const key of ['firstAudioOffsetMs', 'lastAudioOffsetMs']) {
    const av = left.measurements[key]; const bv = right.measurements[key];
    if (av === null || bv === null || !Number.isFinite(av) || !Number.isFinite(bv) || Math.abs(av - bv) > 100) reasons.push(`${key} differs by more than 100 ms or is unmeasured`);
  }
  if (!left.measurements.speakingFrames || !right.measurements.speakingFrames) reasons.push('speech was not sampled in both runs');
  const ordinaryStops = new Set(['exported', 'user_stopped', 'playback_inactive']);
  if (!ordinaryStops.has(left.stoppedReason) || !ordinaryStops.has(right.stoppedReason)) reasons.push('a run was invalidated or remains active');
  return { schema_version: 'avatar-audition-comparison.v1', comparable_conditions: !reasons.length, reasons,
    report_a_sha256: a.reportSha256, report_b_sha256: b.reportSha256,
    asset_a_sha256: left.asset.sha256, asset_b_sha256: right.asset.sha256,
    performance: reasons.length ? null : { a: left.measurements, b: right.measurements },
    likeness_or_quality_winner: null, review_status: 'human_review_required',
    limitation: 'Matched conditions establish comparability only; they do not establish likeness, naturalness, or completed spoken turns.' };
}

async function writeJson(path, value) { await mkdir(dirname(path), { recursive: true }); await writeFile(path, `${JSON.stringify(value, null, 2)}\n`); }
async function main(args) {
  const [command, first, second, ...rest] = args;
  if (command === 'inspect' && first && second) {
    const options = {};
    for (let i = 0; i < rest.length; i += 2) {
      if (!rest[i + 1] || !['--audio', '--provenance'].includes(rest[i])) throw new Error('Supported options: --audio FILE --provenance FILE');
      if (rest[i] === '--audio') options.audio = rest[i + 1];
      else options.provenance = JSON.parse(await readFile(rest[i + 1], 'utf8'));
    }
    const result = await inspectGlb(first, options); await writeJson(second, result);
    console.log(JSON.stringify({ report: second, asset_sha256: result.report.asset.sha256,
      meshes: result.report.measurement.meshes.length, deforming_targets: result.report.measurement.deforming_targets.length,
      decision: result.report.decision })); return;
  }
  if (command === 'compare' && first && second && rest.length === 1) {
    const [a, b] = await Promise.all([first, second].map(async (path) => JSON.parse(await readFile(path, 'utf8'))));
    const result = compareAuditions(a, b); await writeJson(rest[0], result);
    console.log(JSON.stringify({ report: rest[0], comparable_conditions: result.comparable_conditions, reasons: result.reasons })); return;
  }
  throw new Error('Usage: node tools/avatar_audition.mjs inspect INPUT.glb REPORT.json [--provenance FILE] [--audio FILE]\n       node tools/avatar_audition.mjs compare RUN_A.json RUN_B.json REPORT.json');
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).catch((error) => { console.error(`Avatar audition failed: ${error.message}`); process.exitCode = 1; });
}
