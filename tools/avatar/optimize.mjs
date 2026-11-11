/** Offline-only final packaging; preserves skeleton and all native expression targets. */
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { dedup, meshopt, textureCompress } from '@gltf-transform/functions';
import { MeshoptEncoder, MeshoptDecoder } from 'meshoptimizer';
import sharp from 'sharp';

const [input, output, reportPath] = process.argv.slice(2);
if (!input || !output || !reportPath) throw new Error('Usage: node optimize.mjs INPUT.glb OUTPUT.glb REPORT.json');
await Promise.all([MeshoptEncoder.ready, MeshoptDecoder.ready]);
sharp.concurrency(2);
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({
  'meshopt.encoder': MeshoptEncoder, 'meshopt.decoder': MeshoptDecoder,
});
const document = await io.read(input);

function inventory(doc) {
  const root = doc.getRoot(), targets = new Map();
  const meshes = root.listMeshes().map((mesh) => {
    const names = mesh.getExtras().targetNames ?? [];
    let vertices = 0, triangles = 0;
    for (const primitive of mesh.listPrimitives()) {
      vertices += primitive.getAttribute('POSITION').getCount();
      triangles += (primitive.getIndices()?.getCount() ?? primitive.getAttribute('POSITION').getCount()) / 3;
      for (const [index, target] of primitive.listTargets().entries()) {
        const attribute = target.getAttribute('POSITION');
        let maximum = 0;
        if (attribute) for (let i = 0; i < attribute.getCount(); i++) {
          const value = attribute.getElement(i, []);
          maximum = Math.max(maximum, Math.hypot(...value));
        }
        const name = names[index] ?? `unnamed_${index}`;
        targets.set(name, Math.max(targets.get(name) ?? 0, maximum));
      }
    }
    return {name: mesh.getName(), vertices, triangles, targets: names};
  });
  return {meshes, target_names: [...targets.keys()].sort(),
    nonzero_target_names: [...targets].filter(([, n]) => n > 1e-7).map(([name]) => name).sort(),
    joint_names: [...new Set(root.listSkins().flatMap((s) => s.listJoints().map((j) => j.getName())))].sort(),
    material_count: root.listMaterials().length,
    textures: root.listTextures().map((t) => ({name: t.getName(), mime: t.getMimeType(), bytes: t.getImage()?.length, size: t.getSize()})),
  };
}
const before = inventory(document);
// The eye asset's outer surface also uses alpha; ignoring it makes the eyes black.
// Stable cutouts avoid translucent triangle sorting for those layers and hair cards.
for (const material of document.getRoot().listMaterials()) {
  if (['Eyes', 'Hair', 'Brows'].includes(material.getName())) material.setAlphaMode('MASK').setAlphaCutoff(0.35).setDoubleSided(true);
}
await document.transform(
  dedup(),
  textureCompress({encoder: sharp, targetFormat: 'webp', quality: 86, effort: 70, resize: [2048, 2048]}),
  meshopt({encoder: MeshoptEncoder, level: 'high', quantizePosition: 16, quantizeNormal: 12, quantizeTexcoord: 14, quantizeWeight: 12}),
);
await mkdir(dirname(output), {recursive: true});
await io.write(output, document);
const after = inventory(await io.read(output));
for (const key of ['target_names', 'nonzero_target_names', 'joint_names']) {
  if (JSON.stringify(before[key]) !== JSON.stringify(after[key])) throw new Error(`Optimization changed ${key}`);
}
const bytes = await readFile(output);
const inputBytes = await readFile(input);
const report = {
  schema_version: 1, asset_id: 'research-human-v2', output_bytes: bytes.length,
  output_sha256: createHash('sha256').update(bytes).digest('hex'),
  input_bytes: inputBytes.length,
  input_sha256: createHash('sha256').update(inputBytes).digest('hex'),
  pipeline: {gltf_transform: '4.5.1', meshoptimizer: '1.2.0', sharp: '0.35.5',
    meshopt: 'high', quantize_position_bits: 16, quantize_normal_bits: 12,
    quantize_texcoord_bits: 14, quantize_weight_bits: 12,
    texture_format: 'webp', texture_quality: 86, texture_max_dimensions: [2048, 2048],
    transparent_materials: {names: ['Eyes', 'Hair', 'Brows'], mode: 'MASK', cutoff: 0.35}, geometry_simplification: false},
  before, after, likeness_review: 'not_performed',
};
await mkdir(dirname(reportPath), {recursive: true});
await writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({output, bytes: bytes.length, sha256: report.output_sha256,
  native_targets: after.target_names.length, effective_targets: after.nonzero_target_names.length, joints: after.joint_names.length}));
