import { strict as assert } from 'node:assert';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { canonicalJson, compareAuditions, createIO, inspectGlb, readGlbJson, validateProvenance } from './avatar_audition.mjs';

const require = createRequire(new URL('./avatar/package.json', import.meta.url));
const { Document } = require('@gltf-transform/core');
const hash = (value) => createHash('sha256').update(value).digest('hex');

function header(json) {
  const content = Buffer.from(JSON.stringify(json)); const padding = (4 - content.length % 4) % 4;
  const bytes = Buffer.alloc(20 + content.length + padding, 0x20);
  bytes.writeUInt32LE(0x46546c67, 0); bytes.writeUInt32LE(2, 4); bytes.writeUInt32LE(bytes.length, 8);
  bytes.writeUInt32LE(content.length + padding, 12); bytes.writeUInt32LE(0x4e4f534a, 16); content.copy(bytes, 20); return bytes;
}
function exampleDocument() {
  const doc = new Document(); const buffer = doc.createBuffer();
  const position = doc.createAccessor().setType('VEC3').setArray(new Float32Array([0, 0, 0, 1, 0, 0, 0, 2, 0])).setBuffer(buffer);
  const moving = doc.createAccessor().setType('VEC3').setArray(new Float32Array([0, 0, 0, 0, 0.2, 0, 0, 0, 0])).setBuffer(buffer);
  const stationary = doc.createAccessor().setType('VEC3').setArray(new Float32Array(9)).setBuffer(buffer);
  const primitive = doc.createPrimitive().setAttribute('POSITION', position)
    .addTarget(doc.createPrimitiveTarget('jawOpen').setAttribute('POSITION', moving))
    .addTarget(doc.createPrimitiveTarget('eyeBlinkLeft').setAttribute('POSITION', stationary))
    .addTarget(doc.createPrimitiveTarget('viseme_sil').setAttribute('POSITION', stationary));
  primitive.setMaterial(doc.createMaterial('Skin').setAlphaMode('MASK').setDoubleSided(true));
  const mesh = doc.createMesh('Face').setExtras({ targetNames: ['jawOpen', 'eyeBlinkLeft', 'viseme_sil'] }).addPrimitive(primitive);
  doc.createScene().addChild(doc.createNode('Head').setMesh(mesh)); return doc;
}

test('rejects incomplete, oversized, external-resource and unsupported compressed GLBs before decoding', () => {
  assert.throws(() => readGlbJson(Buffer.from('glTF')));
  const bytes = header({ asset: { version: '2.0' } }); bytes.writeUInt32LE(bytes.length + 8, 8);
  assert.throws(() => readGlbJson(bytes));
  for (const uri of ['https://example.com/private.bin', '../texture.png', 'file:///etc/passwd']) {
    assert.throws(() => readGlbJson(header({ asset: { version: '2.0' }, images: [{ uri }] })), /self-contained/);
  }
  assert.throws(() => readGlbJson(header({ asset: { version: '2.0' }, extensionsRequired: ['KHR_draco_mesh_compression'] })), /decoder/);
});

test('measures actual accessors and keeps nominal, neutral and usable facial targets separate', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'bias-avatar-audition-'));
  try {
    const io = await createIO(); const bytes = await io.writeBinary(exampleDocument()); const asset = join(dir, 'example.glb');
    await writeFile(asset, bytes); const result = await inspectGlb(asset);
    assert.equal(result.report.asset.sha256, hash(bytes));
    assert.equal(result.report.measurement.meshes.length, 1);
    assert.deepEqual(result.report.measurement.deforming_targets, ['jawOpen']);
    assert.deepEqual(result.report.controls.face_with_position_deformation, ['jawOpen']);
    assert.ok(result.report.controls.missing_face.includes('eyeBlinkLeft'));
    assert.deepEqual(result.report.controls.native_visemes, ['sil']);
    assert.equal(result.report.review.likeness, 'ungraded');
    assert.equal(result.report.decision, 'inspection_only');
    assert.equal(result.reportSha256, hash(canonicalJson(result.report)));
    assert.equal(result.report.measurement.materials[0].alpha_mode, 'MASK');
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('records operator-supplied provenance without inferring permission from a title', () => {
  assert.equal(validateProvenance({ creator: 'Someone' }).status, 'unverified');
  assert.throws(() => validateProvenance({ status: 'license_recorded', creator: 'Someone' }), /needs/);
  assert.throws(() => validateProvenance({ permitted_distribution: 'obviously-free' }));
  assert.throws(() => validateProvenance({ source_url: 'file:///tmp/model' }));
});

function audition() {
  const report = { schema_version: 'avatar-audition-run.v1', protocol: 'same-audio-fixed-camera.v1',
    audioSha256: 'a'.repeat(64), speechTimingSha256: null, camera: { viewport: [640, 480] },
    settings: { reducedMotion: false }, stoppedReason: 'user_stopped', asset: { sha256: 'b'.repeat(64) },
    measurements: { speakingFrames: 60, firstAudioOffsetMs: 0, lastAudioOffsetMs: 1000 } };
  return { report, reportSha256: hash(canonicalJson(report)) };
}

test('requires intact hashes, identical media and matching sampled windows for comparison', () => {
  const a = audition(); const b = audition();
  assert.equal(compareAuditions(a, b).comparable_conditions, true);
  b.report.measurements.lastAudioOffsetMs = 2000;
  assert.throws(() => compareAuditions(a, b), /changed/);
  b.reportSha256 = hash(canonicalJson(b.report));
  const compared = compareAuditions(a, b);
  assert.equal(compared.comparable_conditions, false); assert.equal(compared.performance, null);
  assert.equal(compared.likeness_or_quality_winner, null);
});
