import { BufferGeometry, Float32BufferAttribute, Group, Mesh, MeshStandardMaterial } from 'three';
import { describe, expect, it } from 'vitest';
import { AUDITION_PROTOCOL, AuditionRecorder, auditionComparable, canonicalJson, effectiveFaceControls,
  inspectAvatarAsset, sealAudition, type AuditionFrame, type AuditionRun } from './audition';
import { createRetargetedRig } from './retarget';

function makeModel(relative = true) {
  const model = new Group(); const geometry = new BufferGeometry();
  geometry.setAttribute('position', new Float32BufferAttribute([0, 0, 0, 1, 0, 0, 0, 2, 0], 3));
  geometry.morphTargetsRelative = relative;
  geometry.morphAttributes.position = [new Float32BufferAttribute(relative ? [0, 0, 0, 0, 0.2, 0, 0, 0, 0] : [0, 0, 0, 1, 0.2, 0, 0, 2, 0], 3),
    new Float32BufferAttribute(relative ? Array(9).fill(0) : [0, 0, 0, 1, 0, 0, 0, 2, 0], 3)];
  const mesh = new Mesh(geometry, new MeshStandardMaterial({ name: 'Skin' })); mesh.name = 'Face';
  mesh.morphTargetDictionary = { jawOpen: 0, eyeBlinkLeft: 1, eyeBlinkRight: 2 };
  mesh.morphTargetInfluences = [0, 0, 0]; model.add(mesh);
  return { model, mesh };
}

const hashA = 'a'.repeat(64); const hashB = 'b'.repeat(64);
function base(): Omit<AuditionRun, 'measurements' | 'samples' | 'unmeasured' | 'stoppedReason'> {
  const inspection = inspectAvatarAsset(makeModel().model); inspection.sha256 = hashB;
  return { schema_version: 'avatar-audition-run.v1', protocol: AUDITION_PROTOCOL,
    asset: inspection, resolvedMapping: {}, resolvedTargets: [], resolvedBones: [], audioSha256: hashA, speechTimingSha256: null,
    camera: { framing: 'portrait', position: [0, 2, 2], target: [0, 2, 0], fov: 34, viewport: [640, 480], pixelRatio: 1 },
    settings: { reducedMotion: false, geometryPreview: false, sourceClipSha256: null }, startedAt: '2026-11-11T00:00:00Z' };
}
const frame = (monotonicMs: number, audioMs: number | null = monotonicMs - 1_000): AuditionFrame => ({
  monotonicMs, audioMs, speaking: true, drawCalls: 7, triangles: 10_000, geometryCount: 7, textureCount: 4,
  telemetry: { faceSource: 'designed', boneSource: 'designed', speechSource: 'audio_recognition', unsupportedFace: [], unsupportedVisemes: [] },
});
function run() {
  const recorder = new AuditionRecorder(base(), 1_000);
  for (let n = 0; n < 60; n++) recorder.add(frame(1_000 + n * 1000 / 60), hashA);
  recorder.stop('user_stopped', 2_000); return recorder.report();
}

describe('actual asset qualification', () => {
  it.each([true, false])('measures relative=%s target deltas, retaining zero/missing targets as such', (relative) => {
    const { model } = makeModel(relative); const inspection = inspectAvatarAsset(model);
    expect(inspection.morphs[0]).toMatchObject({ target: 'jawOpen', movedVertices: 1, status: 'deforms' });
    expect(inspection.morphs[0].maxDisplacement).toBeCloseTo(0.2);
    expect(inspection.morphs[1]).toMatchObject({ status: 'zero', movedVertices: 0, maxDisplacement: 0 });
    expect(inspection.morphs[2]).toMatchObject({ status: 'missing_position', movedVertices: null, maxDisplacement: null });
    expect(inspection.review.likeness).toBe('ungraded');
  });

  it('keeps nominal target mapping distinct from verified position deformation', () => {
    const { model } = makeModel(); const rig = createRetargetedRig(model);
    expect(rig.diagnostics.morphs).toContain('eyeBlinkLeft');
    expect(effectiveFaceControls(rig.diagnostics)).toEqual(['jawOpen']);
    expect(rig.diagnostics.resolvedTargets).toEqual(expect.arrayContaining([expect.objectContaining({ control: 'jawOpen', kind: 'face', mesh: 'Face', target: 'jawOpen' })]));
    rig.dispose();
  });

  it('marks non-finite displacement invalid instead of passing an advertised channel', () => {
    const { model, mesh } = makeModel(); mesh.geometry.morphAttributes.position![0].setX(1, Number.NaN);
    expect(inspectAvatarAsset(model).morphs[0]).toMatchObject({ status: 'invalid', maxDisplacement: null });
  });
});

describe('same-audio audition evidence', () => {
  it('records actual frames and separate audio offsets without assigning human judgments or acoustic latency', () => {
    const report = run();
    expect(report.measurements).toMatchObject({ durationMs: 1_000, renderedFrames: 60, speakingFrames: 60, firstAudioClockObservedMs: 0, firstAudioOffsetMs: 0 });
    expect(report.measurements.frameIntervalP95Ms).toBeCloseTo(1000 / 60);
    expect(report.samples.length).toBeLessThanOrEqual(10);
    expect(report.unmeasured).toContain('acoustic loudspeaker latency');
    expect(report.asset.review.likeness).toBe('ungraded');
  });

  it('stops before admitting frames from a different audio file and bounds recording duration', () => {
    const recorder = new AuditionRecorder(base(), 1_000); recorder.add(frame(1_000), hashA);
    recorder.add(frame(1_016), hashB);
    expect(recorder.report().stoppedReason).toBe('audio_changed');
    expect(recorder.report().measurements.renderedFrames).toBe(1);
    const bounded = new AuditionRecorder(base(), 1_000); bounded.add(frame(181_001), hashA);
    expect(bounded.report().stoppedReason).toBe('recording_limit');
    expect(bounded.report().measurements.durationMs).toBe(180_000);
  });

  it('rejects seek-contaminated or mismatched framing, speech timing, and sampled audio windows', () => {
    const first = run(); const second = run();
    expect(auditionComparable(first, second).comparable).toBe(true);
    second.camera.fov = 40; second.speechTimingSha256 = hashB; second.measurements.firstAudioOffsetMs = 500;
    expect(auditionComparable(first, second).reasons).toContain('camera/viewport differs');
    expect(auditionComparable(first, second).reasons).toContain('speech timing differs');
    const recorder = new AuditionRecorder(base(), 1_000);
    recorder.add(frame(1_000, 200), hashA); recorder.add(frame(1_020, 100), hashA);
    expect(recorder.report().stoppedReason).toBe('audio_seek');
    expect(auditionComparable(first, recorder.report()).comparable).toBe(false);
  });

  it('binds every report field to the digest and rejects invalid data', async () => {
    const first = run(); const sealed = await sealAudition(first);
    const changed = structuredClone(first); changed.resolvedMapping = { yaw: 1 };
    expect((await sealAudition(changed)).reportSha256).not.toBe(sealed.reportSha256);
    expect(canonicalJson({ z: 1, a: [2, 3] })).toBe(canonicalJson({ a: [2, 3], z: 1 }));
    expect(() => canonicalJson({ nan: Number.NaN })).toThrow();
    expect(() => new AuditionRecorder({ ...base(), audioSha256: 'unknown' }, 0)).toThrow();
    expect(() => new AuditionRecorder({ ...base(), asset: { ...base().asset, sha256: null } }, 0)).toThrow();
  });
});
