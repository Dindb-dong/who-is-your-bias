import { PerspectiveCamera, Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import { avatarView } from './builtin';
import { FINGER_CHANNELS } from './channels';
import { sampleMotion } from './motion';
import { parseAvatarClip } from './validation';

const clip = {
  schema_version: 'avatar-clip.v1', duration_ms: 1000, max_gap_ms: 200,
  source: { source_id: 'fixture', source_timeline: 'video', source_start_seconds: 2,
    video_sha256: 'a'.repeat(64), observations_sha256: 'b'.repeat(64) },
  review: { status: 'geometry_preview', reviewer: 'test', track_id: 'one', track_continuity_reviewed: false,
    source_clock_reviewed: false, speaker_association_reviewed: false, attribution_basis: 'test geometry only' },
  calibration: { neutral_frame_ms: 0, coordinate_basis: 'right_handed_y_up', bone_rotation_space: 'local_delta',
    face_neutral: {}, bone_neutral: {} }, limitations: ['Synthetic geometry test; no collected source.'],
  frames: [{ time_ms: 0, bones: { leftIndexProximal: [0, 0, 0, 1] } },
    { time_ms: 100, bones: { leftIndexProximal: [Math.sin(0.3), 0, 0, Math.cos(0.3)] } }],
};

describe('hand performance contract', () => {
  it('preserves v1 clips while accepting 30 independent finger joints', () => {
    expect(new Set(FINGER_CHANNELS).size).toBe(30);
    const parsed = parseAvatarClip(clip, true);
    const sample = sampleMotion(parsed, 50, true)!;
    expect(sample.bones!.leftIndexProximal![0]).toBeCloseTo(Math.sin(0.15));
    expect(sample.bones!.rightIndexProximal).toBeUndefined();
    expect(sampleMotion(parsed, 500, true)).toBeNull();
    expect(() => parseAvatarClip(clip)).toThrow(/preview/);
    expect(() => parseAvatarClip({ ...clip, frames: [{ time_ms: 0, bones: { leftIndexProximal: [0, 0, 0, 0] } }] }, true)).toThrow();
    expect(() => parseAvatarClip({ ...clip, frames: [{ time_ms: 0, bones: { leftExtraFinger: [0, 0, 0, 1] } }] }, true)).toThrow();
  });

  it('retains the exact optional hand-model provenance without requiring it in older clips', () => {
    const measurement = { engine: 'mediapipe', version: '0.10.21', face_model_sha256: 'c'.repeat(64), hand_model_sha256: 'd'.repeat(64) };
    expect(parseAvatarClip({ ...clip, measurement }, true).measurement).toEqual(measurement);
    expect(() => parseAvatarClip({ ...clip, measurement: { ...measurement, hand_model_sha256: 'invalid' } }, true)).toThrow();
    expect(parseAvatarClip(clip, true).measurement).toBeUndefined();
  });

  it.each([0.48, 0.75, 1.0, 1.6])('keeps the declared hand envelope in frame at aspect %s', (aspect) => {
    const view = avatarView('glb', 'upper_body', aspect);
    const camera = new PerspectiveCamera(view.fov, aspect, 0.1, 30);
    camera.position.fromArray(view.position); camera.lookAt(new Vector3().fromArray(view.target)); camera.updateMatrixWorld();
    for (const x of [-1.15, 1.15]) for (const y of [0.8, 2.4]) {
      const screen = new Vector3(x, y, 0.1).project(camera);
      expect(Math.abs(screen.x)).toBeLessThan(0.96);
      expect(Math.abs(screen.y)).toBeLessThan(0.96);
    }
  });
});
