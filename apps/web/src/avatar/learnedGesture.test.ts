import { describe, expect, it } from 'vitest';
import { neutralPose } from './timeline';
import { samplePerformance } from './performance';
import { parseGestureBasis, sampleConversationalGesture, sampleLearnedGesture, type GestureBasisGroup, type GestureBasisModel } from './learnedGesture';
import type { RigDiagnostics, SpeechTiming } from './types';

function model(): GestureBasisModel {
  const group = (side: 'left' | 'right'): GestureBasisGroup => ({
    channels: [`${side}UpperArm`, `${side}Forearm`, `${side}Hand`], rank: 1,
    mean: Array(36).fill(.1), basis: [Array(36).fill(.2)], latent_examples: [[1], [-1]],
    latent_ids: ['event:0', 'event:2000'], training_source_events: ['event'],
    support_counts: Array(36).fill(2), max_angle_rad: .4, iterations_completed: 60,
  });
  return { schema_version: 'gesture-basis.v1', persona_id: 'jensen-huang', review_status: 'geometry_preview',
    coordinate_basis: 'right_handed_y_up', rotation_space: 'local_delta', flattening: 'time_channel_xyz',
    id: 'synthetic-contract-fixture', dataset_sha256: 'a'.repeat(64), sample_count: 4, duration_ms: 2000,
    groups: { left_arm: group('left'), right_arm: group('right') }, limitations: ['Synthetic test fixture only.'] };
}
const diagnostics: RigDiagnostics = { kind: 'glb', morphs: [], bones: [], missing: [] };

describe('learned gesture runtime', () => {
  it('rejects wrong subjects, review promotion, bad coordinates and invalid arrays', () => {
    expect(parseGestureBasis(model()).groups.left_arm.rank).toBe(1);
    for (const change of [{ persona_id: 'elon-musk' }, { review_status: 'speaker_reviewed' }, { rotation_space: 'world' }]) {
      expect(() => parseGestureBasis({ ...model(), ...change })).toThrow();
    }
    const invalid = model(); invalid.groups.left_arm.basis[0][0] = NaN;
    expect(() => parseGestureBasis(invalid)).toThrow();
  });

  it('coordinates both hands by shared source-window IDs and reproduces seeks', () => {
    const fitted = model();
    fitted.groups.right_arm.latent_ids.reverse(); fitted.groups.right_arm.latent_examples.reverse();
    const a = sampleLearnedGesture(fitted, 1000)!;
    expect(a.leftUpperArm).toEqual(a.rightUpperArm);
    expect(sampleLearnedGesture(fitted, 3500)!.leftUpperArm![0]).toBeLessThan(0);
    expect(sampleLearnedGesture(fitted, 1000)).toEqual(a);
    expect(Math.hypot(...a.leftUpperArm!)).toBeCloseTo(1);
    expect(2 * Math.acos(a.leftUpperArm![3])).toBeLessThanOrEqual(.400001);
  });

  it('does not invent unsupported joints or join unmatched left/right examples', () => {
    const fitted = model(); fitted.groups.left_arm.support_counts.fill(0, 0, 36);
    expect(sampleLearnedGesture(fitted, 1000)!.leftUpperArm).toBeUndefined();
    fitted.groups.right_arm.latent_ids = ['other:0', 'other:2000'];
    expect(sampleLearnedGesture(fitted, 1000)).toBeNull();
  });

  it('returns neutral endpoints and rests without jumping between examples', () => {
    const fitted = model();
    expect(sampleLearnedGesture(fitted, 0)!.leftUpperArm).toEqual([0, 0, 0, 1]);
    expect(sampleLearnedGesture(fitted, 1999)!.leftUpperArm![3]).toBeCloseTo(1, 7);
    expect(sampleLearnedGesture(fitted, 2200)).toBeNull();
    expect(sampleLearnedGesture(fitted, 2500)!.leftUpperArm).toEqual([0, 0, 0, 1]);
    expect(sampleLearnedGesture(fitted, NaN)).toBeNull();
  });

  it('limits a reply to two soft gestures and leaves long rests between them', () => {
    const fitted = model();
    const cues = [2000, 3000, 11000, 15000, 19000].map((start_ms) => ({
      start_ms, duration_ms: 1800, channel: 'gesture' as const, name: 'emphasis', intensity: .7, provenance: 'designed' as const,
    }));
    expect(sampleConversationalGesture(fitted, 500, undefined, cues)).toBeNull();
    const active = sampleConversationalGesture(fitted, 2700, undefined, cues)!;
    expect(active.leftUpperArm).toEqual(active.rightUpperArm);
    expect(2 * Math.acos(active.leftUpperArm![3])).toBeLessThan(.23);
    for (const time of [5000, 8000, 15000, 20000]) expect(sampleConversationalGesture(fitted, time, undefined, cues)).toBeNull();
    expect(sampleConversationalGesture(fitted, 12000, undefined, cues)).not.toBeNull();
    expect(sampleConversationalGesture(fitted, 2700, undefined, cues)).toEqual(active);
  });

  it('requires explicit geometry preview and follows the actual audio clock', () => {
    const input = { ...neutralPose(), elapsedMs: 500, audioElapsedMs: 3100, idleMs: 0, speaking: true, amplitude: .2, learnedGesture: model() };
    expect(samplePerformance(input, diagnostics).telemetry.boneSource).toBe('designed');
    const active = samplePerformance({ ...input, allowGeometryPreview: true }, diagnostics);
    expect(active.telemetry.boneSource).toBe('trained_geometry_preview');
    expect(active.pose.bones?.leftUpperArm?.[0]).toBeGreaterThan(0);
    for (const settings of [{ speaking: false }, { reducedMotion: true }]) {
      expect(samplePerformance({ ...input, allowGeometryPreview: true, ...settings }, diagnostics).telemetry.boneSource).toBe('designed');
    }
  });

  it('preserves designed cues until a learned gesture fits and is playing', () => {
    const timing = (duration: number): SpeechTiming => ({
      schema_version: 'speech-timing.v1', audio_sha256: 'b'.repeat(64), audio_duration_ms: duration,
      timing_basis: 'manual', label_set: 'rhubarb', provenance: { engine: 'synthetic-fixture' },
      segments: [{ start_ms: 0, end_ms: duration, label: 'D' }],
    });
    const input = {
      speaking: true, elapsedMs: 3000, audioElapsedMs: 3000, idleMs: 0, audioSha256: 'b'.repeat(64),
      allowGeometryPreview: true, learnedGesture: { ...model(), duration_ms: 5000 },
      plan: { duration_ms: 12000, cues: [{ channel: 'gesture' as const, name: 'open_palm', start_ms: 0, duration_ms: 8000,
        intensity: .5, provenance: 'designed' as const }] },
    };

    const short = samplePerformance({ ...input, speechTiming: timing(8000) }, diagnostics);
    expect(short.telemetry.boneSource).toBe('designed');
    expect(short.pose.openPalm).toBe(.5);

    const long = { ...input, speechTiming: timing(12000) };
    const before = samplePerformance({ ...long, elapsedMs: 1000, audioElapsedMs: 1000 }, diagnostics);
    expect(before.telemetry.boneSource).toBe('designed');
    expect(before.pose.openPalm).toBe(.5);

    const active = samplePerformance(long, diagnostics);
    expect(active.telemetry.boneSource).toBe('trained_geometry_preview');
    expect(active.pose.openPalm).toBe(0);
    expect(active.pose.bones?.leftUpperArm?.[0]).toBeGreaterThan(0);
  });
});
