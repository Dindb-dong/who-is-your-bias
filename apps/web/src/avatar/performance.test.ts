import { Quaternion, Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import { FACE_CHANNELS } from './channels';
import { interpolateRotation, sampleMotion } from './motion';
import { samplePerformance } from './performance';
import { labelToViseme, sampleSpeech } from './speech';
import { dampPose, neutralPose } from './timeline';
import type { AvatarClip, QuaternionTuple, RigDiagnostics, SpeechTiming } from './types';
import { parseAvatarClip, parseRigMapping, parseSpeechTiming } from './validation';

const hash = 'a'.repeat(64);
const rotation = (angle: number): QuaternionTuple => new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), angle).toArray() as QuaternionTuple;
const diagnostics: RigDiagnostics = { kind: 'glb', morphs: ['jawOpen', 'mouthClose', 'eyeBlinkLeft'], bones: ['head'], visemes: ['aa', 'PP'], missing: [] };

export const timingFixture = (): SpeechTiming => ({
  schema_version: 'speech-timing.v1', audio_sha256: hash, audio_duration_ms: 1000,
  timing_basis: 'audio_recognition', label_set: 'rhubarb', provenance: { engine: 'synthetic-test-fixture' },
  segments: [{ start_ms: 0, end_ms: 500, label: 'A' }, { start_ms: 500, end_ms: 1000, label: 'D' }],
});

export const clipFixture = (): AvatarClip => ({
  schema_version: 'avatar-clip.v1', duration_ms: 1000, max_gap_ms: 150,
  source: { source_id: 'synthetic', source_timeline: 'fixture', source_start_seconds: 20, video_sha256: hash, observations_sha256: hash },
  review: { status: 'speaker_reviewed', reviewer: 'test fixture', track_id: 'one', track_continuity_reviewed: true, source_clock_reviewed: true, speaker_association_reviewed: true, attribution_basis: 'Synthetic data only' },
  calibration: { neutral_frame_ms: 0, face_neutral: { jawOpen: 0.2 }, coordinate_basis: 'right_handed_y_up', bone_rotation_space: 'local_delta' },
  frames: [{ time_ms: 0, face: { jawOpen: 0.2, eyeBlinkLeft: 0 }, bones: { head: rotation(0) } },
    { time_ms: 100, face: { jawOpen: 0.6, eyeBlinkLeft: 1 }, bones: { head: rotation(0.4) } },
    { time_ms: 900, face: { jawOpen: 1 }, bones: { head: rotation(0) } }], limitations: ['Synthetic fixture'],
});

describe('measured geometry and review boundaries', () => {
  it('interpolates observations with neutral calibration but never bridges a detection gap', () => {
    const clip = parseAvatarClip(clipFixture());
    const sample = sampleMotion(clip, 50)!;
    expect(sample.face?.jawOpen).toBeCloseTo(0.25);
    expect(sample.face?.eyeBlinkLeft).toBeCloseTo(0.5);
    expect(new Quaternion(...sample.bones!.head!).angleTo(new Quaternion())).toBeCloseTo(0.2);
    expect(sampleMotion(clip, 200)).toBeNull();
    expect(sampleMotion(clip, 1100)).toBeNull();
  });

  it('requires separate preview opt-in and all review fields for speaker replay', () => {
    const clip = clipFixture(); clip.review.status = 'geometry_preview'; clip.review.speaker_association_reviewed = false;
    expect(() => parseAvatarClip(clip)).toThrow(/preview/);
    expect(parseAvatarClip(clip, true).review.status).toBe('geometry_preview');
    expect(sampleMotion(clip, 50)).toBeNull();
    expect(sampleMotion(clip, 50, true)).not.toBeNull();
    clip.review.status = 'speaker_reviewed';
    expect(() => parseAvatarClip(clip, true)).toThrow(/requires reviewed/);
  });

  it('rejects invalid weights, duplicate timestamps, forged rotations and unknown channels', () => {
    const negative = clipFixture(); negative.frames[0].face!.jawOpen = -1;
    expect(() => parseAvatarClip(negative)).toThrow();
    const duplicate = clipFixture(); duplicate.frames[1].time_ms = 0;
    expect(() => parseAvatarClip(duplicate)).toThrow(/increase/);
    const invalid = clipFixture(); invalid.frames[0].bones!.head = [0, 0, 0, 0];
    expect(() => parseAvatarClip(invalid)).toThrow(/unit quaternion/);
    expect(() => parseRigMapping({ morphs: { fabricatedEmotion: 'Happy' } })).toThrow(/Unsupported/);
    expect(FACE_CHANNELS).toHaveLength(52);
  });

  it('uses shortest-path quaternion interpolation, including antipodal encodings', () => {
    const q = rotation(Math.PI - 0.05); const negative = q.map((value) => -value) as QuaternionTuple;
    const midpoint = interpolateRotation(q, negative, 0.5);
    expect(Math.abs(new Quaternion(...midpoint).dot(new Quaternion(...q)))).toBeCloseTo(1);
    const nearPi = interpolateRotation(rotation(3.0), rotation(-3.0), 0.5);
    expect(Math.abs(new Quaternion(...nearPi).w)).toBeLessThan(0.01);
  });

  it('reports face and joint provenance separately when a clip contains only body measurements', () => {
    const clip = clipFixture(); clip.frames.forEach((frame) => { delete frame.face; });
    const observed = samplePerformance({ speaking: false, elapsedMs: 50, idleMs: 0, performanceClip: clip }, diagnostics);
    expect(observed.pose.face).toBeUndefined();
    expect(observed.pose.bones?.head).toBeDefined();
    expect(observed.telemetry.faceSource).toBe('designed');
    expect(observed.telemetry.boneSource).toBe('observed');
    clip.review.status = 'geometry_preview'; clip.review.speaker_association_reviewed = false;
    const preview = samplePerformance({ speaking: false, elapsedMs: 50, idleMs: 0, performanceClip: clip, allowGeometryPreview: true }, diagnostics);
    expect(preview.telemetry.faceSource).toBe('designed');
    expect(preview.telemetry.boneSource).toBe('geometry_preview');
  });
});

describe('audio-bound speech articulation', () => {
  it('uses the audio clock even when the estimated behavior plan has another duration', () => {
    const result = samplePerformance({ speaking: true, elapsedMs: 100, audioElapsedMs: 750, audioSha256: hash,
      idleMs: 0, amplitude: 0.01, speechTiming: timingFixture() }, diagnostics);
    expect(result.pose.face?.jawOpen).toBeGreaterThan(0.6);
    expect(result.pose.visemes?.aa).toBe(1);
    expect(result.telemetry.speechSource).toBe('audio_recognition');
    const mismatched = samplePerformance({ speaking: true, elapsedMs: 750, audioElapsedMs: 750, audioSha256: 'b'.repeat(64),
      idleMs: 0, amplitude: 0.2, speechTiming: timingFixture() }, diagnostics);
    expect(mismatched.pose.visemes).toBeUndefined();
    expect(mismatched.telemetry.speechSource).toBe('amplitude');
    expect(mismatched.telemetry.warning).toMatch(/matching audio/);
  });

  it('distinguishes amplitude, estimated, silent and observed timing sources', () => {
    expect(samplePerformance({ speaking: true, elapsedMs: 100, idleMs: 0 }, diagnostics).telemetry.speechSource).toBe('estimated');
    expect(samplePerformance({ speaking: false, elapsedMs: 100, idleMs: 0, speechTiming: timingFixture() }, diagnostics).pose.visemes).toBeUndefined();
    expect(sampleSpeech(timingFixture(), 1001, hash)?.visemes).toEqual({ sil: 1 });
  });

  it('maps actual label vocabularies and rejects overlapping timing imports', () => {
    expect(labelToViseme('G', 'rhubarb')).toBe('FF');
    expect(labelToViseme('AA1', 'arpabet')).toBe('aa');
    expect(labelToViseme('θ', 'ipa')).toBe('TH');
    expect(labelToViseme('UNKNOWN', 'ipa')).toBeUndefined();
    const overlap = timingFixture(); overlap.segments[1].start_ms = 450;
    expect(() => parseSpeechTiming(overlap)).toThrow();
    const malformed = timingFixture(); malformed.audio_sha256 = 'not-a-hash';
    expect(() => parseSpeechTiming(malformed)).toThrow(/SHA-256/);
    const empty = timingFixture(); empty.segments = [];
    expect(() => parseSpeechTiming(empty)).toThrow(/1–10000/);
    const unknown = timingFixture(); unknown.segments[0].label = 'not-a-mouth-cue';
    expect(() => parseSpeechTiming(unknown)).toThrow(/Rhubarb/);
    unknown.label_set = 'viseme_15';
    expect(() => parseSpeechTiming(unknown)).toThrow(/unsupported/);
  });

  it('keeps the amplitude fallback when an active imported phoneme has no supported articulation', () => {
    const unknown = timingFixture(); unknown.label_set = 'ipa'; unknown.segments[0].label = 'unsupported';
    const parsed = parseSpeechTiming(unknown);
    const sampled = samplePerformance({ speaking: true, elapsedMs: 200, audioElapsedMs: 200, audioSha256: hash,
      idleMs: 0, amplitude: 0.4, speechTiming: parsed }, diagnostics);
    expect(sampled.telemetry.speechSource).toBe('amplitude');
    expect(sampled.pose.visemes).toBeUndefined();
    expect(sampled.pose.jawOpen).toBeCloseTo(0.34);
    expect(sampled.telemetry.warning).toMatch(/Unsupported speech labels/);
  });

  it('smoothly releases observed rotations and native visemes so designed controls can resume', () => {
    let pose = { ...neutralPose(), bones: { head: rotation(0.8) }, face: { jawOpen: 1 }, visemes: { aa: 1 } };
    for (let frame = 0; frame < 120; frame++) pose = dampPose(pose, neutralPose(), 1 / 60) as typeof pose;
    expect(pose.bones).toBeUndefined(); expect(pose.visemes).toBeUndefined(); expect(pose.face).toBeUndefined();
  });
});
