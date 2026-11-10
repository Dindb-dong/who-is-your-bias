import { describe, expect, it } from 'vitest';
import { cueWeight, dampPose, neutralPose, samplePose } from './timeline';
import type { BehaviorCue } from './types';

const cue: BehaviorCue = { start_ms: 100, duration_ms: 1000, channel: 'face', name: 'smile', intensity: 0.8, provenance: 'designed' };

describe('behavior timing', () => {
  it('enters and leaves a bounded smooth envelope on the audio clock', () => {
    expect(cueWeight(cue, 99)).toBe(0);
    expect(cueWeight(cue, 100)).toBe(0);
    expect(cueWeight(cue, 190)).toBeCloseTo(0.4);
    expect(cueWeight(cue, 600)).toBe(0.8);
    expect(cueWeight(cue, 1100)).toBe(0);
  });

  it('rejects invalid time ranges and cannot produce NaN from malformed values', () => {
    for (const duration_ms of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) expect(cueWeight({ ...cue, duration_ms }, 500)).toBe(0);
    expect(cueWeight({ ...cue, intensity: Number.NaN }, 500)).toBe(0);
    expect(cueWeight({ ...cue, intensity: 100 }, 500)).toBe(1);
    expect(cueWeight({ ...cue, start_ms: -5 }, 500)).toBe(0);
  });

  it('is deterministic and applies no stale cues after a seek beyond the plan', () => {
    const input = { elapsedMs: 600, idleMs: 1000, speaking: false, plan: { cues: [cue], duration_ms: 1200 } };
    expect(samplePose(input)).toEqual(samplePose(input));
    expect(samplePose(input).mouthSmile).toBeGreaterThan(0);
    expect(samplePose({ ...input, elapsedMs: 1500 }).mouthSmile).toBe(0);
  });

  it('bounds overlapping cues and lets silence close the jaw', () => {
    const pose = samplePose({ elapsedMs: 600, idleMs: 0, speaking: false, amplitude: 1, plan: { duration_ms: 1200, cues: Array(100).fill(cue) } });
    expect(pose.mouthSmile).toBe(1);
    expect(pose.jawOpen).toBe(0);
    expect(samplePose({ elapsedMs: 600, idleMs: 0, speaking: true, amplitude: 0 }).jawOpen).toBe(0);
  });

  it('uses actual amplitude where supplied and keeps reduced-motion idle still', () => {
    expect(samplePose({ elapsedMs: 500, idleMs: 2000, speaking: true, amplitude: 0.6 }).jawOpen).toBeCloseTo(0.51);
    const pose = samplePose({ elapsedMs: 500, idleMs: 4700, speaking: false, reducedMotion: true });
    expect(pose).toEqual(neutralPose());
  });

  it('damps without overshooting or accumulating invalid frame deltas', () => {
    const target = { ...neutralPose(), jawOpen: 1 };
    expect(dampPose(neutralPose(), target, 1 / 60).jawOpen).toBeGreaterThan(0);
    expect(dampPose(neutralPose(), target, 100).jawOpen).toBeLessThan(1);
    expect(dampPose(neutralPose(), target, Number.NaN).jawOpen).toBe(0);
  });
});
