import { describe, expect, it } from 'vitest';
import { containRect, displayBox, FaceTracker, TRACK_LOSS_MS, validFaceBox } from './tracker';
import type { FaceBox } from './types';
const face = (x: number, y = 0.2, width = 0.16): FaceBox => ({ x, y, width, height: 0.24, score: 0.9 });

describe('participant tracking without identity inference', () => {
  it('does not choose a participant until an explicit selection', () => {
    const tracker = new FaceTracker();
    const frame = tracker.update([face(0.1), face(0.65)], 10);
    expect(frame.tracks).toHaveLength(2); expect(frame.selected).toBeNull();
    expect(tracker.select(frame.tracks[1].id, 11).selected?.box.x).toBe(0.65);
  });

  it('retains the selected track through detector-order changes and bounded motion', () => {
    const tracker = new FaceTracker(); const first = tracker.update([face(0.1), face(0.65)], 10);
    const id = first.tracks[0].id; tracker.select(id, 11);
    const next = tracker.update([face(0.63), face(0.12)], 110);
    expect(next.selected?.id).toBe(id); expect(next.selected?.box.x).toBe(0.12);
    expect(next.selected?.observedAtMs).toBe(110);
  });

  it('holds a briefly missing selection without inventing fresh observations', () => {
    const tracker = new FaceTracker(); const first = tracker.update([face(0.1)], 10); tracker.select(first.tracks[0].id, 11);
    const gap = tracker.update([], 110);
    expect(gap.selected?.missing).toBe(true); expect(gap.selected?.observedAtMs).toBe(10);
    const returned = tracker.update([face(0.13)], 210);
    expect(returned.selected?.id).toBe(first.tracks[0].id); expect(returned.selected?.missing).toBe(false);
  });

  it('expires a stalled video feed and requires reselection after a long absence', () => {
    const tracker = new FaceTracker(); const first = tracker.update([face(0.1)], 10); tracker.select(first.tracks[0].id, 11);
    const expired = tracker.snapshot(10 + TRACK_LOSS_MS + 1);
    expect(expired.selected).toBeNull(); expect(expired.selectionLost).toBe(true);
    const returned = tracker.update([face(0.1)], 10 + TRACK_LOSS_MS + 20);
    expect(returned.selected).toBeNull(); expect(returned.tracks[0].id).not.toBe(first.tracks[0].id);
  });

  it('drops selection during an ambiguous crossing rather than switching people', () => {
    const tracker = new FaceTracker(); const first = tracker.update([face(0.22), face(0.5)], 10); tracker.select(first.tracks[0].id, 11);
    tracker.update([face(0.28), face(0.44)], 110);
    const crossed = tracker.update([face(0.33), face(0.38)], 210);
    expect(crossed.selected).toBeNull(); expect(crossed.selectionLost).toBe(true);
    const apart = tracker.update([face(0.2), face(0.52)], 310);
    expect(apart.selected).toBeNull();
  });

  it('does not allow an out-of-order result to rewind a selected location', () => {
    const tracker = new FaceTracker(); tracker.update([face(0.1)], 10); tracker.select(1, 11);
    tracker.update([face(0.15)], 110);
    const late = tracker.update([face(0.08)], 80);
    expect(late.selected?.box.x).toBe(0.15); expect(late.selected?.observedAtMs).toBe(110);
  });

  it('rejects malformed, low-confidence and degenerate boxes', () => {
    const values = [face(NaN), face(-0.1), face(0.9), { ...face(0.1), score: 0.2 }, { ...face(0.1), width: 0 }];
    expect(values.some(validFaceBox)).toBe(false);
    expect(new FaceTracker().update(values, 10).tracks).toHaveLength(0);
  });

  it('clears selection explicitly without reselecting on the following frame', () => {
    const tracker = new FaceTracker(); tracker.update([face(0.1)], 10); tracker.select(1, 11);
    tracker.select(null, 12); expect(tracker.update([face(0.11)], 110).selected).toBeNull();
  });
});

describe('preview geometry', () => {
  it('mirrors camera boxes once and leaves screen boxes unchanged', () => {
    expect(displayBox(face(0.1), true).x).toBeCloseTo(0.74);
    expect(displayBox(face(0.1), false).x).toBe(0.1);
    expect(displayBox(displayBox(face(0.1), true), true).x).toBeCloseTo(0.1);
  });
  it('positions the overlay on the image content inside letterboxing', () => {
    expect(containRect(640, 480, 640, 360)).toEqual({ x: 80, y: 0, width: 480, height: 360 });
    expect(containRect(1920, 1080, 400, 400)).toEqual({ x: 0, y: 87.5, width: 400, height: 225 });
    expect(containRect(0, 0, 400, 400).width).toBe(0);
  });
});
