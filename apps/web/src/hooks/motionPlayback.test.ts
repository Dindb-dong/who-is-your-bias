import { afterEach, describe, expect, it, vi } from 'vitest';
import { motionFixture } from '../test/fixtures/motion';
import { createMotionPlaybackController } from './motionPlayback';
import type { MotionPlaybackEnvironment, MotionPlaybackState } from './motionPlayback';
import type { AvatarClip } from '../avatar/types';

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((yes) => { resolve = yes; });
  return { resolve, promise };
}

function setup() {
  const fixture = motionFixture(); const states: MotionPlaybackState[] = [];
  let clock = 0; let allowed = true; let frameId = 0;
  const frames = new Map<number, FrameRequestCallback>();
  const load = vi.fn<MotionPlaybackEnvironment['load']>(async () => fixture.clip);
  const environment: MotionPlaybackEnvironment = {
    load, canPlay: () => allowed, now: () => clock,
    requestFrame(callback) { frames.set(++frameId, callback); return frameId; },
    cancelFrame(id) { frames.delete(id); }, loadTimeoutMs: 50,
  };
  const controller = createMotionPlaybackController((state) => states.push(state), environment);
  return { ...fixture, states, frames, load, controller,
    tick(time: number) { clock = time; const pending = [...frames.values()]; frames.clear(); pending.forEach((callback) => callback(time)); },
    setAllowed(value: boolean) { allowed = value; },
  };
}

afterEach(() => { vi.useRealTimers(); });

describe('one-shot measured motion ownership', () => {
  it('uses the original source clock, ends once, and clears measured channels', async () => {
    const f = setup(); await f.controller.play(f.selection, 'geometry_preview');
    expect(f.states.at(-1)?.elapsedMs).toBe(0);
    f.tick(420); expect(f.states.at(-1)?.elapsedMs).toBe(420);
    f.tick(1000); expect(f.states.at(-1)?.status).toBe('completed');
    expect(f.states.at(-1)?.clip).toBeUndefined(); expect(f.frames.size).toBe(0);
    const count = f.states.length; f.tick(5000); expect(f.states).toHaveLength(count); f.controller.dispose();
  });

  it('does not fetch while disabled, reduced, hidden, or behind the review gate', async () => {
    const f = setup();
    await f.controller.play(f.selection, 'disabled');
    await f.controller.play(f.selection, 'reviewed');
    f.setAllowed(false); await f.controller.play(f.selection, 'geometry_preview');
    expect(f.load).not.toHaveBeenCalled(); f.controller.dispose();
  });

  it('stops on visibility/reduced-motion gating and never resumes itself', async () => {
    const f = setup(); await f.controller.play(f.selection, 'geometry_preview');
    f.setAllowed(false); f.tick(100);
    expect(f.states.at(-1)?.status).toBe('idle'); expect(f.frames.size).toBe(0);
    f.setAllowed(true); f.tick(200); expect(f.states.at(-1)?.status).toBe('idle'); f.controller.dispose();
  });

  it('discards a fetch that succeeds after cancellation', async () => {
    const f = setup(); const pending = deferred<AvatarClip>(); f.load.mockReturnValue(pending.promise);
    const work = f.controller.play(f.selection, 'geometry_preview');
    const signal = f.load.mock.calls[0][2]; f.controller.stop(); expect(signal.aborted).toBe(true);
    pending.resolve(f.clip); await work;
    expect(f.states.at(-1)?.status).toBe('idle'); expect(f.frames.size).toBe(0); f.controller.dispose();
  });

  it('an older request cannot replace a newer replay even if it ignores abort', async () => {
    const f = setup(); const pending = deferred<AvatarClip>();
    f.load.mockReturnValueOnce(pending.promise).mockResolvedValueOnce(f.clip);
    const old = f.controller.play(f.selection, 'geometry_preview');
    await f.controller.play(f.selection, 'geometry_preview');
    f.tick(250); const count = f.states.length;
    pending.resolve(f.clip); await old;
    expect(f.states).toHaveLength(count); expect(f.states.at(-1)?.elapsedMs).toBe(250); f.controller.dispose();
  });

  it('rechecks the gate after loading and does not animate a late opt-out', async () => {
    const f = setup(); const pending = deferred<AvatarClip>(); f.load.mockReturnValueOnce(pending.promise);
    const work = f.controller.play(f.selection, 'geometry_preview'); f.setAllowed(false);
    pending.resolve(f.clip); await work;
    expect(f.states.at(-1)?.status).toBe('idle'); expect(f.frames.size).toBe(0); f.controller.dispose();
  });

  it('timeouts abort loading and cannot be revived by a late response', async () => {
    vi.useFakeTimers(); const f = setup(); const pending = deferred<AvatarClip>(); f.load.mockReturnValueOnce(pending.promise);
    const work = f.controller.play(f.selection, 'geometry_preview');
    vi.advanceTimersByTime(51);
    expect(f.states.at(-1)?.error).toContain('timed out'); expect(f.load.mock.calls[0][2].aborted).toBe(true);
    const count = f.states.length; pending.resolve(f.clip); await work;
    expect(f.states).toHaveLength(count); expect(f.frames.size).toBe(0); f.controller.dispose();
  });

  it('disposal cancels all work without publishing state after unmount', async () => {
    const f = setup(); const pending = deferred<AvatarClip>(); f.load.mockReturnValueOnce(pending.promise);
    const work = f.controller.play(f.selection, 'geometry_preview'); f.controller.dispose();
    const count = f.states.length; pending.resolve(f.clip); await work;
    expect(f.states).toHaveLength(count); expect(f.frames.size).toBe(0);
  });
});
