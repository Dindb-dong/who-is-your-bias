import type { AvatarClip, MotionMode, MotionSelection } from '../avatar/types';

export interface MotionPlaybackState {
  status: 'idle' | 'loading' | 'playing' | 'completed' | 'error';
  clip?: AvatarClip;
  selection?: MotionSelection;
  elapsedMs: number;
  error: string | null;
}

export const initialMotion: MotionPlaybackState = { status: 'idle', elapsedMs: 0, error: null };

export interface MotionPlaybackEnvironment {
  load: (selection: MotionSelection, mode: MotionMode, signal: AbortSignal) => Promise<AvatarClip>;
  canPlay: (mode: MotionMode) => boolean;
  now: () => number;
  requestFrame: (callback: FrameRequestCallback) => number;
  cancelFrame: (id: number) => void;
  loadTimeoutMs?: number;
}

export interface MotionPlaybackController {
  play: (selection: MotionSelection | null | undefined, mode: MotionMode) => Promise<void>;
  stop: () => void;
  dispose: () => void;
}

/** One source-time replay, independent of audio duration and response length. */
export function createMotionPlaybackController(emit: (state: MotionPlaybackState) => void, environment: MotionPlaybackEnvironment): MotionPlaybackController {
  let epoch = 0;
  let disposed = false;
  let abort: AbortController | undefined;
  let frame: number | undefined;
  let timeout: ReturnType<typeof setTimeout> | undefined;

  function release() {
    abort?.abort(); abort = undefined;
    if (frame !== undefined) environment.cancelFrame(frame);
    frame = undefined;
    if (timeout !== undefined) clearTimeout(timeout);
    timeout = undefined;
  }
  function stop() {
    epoch += 1; release();
    if (!disposed) emit(initialMotion);
  }
  async function play(selection: MotionSelection | null | undefined, mode: MotionMode) {
    stop();
    if (disposed || !selection || mode === 'disabled' || !environment.canPlay(mode)) return;
    if (selection.review_status === 'geometry_preview' && mode !== 'geometry_preview') {
      emit({ ...initialMotion, status: 'error', error: 'This source geometry requires the explicit preview setting.' });
      return;
    }
    const ticket = epoch;
    const current = () => !disposed && ticket === epoch;
    const controller = new AbortController(); abort = controller;
    emit({ ...initialMotion, status: 'loading', selection });
    timeout = setTimeout(() => {
      if (!current()) return;
      epoch += 1; release();
      emit({ ...initialMotion, status: 'error', selection, error: 'The measured-motion download timed out.' });
    }, environment.loadTimeoutMs ?? 15000);
    try {
      const clip = await environment.load(selection, mode, controller.signal);
      if (!current() || controller.signal.aborted) return;
      if (timeout !== undefined) clearTimeout(timeout);
      timeout = undefined;
      if (!environment.canPlay(mode)) { stop(); return; }
      const started = environment.now();
      if (!Number.isFinite(started)) throw new Error('The motion clock is unavailable.');
      emit({ status: 'playing', clip, selection, elapsedMs: 0, error: null });
      const tick = () => {
        if (!current()) return;
        if (!environment.canPlay(mode)) { stop(); return; }
        const elapsed = environment.now() - started;
        if (!Number.isFinite(elapsed)) { stop(); return; }
        if (elapsed >= clip.duration_ms) {
          epoch += 1; release();
          emit({ status: 'completed', selection, elapsedMs: clip.duration_ms, error: null });
          return;
        }
        emit({ status: 'playing', clip, selection, elapsedMs: Math.max(0, elapsed), error: null });
        frame = environment.requestFrame(tick);
      };
      frame = environment.requestFrame(tick);
    } catch (error) {
      if (!current() || controller.signal.aborted) return;
      epoch += 1; release();
      emit({ ...initialMotion, status: 'error', selection,
        error: error instanceof Error ? error.message : 'The measured-motion clip is unavailable.' });
    }
  }
  return { play, stop, dispose() { disposed = true; epoch += 1; release(); } };
}
