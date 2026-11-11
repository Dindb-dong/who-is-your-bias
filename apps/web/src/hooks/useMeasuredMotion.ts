import { useCallback, useEffect, useRef, useState } from 'react';
import { getMotionClip } from '../api';
import type { MotionMode, MotionSelection } from '../avatar/types';
import { createMotionPlaybackController, initialMotion } from './motionPlayback';
import type { MotionPlaybackController, MotionPlaybackState } from './motionPlayback';

export function useMeasuredMotion(options: { mode: MotionMode; enabled: boolean; reducedMotion: boolean }) {
  const [state, setState] = useState<MotionPlaybackState>(initialMotion);
  const latest = useRef(options); latest.current = options;
  const controller = useRef<MotionPlaybackController | null>(null);
  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const instance = createMotionPlaybackController(setState, {
      load: getMotionClip,
      canPlay: (mode) => latest.current.enabled && latest.current.mode === mode && mode !== 'disabled'
        && !latest.current.reducedMotion && !media.matches && document.visibilityState === 'visible',
      now: () => performance.now(),
      requestFrame: (callback) => requestAnimationFrame(callback), cancelFrame: (id) => cancelAnimationFrame(id),
    });
    controller.current = instance;
    const hide = () => { if (document.visibilityState !== 'visible') instance.stop(); };
    const reduce = () => { if (media.matches) instance.stop(); };
    document.addEventListener('visibilitychange', hide);
    media.addEventListener('change', reduce);
    return () => {
      document.removeEventListener('visibilitychange', hide);
      media.removeEventListener('change', reduce);
      instance.dispose();
      if (controller.current === instance) controller.current = null;
    };
  }, []);
  useEffect(() => { controller.current?.stop(); }, [options.mode, options.enabled, options.reducedMotion]);
  const stop = useCallback(() => controller.current?.stop(), []);
  const play = useCallback((selection: MotionSelection | null | undefined) => controller.current?.play(selection, latest.current.mode) ?? Promise.resolve(), []);
  return { ...state, play, stop };
}
