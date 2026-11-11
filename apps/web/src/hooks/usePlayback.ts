import { useCallback, useEffect, useRef, useState } from 'react';
import type { Turn } from '../api';
import { browserPlaybackEnvironment, createPlaybackController, initialPlayback } from './playback';
import type { PlaybackController, PlaybackState, VoiceMode } from './playback';
import type { PlaybackEvent } from './liveMetrics';

export { planTime } from './playback';
export type { VoiceMode } from './playback';

export function usePlayback() {
  const [state, setState] = useState<PlaybackState>(initialPlayback);
  const controller = useRef<PlaybackController | null>(null);
  useEffect(() => {
    // Each effect mount owns its controller, including StrictMode's remount.
    const instance = createPlaybackController(setState, browserPlaybackEnvironment());
    controller.current = instance;
    return () => { instance.dispose(); if (controller.current === instance) controller.current = null; };
  }, []);
  const stop = useCallback(() => controller.current?.stop(), []);
  const play = useCallback((turn: Turn, mode: VoiceMode, observe?: (event: PlaybackEvent) => void) => controller.current?.play(turn, mode, observe) ?? Promise.resolve(), []);
  return { ...state, play, stop };
}

