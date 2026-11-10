import { useCallback, useEffect, useRef, useState } from 'react';
import type { Language } from '../api';
import { createMicrophoneController, recognitionConstructor } from './microphone';
import type { MicrophoneController, MicrophoneState } from './microphone';

export function useMicrophone(language: Language, onText: (text: string) => void) {
  const callback = useRef(onText); callback.current = onText;
  const controller = useRef<MicrophoneController | null>(null);
  const [state, setState] = useState<MicrophoneState>({ listening: false, error: null });
  const [supported] = useState(() => Boolean(recognitionConstructor()));
  useEffect(() => {
    const instance = createMicrophoneController(setState, (text) => callback.current(text), recognitionConstructor);
    controller.current = instance;
    return () => { instance.dispose(); if (controller.current === instance) controller.current = null; };
  }, []);
  // A recording keeps its original recognition language. Stop it before the UI
  // switches language rather than allowing late text from the previous capture.
  useEffect(() => { controller.current?.stop(); }, [language]);
  const stop = useCallback(() => controller.current?.stop(), []);
  const toggle = useCallback(() => controller.current?.toggle(language), [language]);
  return { ...state, supported, toggle, stop };
}

