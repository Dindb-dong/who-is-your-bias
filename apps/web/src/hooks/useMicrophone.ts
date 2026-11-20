import { useCallback, useEffect, useRef, useState } from 'react';
import type { Language } from '../api';
import { createMicrophoneController, recognitionConstructor } from './microphone';
import type { MicrophoneController, MicrophoneState, OnQuestion } from './microphone';
import { createRecordedQuestionController, recordingMimeType } from './recordedQuestion';

export function useMicrophone(language: Language, onText: (text: string) => void,
  onQuestion?: OnQuestion, serverTranscription = false) {
  const callback = useRef(onText); callback.current = onText;
  const question = useRef(onQuestion); question.current = onQuestion;
  const controller = useRef<MicrophoneController | null>(null);
  const [state, setState] = useState<MicrophoneState>({ listening: false, error: null });
  const autoSubmit = Boolean(onQuestion);
  const recordingType = autoSubmit && serverTranscription ? recordingMimeType() : undefined;
  const supported = Boolean(recordingType || recognitionConstructor());
  useEffect(() => {
    const submit: OnQuestion = (text, endedAt) => question.current?.(text, endedAt);
    const instance = recordingType
      ? createRecordedQuestionController(recordingType, setState, submit)
      : createMicrophoneController(setState, (text) => callback.current(text), recognitionConstructor, autoSubmit ? submit : undefined);
    controller.current = instance;
    instance.stop();
    return () => { instance.dispose(); if (controller.current === instance) controller.current = null; };
  }, [autoSubmit, recordingType]);
  // A recording keeps its original recognition language. Stop it before the UI
  // switches language rather than allowing late text from the previous capture.
  useEffect(() => { controller.current?.stop(); }, [language]);
  const stop = useCallback(() => controller.current?.stop(), []);
  const toggle = useCallback(() => controller.current?.toggle(language), [language]);
  return { ...state, supported, backend: recordingType ? 'server' : 'browser', toggle, stop };
}

