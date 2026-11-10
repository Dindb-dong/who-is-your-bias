import type { Language } from '../api';

export interface RecognitionResult { isFinal: boolean; [index: number]: { transcript: string } }
export interface RecognitionEvent { resultIndex: number; results: { length: number; [index: number]: RecognitionResult } }
export interface Recognition {
  lang: string; interimResults: boolean; continuous: boolean;
  onresult: ((event: RecognitionEvent) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
  start(): void; stop(): void; abort(): void;
}
export type RecognitionConstructor = new () => Recognition;
type RecognitionWindow = Window & { SpeechRecognition?: RecognitionConstructor; webkitSpeechRecognition?: RecognitionConstructor };

export function recognitionConstructor(): RecognitionConstructor | undefined {
  if (typeof window === 'undefined') return undefined;
  return (window as RecognitionWindow).SpeechRecognition ?? (window as RecognitionWindow).webkitSpeechRecognition;
}

export interface MicrophoneState { listening: boolean; error: string | null }
export interface MicrophoneController {
  toggle(language: Language): void;
  stop(): void;
  dispose(): void;
}

/** Results belong to one active capture; old sessions cannot edit a new draft. */
export function createMicrophoneController(emit: (state: MicrophoneState) => void,
  onText: (text: string) => void, getConstructor: () => RecognitionConstructor | undefined): MicrophoneController {
  let active: Recognition | null = null; let disposed = false;
  const detach = (instance: Recognition) => { instance.onresult = null; instance.onend = null; instance.onerror = null; };
  const publish = (state: MicrophoneState) => { if (!disposed) emit(state); };
  const stop = () => {
    const instance = active; active = null;
    if (instance) { detach(instance); try { instance.abort(); } catch { /* Already ended. */ } }
    publish({ listening: false, error: null });
  };
  return {
    stop,
    dispose() { disposed = true; stop(); },
    toggle(language) {
      if (disposed) return;
      if (active) { stop(); return; }
      const Constructor = getConstructor();
      if (!Constructor) { publish({ listening: false, error: 'Microphone transcription is not available in this browser. You can type your question.' }); return; }
      let instance: Recognition;
      try { instance = new Constructor(); }
      catch { publish({ listening: false, error: 'The microphone could not start. Please try again.' }); return; }
      active = instance;
      instance.lang = language === 'ko' ? 'ko-KR' : 'en-US';
      instance.interimResults = true; instance.continuous = false;
      const isCurrent = () => !disposed && active === instance;
      const finish = (error: string | null = null) => {
        if (!isCurrent()) return;
        active = null; detach(instance);
        try { instance.abort(); } catch { /* A completed recognition may already be closed. */ }
        publish({ listening: false, error });
      };
      instance.onresult = (event) => {
        if (!isCurrent()) return;
        let text = '';
        for (let index = 0; index < event.results.length; index += 1) {
          const transcript = event.results[index]?.[0]?.transcript;
          if (typeof transcript === 'string') text += transcript;
        }
        onText(text);
      };
      instance.onerror = (event) => {
        if (!isCurrent()) return;
        const message = event.error === 'aborted' || event.error === 'no-speech' ? null
          : event.error === 'not-allowed' ? 'Microphone permission was not granted.'
            : 'Could not transcribe audio. Please try again or type.';
        finish(message);
      };
      instance.onend = () => finish();
      publish({ listening: true, error: null });
      try { instance.start(); }
      catch { finish('The microphone could not start. Please try again.'); }
    },
  };
}
