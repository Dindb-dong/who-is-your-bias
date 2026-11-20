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

export interface MicrophoneState { listening: boolean; error: string | null; processing?: boolean }
export type OnQuestion = (text: string, endedAt: number) => void;
export interface MicrophoneController {
  toggle(language: Language): void;
  stop(): void;
  dispose(): void;
}

/** Results belong to one active capture; old sessions cannot edit a new draft. */
export function createMicrophoneController(emit: (state: MicrophoneState) => void,
  onText: (text: string) => void, getConstructor: () => RecognitionConstructor | undefined,
  onQuestion?: OnQuestion): MicrophoneController {
  let active: Recognition | null = null; let disposed = false;
  let complete: (() => void) | undefined;
  const detach = (instance: Recognition) => { instance.onresult = null; instance.onend = null; instance.onerror = null; };
  const publish = (state: MicrophoneState) => { if (!disposed) emit(state); };
  const stop = () => {
    const instance = active; active = null; complete = undefined;
    if (instance) { detach(instance); try { instance.abort(); } catch { /* Already ended. */ } }
    publish({ listening: false, error: null });
  };
  return {
    stop,
    dispose() { disposed = true; stop(); },
    toggle(language) {
      if (disposed) return;
      if (active) {
        if (onQuestion) complete?.();
        else stop();
        return;
      }
      const Constructor = getConstructor();
      if (!Constructor) { publish({ listening: false, error: 'Microphone transcription is not available in this browser. You can type your question.' }); return; }
      let instance: Recognition;
      try { instance = new Constructor(); }
      catch { publish({ listening: false, error: 'The microphone could not start. Please try again.' }); return; }
      active = instance;
      instance.lang = language === 'ko' ? 'ko-KR' : 'en-US';
      instance.interimResults = true;
      instance.continuous = Boolean(onQuestion);
      let finalText = '';
      let endedAt: number | undefined;
      const isCurrent = () => !disposed && active === instance;
      const finish = (error: string | null = null) => {
        if (!isCurrent()) return;
        active = null; complete = undefined; detach(instance);
        try { instance.abort(); } catch { /* A completed recognition may already be closed. */ }
        publish({ listening: false, error });
      };
      complete = () => {
        if (endedAt !== undefined) return;
        endedAt = performance.now();
        publish({ listening: false, processing: true, error: null });
        try { instance.stop(); }
        catch { finish('The microphone could not finish. Please try again or type.'); }
      };
      instance.onresult = (event) => {
        if (!isCurrent()) return;
        const transcriptParts: string[] = [];
        const finalParts: string[] = [];
        for (let index = 0; index < event.results.length; index += 1) {
          const result = event.results[index];
          const transcript = result?.[0]?.transcript?.trim();
          if (!transcript) continue;
          transcriptParts.push(transcript);
          if (result.isFinal) finalParts.push(transcript);
        }
        finalText = finalParts.join(' ');
        onText(transcriptParts.join(' '));
      };
      instance.onerror = (event) => {
        if (!isCurrent()) return;
        const message = event.error === 'aborted' || event.error === 'no-speech' ? null
          : event.error === 'not-allowed' ? 'Microphone permission was not granted.'
            : 'Could not transcribe audio. Please try again or type.';
        finish(message);
      };
      instance.onend = () => {
        if (!isCurrent()) return;
        const text = finalText.trim();
        finish();
        if (onQuestion && text) onQuestion(text, endedAt ?? performance.now());
      };
      publish({ listening: true, error: null });
      try { instance.start(); }
      catch { finish('The microphone could not start. Please try again.'); }
    },
  };
}
