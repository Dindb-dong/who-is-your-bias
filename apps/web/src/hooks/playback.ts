import { getPreparedSpeech, getSpeech } from '../api';
import type { Turn } from '../api';
import type { SpeechTiming } from '../avatar/types';
import type { PlaybackEvent, PlaybackMark } from './liveMetrics';

export type VoiceMode = 'off' | 'browser' | 'provider';
export type PlaybackState = {
  status: 'idle' | 'loading' | 'playing'; elapsedMs: number; audioElapsedMs: number;
  amplitude: number; error: string | null; audioSha256?: string; speechTiming?: SpeechTiming;
  timingStatus?: string; timingMessage?: string;
};
export const initialPlayback: PlaybackState = { status: 'idle', elapsedMs: 0, audioElapsedMs: 0, amplitude: 0, error: null };

export function planTime(audioSeconds: number, audioDurationSeconds: number, planDurationMs: number): number {
  if (!Number.isFinite(audioSeconds) || audioSeconds < 0) return 0;
  if (!Number.isFinite(audioDurationSeconds) || audioDurationSeconds <= 0 || !Number.isFinite(planDurationMs) || planDurationMs <= 0) return audioSeconds * 1000;
  return Math.min(planDurationMs, audioSeconds / audioDurationSeconds * planDurationMs);
}

export interface PlaybackEnvironment {
  getSpeech: typeof getSpeech;
  prepareSpeech?: typeof getPreparedSpeech;
  createContext(): AudioContext;
  createAudio(url: string): HTMLAudioElement;
  createUtterance(text: string): SpeechSynthesisUtterance;
  synthesis?: Pick<SpeechSynthesis, 'getVoices' | 'speak' | 'cancel'>;
  createUrl(blob: Blob): string;
  revokeUrl(url: string): void;
  requestFrame(callback: FrameRequestCallback): number;
  cancelFrame(id: number): void;
  now(): number;
}

export function browserPlaybackEnvironment(): PlaybackEnvironment {
  return {
    getSpeech, prepareSpeech: getPreparedSpeech, createContext: () => new AudioContext(), createAudio: (url) => new Audio(url),
    createUtterance: (text) => new SpeechSynthesisUtterance(text),
    synthesis: 'speechSynthesis' in window ? window.speechSynthesis : undefined,
    createUrl: (blob) => URL.createObjectURL(blob), revokeUrl: (url) => URL.revokeObjectURL(url),
    requestFrame: (callback) => requestAnimationFrame(callback), cancelFrame: (id) => cancelAnimationFrame(id), now: () => performance.now(),
  };
}

export interface PlaybackController {
  play(turn: Turn, mode: VoiceMode, observe?: (event: PlaybackEvent) => void): Promise<void>;
  stop(): void;
  dispose(): void;
}

const noop = () => undefined;
const safe = (action: () => void): void => { try { action(); } catch { /* Complete the rest of cleanup. */ } };

/** One owner per playback; cancellation invalidates pending work before cleanup. */
export function createPlaybackController(emit: (state: PlaybackState) => void, environment: PlaybackEnvironment): PlaybackController {
  let generation = 0; let disposed = false; let cleanup: () => void = noop;
  const publish = (state: PlaybackState) => { if (!disposed) emit(state); };
  const stop = () => {
    ++generation;
    const release = cleanup; cleanup = noop; release();
    publish({ ...initialPlayback });
  };
  return {
    stop,
    dispose() { disposed = true; ++generation; const release = cleanup; cleanup = noop; release(); },
    async play(turn, mode, observe) {
      if (disposed) return;
      stop();
      if (mode === 'off') return;
      const ticket = generation;
      let settled = false; let released = false;
      let signalled = false;
      const mark = (type: PlaybackMark, speechServer?: PlaybackEvent['speechServer']) => observe?.({ type, atMs: environment.now(), turnId: turn.id, voiceMode: mode, speechServer });
      const speechText = turn.text.replace(/\[source:[A-Za-z0-9_.-]+\]/g, '').trim();
      const isCurrent = () => !disposed && !settled && ticket === generation;
      const abort = new AbortController();
      let raf = 0;
      let audio: HTMLAudioElement | undefined;
      let audioContext: AudioContext | undefined;
      let audioUrl: string | undefined;
      let source: MediaElementAudioSourceNode | undefined;
      let analyser: AnalyserNode | undefined;
      let utterance: SpeechSynthesisUtterance | undefined;
      let startTimer: ReturnType<typeof setTimeout> | undefined;
      const release = () => {
        if (released) return;
        released = true;
        if (!settled) mark('cancelled');
        clearTimeout(startTimer);
        // Detach first: load(), pause(), and cancel() may synchronously emit
        // media events, and stale callbacks must never reenter finish().
        if (audio) { audio.onended = null; audio.onerror = null; }
        if (utterance) { utterance.onstart = null; utterance.onend = null; utterance.onerror = null; }
        abort.abort(); environment.cancelFrame(raf);
        if (audio) { safe(() => audio!.pause()); safe(() => audio!.removeAttribute('src')); safe(() => audio!.load()); }
        if (utterance) safe(() => environment.synthesis?.cancel());
        safe(() => source?.disconnect()); safe(() => analyser?.disconnect());
        if (audioContext && audioContext.state !== 'closed') safe(() => { void audioContext!.close().catch(noop); });
        if (audioUrl) environment.revokeUrl(audioUrl);
      };
      cleanup = release;
      const finish = (error: string | null = null, cancelled = false) => {
        if (!isCurrent()) return;
        mark(cancelled ? 'cancelled' : error ? 'failed' : 'playback_completed');
        settled = true; ++generation;
        cleanup = noop; release(); publish({ ...initialPlayback, error });
      };
      publish({ ...initialPlayback, status: 'loading' });
      mark('preparation_started');
      try {
        if (mode === 'provider') {
          // Every await is followed by an ownership check before more work.
          audioContext = environment.createContext();
          await audioContext.resume();
          if (!isCurrent()) return;
          const prepared = environment.prepareSpeech ? await environment.prepareSpeech(speechText, turn.language, abort.signal) : undefined;
          if (!isCurrent()) return;
          const blob = prepared?.audio ?? await environment.getSpeech(speechText, abort.signal);
          if (!isCurrent()) return;
          mark('audio_ready', prepared?.serverTiming);
          audioUrl = environment.createUrl(blob);
          audio = environment.createAudio(audioUrl);
          source = audioContext.createMediaElementSource(audio);
          analyser = audioContext.createAnalyser(); analyser.fftSize = 256;
          source.connect(analyser); analyser.connect(audioContext.destination);
          const samples = new Uint8Array(analyser.fftSize);
          const tick = () => {
            if (!isCurrent() || !audio || !analyser) return;
            analyser.getByteTimeDomainData(samples);
            let squareSum = 0;
            for (const value of samples) squareSum += ((value - 128) / 128) ** 2;
            // This measures a non-silent Web Audio frame, not the listener's
            // speaker volume or acoustic output. Silent lead-in stays visible.
            if (!signalled && Math.sqrt(squareSum / samples.length) > 0.002) { signalled = true; mark('first_signal'); }
            publish({ status: 'playing', elapsedMs: planTime(audio.currentTime, audio.duration, turn.behavior.duration_ms),
              audioElapsedMs: Number.isFinite(audio.currentTime) ? Math.max(0, audio.currentTime * 1000) : 0,
              amplitude: Math.min(1, Math.sqrt(squareSum / samples.length) * 5), error: null,
              audioSha256: prepared?.audioSha256, speechTiming: prepared?.speechTiming,
              timingStatus: prepared?.timingStatus ?? 'not_configured', timingMessage: prepared?.timingMessage });
            raf = environment.requestFrame(tick);
          };
          audio.onended = () => finish();
          audio.onerror = () => finish('The audio could not be played. Check the speech service format.');
          await audio.play();
          if (isCurrent()) { mark('playback_started'); tick(); }
        } else {
          const synthesis = environment.synthesis;
          if (!synthesis) throw new Error('This browser does not support built-in speech.');
          utterance = environment.createUtterance(speechText);
          utterance.lang = turn.language === 'ko' ? 'ko-KR' : 'en-US';
          const rate = turn.behavior.speech?.rate ?? 1; const pitch = turn.behavior.speech?.pitch ?? 1;
          utterance.rate = Number.isFinite(rate) ? Math.min(2, Math.max(0.5, rate)) : 1;
          utterance.pitch = Number.isFinite(pitch) ? Math.min(2, Math.max(0, pitch)) : 1;
          const voice = synthesis.getVoices().find((item) => item.lang.toLowerCase().startsWith(turn.language));
          if (voice) utterance.voice = voice;
          let startedAt = 0;
          const tick = () => {
            if (!isCurrent()) return;
            const elapsed = Math.max(0, environment.now() - startedAt);
            // Browser synthesis exposes no PCM waveform. This is designed
            // articulation rather than measured speech energy.
            publish({ status: 'playing', elapsedMs: planTime(elapsed / 1000, turn.behavior.duration_ms / 1000, turn.behavior.duration_ms),
              audioElapsedMs: elapsed, amplitude: 0.12 + (Math.sin(elapsed / 83) + 1) * 0.17, error: null,
              timingStatus: 'estimated' });
            raf = environment.requestFrame(tick);
          };
          utterance.onstart = () => {
            if (!isCurrent()) return;
            clearTimeout(startTimer); startedAt = environment.now(); mark('audio_ready'); mark('playback_started'); tick();
          };
          utterance.onend = () => finish();
          utterance.onerror = (event) => event.error === 'canceled' || event.error === 'interrupted'
            ? finish(null, true) : finish('Built-in speech is unavailable. Try another voice or use text.');
          startTimer = setTimeout(() => finish('The browser did not start speech. Select Listen to try again.'), 12_000);
          synthesis.speak(utterance);
        }
      } catch (error) {
        if (isCurrent()) finish(error instanceof Error ? error.message : 'Audio playback failed.');
      }
    },
  };
}
