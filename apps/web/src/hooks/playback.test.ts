import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { PreparedSpeech, Turn } from '../api';
import { createPlaybackController, planTime } from './playback';
import type { PlaybackEnvironment, PlaybackState } from './playback';

function deferred<T>() {
  let resolve!: (value: T) => void; let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
const flush = async () => { await Promise.resolve(); await Promise.resolve(); };
const turn = { text: 'An evidence-grounded answer.', language: 'en', behavior: { duration_ms: 10_000, cues: [], speech: { rate: 1, pitch: 1, pause_ms: 0 } } } as unknown as Turn;

class FakeAudio {
  currentTime = 3; duration = 6;
  onended: (() => void) | null = null;
  onerror: (() => void) | null = null;
  play = vi.fn(async () => undefined);
  pause = vi.fn();
  removeAttribute = vi.fn();
  // A deliberately reentrant browser: cleanup must detach this callback first.
  load = vi.fn(() => this.onerror?.());
}

function setup() {
  const states: PlaybackState[] = []; const audio: FakeAudio[] = [];
  const contexts: Array<{ resume: ReturnType<typeof vi.fn>; close: ReturnType<typeof vi.fn>; state: string }> = [];
  const utterances: SpeechSynthesisUtterance[] = [];
  let currentUtterance: SpeechSynthesisUtterance | null = null;
  const frames = new Map<number, FrameRequestCallback>(); let frameId = 0;
  const speech = vi.fn<PlaybackEnvironment['getSpeech']>(async () => new Blob(['audio'], { type: 'audio/wav' }));
  const environment: PlaybackEnvironment = {
    getSpeech: speech,
    createContext() {
      const node = { connect: vi.fn(), disconnect: vi.fn() };
      const analyser = { ...node, fftSize: 256, getByteTimeDomainData: (array: Uint8Array) => array.fill(128) };
      const context = { state: 'running', resume: vi.fn(async () => undefined), close: vi.fn(async () => { context.state = 'closed'; }),
        createMediaElementSource: () => node, createAnalyser: () => analyser, destination: {} };
      contexts.push(context); return context as unknown as AudioContext;
    },
    createAudio() { const instance = new FakeAudio(); audio.push(instance); return instance as unknown as HTMLAudioElement; },
    createUtterance(text) { const instance = { text, onstart: null, onend: null, onerror: null } as unknown as SpeechSynthesisUtterance; utterances.push(instance); return instance; },
    synthesis: {
      getVoices: () => [],
      speak: vi.fn((utterance) => { currentUtterance = utterance; }),
      cancel: vi.fn(() => currentUtterance?.onend?.call(currentUtterance, {} as SpeechSynthesisEvent)),
    },
    createUrl: vi.fn(() => `blob:audio-${audio.length}`), revokeUrl: vi.fn(),
    requestFrame(callback) { frames.set(++frameId, callback); return frameId; },
    cancelFrame: (id) => { frames.delete(id); }, now: () => 1000,
  };
  const controller = createPlaybackController((state) => states.push(state), environment);
  return { controller, environment, states, audio, contexts, speech, utterances, frames };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => { vi.clearAllTimers(); vi.useRealTimers(); });

describe('speech playback ownership', () => {
  it('keeps the raw audio clock separate from the behavior plan clock', async () => {
    const f = setup();
    f.environment.prepareSpeech = vi.fn(async (): Promise<PreparedSpeech> => ({ audio: new Blob(['audio']), audioSha256: 'a'.repeat(64),
      timingStatus: 'available', speechTiming: { schema_version: 'speech-timing.v1', audio_sha256: 'a'.repeat(64),
        audio_duration_ms: 6000, timing_basis: 'audio_recognition', label_set: 'rhubarb',
        segments: [{ start_ms: 0, end_ms: 6000, label: 'B' }], provenance: { engine: 'rhubarb-lip-sync' } } }));
    await f.controller.play(turn, 'provider');
    expect(f.speech).not.toHaveBeenCalled();
    expect(f.states.at(-1)?.audioElapsedMs).toBe(3000);
    expect(f.states.at(-1)?.elapsedMs).toBe(5000);
    expect(f.states.at(-1)?.speechTiming?.audio_sha256).toBe(f.states.at(-1)?.audioSha256);
    f.controller.stop(); expect(f.states.at(-1)?.speechTiming).toBeUndefined(); f.controller.dispose();
  });

  it('discards a prepared package that finishes after cancellation', async () => {
    const f = setup(); const preparation = deferred<PreparedSpeech>();
    f.environment.prepareSpeech = vi.fn(() => preparation.promise);
    const work = f.controller.play(turn, 'provider'); await flush(); f.controller.stop();
    preparation.resolve({ audio: new Blob(['late']), audioSha256: 'a'.repeat(64), timingStatus: 'not_configured' });
    await work;
    expect(f.environment.createUrl).not.toHaveBeenCalled(); expect(f.speech).not.toHaveBeenCalled();
    expect(f.states.at(-1)?.status).toBe('idle'); f.controller.dispose();
  });

  it('does not request synthesis after cancellation during context resume', async () => {
    const f = setup(); const resume = deferred<void>(); const createContext = f.environment.createContext;
    f.environment.createContext = () => { const context = createContext(); context.resume = () => resume.promise; return context; };
    const work = f.controller.play(turn, 'provider'); f.controller.stop(); resume.resolve(); await work;
    expect(f.speech).not.toHaveBeenCalled(); expect(f.contexts[0].close).toHaveBeenCalledTimes(1);
    expect(f.states.at(-1)?.status).toBe('idle'); f.controller.dispose();
  });

  it('aborts pending synthesis and discards a late successful response', async () => {
    const f = setup(); const speech = deferred<Blob>(); f.speech.mockReturnValue(speech.promise);
    const work = f.controller.play(turn, 'provider'); await flush();
    const signal = f.speech.mock.calls[0][1]; f.controller.stop();
    expect(signal.aborted).toBe(true);
    speech.resolve(new Blob(['late'])); await work;
    expect(f.environment.createUrl).not.toHaveBeenCalled(); expect(f.audio).toHaveLength(0);
    expect(f.states.at(-1)?.status).toBe('idle'); f.controller.dispose();
  });

  it('ignores late old audio events and uses the actual transport duration', async () => {
    const f = setup(); await f.controller.play(turn, 'provider');
    expect(f.states.at(-1)?.elapsedMs).toBe(5000); expect(f.states.at(-1)?.amplitude).toBe(0);
    const lateError = f.audio[0].onerror;
    await f.controller.play(turn, 'provider');
    expect(f.audio[0].onerror).toBeNull(); const count = f.states.length;
    lateError?.();
    expect(f.states).toHaveLength(count); expect(f.states.at(-1)?.status).toBe('playing');
    f.audio[1].onended?.();
    expect(f.states.at(-1)?.error).toBeNull(); expect(f.states.at(-1)?.status).toBe('idle');
    expect(f.environment.revokeUrl).toHaveBeenCalledTimes(2); f.controller.dispose();
  });

  it('does not replay after unmount even if the provider ignores abort', async () => {
    const f = setup(); const speech = deferred<Blob>(); f.speech.mockReturnValue(speech.promise);
    const work = f.controller.play(turn, 'provider'); await flush(); f.controller.dispose();
    const count = f.states.length; speech.resolve(new Blob(['late'])); await work;
    expect(f.states).toHaveLength(count); expect(f.audio).toHaveLength(0);
  });

  it('detaches browser speech callbacks before cancel, preventing cleanup recursion', async () => {
    const f = setup(); await f.controller.play(turn, 'browser');
    const utterance = f.utterances[0];
    utterance.onstart?.call(utterance, {} as SpeechSynthesisEvent);
    expect(f.states.at(-1)?.status).toBe('playing');
    utterance.onend?.call(utterance, {} as SpeechSynthesisEvent);
    expect(f.environment.synthesis?.cancel).toHaveBeenCalledTimes(1);
    expect(utterance.onend).toBeNull(); expect(f.frames.size).toBe(0); expect(f.states.at(-1)?.status).toBe('idle');
    f.controller.dispose();
  });

  it('ends a browser utterance that never starts instead of remaining loading', async () => {
    const f = setup(); await f.controller.play(turn, 'browser');
    vi.advanceTimersByTime(12_000);
    expect(f.states.at(-1)?.status).toBe('idle'); expect(f.states.at(-1)?.error).toContain('did not start');
    expect(f.utterances[0].onstart).toBeNull(); f.controller.dispose();
  });

  it('keeps invalid duration metadata out of animation coordinates', () => {
    expect(planTime(2, 5, Number.NaN)).toBe(2000);
    expect(planTime(2, 5, Number.POSITIVE_INFINITY)).toBe(2000);
    expect(planTime(Number.POSITIVE_INFINITY, 5, 1000)).toBe(0);
  });
});
