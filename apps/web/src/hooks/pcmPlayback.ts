/** Bounded, request-owned PCM scheduling on the actual Web Audio clock. */
export interface PcmChunk {
  turn_id: string; segment_id: string; sequence: number; segment_sequence: number;
  sample_rate: number; channels: number; sample_offset: number; sample_count: number;
  encoding: 'pcm_s16le'; pcm_base64: string;
}
export interface PcmState {
  status: 'idle' | 'loading' | 'playing' | 'completed' | 'cancelled' | 'failed';
  elapsedMs: number; audioElapsedMs: number; amplitude: number;
  firstDigitalPcmMs: number | null; underruns: number; queuedMs: number;
  interruptionMs: number | null; acousticLatencyMs: null; error: string | null;
}
export const initialPcmState: PcmState = { status: 'idle', elapsedMs: 0, audioElapsedMs: 0, amplitude: 0,
  firstDigitalPcmMs: null, underruns: 0, queuedMs: 0, interruptionMs: null, acousticLatencyMs: null, error: null };

export class PcmSequence {
  turnId: string | undefined;
  nextSequence = 0;
  nextSample = 0;
  format: [number, number] | undefined;
  accept(chunk: PcmChunk): Float32Array[] {
    if (!Number.isSafeInteger(chunk.sequence) || chunk.sequence !== this.nextSequence ||
        typeof chunk.turn_id !== 'string' || !chunk.turn_id ||
        typeof chunk.segment_id !== 'string' || !chunk.segment_id ||
        !Number.isSafeInteger(chunk.segment_sequence) || chunk.segment_sequence < 0 ||
        !Number.isSafeInteger(chunk.sample_offset) || chunk.sample_offset !== this.nextSample ||
        !Number.isSafeInteger(chunk.sample_count) || chunk.sample_count < 1 || chunk.sample_count > 16384 ||
        !Number.isSafeInteger(this.nextSample + chunk.sample_count) ||
        !Number.isSafeInteger(chunk.sample_rate) || chunk.sample_rate < 8000 || chunk.sample_rate > 48000 ||
        ![1, 2].includes(chunk.channels) || chunk.encoding !== 'pcm_s16le' ||
        (this.turnId !== undefined && this.turnId !== chunk.turn_id) ||
        (this.format && (this.format[0] !== chunk.sample_rate || this.format[1] !== chunk.channels))) {
      throw new Error('The streaming voice sequence or PCM format changed.');
    }
    if (chunk.pcm_base64.length > 90_000) throw new Error('The PCM packet exceeds its size limit.');
    const raw = atob(chunk.pcm_base64);
    if (raw.length !== chunk.sample_count * chunk.channels * 2) throw new Error('The PCM packet has incomplete frames.');
    const bytes = Uint8Array.from(raw, (char) => char.charCodeAt(0));
    const view = new DataView(bytes.buffer);
    const channels = Array.from({ length: chunk.channels }, () => new Float32Array(chunk.sample_count));
    for (let frame = 0; frame < chunk.sample_count; frame++) {
      for (let channel = 0; channel < chunk.channels; channel++) channels[channel][frame] = view.getInt16((frame * chunk.channels + channel) * 2, true) / 32768;
    }
    this.turnId = chunk.turn_id; this.format = [chunk.sample_rate, chunk.channels];
    this.nextSequence++; this.nextSample += chunk.sample_count;
    return channels;
  }
}

export function createPcmPlayback(publish: (state: PcmState) => void, options: {
  context?: AudioContext; now?: () => number; requestFrame?: typeof requestAnimationFrame;
  cancelFrame?: typeof cancelAnimationFrame; maxBufferedSeconds?: number;
} = {}) {
  const maxBuffered = options.maxBufferedSeconds ?? 8;
  if (!Number.isFinite(maxBuffered) || maxBuffered <= 0) throw new Error('The PCM buffer limit must be positive.');
  const context = options.context ?? new AudioContext();
  const now = options.now ?? (() => performance.now());
  const frame = options.requestFrame ?? requestAnimationFrame;
  const cancelFrame = options.cancelFrame ?? cancelAnimationFrame;
  const start = now();
  const sequence = new PcmSequence();
  const nodes = new Set<AudioBufferSourceNode>();
  const schedule: { from: number; to: number; sampleOffset: number; rate: number }[] = [];
  const analyser = context.createAnalyser(); analyser.fftSize = 256; analyser.connect(context.destination);
  const samples = new Float32Array(analyser.fftSize);
  let state: PcmState = { ...initialPcmState, status: 'loading' };
  let stopped = false; let ended = false; let nextStart = 0; let raf = 0;
  let finishResolve: (() => void) | undefined;
  let finishReject: ((error: Error) => void) | undefined;
  let releaseStopped: (() => void) | undefined;
  const stoppedPromise = new Promise<null>((resolve) => { releaseStopped = () => resolve(null); });
  // Resume immediately within the submit gesture, and attach a rejection handler
  // before any network wait. A suspended context need not settle resume on close.
  const ready = context.resume().then(() => ({ ok: true as const }), (error: unknown) => ({ ok: false as const, error }));
  const snapshot = () => ({ ...state });
  const notify = () => publish(snapshot());
  const cleanup = () => {
    releaseStopped?.();
    cancelFrame(raf); analyser.disconnect();
    for (const node of nodes) { node.onended = null; try { node.stop(); } catch { /* already ended */ } node.disconnect(); }
    nodes.clear(); schedule.splice(0);
    void context.close().catch(() => undefined);
  };
  const complete = () => {
    if (stopped || !ended || nodes.size) return;
    const elapsed = sequence.format ? sequence.nextSample / sequence.format[0] * 1000 : 0;
    stopped = true; state = { ...state, status: 'completed', elapsedMs: elapsed, audioElapsedMs: elapsed, amplitude: 0, queuedMs: 0 };
    cleanup(); notify(); finishResolve?.();
  };
  const tick = () => {
    if (stopped) return;
    const clock = context.currentTime;
    while (schedule.length > 1 && schedule[0].to <= clock) schedule.shift();
    const active = schedule.find((span) => span.from <= clock && clock < span.to);
    analyser.getFloatTimeDomainData(samples);
    const rms = Math.sqrt(samples.reduce((sum, value) => sum + value * value, 0) / samples.length);
    if (active) {
      const elapsed = (active.sampleOffset / active.rate + clock - active.from) * 1000;
      state.elapsedMs = elapsed; state.audioElapsedMs = elapsed;
      if (state.firstDigitalPcmMs === null && rms > 0.002) state.firstDigitalPcmMs = now() - start;
    }
    state.status = active ? 'playing' : 'loading';
    state.amplitude = active ? Math.min(1, rms * 5) : 0;
    state.queuedMs = Math.max(0, nextStart - clock) * 1000;
    notify(); raf = frame(tick);
  };
  raf = frame(tick);
  notify();
  return {
    snapshot,
    async push(chunk: PcmChunk) {
      if (stopped) return;
      if (ended) throw new Error('PCM arrived after stream completion.');
      const resumed = await Promise.race([ready, stoppedPromise]);
      if (stopped || resumed === null) return;
      if (ended) throw new Error('PCM arrived after stream completion.');
      if (!resumed.ok) throw resumed.error instanceof Error ? resumed.error : new Error('The audio context could not start.');
      const duration = chunk.sample_count / chunk.sample_rate;
      if (!Number.isFinite(duration) || duration <= 0 || duration > maxBuffered) throw new Error('The PCM packet exceeds the playback buffer limit.');
      // Await playback to bound scheduled audio and let fetch/HTTP backpressure
      // reach the server's bounded queues. Include the incoming packet, and let
      // cancellation unwind both this wait and a still-pending context resume.
      while (!stopped && Math.max(0, nextStart - context.currentTime) + duration > maxBuffered) {
        await Promise.race([new Promise((resolve) => setTimeout(resolve, 20)), stoppedPromise]);
      }
      if (stopped) return;
      if (ended) throw new Error('PCM arrived after stream completion.');
      const decoded = sequence.accept(chunk);
      const buffer = context.createBuffer(chunk.channels, chunk.sample_count, chunk.sample_rate);
      decoded.forEach((channel, index) => buffer.copyToChannel(channel as Float32Array<ArrayBuffer>, index));
      const source = context.createBufferSource(); source.buffer = buffer; source.connect(analyser);
      const clock = context.currentTime;
      const contiguous = nextStart > 0 && nextStart >= clock;
      if (nextStart && !contiguous) state.underruns++;
      // Preserve future contiguous audio even when less than 25 ms is queued.
      // Only a fresh start or a real underrun needs scheduling lead time.
      const at = contiguous ? nextStart : clock + Math.min(0.025, maxBuffered - duration);
      nextStart = at + duration;
      schedule.push({ from: at, to: nextStart, sampleOffset: chunk.sample_offset, rate: chunk.sample_rate });
      nodes.add(source);
      source.onended = () => { nodes.delete(source); source.disconnect(); complete(); };
      source.start(at);
    },
    finish(sampleCount: number) {
      if (sampleCount !== sequence.nextSample) return Promise.reject(new Error('The voice stream ended with missing PCM samples.'));
      ended = true;
      if (stopped) return Promise.resolve();
      return new Promise<void>((resolve, reject) => { finishResolve = resolve; finishReject = reject; complete(); });
    },
    stop(error?: string) {
      if (stopped) return;
      const before = now(); stopped = true;
      state = { ...state, status: error ? 'failed' : 'cancelled', amplitude: 0, queuedMs: 0, error: error ?? null };
      cleanup(); state.interruptionMs = now() - before; notify();
      if (error) finishReject?.(new Error(error)); else finishResolve?.();
    },
  };
}
