import { afterEach, describe, expect, it, vi } from 'vitest';
import { createPcmPlayback, PcmSequence } from './pcmPlayback';
import type { PcmChunk, PcmState } from './pcmPlayback';

const packet = (extra: Partial<PcmChunk> = {}): PcmChunk => ({ turn_id: 'one', segment_id: 'one:0',
  sequence: 0, segment_sequence: 0, sample_rate: 16000, channels: 1, sample_offset: 0, sample_count: 2,
  encoding: 'pcm_s16le', pcm_base64: btoa('\x00\x40\x00\xc0'), ...extra });

describe('PCM stream integrity', () => {
  it('decodes signed little-endian frames and preserves sample clock continuity', () => {
    const sequence = new PcmSequence();
    expect(Array.from(sequence.accept(packet())[0])).toEqual([.5, -.5]);
    sequence.accept(packet({ sequence: 1, sample_offset: 2 }));
    expect(sequence.nextSample).toBe(4);
  });
  it.each([
    { sequence: 2 }, { sample_offset: 1 }, { sample_count: 3 }, { channels: 3 },
    { sample_rate: 96000 }, { sample_count: 0 }, { sequence: NaN },
    { turn_id: '' }, { segment_id: '' }, { segment_sequence: -1 },
    { pcm_base64: '???' }, { pcm_base64: btoa('\x00') },
  ])('rejects malformed or reordered data without advancing the clock: %j', (change) => {
    const sequence = new PcmSequence();
    expect(() => sequence.accept(packet(change))).toThrow();
    expect(sequence.nextSample).toBe(0);
  });
  it('rejects stale-turn and mid-turn format changes', () => {
    for (const change of [{ turn_id: 'other' }, { sample_rate: 24000 }]) {
      const sequence = new PcmSequence(); sequence.accept(packet());
      expect(() => sequence.accept(packet({ sequence: 1, sample_offset: 2, ...change }))).toThrow();
      expect(sequence.nextSample).toBe(2);
    }
  });
});

function environment(options: { resume?: () => Promise<void>; maxBufferedSeconds?: number } = {}) {
  const starts: number[] = []; let stops = 0; let closed = false; let callback: FrameRequestCallback | undefined;
  const nodes: { onended: (() => void) | null }[] = [];
  const context = {
    currentTime: 1, state: 'running', destination: {}, resume: options.resume ?? (async () => undefined),
    close: async () => { closed = true; },
    createAnalyser: () => ({ fftSize: 256, connect() {}, disconnect() {}, getFloatTimeDomainData: (out: Float32Array) => out.fill(.1) }),
    createBuffer: () => ({ copyToChannel() {} }),
    createBufferSource: () => {
      const node = { onended: null as (() => void) | null, connect() {}, disconnect() {}, stop() { stops++; }, start(at: number) { starts.push(at); } };
      nodes.push(node); return node;
    },
  };
  const states: PcmState[] = [];
  const player = createPcmPlayback((state) => states.push(state), { context: context as unknown as AudioContext,
    now: () => context.currentTime * 1000, requestFrame: (cb) => { callback = cb; return 1; }, cancelFrame() {}, maxBufferedSeconds: options.maxBufferedSeconds });
  return { context, player, nodes, states, starts, tick() { callback?.(0); }, get stops() { return stops; }, get closed() { return closed; } };
}
afterEach(() => vi.useRealTimers());
const fullPacket = (sequence = 0, sampleCount = 8000, sampleRate = 8000) => packet({ sequence,
  sample_offset: sequence * sampleCount, sample_count: sampleCount, sample_rate: sampleRate,
  pcm_base64: btoa('\0'.repeat(sampleCount * 2)),
});

it('schedules contiguous packets on AudioContext time, not wall-clock arrival', async () => {
  const env = environment();
  await env.player.push(packet());
  await env.player.push(packet({ sequence: 1, sample_offset: 2 }));
  expect(env.starts[1] - env.starts[0]).toBeCloseTo(2 / 16000, 7);
  env.context.currentTime = env.starts[0] + 1 / 16000; env.tick();
  expect(env.player.snapshot().audioElapsedMs).toBeCloseTo(1000 / 16000, 6);
  expect(env.player.snapshot().acousticLatencyMs).toBeNull();
  env.player.stop(); expect(env.stops).toBe(2); expect(env.closed).toBe(true);
});

it('does not mark a network-complete stream as played until every buffer ends', async () => {
  const env = environment(); await env.player.push(packet());
  let finished = false; const completion = env.player.finish(2).then(() => { finished = true; });
  await Promise.resolve(); expect(finished).toBe(false);
  env.nodes[0].onended?.(); await completion;
  expect(env.player.snapshot().status).toBe('completed'); expect(env.closed).toBe(true);
});

it('seals the exact final sample clock even when animation frames were throttled', async () => {
  const env = environment(); await env.player.push(fullPacket());
  const finished = env.player.finish(8000);
  env.context.currentTime = env.starts[0] + 1;
  env.nodes[0].onended?.(); await finished;
  expect(env.player.snapshot()).toMatchObject({ status: 'completed', elapsedMs: 1000, audioElapsedMs: 1000, amplitude: 0, queuedMs: 0 });
});

it('keeps an imminent buffer contiguous and counts a genuinely starved restart', async () => {
  const env = environment(); await env.player.push(fullPacket());
  env.context.currentTime = env.starts[0] + .99;
  await env.player.push(fullPacket(1));
  expect(env.starts[1] - env.starts[0]).toBeCloseTo(1, 7);
  expect(env.player.snapshot().underruns).toBe(0);
  env.context.currentTime = env.starts[1] + 1.05;
  await env.player.push(fullPacket(2));
  expect(env.starts[2]).toBeCloseTo(env.context.currentTime + .025, 7);
  expect(env.player.snapshot().underruns).toBe(1);
  env.player.stop();
});

it('includes the incoming packet in the eight-second buffer ceiling', async () => {
  vi.useFakeTimers(); const env = environment();
  for (let index = 0; index < 3; index++) await env.player.push(fullPacket(index, 16384));
  const pending = env.player.push(fullPacket(3, 16384));
  await vi.advanceTimersByTimeAsync(0);
  expect(env.starts).toHaveLength(3);
  env.context.currentTime += .218;
  await vi.advanceTimersByTimeAsync(20); await pending; env.tick();
  expect(env.starts).toHaveLength(4);
  expect(env.player.snapshot().queuedMs).toBeLessThanOrEqual(8000);
  env.player.stop();
});

it('rejects a packet larger than its buffer budget instead of waiting forever', async () => {
  const env = environment({ maxBufferedSeconds: .1 });
  await expect(env.player.push(fullPacket())).rejects.toThrow('buffer limit');
  expect(env.starts).toHaveLength(0); env.player.stop();
});

it('cancellation releases a consumer blocked by backpressure immediately', async () => {
  vi.useFakeTimers(); const env = environment();
  for (let index = 0; index < 3; index++) await env.player.push(fullPacket(index, 16384));
  const pending = env.player.push(fullPacket(3, 16384));
  await vi.advanceTimersByTimeAsync(0);
  env.player.stop(); await pending;
  expect(env.starts).toHaveLength(3);
  expect(env.closed).toBe(true);
  expect(env.player.snapshot().status).toBe('cancelled');
});

it('cancellation releases pending context resume and prevents late scheduling', async () => {
  let resume: (() => void) | undefined;
  const ready = new Promise<void>((resolve) => { resume = resolve; });
  const env = environment({ resume: () => ready });
  const pending = env.player.push(packet());
  env.player.stop(); await pending;
  expect(env.closed).toBe(true); expect(env.starts).toHaveLength(0);
  resume?.(); await ready;
  await env.player.push(packet());
  expect(env.starts).toHaveLength(0);
});

it('handles a rejected resume before the first packet arrives', async () => {
  const env = environment({ resume: () => Promise.reject(new Error('Audio permission unavailable')) });
  await Promise.resolve();
  await expect(env.player.push(packet())).rejects.toThrow('Audio permission unavailable');
  expect(env.starts).toHaveLength(0); env.player.stop();
});

it('cancellation invalidates scheduled and later-arriving PCM', async () => {
  const env = environment(); await env.player.push(packet());
  const complete = env.player.finish(2); env.player.stop(); await complete;
  await env.player.push(packet({ sequence: 1, sample_offset: 2 }));
  expect(env.starts).toHaveLength(1); expect(env.nodes[0].onended).toBeNull();
  expect(env.player.snapshot().status).toBe('cancelled');
});

it('does not append PCM after delivery has finished', async () => {
  const env = environment(); await env.player.push(packet());
  const completion = env.player.finish(2);
  await expect(env.player.push(packet({ sequence: 1, sample_offset: 2 }))).rejects.toThrow('after stream completion');
  env.player.stop(); await completion;
});
