import { afterEach, describe, expect, it, vi } from 'vitest';
import { createEventParser, streamChat } from './api';
import type { StreamEvent, Turn } from './api';
import { planTime } from './hooks/usePlayback';

describe('conversation event transport', () => {
  it('preserves Korean tokens across split frames and ignores keepalives', () => {
    const events: unknown[] = []; const parser = createEventParser((event) => events.push(event));
    parser.push(': keepalive\r\n\r\nevent: token\r');
    parser.push('\ndata: {"text":"안녕"}\r\n\r\nevent: done\ndata: {}\n\n');
    parser.finish();
    expect(events).toEqual([{ event: 'token', data: { text: '안녕' } }, { event: 'done', data: {} }]);
  });
  it('rejects malformed payloads instead of silently treating them as responses', () => {
    const parser = createEventParser(() => undefined);
    expect(() => parser.push('event: token\ndata: not-json\n\n')).toThrow();
  });
  it.each([
    ['token', { text: 123 }],
    ['status', {}],
    ['meta', { session_id: null }],
    ['result', { turn: { text: 'Missing turn structure' } }],
    ['error', { message: 42 }],
    ['done', []],
  ])('rejects invalid %s event fields before they reach the UI', (event, data) => {
    const parser = createEventParser(() => undefined);
    expect(() => parser.push(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)).toThrow();
  });
});

const turn: Turn = {
  id: 'turn-1', session_id: 'session-1', text: '안녕하세요', language: 'ko', mode: 'rag', provider: 'evidence', citations: [],
  behavior: { duration_ms: 1200, cues: [], speech: { rate: 1, pitch: 1, pause_ms: 260 } },
  grounding: { status: 'insufficient', retrieved_ids: [], invalid_citation_ids: [] },
  metrics: { retrieval_ms: 1, generation_ms: 2, total_ms: 3 },
};
const frame = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
const request = { message: '안녕', mode: 'rag', language: 'ko' } as const;
function mockStream(text: string, oneByte = false) {
  const bytes = new TextEncoder().encode(text);
  const body = new ReadableStream<Uint8Array>({ start(controller) {
    if (oneByte) for (let index = 0; index < bytes.length; index++) controller.enqueue(bytes.slice(index, index + 1));
    else controller.enqueue(bytes);
    controller.close();
  } });
  const response = new Response(body, { headers: { 'content-type': 'text/event-stream' } });
  vi.stubGlobal('fetch', vi.fn(async () => response)); return response;
}
afterEach(() => vi.unstubAllGlobals());

describe('stream completion and cancellation', () => {
  it('decodes UTF-8 split inside Korean characters and completes one result', async () => {
    mockStream(frame('meta', { session_id: 'session-1' }) + frame('token', { text: '안녕' }) + frame('result', { turn }) + frame('done', {}), true);
    const events: StreamEvent[] = [];
    await streamChat(request, new AbortController().signal, (event) => events.push(event));
    expect(events.find((event) => event.event === 'token')?.data).toEqual({ text: '안녕' });
    expect(events.at(-1)?.event).toBe('done');
  });

  it('rejects a done event without a complete turn', async () => {
    const response = mockStream(frame('done', {}));
    await expect(streamChat(request, new AbortController().signal, () => undefined)).rejects.toThrow();
    expect(response.body?.locked).toBe(false);
  });

  it('rejects premature EOF and releases the reader', async () => {
    const response = mockStream(frame('result', { turn }));
    await expect(streamChat(request, new AbortController().signal, () => undefined)).rejects.toThrow();
    expect(response.body?.locked).toBe(false);
  });

  it('rejects conflicting events after completion', async () => {
    mockStream(frame('result', { turn }) + frame('done', {}) + frame('token', { text: 'late text' }));
    await expect(streamChat(request, new AbortController().signal, () => undefined)).rejects.toThrow();
  });

  it('does not emit buffered tokens after the consumer cancels', async () => {
    mockStream(frame('meta', { session_id: 'session-1' }) + frame('token', { text: 'late' }) + frame('result', { turn }) + frame('done', {}));
    const abort = new AbortController(); const events: string[] = [];
    await expect(streamChat(request, abort.signal, (event) => { events.push(event.event); if (event.event === 'meta') abort.abort(); })).rejects.toMatchObject({ name: 'AbortError' });
    expect(events).toEqual(['meta']);
  });
});

describe('audio and behavior clock', () => {
  it('uses actual audio position to retime an estimated gesture plan', () => {
    expect(planTime(3, 6, 10_000)).toBe(5000);
    expect(planTime(8, 6, 10_000)).toBe(10_000);
  });
  it('avoids invalid clock values before metadata arrives', () => {
    expect(planTime(1, NaN, 10_000)).toBe(1000);
    expect(planTime(-1, 6, 10_000)).toBe(0);
  });
});

