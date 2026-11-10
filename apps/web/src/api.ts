import type { BehaviorPlan } from './avatar/types';
import type { SpeechTiming } from './avatar/types';
import { parseSpeechTiming } from './avatar/validation';

export type Language = 'en' | 'ko';
export type ExperimentMode = 'profile' | 'rag' | 'behavior';
export interface Source {
  id: string; title: string; url: string; publisher: string; published_at: string;
  source_type: string; split: 'train' | 'test'; attribution: string;
}
export interface Locator { label: string; start_seconds?: number | null; end_seconds?: number | null; timeline?: string | null }
export interface Citation { evidence_id: string; source_id: string; title: string; url: string; locator: Locator; text: string }
export interface Turn {
  id: string; session_id: string; text: string; language: Language; mode: ExperimentMode;
  provider: 'evidence' | 'openai'; citations: Citation[]; behavior: BehaviorPlan & { speech: { rate: number; pitch: number; pause_ms: number } };
  grounding: { status: 'grounded' | 'insufficient' | 'unverified'; retrieved_ids: string[]; invalid_citation_ids: string[] };
  metrics: { retrieval_ms: number; generation_ms: number; total_ms: number };
}
export interface Persona {
  id: string; name: string; tagline: string; description: string;
  example_questions: { en: string[]; ko: string[] };
  stats: { sources: number; evidence: number; held_out_sources: number };
  capabilities: { llm: boolean; speech: boolean; transcription?: boolean };
  notice: string;
}
export interface ChatRequest { message: string; session_id?: string; mode: ExperimentMode; language: Language }
export type StreamEvent =
  | { event: 'meta'; data: { session_id: string } }
  | { event: 'status'; data: { stage: string } }
  | { event: 'token'; data: { text: string } }
  | { event: 'result'; data: { turn: Turn } }
  | { event: 'done'; data: Record<string, never> }
  | { event: 'error'; data: { code: string; message: string } };

export class ApiError extends Error {
  constructor(message: string, public status = 0) { super(message); this.name = 'ApiError'; }
}

async function failure(response: Response): Promise<never> {
  let message = `The service returned ${response.status}. Please try again.`;
  try {
    const value = await response.json();
    const detail = value.detail ?? value.error ?? value.message;
    if (typeof detail === 'string') message = detail;
    else if (detail?.message) message = detail.message;
  } catch { /* Keep a readable error for non-JSON proxy responses. */ }
  throw new ApiError(message, response.status);
}

export async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal });
  if (!response.ok) return failure(response);
  return response.json() as Promise<T>;
}

function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
function strings(value: unknown): value is string[] { return Array.isArray(value) && value.every((item) => typeof item === 'string'); }
function positive(value: unknown, zero = true): value is number { return typeof value === 'number' && Number.isFinite(value) && (zero ? value >= 0 : value > 0); }
function isTurn(value: unknown): value is Turn {
  if (!object(value) || !['id', 'session_id', 'text'].every((key) => typeof value[key] === 'string')) return false;
  if (!['en', 'ko'].includes(String(value.language)) || !['profile', 'rag', 'behavior'].includes(String(value.mode)) || !['evidence', 'openai'].includes(String(value.provider))) return false;
  if (!Array.isArray(value.citations) || !value.citations.every((citation) => object(citation) && ['evidence_id', 'source_id', 'title', 'url', 'text'].every((key) => typeof citation[key] === 'string') && /^https?:\/\//.test(String(citation.url)) && object(citation.locator) && typeof citation.locator.label === 'string')) return false;
  const plan = value.behavior;
  if (!object(plan) || !positive(plan.duration_ms) || !Array.isArray(plan.cues)) return false;
  if (!plan.cues.every((cue) => object(cue) && positive(cue.start_ms) && positive(cue.duration_ms, false) && Number(cue.start_ms) + Number(cue.duration_ms) <= Number(plan.duration_ms) && ['face', 'gaze', 'head', 'gesture'].includes(String(cue.channel)) && typeof cue.name === 'string' && positive(cue.intensity) && cue.intensity <= 1 && ['designed', 'observed'].includes(String(cue.provenance)) && (cue.evidence_ids === undefined || strings(cue.evidence_ids)))) return false;
  if (!object(plan.speech) || !positive(plan.speech.rate, false) || !positive(plan.speech.pitch, false) || !positive(plan.speech.pause_ms)) return false;
  const grounding = value.grounding;
  if (!object(grounding) || !['grounded', 'insufficient', 'unverified'].includes(String(grounding.status)) || !strings(grounding.retrieved_ids) || !strings(grounding.invalid_citation_ids)) return false;
  const metrics = value.metrics;
  return object(metrics) && ['retrieval_ms', 'generation_ms', 'total_ms'].every((key) => positive(metrics[key]));
}

function validateEvent(event: string, data: unknown): StreamEvent | null {
  if (!['meta', 'status', 'token', 'result', 'done', 'error'].includes(event)) return null;
  if (!object(data)) throw new ApiError('Invalid event from the conversation service.');
  const valid = event === 'meta' ? typeof data.session_id === 'string'
    : event === 'status' ? typeof data.stage === 'string'
    : event === 'token' ? typeof data.text === 'string'
    : event === 'result' ? isTurn(data.turn)
    : event === 'error' ? typeof data.code === 'string' && typeof data.message === 'string'
    : true;
  if (!valid) throw new ApiError(`Invalid ${event} event from the conversation service.`);
  return { event, data } as StreamEvent;
}

/** Parses incremental SSE, including CRLF, comments and multi-line data. */
export function createEventParser(emit: (event: StreamEvent) => void) {
  let buffer = '';
  function frame(block: string) {
    let event = 'message';
    const lines: string[] = [];
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      if (line.startsWith('data:')) lines.push(line.slice(5).replace(/^ /, ''));
    }
    if (!lines.length) return;
    const data: unknown = JSON.parse(lines.join('\n'));
    const parsed = validateEvent(event, data);
    if (parsed) emit(parsed);
  }
  return {
    push(chunk: string) {
      buffer += chunk;
      buffer = buffer.replace(/\r\n/g, '\n');
      if (buffer.length > 1_048_576) throw new ApiError('The conversation stream exceeded its size limit.');
      let boundary = buffer.indexOf('\n\n');
      while (boundary !== -1) {
        frame(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        boundary = buffer.indexOf('\n\n');
      }
    },
    finish() { if (buffer.trim()) frame(buffer); buffer = ''; },
  };
}

export async function streamChat(request: ChatRequest, signal: AbortSignal, emit: (event: StreamEvent) => void) {
  const response = await fetch('/api/chat/stream', {
    method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(request), signal,
  });
  if (!response.ok) return failure(response);
  if (!response.body || !response.headers.get('content-type')?.includes('text/event-stream')) {
    throw new ApiError('The conversation service did not return a stream.');
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let completed = false;
  let receivedResult = false;
  const parser = createEventParser((event) => {
    signal.throwIfAborted();
    if (completed) throw new ApiError('The conversation service sent data after completion.');
    if (event.event === 'error') throw new ApiError(event.data.message);
    if (event.event === 'result') {
      if (receivedResult) throw new ApiError('The conversation service sent a duplicate result.');
      receivedResult = true;
    }
    if (event.event === 'done') {
      if (!receivedResult) throw new ApiError('The conversation ended without a result.');
      completed = true;
    }
    emit(event);
  });
  try {
    while (true) {
      signal.throwIfAborted();
      const part = await reader.read();
      if (part.done) break;
      parser.push(decoder.decode(part.value, { stream: true }));
    }
    parser.push(decoder.decode());
    parser.finish();
    if (!completed && !signal.aborted) throw new ApiError('The response ended early. Please retry your question.');
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

export async function getSpeech(text: string, signal: AbortSignal): Promise<Blob> {
  const response = await fetch('/api/speech', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }), signal,
  });
  if (!response.ok) return failure(response);
  const mime = response.headers.get('content-type') ?? '';
  if (!mime.startsWith('audio/')) throw new ApiError('The voice service returned an unsupported audio format.');
  const blob = await response.blob();
  if (!blob.size) throw new ApiError('The voice service returned empty audio.');
  return blob;
}

export interface PreparedSpeech {
  audio: Blob;
  audioSha256: string;
  speechTiming?: SpeechTiming;
  timingStatus: string;
  timingMessage?: string;
}

async function boundedBody(response: Response, maximum: number): Promise<Uint8Array<ArrayBuffer>> {
  if (!response.body) throw new ApiError('The voice service returned an empty response.');
  const reader = response.body.getReader(); const parts: Uint8Array[] = []; let length = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > maximum) throw new ApiError('The voice service response exceeded its size limit.');
      parts.push(value);
    }
  } finally { await reader.cancel().catch(() => undefined); reader.releaseLock(); }
  const bytes = new Uint8Array(length); let offset = 0;
  for (const part of parts) { bytes.set(part, offset); offset += part.byteLength; }
  return bytes;
}

/** Mouth cues are usable only for the exact verified audio bytes they describe. */
export async function getPreparedSpeech(text: string, language: Language, signal: AbortSignal): Promise<PreparedSpeech> {
  const response = await fetch('/api/speech/prepare', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, language }), signal,
  });
  if (!response.ok) return failure(response);
  let value: unknown;
  try { value = JSON.parse(new TextDecoder().decode(await boundedBody(response, 2_000_000))); }
  catch (error) { throw error instanceof ApiError ? error : new ApiError('The voice service returned invalid timing metadata.'); }
  if (!object(value) || typeof value.audio_url !== 'string' || !/^\/api\/speech\/clips\/[a-f0-9]{32}$/.test(value.audio_url)
    || typeof value.audio_sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(value.audio_sha256)
    || typeof value.timing_status !== 'string' || !['available', 'not_configured', 'unsupported_audio', 'unavailable', 'failed', 'timeout'].includes(value.timing_status)) {
    throw new ApiError('The voice service returned invalid audio metadata.');
  }
  const timing = value.speech_timing == null ? undefined : parseSpeechTiming(value.speech_timing);
  if ((value.timing_status === 'available') !== Boolean(timing) || (timing && timing.audio_sha256 !== value.audio_sha256)) {
    throw new ApiError('Mouth cues do not match the generated audio.');
  }
  signal.throwIfAborted();
  const clip = await fetch(value.audio_url, { signal });
  if (!clip.ok) return failure(clip);
  const mime = clip.headers.get('content-type')?.split(';')[0].trim();
  if (!mime || !['audio/wav', 'audio/x-wav', 'audio/mpeg', 'audio/ogg'].includes(mime)) throw new ApiError('The voice service returned an unsupported audio format.');
  const bytes = await boundedBody(clip, 20_000_000);
  if (!bytes.byteLength) throw new ApiError('The voice service returned empty audio.');
  signal.throwIfAborted();
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  const hash = Array.from(digest, (part) => part.toString(16).padStart(2, '0')).join('');
  if (hash !== value.audio_sha256) throw new ApiError('The downloaded audio did not match its timing metadata. Generate the voice again.');
  signal.throwIfAborted();
  return { audio: new Blob([bytes], { type: mime }), audioSha256: hash, speechTiming: timing,
    timingStatus: value.timing_status,
    timingMessage: typeof value.timing_message === 'string' ? value.timing_message.slice(0, 500) : undefined };
}

