import type { AvatarClip, BehaviorPlan, MotionMode, MotionSelection } from './avatar/types';
import type { SpeechTiming } from './avatar/types';
import { parseAvatarClip, parseMotionSelection, parseSpeechTiming } from './avatar/validation';
import type { SpeechServerTiming } from './hooks/liveMetrics';
import type { PcmChunk } from './hooks/pcmPlayback';

export type Language = 'en' | 'ko';
export type ExperimentMode = 'profile' | 'rag' | 'behavior';
export interface Source {
  id: string; title: string; url: string; publisher: string; published_at: string;
  source_type: string; split: 'train' | 'validation' | 'test'; attribution: string;
}
export interface Locator { label: string; start_seconds?: number | null; end_seconds?: number | null; timeline?: string | null }
export interface Citation { evidence_id: string; source_id: string; title: string; url: string; locator: Locator; text: string }
export interface Grounding {
  status: 'grounded' | 'insufficient' | 'unverified'; retrieved_ids: string[]; invalid_citation_ids: string[];
  reason?: string | null;
  citation_status?: 'complete' | 'partial' | 'missing' | 'invalid' | 'not_required';
  citation_unit_count?: number; cited_unit_count?: number; malformed_citations?: string[];
  semantic_status?: 'unreviewed' | 'curated_source';
}
export interface Turn {
  id: string; session_id: string; text: string; language: Language; mode: ExperimentMode;
  provider: 'evidence' | 'openai'; citations: Citation[]; behavior: BehaviorPlan & { speech: { rate: number; pitch: number; pause_ms: number } };
  grounding: Grounding;
  metrics: { retrieval_ms: number; generation_ms: number; total_ms: number };
  context?: { snapshot?: { id: string; as_of: string } | null; retrieval_backend?: string; query_resolution?: { original: string; query: string; method: string; supporting_turn_ids: string[] } } | null;
}
export interface Persona {
  id: string; name: string; tagline: string; description: string;
  example_questions: { en: string[]; ko: string[] };
  stats: { sources: number; evidence: number; held_out_sources: number };
  qa_corpus?: { corpus_id: string | null; materialization: string; question_answer_records: number; training_question_answers: number; complete_exchanges: number; excerpts: number };
  capabilities: { llm: boolean; speech: boolean; transcription?: boolean; live_speech?: boolean };
  snapshot?: { id: string; as_of: string; current_dated_training_sources: number; coverage_note: string } | null;
  notice: string;
  presentation?: { initials: string; topics: string[]; avatar_status: string; avatar_label: string; motion_status: string; voice_status: string };
  bindings?: { persona_id: string; voice: { status: string; persona_id: string | null; voice: string }; motion: { status: string; persona_id: string | null } };
}
export interface ChatRequest { message: string; session_id?: string; mode: ExperimentMode; language: Language; motion_mode?: MotionMode }
export type StreamEvent =
  | { event: 'meta'; data: { session_id: string; turn_id?: string; snapshot_id?: string | null; protocol?: string } }
  | { event: 'status'; data: { stage: string } }
  | { event: 'token'; data: { text: string } }
  | { event: 'result'; data: { turn: Turn } }
  | { event: 'segment'; data: { turn_id: string; segment_id: string; sequence: number; text: string; evidence_ids: string[]; state: 'committed' } }
  | { event: 'audio_segment'; data: { turn_id: string; segment_id: string; sequence: number; audio_sha256: string; sample_rate: number; channels: number; sample_offset: number; sample_count: number; synthesis_ms: number } }
  | { event: 'audio'; data: PcmChunk }
  | { event: 'audio_segment_end'; data: { turn_id: string; segment_id: string; sequence: number; sample_end: number; state: 'delivered' } }
  | { event: 'audio_end'; data: { turn_id: string; chunks: number; sample_count: number; sample_rate: number | null; first_pcm_ready_ms: number | null; total_ms: number; state: 'delivered'; spoken_complete: false; acoustic_latency_ms: null } }
  | { event: 'turn_state'; data: { turn_id: string | null; state: 'failed'; text_complete: boolean; delivered_audio_chunks: number; spoken_complete: false } }
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
  // Older servers omit these additive diagnostics. Present fields must still
  // be validated before a coverage count or semantic label reaches the UI.
  if (grounding.reason != null && typeof grounding.reason !== 'string') return false;
  if (grounding.citation_status !== undefined && !['complete', 'partial', 'missing', 'invalid', 'not_required'].includes(String(grounding.citation_status))) return false;
  if (grounding.semantic_status !== undefined && !['unreviewed', 'curated_source'].includes(String(grounding.semantic_status))) return false;
  if (grounding.malformed_citations !== undefined && !strings(grounding.malformed_citations)) return false;
  for (const key of ['citation_unit_count', 'cited_unit_count']) {
    if (grounding[key] !== undefined && (!positive(grounding[key]) || !Number.isSafeInteger(grounding[key]))) return false;
  }
  if (typeof grounding.citation_unit_count === 'number' && typeof grounding.cited_unit_count === 'number' && grounding.cited_unit_count > grounding.citation_unit_count) return false;
  if (grounding.semantic_status === 'curated_source' && value.provider !== 'evidence') return false;
  if (plan.motion != null) {
    if (value.mode !== 'behavior' || grounding.retrieved_ids.length === 0) return false;
    try { parseMotionSelection(plan.motion); } catch { return false; }
  }
  const metrics = value.metrics;
  return object(metrics) && ['retrieval_ms', 'generation_ms', 'total_ms'].every((key) => positive(metrics[key]));
}

function validateEvent(event: string, data: unknown): StreamEvent | null {
  if (!['meta', 'status', 'token', 'result', 'done', 'error', 'segment', 'audio_segment', 'audio', 'audio_segment_end', 'audio_end', 'turn_state'].includes(event)) return null;
  if (!object(data)) throw new ApiError('Invalid event from the conversation service.');
  const integer = (value: unknown) => positive(value) && Number.isSafeInteger(value);
  const identity = (value: unknown) => typeof value === 'string' && value.length > 0;
  const rate = (value: unknown) => integer(value) && Number(value) >= 8000 && Number(value) <= 48000;
  const segment = () => identity(data.turn_id) && identity(data.segment_id) && integer(data.sequence);
  const valid = event === 'meta' ? identity(data.session_id) && (data.turn_id === undefined || identity(data.turn_id)) && (data.protocol === undefined || typeof data.protocol === 'string')
    : event === 'status' ? typeof data.stage === 'string'
    : event === 'token' ? typeof data.text === 'string'
    : event === 'result' ? isTurn(data.turn)
    : event === 'error' ? typeof data.code === 'string' && typeof data.message === 'string'
    : event === 'segment' ? segment() && typeof data.text === 'string' && data.text.length <= 6000 && strings(data.evidence_ids) && data.state === 'committed'
    : event === 'audio_segment' ? segment() && typeof data.audio_sha256 === 'string' && /^[a-f0-9]{64}$/.test(data.audio_sha256) && rate(data.sample_rate) && [1, 2].includes(Number(data.channels)) && integer(data.sample_offset) && integer(data.sample_count) && Number(data.sample_count) > 0 && Number(data.sample_count) <= Number(data.sample_rate) * 180 && positive(data.synthesis_ms)
    : event === 'audio' ? segment() && integer(data.segment_sequence) && integer(data.sample_offset) && integer(data.sample_count) && Number(data.sample_count) > 0 && Number(data.sample_count) <= 16384 && rate(data.sample_rate) && [1, 2].includes(Number(data.channels)) && data.encoding === 'pcm_s16le' && typeof data.pcm_base64 === 'string' && data.pcm_base64.length <= 90000
    : event === 'audio_segment_end' ? segment() && integer(data.sample_end) && data.state === 'delivered'
    : event === 'audio_end' ? identity(data.turn_id) && integer(data.chunks) && integer(data.sample_count) && (data.sample_rate === null || rate(data.sample_rate)) && (data.first_pcm_ready_ms === null || positive(data.first_pcm_ready_ms)) && positive(data.total_ms) && data.state === 'delivered' && data.spoken_complete === false && data.acoustic_latency_ms === null
    : event === 'turn_state' ? (data.turn_id === null || typeof data.turn_id === 'string') && data.state === 'failed' && typeof data.text_complete === 'boolean' && integer(data.delivered_audio_chunks) && data.spoken_complete === false
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

/** Bind live PCM to this response's committed text before it reaches playback. */
class LiveStreamIntegrity {
  private owner: { turnId: string; sessionId: string } | undefined;
  private segments: { id: string; sequence: number }[] = [];
  private segmentIds = new Set<string>();
  private nextSegment = 0;
  private nextChunk = 0;
  private nextSample = 0;
  private format: [number, number] | undefined;
  private active: Extract<StreamEvent, { event: 'audio_segment' }>['data'] | undefined;
  private textComplete = false;
  private audioComplete = false;
  private failed = false;

  accept(event: StreamEvent): void {
    const invalid = (detail: string): never => { throw new ApiError(`Invalid live voice stream: ${detail}.`); };
    if (event.event === 'meta') {
      if (this.owner || !event.data.turn_id || event.data.protocol !== 'live-segments-pcm.v1') invalid('missing or changed turn ownership');
      this.owner = { turnId: event.data.turn_id!, sessionId: event.data.session_id };
      return;
    }
    if (!this.owner) invalid('data arrived before turn ownership');
    const owner = this.owner!;
    if (this.failed) invalid('data arrived after failure');
    if ('turn_id' in event.data && event.data.turn_id !== owner.turnId) invalid('turn identity changed');
    if (this.audioComplete && ['segment', 'audio_segment', 'audio', 'audio_segment_end', 'audio_end'].includes(event.event)) invalid('audio arrived after completion');
    switch (event.event) {
      case 'result':
        if (event.data.turn.id !== owner.turnId || event.data.turn.session_id !== owner.sessionId) invalid('result ownership changed');
        this.textComplete = true;
        break;
      case 'token':
        if (this.textComplete) invalid('text arrived after its result');
        break;
      case 'segment': {
        const data = event.data;
        if (this.textComplete || data.sequence !== this.nextSegment || this.segmentIds.has(data.segment_id)) invalid('committed segments are reordered or duplicated');
        this.segments.push({ id: data.segment_id, sequence: data.sequence });
        this.segmentIds.add(data.segment_id); this.nextSegment++;
        break;
      }
      case 'audio_segment': {
        const data = event.data; const committed = this.segments[0];
        if (this.active || !committed || data.segment_id !== committed.id || data.sequence !== committed.sequence) invalid('audio does not match the next committed segment');
        if (data.sample_offset !== this.nextSample || !Number.isSafeInteger(data.sample_offset + data.sample_count)) invalid('segment sample boundary changed');
        if (this.format && (data.sample_rate !== this.format[0] || data.channels !== this.format[1])) invalid('PCM format changed');
        this.format = [data.sample_rate, data.channels]; this.active = data;
        break;
      }
      case 'audio': {
        const data = event.data; const active = this.active;
        if (!active || data.segment_id !== active.segment_id || data.segment_sequence !== active.sequence) invalid('PCM has no matching committed segment');
        if (data.sequence !== this.nextChunk || data.sample_offset !== this.nextSample || data.sample_offset + data.sample_count > active!.sample_offset + active!.sample_count) invalid('PCM sequence or sample boundary changed');
        if (data.sample_rate !== active!.sample_rate || data.channels !== active!.channels) invalid('PCM format differs from its segment');
        this.nextChunk++; this.nextSample += data.sample_count;
        break;
      }
      case 'audio_segment_end': {
        const data = event.data; const active = this.active;
        if (!active || data.segment_id !== active.segment_id || data.sequence !== active.sequence || data.sample_end !== this.nextSample || this.nextSample !== active.sample_offset + active.sample_count) invalid('segment ended with missing or unexpected samples');
        this.active = undefined; this.segments.shift();
        break;
      }
      case 'audio_end': {
        const data = event.data;
        if (!this.textComplete || this.active || this.segments.length) invalid('audio ended before all committed segments were delivered');
        if (data.chunks !== this.nextChunk || data.sample_count !== this.nextSample || data.sample_rate !== (this.format?.[0] ?? null)) invalid('audio completion counts or format differ');
        if ((this.nextChunk === 0) !== (data.first_pcm_ready_ms === null)) invalid('first PCM timing disagrees with delivery');
        this.audioComplete = true;
        break;
      }
      case 'turn_state':
        if (event.data.delivered_audio_chunks !== this.nextChunk) invalid('failure delivery count differs');
        this.failed = true;
        break;
      case 'done':
        if (!this.textComplete || !this.audioComplete) invalid('turn completed before text and audio delivery');
        break;
    }
  }
}

export async function streamChat(request: ChatRequest, signal: AbortSignal, emit: (event: StreamEvent) => unknown | Promise<unknown>, options: { live?: boolean } = {}) {
  const response = await fetch(options.live ? '/api/chat/live' : '/api/chat/stream', {
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
  let receivedAudioEnd = false;
  const liveIntegrity = options.live ? new LiveStreamIntegrity() : undefined;
  const pending: StreamEvent[] = [];
  const parser = createEventParser((event) => {
    signal.throwIfAborted();
    if (completed) throw new ApiError('The conversation service sent data after completion.');
    if (event.event === 'error') throw new ApiError(event.data.message);
    liveIntegrity?.accept(event);
    if (event.event === 'result') {
      if (receivedResult) throw new ApiError('The conversation service sent a duplicate result.');
      receivedResult = true;
    }
    if (event.event === 'done') {
      if (!receivedResult) throw new ApiError('The conversation ended without a result.');
      if (options.live && !receivedAudioEnd) throw new ApiError('The conversation ended without complete audio delivery.');
      completed = true;
    }
    if (event.event === 'audio_end') {
      if (receivedAudioEnd) throw new ApiError('The voice stream sent a duplicate completion.');
      receivedAudioEnd = true;
    }
    pending.push(event);
  });
  async function deliver() {
    for (const event of pending.splice(0)) { signal.throwIfAborted(); await emit(event); }
  }
  try {
    while (true) {
      signal.throwIfAborted();
      const part = await reader.read();
      if (part.done) break;
      parser.push(decoder.decode(part.value, { stream: true }));
      await deliver();
    }
    parser.push(decoder.decode());
    parser.finish();
    await deliver();
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
  serverTiming?: SpeechServerTiming;
}

async function boundedBody(response: Response, maximum: number, service = 'voice service'): Promise<Uint8Array<ArrayBuffer>> {
  if (!response.body) throw new ApiError(`The ${service} returned an empty response.`);
  const reader = response.body.getReader(); const parts: Uint8Array[] = []; let length = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > maximum) throw new ApiError(`The ${service} response exceeded its size limit.`);
      parts.push(value);
    }
  } finally { await reader.cancel().catch(() => undefined); reader.releaseLock(); }
  const bytes = new Uint8Array(length); let offset = 0;
  for (const part of parts) { bytes.set(part, offset); offset += part.byteLength; }
  return bytes;
}

export interface MotionLibraryStatus {
  available: boolean;
  status: 'available' | 'not_configured' | 'invalid_catalog';
  id?: string; version?: string;
  reviewed_runtime_primitives: number; preview_runtime_primitives: number;
}

export async function getMotionLibrary(signal: AbortSignal): Promise<MotionLibraryStatus> {
  const response = await fetch('/api/motion/library', { signal, redirect: 'error' });
  if (!response.ok) return failure(response);
  const value: unknown = JSON.parse(new TextDecoder().decode(await boundedBody(response, 1_000_000, 'motion service')));
  if (!object(value) || typeof value.available !== 'boolean' || !['available', 'not_configured', 'invalid_catalog'].includes(String(value.status))) throw new ApiError('Invalid motion library status.');
  for (const key of ['reviewed_runtime_primitives', 'preview_runtime_primitives']) {
    if (!positive(value[key]) || !Number.isSafeInteger(value[key]) || value[key] > 20000) throw new ApiError('Invalid motion library counts.');
  }
  if ((value.available && (value.status !== 'available' || typeof value.id !== 'string' || typeof value.version !== 'string'))
    || (!value.available && (value.status === 'available' || value.preview_runtime_primitives !== 0 || value.reviewed_runtime_primitives !== 0))
    || Number(value.reviewed_runtime_primitives) > Number(value.preview_runtime_primitives)) throw new ApiError('Inconsistent motion library status.');
  signal.throwIfAborted();
  return value as unknown as MotionLibraryStatus;
}

/** Verify raw canonical response bytes before trusting any measured channels. */
export async function getMotionClip(selection: MotionSelection, mode: MotionMode, signal: AbortSignal): Promise<AvatarClip> {
  const chosen = parseMotionSelection(selection);
  const preview = mode === 'geometry_preview';
  if (!['reviewed', 'geometry_preview'].includes(mode) || (chosen.review_status === 'geometry_preview' && !preview)) {
    throw new ApiError('Source geometry requires an explicit preview opt-in.');
  }
  signal.throwIfAborted();
  const response = await fetch(chosen.clip_url + (preview ? '?preview=true' : ''), { signal, redirect: 'error' });
  if (!response.ok) return failure(response);
  if (response.headers.get('content-type')?.split(';')[0].trim() !== 'application/json') throw new ApiError('The motion service returned an unsupported format.');
  const bytes = await boundedBody(response, 8 * 1024 * 1024, 'motion service');
  signal.throwIfAborted();
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  const checksum = Array.from(digest, (part) => part.toString(16).padStart(2, '0')).join('');
  if (checksum !== chosen.clip_sha256) throw new ApiError('The downloaded motion did not match its selection hash.');
  const clip = parseAvatarClip(JSON.parse(new TextDecoder().decode(bytes)), preview);
  if (clip.id !== chosen.primitive_id || clip.source.source_id !== chosen.source_id || clip.source.source_timeline !== chosen.source_timeline
    || Math.abs(clip.duration_ms - chosen.duration_ms) > 0.01 || Math.abs(clip.source.source_start_seconds - chosen.source_start_seconds) > 0.00001
    || clip.review.status !== chosen.review_status || clip.calibration.parent_reference?.clip_sha256 !== chosen.parent_clip_sha256) {
    throw new ApiError('Motion provenance does not match the selected source window.');
  }
  const bodyBones = new Set(['chest', 'leftUpperArm', 'rightUpperArm', 'leftForearm', 'rightForearm', 'leftHand', 'rightHand']);
  for (const frame of clip.frames) {
    if ((chosen.family === 'upper_body' && frame.face && Object.keys(frame.face).length > 0)
      || Object.keys(frame.bones ?? {}).some((name) => chosen.family === 'face' ? name !== 'head' : !bodyBones.has(name))) {
      throw new ApiError('Motion channels do not match the selected geometry family.');
    }
  }
  signal.throwIfAborted();
  return clip;
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
  if (value.server_timing !== undefined && (!object(value.server_timing) || !['synthesis_ms', 'alignment_ms', 'total_ms'].every((key) => positive((value.server_timing as Record<string, unknown>)[key])))) throw new ApiError('The voice service returned invalid performance timing.');
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
    serverTiming: value.server_timing as SpeechServerTiming | undefined,
    timingStatus: value.timing_status,
    timingMessage: typeof value.timing_message === 'string' ? value.timing_message.slice(0, 500) : undefined };
}


