/** One monotonic browser clock; missing measurements remain null. */
export type PlaybackMark = 'preparation_started' | 'audio_ready' | 'playback_started' | 'first_signal' | 'playback_completed' | 'failed' | 'cancelled';
export interface SpeechServerTiming { synthesis_ms: number; alignment_ms: number; total_ms: number }
export interface PlaybackEvent { type: PlaybackMark; atMs: number; turnId: string; voiceMode: 'browser' | 'provider'; speechServer?: SpeechServerTiming }
export interface LiveMeasurement {
  schema_version: 'live-turn.v1'; id: string; input_basis: 'question_submitted';
  submitted_at: string;
  voice: 'off' | 'browser' | 'provider'; provider: 'openai' | 'evidence' | null;
  first_visible_text_ms: number | null; answer_complete_ms: number | null;
  stream_complete_ms: number | null;
  speech_preparation_ms: number | null; audio_start_ms: number | null;
  first_audio_signal_ms: number | null; turn_complete_ms: number | null;
  status: 'running' | 'text_only' | 'speaking' | 'completed' | 'cancelled' | 'failed';
  server: { retrieval_ms: number; generation_ms: number; total_ms: number } | null;
  speech_server: SpeechServerTiming | null;
  audio_basis: 'not_requested' | 'browser_start_event_no_pcm' | 'web_audio_signal';
}

export class LiveTurnClock {
  private preparationAt: number | null = null;
  readonly startedAt: number;
  private value: LiveMeasurement;
  constructor(id: string, startedAt: number, voice: LiveMeasurement['voice'], submittedAt = new Date().toISOString()) {
    if (!Number.isFinite(startedAt)) throw new Error('Invalid monotonic start');
    this.startedAt = startedAt;
    this.value = { schema_version: 'live-turn.v1', id, input_basis: 'question_submitted', submitted_at: submittedAt, voice, provider: null,
      first_visible_text_ms: null, answer_complete_ms: null, stream_complete_ms: null, speech_preparation_ms: null, audio_start_ms: null,
      first_audio_signal_ms: null, turn_complete_ms: null, status: 'running', server: null, speech_server: null,
      audio_basis: voice === 'off' ? 'not_requested' : voice === 'browser' ? 'browser_start_event_no_pcm' : 'web_audio_signal' };
  }
  snapshot(): LiveMeasurement { return { ...this.value, server: this.value.server ? { ...this.value.server } : null, speech_server: this.value.speech_server ? { ...this.value.speech_server } : null }; }
  configureVoice(voice: LiveMeasurement['voice']): void {
    if (!['running', 'text_only'].includes(this.value.status) || this.preparationAt !== null) return;
    this.value.voice = voice; this.value.audio_basis = voice === 'off' ? 'not_requested' : voice === 'browser' ? 'browser_start_event_no_pcm' : 'web_audio_signal';
    if (this.value.answer_complete_ms !== null) {
      this.value.status = voice === 'off' ? 'text_only' : 'running';
      this.value.turn_complete_ms = voice === 'off' ? this.value.answer_complete_ms : null;
    }
  }
  private elapsed(at: number): number | null { return Number.isFinite(at) && at >= this.startedAt ? Math.round((at - this.startedAt) * 1000) / 1000 : null; }
  visible(at: number): void {
    if (this.value.status === 'cancelled' || this.value.status === 'failed') return;
    if (this.value.first_visible_text_ms === null) this.value.first_visible_text_ms = this.elapsed(at);
  }
  answer(at: number, provider: LiveMeasurement['provider'], server: LiveMeasurement['server']): void {
    if (this.value.status !== 'running') return;
    this.value.answer_complete_ms = this.elapsed(at); this.value.provider = provider; this.value.server = server;
    if (this.value.voice === 'off') { this.value.status = 'text_only'; this.value.turn_complete_ms = this.elapsed(at); }
  }
  streamComplete(at: number): void { this.value.stream_complete_ms = this.elapsed(at); }
  playback(event: PlaybackEvent): void {
    if (this.value.status === 'cancelled' || this.value.status === 'failed' || this.value.status === 'completed' || this.value.voice === 'off') return;
    const elapsed = this.elapsed(event.atMs);
    if (elapsed === null || event.voiceMode !== this.value.voice) return;
    if (event.type === 'preparation_started') this.preparationAt = event.atMs;
    if (event.type === 'audio_ready' && this.preparationAt !== null && event.atMs >= this.preparationAt) this.value.speech_preparation_ms = Math.round((event.atMs - this.preparationAt) * 1000) / 1000;
    if (event.type === 'audio_ready' && event.speechServer) this.value.speech_server = { ...event.speechServer };
    if (event.type === 'playback_started') { this.value.audio_start_ms ??= elapsed; this.value.status = 'speaking'; }
    if (event.type === 'first_signal' && event.voiceMode === 'provider' && this.value.audio_start_ms !== null) this.value.first_audio_signal_ms ??= elapsed;
    if (event.type === 'playback_completed') { this.value.turn_complete_ms = elapsed; this.value.status = 'completed'; }
    if (event.type === 'cancelled' || event.type === 'failed') this.value.status = event.type;
  }
  finish(status: 'cancelled' | 'failed'): void {
    if (this.value.status === 'completed' || (this.value.status === 'text_only' && status === 'cancelled')) return;
    this.value.status = status; this.value.turn_complete_ms = null;
  }
}
