import type { PreparedSpeech, Turn } from '../api';
import { sha256Bytes } from './audition';
import { parseSpeechTiming } from './validation';

export const MAX_AUDITION_AUDIO_BYTES = 16 * 1024 * 1024;
export interface AuditionAudio extends PreparedSpeech { name: string }

export async function prepareAuditionAudio(blob: Blob, name: string): Promise<AuditionAudio> {
  if (!blob.size || blob.size > MAX_AUDITION_AUDIO_BYTES) throw new Error('Choose a WAV sample up to 16 MiB.');
  const bytes = await blob.arrayBuffer(); const header = new Uint8Array(bytes);
  const magic = (offset: number, text: string) => [...text].every((char, index) => header[offset + index] === char.charCodeAt(0));
  if (bytes.byteLength < 44 || !magic(0, 'RIFF') || !magic(8, 'WAVE')) throw new Error('Choose a standard RIFF/WAVE audio file.');
  if (new DataView(bytes).getUint32(4, true) + 8 > bytes.byteLength) throw new Error('The WAV sample is truncated.');
  return { name, audio: new Blob([bytes], { type: 'audio/wav' }), audioSha256: await sha256Bytes(bytes),
    timingStatus: 'amplitude', timingMessage: 'Mouth motion uses measured audio energy until matching speech timing is imported.' };
}

export function attachAuditionTiming(sample: AuditionAudio, value: unknown): AuditionAudio {
  const timing = parseSpeechTiming(value);
  if (timing.audio_sha256 !== sample.audioSha256) throw new Error('Speech timing belongs to a different audio file.');
  return { ...sample, speechTiming: timing, timingStatus: timing.timing_basis, timingMessage: undefined };
}

/** Existing transport consumes this envelope only to play the provided local
 * bytes. It never sends this diagnostic turn to the conversation service. */
export function auditionPlaybackTurn(sample: AuditionAudio): Turn {
  return { id: `audition-${sample.audioSha256}`, session_id: 'local-avatar-audition', text: 'Local audition audio',
    language: 'en', mode: 'profile', provider: 'evidence', citations: [],
    behavior: { duration_ms: 0, cues: [], speech: { rate: 1, pitch: 1, pause_ms: 0 } },
    grounding: { status: 'unverified', retrieved_ids: [], invalid_citation_ids: [] },
    metrics: { retrieval_ms: 0, generation_ms: 0, total_ms: 0 } };
}
