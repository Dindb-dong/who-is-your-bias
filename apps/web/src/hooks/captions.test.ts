import { describe, expect, it } from 'vitest';
import type { StreamEvent } from '../api';
import { CaptionTrack } from './captions';

const owner = { turn_id: 'turn-1', segment_id: 'turn-1:0', sequence: 0 };
const segment = (sequence = 0): StreamEvent => ({ event: 'segment', data: { ...owner,
  segment_id: `turn-1:${sequence}`, sequence, text: sequence ? 'Second sentence.' : 'First sentence. [source:example]', evidence_ids: [], state: 'committed' } });
const audio = (sequence = 0): StreamEvent => ({ event: 'audio_segment', data: { ...owner,
  segment_id: `turn-1:${sequence}`, sequence, sample_offset: sequence * 16000, sample_count: 16000,
  sample_rate: 16000, channels: 1, audio_sha256: 'a'.repeat(64), synthesis_ms: 10 } });
const subtitle = (sequence = 0): Extract<StreamEvent, { event: 'subtitle' }> => ({ event: 'subtitle', data: { ...owner,
  segment_id: `turn-1:${sequence}`, sequence, language: 'ko', text: sequence ? '두 번째 문장.' : '첫 번째 문장.', state: 'ready' } });
function track(live = true) {
  const result = new CaptionTrack('ko', 'en', live);
  result.accept({ event: 'meta', data: { turn_id: owner.turn_id, session_id: 'session' } });
  return result;
}

describe('sentence captions bound to the PCM media clock', () => {
  it('shows only the currently audible segment, not future generated text', () => {
    const captions = track();
    for (const event of [segment(), segment(1), subtitle(), subtitle(1), audio(), audio(1)]) captions.accept(event);
    expect(captions.active(500, false)).toBeUndefined();
    expect(captions.active(500, true)?.text).toBe('첫 번째 문장.');
    expect(captions.active(1000, true)?.text).toBe('두 번째 문장.');
    expect(captions.active(2000, true)).toBeUndefined();
    expect(captions.transcript()).toBeUndefined();
  });
  it('keeps an English fallback while translation is late, and never revives the previous sentence', () => {
    const captions = track();
    for (const event of [segment(), segment(1), audio(), audio(1)]) captions.accept(event);
    expect(captions.active(500, true)).toMatchObject({ text: 'First sentence.', language: 'en' });
    captions.accept(subtitle());
    expect(captions.active(1200, true)?.text).toBe('Second sentence.');
    captions.accept(subtitle(1));
    expect(captions.active(1200, true)?.text).toBe('두 번째 문장.');
  });
  it('makes unavailable translation visible without claiming it is Korean', () => {
    const captions = track(); captions.accept(segment()); captions.accept(audio());
    captions.accept({ event: 'subtitle', data: { ...owner, language: 'ko', text: '', state: 'unavailable', reason: 'timeout' } });
    expect(captions.active(100, true)).toMatchObject({ language: 'en', label: 'English original · Korean translation unavailable' });
  });
  it('rejects stale, uncommitted, duplicate and wrong-language translations', () => {
    const captions = track();
    expect(() => captions.accept(subtitle())).toThrow('matching committed');
    captions.accept(segment());
    expect(() => captions.accept({ event: 'subtitle', data: { ...subtitle().data, turn_id: 'old-turn' } })).toThrow('ownership');
    expect(() => captions.accept({ event: 'subtitle', data: { ...subtitle().data, language: 'en' } })).toThrow('language');
    captions.accept(subtitle()); expect(() => captions.accept(subtitle())).toThrow('duplicated');
  });
  it('labels text-only translations as unsynchronized and preserves sequence', () => {
    const captions = track(false); captions.accept(segment()); captions.accept(segment(1)); captions.accept(subtitle(1));
    expect(captions.transcript()?.text).toBe('[English original — translation pending] First sentence. 두 번째 문장.');
    captions.accept(subtitle());
    expect(captions.transcript()).toEqual({ text: '첫 번째 문장. 두 번째 문장.', language: 'ko', label: 'Translation transcript · not synchronized to audio' });
    expect(captions.active(500, true)).toBeUndefined();
  });
});
