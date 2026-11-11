import { describe, expect, it } from 'vitest';
import { attachAuditionTiming, auditionPlaybackTurn, prepareAuditionAudio } from './auditionAudio';

function wave(): Blob {
  const data = new Uint8Array(364); const view = new DataView(data.buffer);
  for (const [offset, text] of [[0, 'RIFF'], [8, 'WAVEfmt '], [36, 'data']] as const) {
    [...text].forEach((char, i) => data[offset + i] = char.charCodeAt(0));
  }
  view.setUint32(4, 356, true); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, 16000, true); view.setUint32(28, 32000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  view.setUint32(40, 320, true); return new Blob([data]);
}

describe('local audio audition input', () => {
  it('hashes actual WAV bytes and leaves phoneme alignment explicitly unavailable until supplied', async () => {
    const prepared = await prepareAuditionAudio(wave(), 'sample.wav');
    expect(prepared.audioSha256).toMatch(/^[a-f0-9]{64}$/);
    expect(prepared.timingStatus).toBe('amplitude'); expect(prepared.speechTiming).toBeUndefined();
    expect(auditionPlaybackTurn(prepared).id).toContain(prepared.audioSha256);
    expect(auditionPlaybackTurn(prepared).behavior.duration_ms).toBe(0); // Keep the actual audio clock.
  });
  it('rejects empty, non-WAV and truncated audio', async () => {
    await expect(prepareAuditionAudio(new Blob(), 'empty.wav')).rejects.toThrow();
    await expect(prepareAuditionAudio(new Blob(['x'.repeat(100)]), 'text.wav')).rejects.toThrow('RIFF');
    await expect(prepareAuditionAudio(wave().slice(0, 100), 'truncated.wav')).rejects.toThrow('truncated');
  });
  it('rejects valid timing for different bytes and accepts a matching hash without claiming listening quality', async () => {
    const prepared = await prepareAuditionAudio(wave(), 'sample.wav');
    const timing = { schema_version: 'speech-timing.v1', audio_sha256: 'b'.repeat(64), audio_duration_ms: 10,
      timing_basis: 'manual', label_set: 'viseme_15', provenance: { engine: 'synthetic-test-fixture' },
      segments: [{ start_ms: 0, end_ms: 10, label: 'sil' }] };
    expect(() => attachAuditionTiming(prepared, timing)).toThrow('different audio');
    const accepted = attachAuditionTiming(prepared, { ...timing, audio_sha256: prepared.audioSha256 });
    expect(accepted.speechTiming?.audio_sha256).toBe(prepared.audioSha256); expect(accepted.timingStatus).toBe('manual');
  });
});
