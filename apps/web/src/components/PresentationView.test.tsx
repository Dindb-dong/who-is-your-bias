import { describe, expect, it } from 'vitest';
import { parsePresentation } from './PresentationView';

const manifest = { schema_version: 'presentation.v1', title: 'Research intro', backend: 'Qwen3-TTS → MuseTalk', disclosure: 'AI simulation', generated_at: '2026-11-11', video_url: '/presentation/presentation.mp4', audio_url: '/presentation/narration.wav', captions_url: '/presentation/captions.vtt', duration_seconds: 12.24, timing_verified: true };
describe('prepared presentation metadata', () => {
  it('accepts verified local package links', () => expect(parsePresentation(manifest).duration_seconds).toBe(12.24));
  it.each(['https://unknown.example/clip.mp4', '/presentation/../secret.mp4', 'javascript:alert(1)', '/presentation/presentation.mp4?redirect=1'])('rejects unexpected media path %s', (video_url) => expect(() => parsePresentation({ ...manifest, video_url })).toThrow());
  it('does not claim timing from malformed duration or missing verification', () => {
    expect(() => parsePresentation({ ...manifest, duration_seconds: Number.NaN })).toThrow();
    expect(() => parsePresentation({ ...manifest, timing_verified: 'yes' })).toThrow();
  });
});
