import { describe, expect, it } from 'vitest';
import { LiveTurnClock } from './liveMetrics';

describe('live turn measurements', () => {
  it('keeps result arrival separate from a delayed stream close and export time', () => {
    const clock = new LiveTurnClock('turn', 0, 'off', '2026-11-11T04:00:00Z');
    clock.answer(1000, 'openai', null); clock.streamComplete(5000);
    expect(clock.snapshot()).toMatchObject({ answer_complete_ms: 1000, stream_complete_ms: 5000, submitted_at: '2026-11-11T04:00:00Z' });
    clock.configureVoice('provider');
    expect(clock.snapshot().turn_complete_ms).toBeNull();
    clock.configureVoice('off');
    expect(clock.snapshot().turn_complete_ms).toBe(1000);
  });
  it('measures non-silent audio separately from completed text and playback start', () => {
    const clock = new LiveTurnClock('turn', 1000, 'provider');
    clock.visible(1200); clock.visible(1250);
    clock.answer(1800, 'openai', { retrieval_ms: 5, generation_ms: 750, total_ms: 760 });
    clock.playback({ type: 'preparation_started', atMs: 1800, turnId: 'turn', voiceMode: 'provider' });
    clock.playback({ type: 'audio_ready', atMs: 5000, turnId: 'turn', voiceMode: 'provider', speechServer: { synthesis_ms: 3000, alignment_ms: 100, total_ms: 3101 } });
    clock.playback({ type: 'playback_started', atMs: 5100, turnId: 'turn', voiceMode: 'provider' });
    expect(clock.snapshot().first_audio_signal_ms).toBeNull();
    clock.playback({ type: 'first_signal', atMs: 5600, turnId: 'turn', voiceMode: 'provider' });
    clock.playback({ type: 'playback_completed', atMs: 10000, turnId: 'turn', voiceMode: 'provider' });
    expect(clock.snapshot()).toMatchObject({ first_visible_text_ms: 200, answer_complete_ms: 800, speech_preparation_ms: 3200, audio_start_ms: 4100, first_audio_signal_ms: 4600, turn_complete_ms: 9000, status: 'completed' });
  });
  it('keeps unmeasured browser audio and cancelled completions null', () => {
    const clock = new LiveTurnClock('turn', 10, 'browser');
    clock.answer(20, 'evidence', null);
    clock.playback({ type: 'playback_started', atMs: 30, turnId: 'turn', voiceMode: 'browser' });
    clock.playback({ type: 'first_signal', atMs: 31, turnId: 'turn', voiceMode: 'browser' });
    clock.finish('cancelled');
    clock.playback({ type: 'playback_completed', atMs: 50, turnId: 'turn', voiceMode: 'browser' });
    expect(clock.snapshot()).toMatchObject({ first_audio_signal_ms: null, turn_complete_ms: null, status: 'cancelled', provider: 'evidence' });
  });
  it('rejects backwards/nonfinite clocks and does not fake speech when disabled', () => {
    const clock = new LiveTurnClock('turn', 100, 'provider');
    clock.visible(Number.NaN); clock.visible(50);
    clock.configureVoice('off'); clock.answer(150, 'openai', null);
    clock.playback({ type: 'playback_started', atMs: 200, turnId: 'turn', voiceMode: 'provider' });
    expect(clock.snapshot()).toMatchObject({ first_visible_text_ms: null, status: 'text_only', turn_complete_ms: 50, first_audio_signal_ms: null });
  });
});
