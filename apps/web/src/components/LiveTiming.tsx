import { Download, Timer } from 'lucide-react';
import type { PcmState } from '../hooks/pcmPlayback';
import type { LiveMeasurement } from '../hooks/liveMetrics';

const duration = (value: number | null) => value === null ? '—' : value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`;

export function LiveTiming({ measurement, pcm }: { measurement: LiveMeasurement | null; pcm?: PcmState | null }) {
  if (!measurement) return null;
  const download = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify({ ...measurement, streaming_pcm: pcm ? { ...pcm, timing_basis: 'digital_Web_Audio_clock', acousticLatencyMs: null } : null, exported_at: new Date().toISOString(), environment: { user_agent: navigator.userAgent }, note: 'Browser monotonic timings from question submission. First text is a post-paint opportunity, audio signal measures Web Audio PCM; microphone capture/ASR and acoustic output are not measured.' }, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = `live-turn-${measurement.id}.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 0);
  };
  return <section className="live-timing" aria-label="Live response timing"><div className="live-timing-heading"><strong><Timer size={16}/> This turn</strong><span>{measurement.status.replaceAll('_', ' ')}{measurement.provider === 'evidence' ? ' · source preview' : ''}</span><button onClick={download} aria-label="Download this turn timing"><Download size={15}/> Export</button></div><dl><div><dt>First text</dt><dd>{duration(measurement.first_visible_text_ms)}</dd></div><div><dt>Answer ready</dt><dd>{duration(measurement.answer_complete_ms)}</dd></div><div><dt>Voice preparation</dt><dd>{duration(measurement.speech_preparation_ms)}</dd></div><div><dt>First audio signal</dt><dd>{duration(measurement.first_audio_signal_ms)}</dd></div><div><dt>Turn complete</dt><dd>{duration(measurement.turn_complete_ms)}</dd></div></dl>{pcm && <p>PCM stream: {pcm.underruns} underruns · {Math.round(pcm.queuedMs)} ms buffered · {pcm.status}. Mouth movement follows playback energy; phoneme alignment is not measured.</p>}<p>Measured here from question submission. {measurement.voice === 'browser' ? 'Browser speech exposes its start event, so audio signal timing is unavailable.' : measurement.voice === 'off' ? 'Enable automatic server voice to measure speech playback.' : 'Audio timing begins at the first non-silent playback frame.'}</p></section>;
}
