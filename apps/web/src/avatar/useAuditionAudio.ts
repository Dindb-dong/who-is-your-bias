import { useCallback, useEffect, useRef, useState } from 'react';
import { browserPlaybackEnvironment, createPlaybackController, initialPlayback, type PlaybackController, type PlaybackState } from '../hooks/playback';
import { attachAuditionTiming, auditionPlaybackTurn, MAX_AUDITION_AUDIO_BYTES, prepareAuditionAudio, type AuditionAudio } from './auditionAudio';

/** Uses the same cancellable Web Audio/HTMLAudioElement transport as conversation
 * speech, with local bytes and no synthesis-service request. */
export function useAuditionAudio() {
  const [sample, setSample] = useState<AuditionAudio>();
  const current = useRef<AuditionAudio | undefined>(undefined);
  const controller = useRef<PlaybackController | null>(null);
  const [state, setState] = useState<PlaybackState>(initialPlayback);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const active = useRef(true); const revision = useRef(0); const download = useRef<AbortController | null>(null);
  useEffect(() => {
    active.current = true;
    const environment = browserPlaybackEnvironment();
    const prepared = async () => { if (!current.current) throw new Error('Choose an audition audio sample.'); return current.current; };
    const instance = createPlaybackController(setState, { ...environment, prepareSpeech: prepared, getSpeech: async () => (await prepared()).audio });
    controller.current = instance;
    return () => { active.current = false; ++revision.current; download.current?.abort(); instance.dispose(); controller.current = null; };
  }, []);

  const importAudio = useCallback(async (file: Blob, name: string) => {
    const ticket = ++revision.current; download.current?.abort(); controller.current?.stop(); setError(null); setLoading(true);
    try {
      const prepared = await prepareAuditionAudio(file, name);
      if (active.current && revision.current === ticket) { current.current = prepared; setSample(prepared); }
    } catch (reason) { if (active.current && revision.current === ticket) setError(reason instanceof Error ? reason.message : 'Could not load sample.'); }
    finally { if (active.current && revision.current === ticket) setLoading(false); }
  }, []);

  const loadBundled = useCallback(async () => {
    const ticket = ++revision.current; download.current?.abort(); const request = new AbortController(); download.current = request;
    controller.current?.stop(); setLoading(true); setError(null);
    try {
      const response = await fetch('/presentation/narration.wav', { signal: request.signal });
      if (!response.ok || !response.body) throw new Error('The bundled narration is unavailable. Import a local WAV sample.');
      const reader = response.body.getReader(); const chunks: Uint8Array<ArrayBuffer>[] = []; let bytes = 0;
      try {
        while (true) {
          const result = await reader.read(); if (result.done) break;
          bytes += result.value.byteLength; if (bytes > MAX_AUDITION_AUDIO_BYTES) throw new Error('Audio exceeds the 16 MiB audition limit.');
          chunks.push(new Uint8Array(result.value));
        }
      } finally { await reader.cancel().catch(() => undefined); reader.releaseLock(); }
      const prepared = await prepareAuditionAudio(new Blob(chunks, { type: 'audio/wav' }), 'Bundled generated narration');
      if (active.current && revision.current === ticket) { current.current = prepared; setSample(prepared); }
    } catch (reason) { if (active.current && revision.current === ticket && !request.signal.aborted) setError(reason instanceof Error ? reason.message : 'Could not load sample.'); }
    finally { if (active.current && revision.current === ticket) setLoading(false); }
  }, []);

  const importTiming = useCallback(async (file: File) => {
    const ticket = ++revision.current; const selected = current.current; download.current?.abort();
    controller.current?.stop(); setLoading(false); setError(null);
    try {
      if (!selected) throw new Error('Choose the audio sample first.');
      if (file.size > 2_000_000) throw new Error('Speech timing must be a JSON file up to 2 MB.');
      const updated = attachAuditionTiming(selected, JSON.parse(await file.text()));
      if (active.current && revision.current === ticket && current.current === selected) {
        controller.current?.stop(); current.current = updated; setSample(updated);
      }
    } catch (reason) { if (active.current && revision.current === ticket) setError(reason instanceof Error ? reason.message : 'Invalid speech timing.'); }
  }, []);
  const play = useCallback(() => {
    if (current.current) { setError(null); return controller.current?.play(auditionPlaybackTurn(current.current), 'provider'); }
  }, []);
  const stop = useCallback(() => controller.current?.stop(), []);
  return { sample, state, error: error ?? state.error, loading, importAudio, loadBundled, importTiming, play, stop };
}
