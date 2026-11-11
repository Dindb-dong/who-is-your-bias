import { useEffect, useState } from 'react';
import { ArrowRight, Clapperboard, Download, RefreshCw } from 'lucide-react';
import { getJson } from '../api';

interface Presentation {
  schema_version: 'presentation.v1'; title: string; video_url: string; audio_url: string;
  captions_url: string; duration_seconds: number; backend: string; disclosure: string;
  generated_at: string; timing_verified: boolean;
}

export function parsePresentation(value: unknown): Presentation {
  if (!value || typeof value !== 'object') throw new Error('Invalid presentation metadata.');
  const item = value as Record<string, unknown>;
  if (item.schema_version !== 'presentation.v1' || !['title', 'backend', 'disclosure', 'generated_at'].every((key) => typeof item[key] === 'string' && (item[key] as string).length > 0)
    || !['video_url', 'audio_url', 'captions_url'].every((key) => typeof item[key] === 'string' && /^\/presentation\/[a-zA-Z0-9_-]+\.(mp4|wav|vtt)$/.test(item[key] as string))
    || typeof item.duration_seconds !== 'number' || !Number.isFinite(item.duration_seconds) || item.duration_seconds <= 0 || item.duration_seconds > 3600 || typeof item.timing_verified !== 'boolean') throw new Error('Invalid presentation metadata.');
  return item as unknown as Presentation;
}

export function PresentationView({ onAsk }: { onAsk: () => void }) {
  const [presentation, setPresentation] = useState<Presentation | null>(null);
  const [error, setError] = useState(''); const [reload, setReload] = useState(0);
  useEffect(() => {
    const controller = new AbortController(); setError(''); setPresentation(null);
    getJson<unknown>('/presentation/manifest.json', controller.signal).then(parsePresentation).then(setPresentation).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Presentation files are unavailable.');
    });
    return () => controller.abort();
  }, [reload]);
  return <section className="presentation-page"><div className="section-heading"><div><span className="eyebrow">PREPARED TO PLAY. READY FOR QUESTIONS.</span><h1>Presentation & live Q&A</h1><p>Play the prepared introduction, then open a new conversation for audience questions.</p></div><Clapperboard size={40} strokeWidth={1.1}/></div>
    {error ? <div className="presentation-error" role="alert"><h2>Presentation unavailable</h2><p>{error}</p><button className="secondary-button" onClick={() => setReload((value) => value + 1)}><RefreshCw size={15}/> Try again</button></div> : !presentation ? <p role="status">Loading the prepared presentation…</p> : <>
      <div className="presentation-player"><video controls playsInline preload="metadata" poster="/presentation/portrait.png" aria-label="Prepared AI presentation" onError={() => setError('The prepared video could not be loaded. Check the presentation package.')}><source src={presentation.video_url} type="video/mp4"/><track src={presentation.captions_url} kind="captions" srcLang="en" label="English"/></video><div className="presentation-disclosure">{presentation.disclosure}</div></div>
      <div className="presentation-details"><div><span className="eyebrow">PREPARED INTRODUCTION</span><h2>{presentation.title}</h2><p>{presentation.backend} · {presentation.duration_seconds.toFixed(2)} seconds · {presentation.timing_verified ? 'Audio/video duration checked' : 'Timing verification pending'}</p><div className="presentation-downloads"><a href={presentation.video_url} download><Download size={15}/> Video</a><a href={presentation.audio_url} download><Download size={15}/> Audio</a><a href={presentation.captions_url} download><Download size={15}/> Captions</a></div></div><button className="primary-button" onClick={onAsk}>Open live Q&A <ArrowRight size={16}/></button></div>
      <div className="research-note"><Clapperboard size={19}/><p>The introduction is generated in advance. Live questions use the conversation model and configured voice service. Enable automatic server voice in Studio settings to measure question-to-audio timing on this device.</p></div>
    </>}
  </section>;
}
