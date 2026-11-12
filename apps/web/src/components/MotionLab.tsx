import { useEffect, useId, useRef, useState } from 'react';
import { Play, RotateCcw, SlidersHorizontal, Square, Upload } from 'lucide-react';
import { AvatarStage } from './AvatarStage';
import { parseAvatarClip, parseRigMapping } from '../avatar/validation';
import type { AvatarClip, RigMapping } from '../avatar/types';
import type { AvatarFraming } from '../avatar/builtin';
import { useAuditionAudio } from '../avatar/useAuditionAudio';

/** Imported measurements have their own source clock, separate from conversation cues. */
export function MotionLab({ modelUrl, reducedMotion, personaId }: { modelUrl?: string; reducedMotion: boolean; personaId?: string }) {
  const isJensen = personaId === 'jensen-huang';
  const scrubberId = useId();
  const [clip, setClip] = useState<AvatarClip>();
  const [mapping, setMapping] = useState<RigMapping>();
  const [clipName, setClipName] = useState('');
  const [mappingName, setMappingName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [allowPreview, setAllowPreview] = useState(false);
  const [showCalibration, setShowCalibration] = useState(true);
  const [framing, setFraming] = useState<AvatarFraming>(isJensen ? 'upper_body' : 'full_body');
  const [loadingSample, setLoadingSample] = useState<'face' | 'body' | 'jensen' | null>(null);
  const audio = useAuditionAudio(); const [auditionStartKey, setAuditionStartKey] = useState(0);
  const clock = useRef(0); clock.current = elapsed;
  const sampleRequest = useRef<AbortController | null>(null);
  const alive = useRef(true); const importTicket = useRef({ clip: 0, mapping: 0 });
  useEffect(() => { alive.current = true; return () => { alive.current = false; sampleRequest.current?.abort(); }; }, []);
  const permitted = Boolean(clip && (clip.review.status === 'speaker_reviewed' || allowPreview));
  useEffect(() => {
    if (!playing || !clip || !permitted) return;
    let frame = 0; const start = performance.now() - clock.current;
    const tick = () => {
      const next = Math.min(clip.duration_ms, Math.max(0, performance.now() - start));
      setElapsed(next);
      if (next >= clip.duration_ms) setPlaying(false);
      else frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, clip, permitted]);

  async function importFile(file: File, kind: 'clip' | 'mapping') {
    if (kind === 'clip') sampleRequest.current?.abort();
    const ticket = ++importTicket.current[kind];
    setError(null); setPlaying(false);
    try {
      if (!file.name.toLowerCase().endsWith('.json') || file.size > (kind === 'clip' ? 8_000_000 : 100_000)) throw new Error('Choose a bounded JSON clip or rig mapping.');
      const value: unknown = JSON.parse(await file.text());
      if (!alive.current || ticket !== importTicket.current[kind]) return;
      if (kind === 'clip') { setClip(parseAvatarClip(value, true)); setClipName(file.name); setElapsed(0); setAllowPreview(false); }
      else { setMapping(parseRigMapping(value)); setMappingName(file.name); }
    } catch (reason) {
      if (alive.current && ticket === importTicket.current[kind]) setError(reason instanceof Error ? reason.message : 'This file could not be loaded.');
    }
  }

  async function loadSample(kind: 'face' | 'body' | 'jensen' = 'face') {
    sampleRequest.current?.abort();
    const controller = new AbortController(); sampleRequest.current = controller;
    setLoadingSample(kind); setError(null);
    const filename = kind === 'jensen' ? 'goldman-0164-8s.json' : kind === 'body' ? 'ted-2013-body-geometry-preview.json' : 'ted-2013-geometry-preview.json';
    if (kind === 'jensen') setFraming('upper_body');
    try {
      const response = await fetch(`${kind === 'jensen' ? '/motion/jensen' : '/samples'}/${filename}`, { signal: controller.signal });
      if (!response.ok) throw new Error('The measured sample is unavailable in this build. You can import a local clip.');
      const blob = await response.blob();
      if (!alive.current || controller.signal.aborted) return;
      await importFile(new File([blob], filename, { type: 'application/json' }), 'clip');
    } catch (reason) {
      if (alive.current && !controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'The measured sample could not be loaded.');
    } finally {
      if (sampleRequest.current === controller) { sampleRequest.current = null; if (alive.current) setLoadingSample(null); }
    }
  }

  return <section className="motion-lab" aria-labelledby="motion-lab-heading">
    <div className="motion-heading"><div><span className="eyebrow">FROM SOURCE VIDEO TO MOVEMENT</span><h2 id="motion-lab-heading">Motion & rig calibration</h2><p>{isJensen ? 'Inspect measured face, arm, hand and finger tracks from the Goldman Sachs 2024 interview. The source geometry remains an unreviewed preview. Choose a research avatar or your own rig in Studio settings.' : 'Replay measured face and pose tracks on the reference rig or your own avatar. Bundled TED clips and narration are historical Musk-era evaluation material; they are not Jensen motion or voice.'}</p></div><SlidersHorizontal size={27}/></div>
    <div className="motion-layout">
      <div className="motion-stage"><AvatarStage speaking={audio.state.status === 'playing'} modelUrl={modelUrl} rigMapping={mapping} framing={framing} portraitStyle={isJensen ? 'jensen' : 'reference'}
        audioElapsedMs={audio.state.audioElapsedMs} elapsedMs={audio.state.elapsedMs} amplitude={audio.state.amplitude}
        audioSha256={audio.state.audioSha256} speechTiming={audio.state.speechTiming} auditionStartKey={auditionStartKey}
        reducedMotion={reducedMotion} performanceClip={permitted ? clip : undefined}
        observationElapsedMs={elapsed} allowGeometryPreview={allowPreview} showCalibration={showCalibration}/></div>
      <div className="motion-controls">
        <label className="switch-row"><span>Camera framing</span><select aria-label="Motion lab camera framing" value={framing} onChange={(event) => setFraming(event.target.value as AvatarFraming)}><option value="portrait">Portrait</option><option value="upper_body">Upper body & hands</option><option value="full_body">Full body</option></select></label>
        {isJensen ? <button className="secondary-button" disabled={loadingSample !== null} onClick={() => void loadSample('jensen')}><Play size={15}/>{loadingSample === 'jensen' ? 'Loading Jensen hand sample…' : 'Load Jensen face & hands'}</button> : <>
          <button className="secondary-button" disabled={loadingSample !== null} onClick={() => void loadSample()}><Play size={15}/>{loadingSample === 'face' ? 'Loading measured sample…' : 'Load measured sample'}</button>
          <button className="secondary-button" disabled={loadingSample !== null} onClick={() => void loadSample('body')}><Play size={15}/>{loadingSample === 'body' ? 'Loading arm-motion sample…' : 'Load arm-motion sample'}</button>
        </>}
        <label className="motion-upload"><Upload size={17}/><span>Import motion clip<small>{clipName || 'Source-linked animation JSON · up to 8 MB'}</small></span><input type="file" accept=".json,application/json" aria-label="Import motion clip" onChange={(event) => { const file = event.target.files?.[0]; if (file) void importFile(file, 'clip'); event.target.value = ''; }}/></label>
        <label className="motion-upload"><Upload size={17}/><span>Import rig mapping<small>{mappingName || 'Morphs, neutral values and bone axes · JSON'}</small></span><input type="file" accept=".json,application/json" aria-label="Import rig mapping" onChange={(event) => { const file = event.target.files?.[0]; if (file) void importFile(file, 'mapping'); event.target.value = ''; }}/></label>
        <label className="motion-check"><input type="checkbox" checked={showCalibration} onChange={(event) => setShowCalibration(event.target.checked)}/> Show calibration controls</label>
        <div className="motion-source">
          <span className="eyebrow">SAME-AUDIO AVATAR AUDITION</span>
          <strong>{audio.sample?.name ?? 'Choose a speech sample'}</strong>
          <p>Replay the same WAV on each candidate. Import matching speech timing for phoneme-based articulation, or compare measured audio-energy motion.</p>
          {!isJensen && <button className="secondary-button" disabled={audio.loading} onClick={() => void audio.loadBundled()}>Load narration sample</button>}
          <label className="motion-upload"><Upload size={17}/><span>Import audition audio<small>Local WAV · up to 16 MiB</small></span><input type="file" accept=".wav,audio/wav" aria-label="Import audition audio" onChange={(event) => { const file = event.target.files?.[0]; if (file) void audio.importAudio(file, file.name); event.target.value = ''; }}/></label>
          <label className="motion-upload"><Upload size={17}/><span>Import matching speech timing<small>Optional speech-timing.v1 JSON</small></span><input type="file" accept=".json,application/json" disabled={!audio.sample} aria-label="Import audition speech timing" onChange={(event) => { const file = event.target.files?.[0]; if (file) void audio.importTiming(file); event.target.value = ''; }}/></label>
          {audio.sample && <p>{audio.sample.timingStatus === 'amplitude' ? 'Articulation: measured audio energy' : `Articulation: ${audio.sample.timingStatus}`} · {(audio.state.audioElapsedMs / 1000).toFixed(2)} s</p>}
          <div className="motion-transport">
            <button className="primary-button" disabled={!audio.sample || audio.loading} onClick={() => { setShowCalibration(true); setPlaying(false); setElapsed(0); setAuditionStartKey((key) => key + 1); void audio.play(); }}><Play size={14}/> Play & record audition</button>
            <button className="icon-button" aria-label="Stop audition audio" disabled={audio.state.status === 'idle'} onClick={audio.stop}><Square size={14}/></button>
          </div>
          <p>Export the recording from Asset audition in the rig inspection panel. Appearance ratings remain ungraded.</p>
          {audio.error && <p className="error-text" role="alert">{audio.error}</p>}
        </div>
        {clip ? <>
          <div className="motion-source"><span className="eyebrow">{clip.review.status === 'speaker_reviewed' ? 'REVIEWED SOURCE TRACK' : 'GEOMETRY PREVIEW'}</span><strong>{clip.source.source_id}</strong><p>{clip.review.attribution_basis}</p><small>{clip.frames.length} measured frames · {(clip.duration_ms / 1000).toFixed(2)} seconds</small></div>
          {clip.review.status === 'geometry_preview' && <label className="motion-check preview-choice"><input type="checkbox" checked={allowPreview} onChange={(event) => { setAllowPreview(event.target.checked); setPlaying(false); }}/> Preview geometry without speaker attribution</label>}
          <div className="motion-transport"><button className="primary-button" disabled={!permitted} onClick={() => { if (elapsed >= clip.duration_ms) setElapsed(0); setPlaying(!playing); }}>{playing ? <Square size={14}/> : <Play size={14}/>} {playing ? 'Pause replay' : 'Replay source motion'}</button><button className="icon-button" aria-label="Reset source replay" onClick={() => { setPlaying(false); setElapsed(0); }}><RotateCcw size={16}/></button></div>
          <div className="motion-scrubber"><span><label htmlFor={scrubberId}>Source position</label><output htmlFor={scrubberId}>{(elapsed / 1000).toFixed(2)} s</output></span><input id={scrubberId} type="range" min={0} max={clip.duration_ms} step={10} value={elapsed} aria-valuetext={`${(elapsed / 1000).toFixed(2)} seconds`} disabled={!permitted} onChange={(event) => { setPlaying(false); setElapsed(Number(event.target.value)); }}/></div>
          <details className="motion-details"><summary>Measurement limitations</summary><ul>{clip.limitations.map((limitation, index) => <li key={index}>{limitation}</li>)}</ul></details>
        </> : <p className="motion-empty">Import a source track to inspect its movements. Calibration controls also let you check individual channels on an avatar before replay.</p>}
        {(clip || mapping) && <button className="text-button" onClick={() => { sampleRequest.current?.abort(); ++importTicket.current.clip; ++importTicket.current.mapping; setClip(undefined); setMapping(undefined); setClipName(''); setMappingName(''); setPlaying(false); setElapsed(0); setAllowPreview(false); setError(null); }}>Clear imported files</button>}
        {error && <p className="error-text" role="alert">{error}</p>}
      </div>
    </div>
  </section>;
}
