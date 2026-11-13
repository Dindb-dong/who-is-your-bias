import { useEffect, useRef, useState } from 'react';
import { Quaternion, Vector3 } from 'three';
import type { AvatarFraming } from '../avatar/builtin';
import { BONE_CHANNELS, FACE_CHANNELS, VISEMES } from '../avatar/channels';
import { AvatarRenderer } from '../avatar/renderer';
import type { PortraitStyle } from '../avatar/procedural';
import type { AttentionTarget } from '../avatar/attention';
import { visemesToFace } from '../avatar/speech';
import { neutralPose } from '../avatar/timeline';
import { parseRigMapping } from '../avatar/validation';
import { canonicalJson, effectiveFaceControls, sha256Bytes } from '../avatar/audition';
import type { AnimationTelemetry, AvatarClip, BehaviorPlan, BoneControl, FaceChannel, QuaternionTuple, RigDiagnostics, RigMapping, SpeechTiming, Viseme } from '../avatar/types';

export interface AvatarStageProps {
  speaking: boolean;
  timeline?: BehaviorPlan;
  elapsedMs?: number;
  amplitude?: number;
  audioElapsedMs?: number;
  audioSha256?: string;
  speechTiming?: SpeechTiming;
  performanceClip?: AvatarClip;
  observationElapsedMs?: number;
  allowGeometryPreview?: boolean;
  attentionTarget?: AttentionTarget | null;
  portraitStyle?: PortraitStyle;
  showCalibration?: boolean;
  framing?: AvatarFraming;
  modelUrl?: string;
  rigMapping?: RigMapping;
  reducedMotion?: boolean;
  onReady?: (diagnostics: RigDiagnostics) => void;
  onError?: (error: Error) => void;
  onTelemetry?: (telemetry: AnimationTelemetry) => void;
  className?: string;
  /** Increment when local sample playback starts to capture from its first frames. */
  auditionStartKey?: number;
}

/** GPU resources are owned by one mount, including under React StrictMode. */
export function AvatarStage({ speaking, timeline, elapsedMs = 0, amplitude, modelUrl,
  rigMapping, reducedMotion, onReady, onError, onTelemetry, className = '',
  audioElapsedMs, audioSha256, speechTiming, performanceClip, observationElapsedMs, allowGeometryPreview = false, attentionTarget, portraitStyle = 'reference',
  showCalibration = false, framing = 'portrait', auditionStartKey = 0 }: AvatarStageProps) {
  const host = useRef<HTMLDivElement>(null);
  const runtime = useRef<AvatarRenderer | null>(null);
  const callbacks = useRef({ onReady, onError, onTelemetry, showCalibration }); callbacks.current = { onReady, onError, onTelemetry, showCalibration };
  const [fatal, setFatal] = useState<string | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [loadingModel, setLoadingModel] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [systemReduced, setSystemReduced] = useState(false);
  const [diagnostics, setDiagnostics] = useState<RigDiagnostics | null>(null);
  const [telemetry, setTelemetry] = useState<AnimationTelemetry | null>(null);
  const [debugEnabled, setDebugEnabled] = useState(false);
  const [control, setControl] = useState('face:jawOpen');
  const [weight, setWeight] = useState(0);
  const [axis, setAxis] = useState<'x' | 'y' | 'z'>('x');
  const [localMapping, setLocalMapping] = useState<RigMapping | undefined>();
  const [mappingText, setMappingText] = useState('{}');
  const [mappingError, setMappingError] = useState<string | null>(null);
  const [recordingAudition, setRecordingAudition] = useState(false);
  const [auditionStatus, setAuditionStatus] = useState<string | null>(null);
  const lastAuditionRequest = useRef(0);

  useEffect(() => { setLocalMapping(undefined); setMappingText(JSON.stringify(rigMapping ?? {}, null, 2)); }, [rigMapping]);

  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setSystemReduced(media.matches); update();
    media.addEventListener('change', update); return () => media.removeEventListener('change', update);
  }, []);

  useEffect(() => {
    const element = host.current; if (!element) return;
    let alive = true;
    setFatal(null); setModelError(null);
    try {
      const instance = new AvatarRenderer(element, {
        onReady: (diagnostics) => { if (alive) { setDiagnostics(diagnostics); callbacks.current.onReady?.(diagnostics); } },
        onError: (error) => {
          if (!alive) return;
          if (error.message.includes('graphics context')) setFatal(error.message); else setModelError(error.message);
          callbacks.current.onError?.(error);
        },
        onModelState: (state) => {
          if (!alive) return;
          setLoadingModel(state === 'loading'); if (state === 'ready') setModelError(null);
        },
        onTelemetry: (value) => {
          if (!alive) return;
          setTelemetry(value);
          callbacks.current.onTelemetry?.(value);
        },
        onAuditionState: (recording) => { if (alive) {
          setRecordingAudition(recording);
          if (!recording) setAuditionStatus('Recording stopped. Export it to keep the measurements.');
        } },
      });
      runtime.current = instance;
    } catch (error) {
      const message = '3D rendering needs WebGL 2. You can continue the conversation or retry the avatar.';
      setFatal(message); callbacks.current.onError?.(error instanceof Error ? error : new Error(message));
    }
    return () => { alive = false; runtime.current?.dispose(); runtime.current = null; };
  }, [epoch]);

  useEffect(() => {
    runtime.current?.setPlayback({ speaking, timeline, elapsedMs, amplitude, reducedMotion: reducedMotion ?? systemReduced,
      audioElapsedMs, audioSha256, speechTiming, performanceClip, observationElapsedMs, allowGeometryPreview, attentionTarget });
  }, [speaking, timeline, elapsedMs, amplitude, reducedMotion, systemReduced, epoch, audioElapsedMs, audioSha256, speechTiming, performanceClip, observationElapsedMs, allowGeometryPreview, attentionTarget]);

  useEffect(() => { runtime.current?.setPortraitStyle(portraitStyle); }, [portraitStyle, epoch]);

  useEffect(() => { void runtime.current?.setModel(modelUrl, localMapping ?? rigMapping); }, [modelUrl, localMapping, rigMapping, epoch]);

  useEffect(() => { runtime.current?.setFraming(framing); }, [framing, epoch]);

  useEffect(() => {
    if (!auditionStartKey || auditionStartKey <= lastAuditionRequest.current || !speaking || !audioSha256 || !diagnostics?.assetInspection?.sha256) return;
    lastAuditionRequest.current = auditionStartKey;
    setDebugEnabled(false); runtime.current?.setCalibrationPose();
    void runtime.current?.startAudition().then(() => setAuditionStatus('Recording this audio sample with a fixed camera.'))
      .catch((error) => setAuditionStatus(error instanceof Error ? error.message : 'Could not record audition.'));
  }, [auditionStartKey, speaking, audioSha256, diagnostics]);

  useEffect(() => {
    if (!debugEnabled || !showCalibration) { runtime.current?.setCalibrationPose(); return; }
    const pose = neutralPose(); const [group, name] = control.split(':');
    if (group === 'face') pose.face = { [name as FaceChannel]: weight };
    if (group === 'viseme') { pose.visemes = { [name as Viseme]: weight }; pose.face = visemesToFace(pose.visemes); }
    if (group === 'bone') {
      const direction = new Vector3(axis === 'x' ? 1 : 0, axis === 'y' ? 1 : 0, axis === 'z' ? 1 : 0);
      pose.bones = { [name as BoneControl]: new Quaternion().setFromAxisAngle(direction, weight * Math.PI / 3).toArray() as QuaternionTuple };
    }
    runtime.current?.setCalibrationPose(pose);
  }, [debugEnabled, showCalibration, control, weight, axis, epoch]);

  const applyMapping = () => {
    try { const mapping = parseRigMapping(JSON.parse(mappingText)); setLocalMapping(mapping); setMappingError(null); setDebugEnabled(false); }
    catch (error) { setMappingError(error instanceof Error ? error.message : 'Invalid rig mapping.'); }
  };
  const exportMapping = () => {
    try {
      const mapping = parseRigMapping(JSON.parse(mappingText));
      const url = URL.createObjectURL(new Blob([JSON.stringify(mapping, null, 2)], { type: 'application/json' }));
      const link = document.createElement('a'); link.href = url; link.download = 'avatar-rig-mapping.json'; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000); setMappingError(null);
    } catch (error) { setMappingError(error instanceof Error ? error.message : 'Invalid rig mapping.'); }
  };
  const downloadReport = (value: unknown, filename: string) => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = filename; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const exportInspection = async () => {
    try {
      if (!diagnostics?.assetInspection) throw new Error('Load a GLB to inspect its actual geometry.');
      const report = { asset: diagnostics.assetInspection, configuredMapping: diagnostics.resolvedMapping ?? {},
        resolvedTargets: diagnostics.resolvedTargets ?? [], resolvedBones: diagnostics.resolvedBones ?? [], unsupportedFace: diagnostics.unsupportedFace ?? [],
        unsupportedBones: diagnostics.unsupportedBones ?? [], effectiveFaceControls: effectiveFaceControls(diagnostics) };
      const reportSha256 = await sha256Bytes(new TextEncoder().encode(canonicalJson(report)).buffer);
      downloadReport({ report, reportSha256 }, 'avatar-asset-inspection.json'); setAuditionStatus('Asset inspection exported.');
    } catch (error) { setAuditionStatus(error instanceof Error ? error.message : 'Could not export inspection.'); }
  };
  const recordAudition = async () => {
    try {
      if (recordingAudition) { runtime.current?.stopAudition(); setAuditionStatus('Recording stopped. Export it to keep the measurements.'); }
      else { await runtime.current?.startAudition(); setAuditionStatus('Recording this audio sample with a fixed camera.'); }
    } catch (error) { setAuditionStatus(error instanceof Error ? error.message : 'Could not start audition.'); }
  };
  const exportAudition = async () => {
    try {
      const report = await runtime.current?.exportAudition();
      if (!report) throw new Error('The avatar is not ready.');
      downloadReport(report, 'avatar-audition.json'); setAuditionStatus('Audition exported. Human appearance ratings remain ungraded.');
    } catch (error) { setAuditionStatus(error instanceof Error ? error.message : 'Could not export audition.'); }
  };

  return (
    <div className={`avatar-stage${showCalibration ? ' avatar-stage--calibration' : ''} ${className}`} data-testid="avatar-stage"
      data-portrait-style={portraitStyle} data-rig-kind={diagnostics?.kind ?? 'initializing'} data-face-controls={diagnostics?.morphs.length ?? 0}
      data-visemes={diagnostics?.visemes?.length ?? 0} data-mapped-joints={diagnostics?.bones.length ?? 0}
      data-asset-sha256={diagnostics?.assetInspection?.sha256 ?? ''}
      data-framing={framing} data-model-state={loadingModel ? 'loading' : modelError ? 'fallback' : diagnostics ? 'ready' : 'initializing'}
      data-attention-phase={telemetry?.attention?.phase ?? 'idle'} data-attention-target={telemetry?.attention?.targetId ?? ''}
      data-attention-eye-yaw={telemetry?.attention?.eyeYaw ?? 0} data-attention-eye-pitch={telemetry?.attention?.eyePitch ?? 0}
      data-attention-head-yaw={telemetry?.attention?.headYaw ?? 0} data-attention-head-pitch={telemetry?.attention?.headPitch ?? 0}
      style={{ position: 'relative', width: '100%', height: showCalibration ? 'auto' : '100%', minHeight: 440, overflow: 'hidden', background: 'radial-gradient(ellipse at 50% 38%, #26301f 0%, #151c13 45%, #10160f 100%)' }}>
      <div className="avatar-visual">
        <div ref={host} style={{ position: 'absolute', inset: 0 }} aria-hidden={fatal ? true : undefined} />
        {loadingModel && <div role="status" className="avatar-model-status" style={{ position: 'absolute', bottom: 20, left: 20, color: '#d8f8a0', fontSize: 12 }}>Loading your avatar…</div>}
        {modelError && !fatal && <div role="status" className="avatar-model-status" style={{ position: 'absolute', bottom: 20, left: 20, right: 20, padding: 12, borderRadius: 12, background: '#172014ed', color: '#e3e8dc', fontSize: 12 }}>
          {modelError} The built-in portrait is still available.
        </div>}
        {fatal && <div role="status" className="avatar-fallback" style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', padding: 36, textAlign: 'center', color: '#c5ccbc', gap: 16 }}>
          <span style={{ fontSize: 12, letterSpacing: '0.16em', textTransform: 'uppercase', color: '#d8f8a0' }}>Avatar paused</span>
          <p style={{ maxWidth: 280, lineHeight: 1.7, fontSize: 14, margin: 0 }}>{fatal}</p>
          <button type="button" onClick={() => setEpoch((value) => value + 1)} style={{ background: '#d8f8a0', border: 0, borderRadius: 8, color: '#14200d', padding: '10px 16px', cursor: 'pointer', font: 'inherit', fontSize: 13 }}>Reload avatar</button>
        </div>}
      </div>
      {showCalibration && !fatal && <details className="avatar-calibration" onToggle={(event) => { if (!event.currentTarget.open) setDebugEnabled(false); }}>
        <summary style={{ cursor: 'pointer', fontWeight: 650, color: '#d8f8a0' }}>Rig inspection & calibration</summary>
        <p>{diagnostics?.morphs.length ?? 0}/52 named face controls · {diagnostics?.visemes?.length ?? 0}/15 native visemes · {diagnostics?.bones.length ?? 0} mapped joints</p>
        <p style={{ color: '#b8c4ad', lineHeight: 1.55 }}>Face: {telemetry?.faceSource ?? 'designed'} · Speech: {telemetry?.speechSource ?? 'silent'}</p>
        <p style={{ color: '#b8c4ad', lineHeight: 1.55 }}>Body: {telemetry?.boneSource ?? 'designed'}</p>
        {telemetry?.attention && telemetry.attention.phase !== 'idle' && <p style={{ color: '#b8c4ad', lineHeight: 1.55 }}>
          Designed attention: {telemetry.attention.phase}{telemetry.attention.unsupported.length > 0 ? ` · Unavailable: ${telemetry.attention.unsupported.join(', ')}` : ''}
        </p>}
        {telemetry?.warning && <p role="status" style={{ color: '#f0d6a0' }}>{telemetry.warning}</p>}
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 10 }}><input type="checkbox" checked={debugEnabled} onChange={(event) => setDebugEnabled(event.target.checked)}/> Manual pose preview</label>
        <label style={{ display: 'grid', gap: 6 }}>Control
          <select aria-label="Calibration control" value={control} onChange={(event) => { setControl(event.target.value); setWeight(0); }} style={{ maxWidth: '100%' }}>
            <optgroup label="Face channels">{FACE_CHANNELS.map((name) => <option key={name} value={`face:${name}`} disabled={!diagnostics?.morphs.includes(name)}>{name}{diagnostics?.morphs.includes(name) ? '' : ' · unavailable'}</option>)}</optgroup>
            <optgroup label="Joints">{BONE_CHANNELS.map((name) => <option key={name} value={`bone:${name}`} disabled={!diagnostics?.bones.includes(name)}>{name}{diagnostics?.bones.includes(name) ? '' : ' · unavailable'}</option>)}</optgroup>
            <optgroup label="Native visemes">{VISEMES.map((name) => <option key={name} value={`viseme:${name}`} disabled={!diagnostics?.visemes?.includes(name)}>{name}</option>)}</optgroup>
          </select>
        </label>
        {control.startsWith('bone:') && <label style={{ display: 'flex', gap: 8, marginTop: 8 }}>Local axis<select aria-label="Calibration joint axis" value={axis} onChange={(event) => setAxis(event.target.value as typeof axis)}><option>x</option><option>y</option><option>z</option></select></label>}
        <label style={{ display: 'grid', gap: 4, marginTop: 10 }}>Preview value: {weight.toFixed(2)}{control.startsWith('bone:') ? ` (${Math.round(weight * 60)}°)` : ''}
          <input aria-label="Calibration preview value" type="range" min={control.startsWith('bone:') ? -1 : 0} max={1} step={0.01} value={weight} onChange={(event) => setWeight(Number(event.target.value))}/>
        </label>
        <div style={{ display: 'flex', gap: 8, margin: '8px 0 12px' }}><button type="button" onClick={() => { setWeight(0); setDebugEnabled(true); }}>Neutral pose</button><button type="button" onClick={() => runtime.current?.resetView()}>Reset view</button></div>
        <details><summary style={{ cursor: 'pointer' }}>Mapping, neutral offsets & local axes</summary>
          <p style={{ lineHeight: 1.5 }}>Map actual target names. Morph ranges use inputNeutral/inputMax/outputNeutral; joint neutral and basis are XYZW quaternions.</p>
          <label style={{ display: 'grid', gap: 6 }}>Rig mapping JSON<textarea aria-label="Rig mapping JSON" rows={8} spellCheck={false} value={mappingText} onChange={(event) => setMappingText(event.target.value)} style={{ width: '100%', boxSizing: 'border-box', fontFamily: 'monospace', fontSize: 11 }}/></label>
          <div style={{ display: 'flex', gap: 8, marginTop: 8 }}><button type="button" onClick={applyMapping}>Apply mapping</button><button type="button" onClick={exportMapping}>Export mapping</button></div>
          {mappingError && <p role="alert" style={{ color: '#efb5a1' }}>{mappingError}</p>}
        </details>
        <details style={{ marginTop: 10 }}><summary style={{ cursor: 'pointer' }}>Available asset targets</summary>
          <p style={{ overflowWrap: 'anywhere', lineHeight: 1.6 }}>{diagnostics?.availableMorphs?.slice(0, 240).map((entry) => `${entry.mesh}/${entry.target}`).join(', ') || 'The built-in rig exposes its supported controls above.'}</p>
          <p style={{ overflowWrap: 'anywhere', lineHeight: 1.6 }}>Asset nodes: {diagnostics?.availableNodes?.slice(0, 240).join(', ') || diagnostics?.bones.join(', ')}</p>
          <p style={{ overflowWrap: 'anywhere', lineHeight: 1.6 }}>Unsupported face channels: {diagnostics?.unsupportedFace?.join(', ') || 'none reported'}</p>
          {diagnostics?.warnings?.map((warning, index) => <p key={index} style={{ color: '#f0d6a0' }}>{warning}</p>)}
        </details>
        <details style={{ marginTop: 10 }}><summary style={{ cursor: 'pointer' }}>Asset audition</summary>
          <p style={{ lineHeight: 1.5 }}>Compare imported avatars using the same recorded audio and framing. Start the audio, then record; the camera stays fixed during the sample.</p>
          {diagnostics?.assetInspection && <p style={{ lineHeight: 1.5 }}>
            {diagnostics.assetInspection.meshes.length} meshes · {effectiveFaceControls(diagnostics).length} face controls with measured vertex movement · {diagnostics.assetInspection.animations.length} embedded animations
          </p>}
          <p style={{ lineHeight: 1.5 }}>The export records file, mapping and audio digests, frame intervals and draw counts. Likeness and material appearance need a separate review.</p>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <button type="button" disabled={!diagnostics?.assetInspection} onClick={() => void exportInspection()}>Export asset inspection</button>
            <button type="button" disabled={!recordingAudition && (!speaking || !audioSha256 || !diagnostics?.assetInspection?.sha256 || debugEnabled)} onClick={() => void recordAudition()}>{recordingAudition ? 'Stop audition' : 'Record audition'}</button>
            <button type="button" onClick={() => void exportAudition()}>Export audition</button>
          </div>
          {auditionStatus && <p role="status" style={{ lineHeight: 1.5 }}>{auditionStatus}</p>}
        </details>
      </details>}
    </div>
  );
}

export default AvatarStage;

