import { useEffect, useRef, useState } from 'react';
import { AvatarRenderer } from '../avatar/renderer';
import type { BehaviorPlan, RigDiagnostics, RigMapping } from '../avatar/types';

export interface AvatarStageProps {
  speaking: boolean;
  timeline?: BehaviorPlan;
  elapsedMs?: number;
  amplitude?: number;
  modelUrl?: string;
  rigMapping?: RigMapping;
  reducedMotion?: boolean;
  onReady?: (diagnostics: RigDiagnostics) => void;
  onError?: (error: Error) => void;
  className?: string;
}

/** GPU resources are owned by one mount, including under React StrictMode. */
export function AvatarStage({ speaking, timeline, elapsedMs = 0, amplitude, modelUrl,
  rigMapping, reducedMotion, onReady, onError, className = '' }: AvatarStageProps) {
  const host = useRef<HTMLDivElement>(null);
  const runtime = useRef<AvatarRenderer | null>(null);
  const callbacks = useRef({ onReady, onError }); callbacks.current = { onReady, onError };
  const [fatal, setFatal] = useState<string | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [loadingModel, setLoadingModel] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [systemReduced, setSystemReduced] = useState(false);

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
        onReady: (diagnostics) => { if (alive) callbacks.current.onReady?.(diagnostics); },
        onError: (error) => {
          if (!alive) return;
          if (error.message.includes('graphics context')) setFatal(error.message); else setModelError(error.message);
          callbacks.current.onError?.(error);
        },
        onModelState: (state) => {
          if (!alive) return;
          setLoadingModel(state === 'loading'); if (state === 'ready') setModelError(null);
        },
      });
      runtime.current = instance;
    } catch (error) {
      const message = '3D rendering needs WebGL 2. You can continue the conversation or retry the avatar.';
      setFatal(message); callbacks.current.onError?.(error instanceof Error ? error : new Error(message));
    }
    return () => { alive = false; runtime.current?.dispose(); runtime.current = null; };
  }, [epoch]);

  useEffect(() => {
    runtime.current?.setPlayback({ speaking, timeline, elapsedMs, amplitude, reducedMotion: reducedMotion ?? systemReduced });
  }, [speaking, timeline, elapsedMs, amplitude, reducedMotion, systemReduced, epoch]);

  useEffect(() => { void runtime.current?.setModel(modelUrl, rigMapping); }, [modelUrl, rigMapping, epoch]);

  return (
    <div className={`avatar-stage ${className}`} data-testid="avatar-stage"
      style={{ position: 'relative', width: '100%', height: '100%', minHeight: 440, overflow: 'hidden', background: 'radial-gradient(ellipse at 50% 38%, #26301f 0%, #151c13 45%, #10160f 100%)' }}>
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
  );
}

export default AvatarStage;
