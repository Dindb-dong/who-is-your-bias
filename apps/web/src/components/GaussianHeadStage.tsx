import { useEffect, useRef, useState } from 'react';

/* Local preview of a FLAME-bound Gaussian head (gaussian-head.v1 bundle).
 * The renderer code is lazy-loaded; the person-specific bundle is fetched at runtime
 * from an ignored public folder, so it never enters the app build or git. */
const BASE = '/models/gaussian-head/';

interface Props { speaking: boolean; amplitude?: number; bundle?: string; className?: string; onReady?: () => void; onError?: (error: Error) => void }

type Params = { expr: Float32Array; pose: Float32Array; trans: Float32Array };

export function GaussianHeadStage({ speaking, amplitude = 0, bundle = 'jensen_a', className = '', onReady, onError }: Props) {
  const callbacks = useRef({ onReady, onError }); callbacks.current = { onReady, onError };
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const live = useRef({ speaking, amplitude }); live.current = { speaking, amplitude };
  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [message, setMessage] = useState('');
  const [fps, setFps] = useState(0);

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    let alive = true, raf = 0;
    (async () => {
      try {
        const core = await import('../avatar/gaussian/gaussianHeadCore');
        const mod = await import('../avatar/gaussian/gaussianHeadRenderer');
        // ?headBundle=<name> picks another local bundle (letters, digits, _ and - only).
        const override = new URLSearchParams(window.location.search).get('headBundle');
        const name = override && /^[\w-]+$/.test(override) ? override : bundle;
        const { header, B } = await core.loadBundle(BASE + name);
        if (!alive) return;
        const renderer = new mod.GaussianHeadRenderer(canvas, header, B);
        const q = new URLSearchParams(window.location.search);
        const dist = Number(q.get('headDistance')) || 1;            // >1 = smaller head on screen
        const resScale = Math.min(1, Math.max(0.25, Number(q.get('headRes')) || 1)); // <1 = render fewer pixels, then upscale
        const cam = header.camera, t = cam.target as number[], p = cam.position as number[];
        const eye = p.map((v, i) => t[i] + (v - t[i]) * dist);
        const view = mod.lookAt(eye, t), fovY = cam.fov_y_deg * Math.PI / 180;
        const E = header.n_expr;
        const cur: Params = { expr: new Float32Array(E), pose: new Float32Array(15), trans: new Float32Array(3) };
        let frame = 0, last = performance.now(), mix = 0, avg = 0, shown = 0;
        setState('ready'); callbacks.current.onReady?.();
        const tick = (now: number) => {
          if (!alive) return;
          const dt = Math.min(0.1, (now - last) / 1000); last = now;
          const { speaking: talk, amplitude: amp } = live.current;
          // speaking: replay the tracked talking motion; idle: same head motion, mouth/expressions relaxed.
          mix += ((talk ? 1 : 0) - mix) * Math.min(1, dt * 6);
          frame += dt * header.fps * (talk ? 1 : 0.35);
          const src: Params = core.frameParams(B, header, Math.floor(frame));
          const k = 0.25 + 0.75 * mix, jaw = 0.15 + 0.85 * mix * (amp > 0 ? Math.min(1.2, 0.6 + amp * 2) : 1);
          for (let i = 0; i < E; i++) cur.expr[i] = src.expr[i] * k;
          for (let i = 0; i < 15; i++) cur.pose[i] = src.pose[i] * (i >= 6 && i < 9 ? jaw : 1);
          cur.trans.set(src.trans);
          const dpr = Math.min(1.5, window.devicePixelRatio || 1) * resScale;
          const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
          if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
          renderer.render(cur, { view, fovY });
          avg = avg * 0.95 + (1 / Math.max(dt, 1e-3)) * 0.05;
          if (now - shown > 500) { shown = now; setFps(Math.round(avg)); }
          raf = requestAnimationFrame(tick);
        };
        raf = requestAnimationFrame(tick);
      } catch (error) {
        if (!alive) return;
        setState('error'); setMessage(error instanceof Error ? error.message : String(error));
        callbacks.current.onError?.(error instanceof Error ? error : new Error(String(error)));
      }
    })();
    return () => { alive = false; cancelAnimationFrame(raf); };
  }, [bundle]);

  return (
    <div className={`avatar-stage ${className}`} data-testid="gaussian-head-stage" data-state={state}
      style={{ position: 'relative', width: '100%', height: '100%', minHeight: 440, overflow: 'hidden', background: 'radial-gradient(ellipse at 50% 38%, #26301f 0%, #151c13 45%, #10160f 100%)' }}>
      <canvas ref={canvasRef} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }} aria-label="Synthetic 3D head preview"/>
      <div style={{ position: 'absolute', right: 16, top: 72, padding: '6px 10px', borderRadius: 8, background: '#0b1009cc', color: '#d8f8a0', fontSize: 11, lineHeight: 1.5 }}>
        Synthetic likeness · not the real person · research preview{state === 'ready' ? ` · ${fps} fps` : ''}
      </div>
      {state === 'loading' && <div role="status" style={{ position: 'absolute', top: 20, left: 20, color: '#d8f8a0', fontSize: 12 }}>Loading 3D head…</div>}
      {state === 'error' && <div role="status" style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', color: '#efb5a1', padding: 24, textAlign: 'center' }}>Could not load the 3D head: {message}</div>}
    </div>
  );
}

export default GaussianHeadStage;
