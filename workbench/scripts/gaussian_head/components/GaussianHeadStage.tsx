import { useEffect, useRef, useState } from 'react';
import type { SpeechTiming } from '../avatar/types';
import {
  blendMouth, createTrackedMouth, energyMouth, mouthTarget, nodPitch, restingMouth, speechNod,
} from '../avatar/gaussian/gaussianSpeech';

// Based on the browser head stage from who-is-your-bias PR #21.
interface Props {
  bundle: string;
  speaking: boolean;
  amplitude: number;
  audioElapsedMs: number;
  speechTiming?: SpeechTiming;
  audioSha256?: string;
  reducedMotion: boolean;
  onReady: () => void;
  onError: (error: Error) => void;
}

export function GaussianHeadStage({ bundle, speaking, amplitude, audioElapsedMs, speechTiming, audioSha256, reducedMotion, onReady, onError }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const live = useRef({ speaking, amplitude, audioElapsedMs, speechTiming, audioSha256, reducedMotion });
  live.current = { speaking, amplitude, audioElapsedMs, speechTiming, audioSha256, reducedMotion };
  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading');

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const controller = new AbortController();
    let raf = 0;
    let dispose: (() => void) | undefined;
    const fail = (error: unknown) => {
      if (controller.signal.aborted) return;
      setState('error');
      onError(error instanceof Error ? error : new Error(String(error)));
    };
    const contextLost = (event: Event) => {
      event.preventDefault();
      cancelAnimationFrame(raf);
      fail(new Error('Gaussian head rendering stopped. Try Photo-fit v4 or select the Gaussian head again.'));
    };
    canvas.addEventListener('webglcontextlost', contextLost);

    void (async () => {
      const [core, mod] = await Promise.all([
        import('../avatar/gaussian/gaussianHeadCore'), import('../avatar/gaussian/gaussianHeadRenderer'),
      ]);
      const { header, B } = await core.loadBundle('/models/gaussian-head/' + bundle, controller.signal);
      if (controller.signal.aborted) return;
      const options = new URLSearchParams(window.location.search);
      const useSpeechCues = options.get('mouth') !== 'energy';
      const cleanMatte = options.get('matte') !== 'original';
      const mode = options.get('motion');
      const motion = mode === 'tracked' || mode === 'still' ? mode : 'subtle';
      if (!(B.v_shaped instanceof Float32Array) || header.n_verts !== 5143) {
        throw new Error('Unsupported Gaussian face topology');
      }
      const applyMouth = createTrackedMouth((index) => core.frameParams(B, header, index));
      const renderer = new mod.GaussianHeadRenderer(canvas, header, B, cleanMatte);
      dispose = () => renderer.dispose();
      const camera = {
        view: mod.lookAt(header.camera.position, header.camera.target),
        fovY: header.camera.fov_y_deg * Math.PI / 180,
      };
      const params = { expr: new Float32Array(header.n_expr), pose: new Float32Array(15), trans: new Float32Array(3) };
      // Frame 1083 is a near-frontal, level-eye pose from the bundle's training portion.
      const restFrame = 1083;
      const rest = core.frameParams(B, header, restFrame);
      let frame = 0, first = true, talking = false, startFrame = 0, last = performance.now();
      let mouth = restingMouth;
      let gestureTime = 0;
      const smoothedOffsets = new Float32Array(15);
      let nod: ReturnType<typeof speechNod> = null;
      function render(now: number) {
        if (controller.signal.aborted) return;
        const dt = Math.min(0.1, (now - last) / 1000);
        last = now;
        const { speaking: talk, amplitude: energy, audioElapsedMs: elapsed, speechTiming: cues, audioSha256: hash, reducedMotion: reduced } = live.current;
        const target = talk && useSpeechCues ? mouthTarget(cues, hash, elapsed) : null;
        if (talk && !talking) {
          startFrame = frame;
          nod = speechNod(cues, hash);
        }
        talking = talk;
        if (motion !== 'tracked') {
          params.pose.set(rest.pose);
          params.trans.set(rest.trans);
          if (motion === 'subtle' && !reduced) {
            gestureTime += dt;
            // Follow a small neighborhood near the frontal pose without replaying
            // its jaw, expression or translation tracks.
            const t = restFrame + 23 * Math.sin(gestureTime * 0.91);
            const frameIndex = Math.floor(t);
            const from = core.frameParams(B, header, frameIndex).pose;
            const to = core.frameParams(B, header, frameIndex + 1).pose;
            const mixIn = 1 - Math.exp(-dt * 6);
            for (let channel = 0; channel < params.pose.length; channel++) {
              if (channel >= 6 && channel < 9) continue;
              const limit = channel < 6 ? 0.034 : 0.024;
              const delta = (from[channel] + (to[channel] - from[channel]) * (t - frameIndex) - rest.pose[channel]) * 0.31;
              const bounded = Math.max(-limit, Math.min(limit, delta));
              smoothedOffsets[channel] += (bounded - smoothedOffsets[channel]) * mixIn;
              params.pose[channel] += smoothedOffsets[channel];
            }
          }
          if (talk && !reduced) params.pose[3] += nodPitch(nod, elapsed);
        } else {
          if (!reduced) {
            frame = talk ? startFrame + Math.max(0, elapsed) * header.fps / 1000 : frame + dt * header.fps * 0.35;
          }
          const tracked = core.frameParams(B, header, Math.floor(frame));
          params.pose.set(tracked.pose);
          params.trans.set(tracked.trans);
        }
        mouth = blendMouth(mouth, target?.weights ?? (talk ? energyMouth(energy) : restingMouth), dt);
        applyMouth(params, mouth);
        const dpr = Math.min(1.5, window.devicePixelRatio || 1);
        const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
        const height = Math.max(1, Math.round(canvas.clientHeight * dpr));
        if (canvas.width !== width || canvas.height !== height) {
          canvas.width = width;
          canvas.height = height;
        }
        renderer.render(params, camera);
        canvas.dataset.mouthMode = target ? 'speech-cues' : talk ? 'audio-energy' : 'idle';
        canvas.dataset.viseme = target?.label ?? '';
        canvas.dataset.headMotion = motion;
        canvas.dataset.headPose = Array.from(params.pose.subarray(0, 6)).map(value => value.toFixed(5)).join(',');
        canvas.dataset.matte = cleanMatte ? 'clean' : 'original';
        canvas.dataset.speaking = String(talk);
        canvas.dataset.audioClockMs = elapsed.toFixed(1);
        canvas.dataset.jaw = params.pose[6].toFixed(4);
        if (first) {
          first = false;
          setState('ready');
          onReady();
        }
        raf = requestAnimationFrame(tick);
      }
      const tick = (now: number) => {
        try { render(now); } catch (error) { fail(error); }
      };
      raf = requestAnimationFrame(tick);
    })().catch(fail);
    return () => {
      controller.abort();
      cancelAnimationFrame(raf);
      canvas.removeEventListener('webglcontextlost', contextLost);
      dispose?.();
    };
  }, [bundle, onReady, onError]);

  return (
    <div className="avatar-stage" data-testid="gaussian-head-stage" data-state={state}
      style={{ position: 'relative', minHeight: 440, height: '100%', overflow: 'hidden',
        background: 'radial-gradient(ellipse at 50% 38%, #26301f 0%, #151c13 45%, #10160f 100%)' }}>
      <canvas ref={canvasRef} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
        aria-label="Gaussian head preview with speech-driven mouth motion"/>
    </div>
  );
}
