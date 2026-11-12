import { useCallback, useEffect, useRef, useState } from 'react';
import { Camera, Eye, MonitorUp, MousePointer2, Square, X } from 'lucide-react';
import { CaptureLease, captureError, requestScreenCapture } from '../attention/capture';
import { LocalFaceDetector } from '../attention/detector';
import { containRect, displayBox, FaceTracker } from '../attention/tracker';
import type { AttentionTarget, FaceTrack, TrackSnapshot } from '../attention/types';
import './AttentionPanel.css';

export interface AttentionPanelProps {
  enabled: boolean;
  reducedMotion: boolean;
  onTarget: (target: AttentionTarget | null) => void;
}
type Input = 'off' | 'camera' | 'screen' | 'demo';
type Phase = 'off' | 'permission' | 'loading' | 'running' | 'error' | 'paused';
const clamp = (value: number) => Math.min(0.96, Math.max(0.04, value));
const MAX_FRAME_SIDE = 640;
const FRAME_INTERVAL_MS = 100;
const MAX_RESULT_AGE_MS = 350;

/** Live capture is optional and never started by mounting or changing a setting. */
export function AttentionPanel({ enabled, reducedMotion, onTarget }: AttentionPanelProps) {
  const [input, setInput] = useState<Input>('off');
  const [phase, setPhase] = useState<Phase>('off');
  const [notice, setNotice] = useState('Choose an input, then select the participant to follow.');
  const [tracks, setTracks] = useState<FaceTrack[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [point, setPoint] = useState({ x: 0.72, y: 0.36 });
  const [sourceSize, setSourceSize] = useState({ width: 640, height: 360 });
  const [previewSize, setPreviewSize] = useState({ width: 640, height: 360 });
  const video = useRef<HTMLVideoElement>(null);
  const preview = useRef<HTMLDivElement>(null);
  const tracker = useRef(new FaceTracker());
  const detector = useRef<LocalFaceDetector | null>(null);
  const lease = useRef<CaptureLease | null>(null);
  const frameId = useRef<number | null>(null);
  const demoTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const generation = useRef(0);
  const alive = useRef(true);
  const onTargetRef = useRef(onTarget); onTargetRef.current = onTarget;
  const inputRef = useRef<Input>('off');
  const pointRef = useRef(point); pointRef.current = point;

  const stop = useCallback((message = 'Tracking is off. Choose an input to begin.', nextPhase: Phase = 'off') => {
    generation.current++;
    if (frameId.current !== null) cancelAnimationFrame(frameId.current);
    frameId.current = null;
    if (demoTimer.current !== null) clearInterval(demoTimer.current);
    demoTimer.current = null;
    detector.current?.close(); detector.current = null;
    lease.current?.stop();
    if (video.current) { video.current.pause(); video.current.srcObject = null; }
    tracker.current.reset(); inputRef.current = 'off';
    onTargetRef.current(null);
    if (alive.current) { setInput('off'); setPhase(nextPhase); setNotice(message); setTracks([]); setSelected(null); }
  }, []);

  if (!lease.current) lease.current = new CaptureLease(() => stop('The video input ended. Choose an input to restart.'));

  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; stop(); };
  }, [stop]);
  useEffect(() => { if (!enabled) stop(); }, [enabled, stop]);
  useEffect(() => {
    const visibility = () => { if (document.hidden && inputRef.current !== 'off') stop('Tracking paused while this page was hidden. Choose an input to restart.', 'paused'); };
    document.addEventListener('visibilitychange', visibility);
    return () => document.removeEventListener('visibilitychange', visibility);
  }, [stop]);
  useEffect(() => {
    const element = preview.current; if (!element) return;
    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (rect?.width && rect.height) setPreviewSize({ width: rect.width, height: rect.height });
    });
    observer.observe(element); return () => observer.disconnect();
  }, []);

  const present = useCallback((snapshot: TrackSnapshot, ticket: number, source: Input) => {
    if (!alive.current || ticket !== generation.current) return;
    setTracks(snapshot.tracks); setSelected(snapshot.selected?.id ?? null);
    if (snapshot.selected) {
      const box = displayBox(snapshot.selected.box, source === 'camera');
      onTargetRef.current({ id: `capture-${ticket}-${snapshot.selected.id}`, x: box.x + box.width / 2,
        y: box.y + box.height / 2, observedAtMs: snapshot.selected.observedAtMs });
      setNotice(snapshot.selected.missing ? 'Participant briefly out of view. Holding the selection.' : `Following participant ${snapshot.selected.id} · eyes first, then head.`);
    } else {
      onTargetRef.current(null);
      setNotice(snapshot.selectionLost ? 'Selection lost or ambiguous. Select a visible participant again.'
        : snapshot.tracks.some((track) => !track.missing) ? 'Select a numbered participant to begin following.' : 'Looking for faces. Move closer or choose a clearer view.');
    }
  }, []);

  async function start(source: 'camera' | 'screen') {
    if (!enabled || document.hidden) return;
    stop();
    const ticket = generation.current;
    inputRef.current = source; setInput(source); setPhase('permission');
    setNotice(source === 'camera' ? 'Allow this page to use the camera.' : 'Choose a tab, window, or screen to share.');
    const devices = navigator.mediaDevices;
    if (!devices || (source === 'camera' ? !devices.getUserMedia : !devices.getDisplayMedia)) {
      stop(source === 'camera' ? 'Camera capture needs HTTPS or localhost and a supported browser. Try the pointer preview.' : 'Screen sharing is unavailable here. Try Camera or the pointer preview.', 'error'); return;
    }
    try {
      // Keep this request ahead of every await, including model initialization.
      const stream = await lease.current!.start(() => source === 'camera'
        ? devices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user', frameRate: { ideal: 30, max: 30 } }, audio: false })
        : requestScreenCapture(devices));
      if (!stream || ticket !== generation.current || !alive.current) return;
      const element = video.current;
      if (!element) { stop(); return; }
      setPhase('loading'); setNotice('Loading the local face detector…');
      element.srcObject = stream;
      await element.play();
      if (ticket !== generation.current || !alive.current) return;
      setSourceSize({ width: element.videoWidth || 640, height: element.videoHeight || 360 });
      const instance = new LocalFaceDetector(); detector.current = instance;
      await instance.ready;
      if (ticket !== generation.current || !alive.current) { instance.close(); return; }
      setPhase('running');
      let busy = false; let lastFrame = -Infinity; let lastVideoTime = -1; let lastSnapshot = -Infinity;
      let previousWidth = element.videoWidth; let previousHeight = element.videoHeight;
      const tick = (now: number) => {
        if (ticket !== generation.current || !alive.current) return;
        frameId.current = requestAnimationFrame(tick);
        if (now - lastSnapshot >= FRAME_INTERVAL_MS) {
          lastSnapshot = now; present(tracker.current.snapshot(now), ticket, source);
        }
        if (busy || element.readyState < 2 || element.currentTime === lastVideoTime || now - lastFrame < FRAME_INTERVAL_MS) return;
        if (!element.videoWidth || !element.videoHeight) return;
        if (element.videoWidth !== previousWidth || element.videoHeight !== previousHeight) {
          previousWidth = element.videoWidth; previousHeight = element.videoHeight;
          tracker.current.reset(); onTargetRef.current(null);
          setSourceSize({ width: previousWidth, height: previousHeight });
        }
        lastFrame = now; lastVideoTime = element.currentTime; busy = true;
        const capturedWidth = element.videoWidth; const capturedHeight = element.videoHeight;
        const scale = Math.min(1, MAX_FRAME_SIDE / Math.max(capturedWidth, capturedHeight));
        void createImageBitmap(element, { resizeWidth: Math.max(1, Math.round(element.videoWidth * scale)),
          resizeHeight: Math.max(1, Math.round(element.videoHeight * scale)), resizeQuality: 'low' })
          .then(async (bitmap) => {
            if (ticket !== generation.current || !alive.current) { bitmap.close(); return; }
            const result = await instance.detect(bitmap, now);
            if (ticket !== generation.current || !alive.current || performance.now() - result.observedAtMs > MAX_RESULT_AGE_MS
              || element.videoWidth !== capturedWidth || element.videoHeight !== capturedHeight) return;
            present(tracker.current.update(result.boxes, result.observedAtMs), ticket, source);
          }).catch((error: unknown) => {
            if (ticket === generation.current && alive.current) stop(captureError(error), 'error');
          }).finally(() => { busy = false; });
      };
      frameId.current = requestAnimationFrame(tick);
    } catch (error) {
      if (ticket === generation.current && alive.current) stop(captureError(error), 'error');
    }
  }

  function startDemo() {
    if (!enabled || document.hidden) return;
    stop();
    const ticket = generation.current;
    inputRef.current = 'demo'; setInput('demo'); setPhase('running');
    setNotice('Pointer preview · move the point to direct the avatar.');
    const update = () => {
      if (ticket === generation.current && alive.current) onTargetRef.current({ id: `demo-${ticket}`, ...pointRef.current, observedAtMs: performance.now() });
    };
    update(); demoTimer.current = setInterval(update, 80);
  }

  function selectParticipant(id: number | null) {
    present(tracker.current.select(id, performance.now()), generation.current, inputRef.current);
  }

  function movePoint(x: number, y: number) {
    const next = { x: clamp(x), y: clamp(y) }; pointRef.current = next; setPoint(next);
    if (inputRef.current === 'demo') onTargetRef.current({ id: `demo-${generation.current}`, ...next, observedAtMs: performance.now() });
  }

  const rect = containRect(sourceSize.width, sourceSize.height, previewSize.width, previewSize.height);
  const capturing = input === 'camera' || input === 'screen';
  return <section className="attention-panel" aria-label="Avatar attention" data-testid="attention-panel" data-input={input} data-phase={phase} data-selected-participant={selected ?? ''}>
    <div className="attention-heading"><div><span className="eyebrow">LOOK, THEN TURN</span><h3><Eye size={17}/> Avatar attention</h3></div><span className={`attention-state ${input !== 'off' ? 'active' : ''}`}>{input === 'demo' ? 'Preview' : phase === 'running' ? 'Local capture' : phase === 'permission' ? 'Permission' : phase === 'loading' ? 'Loading' : 'Off'}</span></div>
    <p className="attention-intro">Choose a participant. The eyes notice them first, and the head follows.</p>
    <div className="attention-controls"><button type="button" onClick={() => void start('camera')} disabled={!enabled || input === 'camera'}><Camera size={15}/> Camera</button><button type="button" onClick={() => void start('screen')} disabled={!enabled || input === 'screen'}><MonitorUp size={15}/> Share screen</button><button type="button" onClick={startDemo} disabled={!enabled || input === 'demo'}><MousePointer2 size={15}/> Try pointer preview</button>{input !== 'off' && <button type="button" className="attention-stop" onClick={() => stop()} aria-label="Stop attention tracking"><Square size={13}/> Stop</button>}</div>
    <div ref={preview} className="attention-preview" hidden={!capturing} data-testid="attention-video-preview">
      <div className="attention-video-content" style={{ left: rect.x, top: rect.y, width: rect.width, height: rect.height }}>
        <video ref={video} muted playsInline aria-label={input === 'screen' ? 'Shared screen preview' : 'Camera preview'} style={{ transform: input === 'camera' ? 'scaleX(-1)' : undefined }}/>
        {capturing && tracks.filter((track) => !track.missing).map((track) => {
          const box = displayBox(track.box, input === 'camera');
          return <button key={track.id} type="button" className={`attention-face ${selected === track.id ? 'selected' : ''}`} aria-label={`Follow participant ${track.id}`} aria-pressed={selected === track.id} onClick={() => selectParticipant(track.id)} style={{ left: `${box.x * 100}%`, top: `${box.y * 100}%`, width: `${box.width * 100}%`, height: `${box.height * 100}%` }}><span>{track.id}</span></button>;
        })}
      </div>
      {phase !== 'running' && <span className="attention-preview-message">{phase === 'permission' ? 'Waiting for your video choice…' : 'Starting local detection…'}</span>}
    </div>
    {capturing && phase === 'running' && <div className="attention-participants" aria-label="Visible participants">{tracks.filter((track) => !track.missing).map((track) => <button key={track.id} type="button" onClick={() => selectParticipant(track.id)} aria-pressed={selected === track.id}>Participant {track.id}</button>)}{selected !== null && <button type="button" onClick={() => selectParticipant(null)}><X size={13}/> Clear selection</button>}</div>}
    {input === 'demo' && <div className="attention-demo"><div className="attention-demo-pad" tabIndex={0} role="group" aria-label="Attention preview target. Use arrow keys to move." data-testid="attention-demo-pad" onPointerMove={(event) => {
      const bounds = event.currentTarget.getBoundingClientRect(); if (bounds.width && bounds.height) movePoint((event.clientX - bounds.left) / bounds.width, (event.clientY - bounds.top) / bounds.height);
    }} onKeyDown={(event) => {
      const delta = event.shiftKey ? 0.16 : 0.06;
      const directions: Record<string, [number, number]> = { ArrowLeft: [-delta, 0], ArrowRight: [delta, 0], ArrowUp: [0, -delta], ArrowDown: [0, delta] };
      if (directions[event.key]) { event.preventDefault(); const [x, y] = directions[event.key]; movePoint(pointRef.current.x + x, pointRef.current.y + y); }
      if (event.key === 'Home') { event.preventDefault(); movePoint(0.5, 0.5); }
    }}><span className="attention-demo-cross horizontal"/><span className="attention-demo-cross vertical"/><span className="attention-demo-target" style={{ left: `clamp(24px, ${point.x * 100}%, calc(100% - 24px))`, top: `clamp(24px, ${point.y * 100}%, calc(100% - 24px))` }}><Eye size={18}/></span></div><span className="attention-demo-label">Move the point · arrow keys also work</span><div className="attention-demo-ranges"><label>Left / right<input type="range" min={4} max={96} value={point.x * 100} onChange={(event) => movePoint(Number(event.target.value) / 100, pointRef.current.y)} aria-label="Preview horizontal position"/></label><label>Up / down<input type="range" min={4} max={96} value={point.y * 100} onChange={(event) => movePoint(pointRef.current.x, Number(event.target.value) / 100)} aria-label="Preview vertical position"/></label></div></div>}
    <p className={`attention-notice ${phase === 'error' ? 'error' : ''}`} role={phase === 'error' ? 'alert' : 'status'} aria-live="polite">{notice}</p>
    {reducedMotion && <p className="attention-reduced">Reduced motion is on. Head movement is limited.</p>}
    <p className="attention-privacy">Video stays in this browser. Select participants yourself; no gender or identity classification.</p>
  </section>;
}
