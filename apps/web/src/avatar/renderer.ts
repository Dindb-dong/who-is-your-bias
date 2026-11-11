import {
  ACESFilmicToneMapping, AmbientLight, CircleGeometry, DirectionalLight,
  FogExp2, HemisphereLight, Mesh, MeshBasicMaterial, MeshStandardMaterial,
  PerspectiveCamera, PlaneGeometry, Scene, SRGBColorSpace, WebGLRenderer,
} from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { avatarView, type AvatarFraming } from './builtin';
import { disposeObject } from './dispose';
import { createProceduralRig, type AvatarRig } from './procedural';
import { loadGlbRig } from './retarget';
import { dampPose, neutralPose } from './timeline';
import { samplePerformance } from './performance';
import { parseAvatarClip, parseSpeechTiming } from './validation';
import type { AnimationTelemetry, AvatarClip, BehaviorPlan, RigDiagnostics, RigMapping, RigPose, SpeechTiming } from './types';

export interface PlaybackState {
  speaking: boolean;
  elapsedMs: number;
  amplitude?: number;
  timeline?: BehaviorPlan;
  reducedMotion?: boolean;
  audioElapsedMs?: number;
  audioSha256?: string;
  speechTiming?: SpeechTiming;
  performanceClip?: AvatarClip;
  observationElapsedMs?: number;
  allowGeometryPreview?: boolean;
}

interface RendererCallbacks {
  onReady?: (diagnostics: RigDiagnostics) => void;
  onError?: (error: Error) => void;
  onModelState?: (state: 'loading' | 'ready' | 'fallback') => void;
  onTelemetry?: (telemetry: AnimationTelemetry) => void;
}

export class AvatarRenderer {
  readonly canvas: HTMLCanvasElement;
  private renderer: WebGLRenderer;
  private scene = new Scene();
  private camera = new PerspectiveCamera(34, 1, 0.1, 30);
  private controls: OrbitControls;
  private rig: AvatarRig;
  private resizeObserver: ResizeObserver;
  private raf = 0;
  private disposed = false;
  private contextLost = false;
  private visible = true;
  private playback: PlaybackState = { speaking: false, elapsedMs: 0 };
  private pose: RigPose = neutralPose();
  private previousFrame = 0;
  private start = performance.now();
  private modelAbort?: AbortController;
  private modelSerial = 0;
  private timingInput?: SpeechTiming;
  private clipInput?: AvatarClip;
  private validatedTiming?: SpeechTiming;
  private validatedClip?: AvatarClip;
  private calibrationPose?: RigPose;
  private lastClock = 0;
  private lastTelemetry = 0;
  private framing: AvatarFraming = 'portrait';

  constructor(private host: HTMLElement, private callbacks: RendererCallbacks = {}) {
    this.renderer = new WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'low-power' });
    this.canvas = this.renderer.domElement as HTMLCanvasElement;
    this.canvas.className = 'avatar-canvas';
    this.canvas.style.cssText = 'width:100%;height:100%;display:block;outline:none;touch-action:pan-y;';
    this.canvas.setAttribute('aria-label', 'Interactive stylized 3D avatar. Drag to change the viewing angle.');
    this.canvas.setAttribute('role', 'img');
    this.canvas.dataset.avatar = 'procedural';
    this.canvas.dataset.framing = this.framing;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
    this.renderer.setClearColor(0x10150f, 0);
    this.renderer.outputColorSpace = SRGBColorSpace;
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.22;
    this.scene.fog = new FogExp2('#141a12', 0.045);
    this.camera.position.set(0.07, 1.58, 4.45);
    this.camera.lookAt(0, 1.40, 0);
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.target.set(0, 1.40, 0);
    this.controls.enablePan = false; this.controls.enableZoom = false;
    this.controls.enableDamping = true; this.controls.dampingFactor = 0.08;
    this.controls.minAzimuthAngle = -0.42; this.controls.maxAzimuthAngle = 0.42;
    this.controls.minPolarAngle = 1.35; this.controls.maxPolarAngle = 1.64;

    this.scene.add(new HemisphereLight('#dce4c9', '#333827', 2.25));
    this.scene.add(new AmbientLight('#d7ddc7', 0.20));
    const key = new DirectionalLight('#ffebd5', 3.2); key.position.set(-2.8, 4.2, 3.4); this.scene.add(key);
    const fill = new DirectionalLight('#d0dfd1', 1.25); fill.position.set(2.5, 2.5, 3.3); this.scene.add(fill);
    const rim = new DirectionalLight('#d8f8a0', 2.3); rim.position.set(1.5, 2.8, -2); this.scene.add(rim);
    const leftRim = new DirectionalLight('#819e91', 1.4); leftRim.position.set(-2, 1.9, -1.2); this.scene.add(leftRim);
    this.createStudio();
    this.rig = createProceduralRig(); this.scene.add(this.rig.root);
    this.host.appendChild(this.canvas);
    this.resizeObserver = new ResizeObserver(this.resize); this.resizeObserver.observe(host);
    document.addEventListener('visibilitychange', this.onVisibility);
    this.canvas.addEventListener('webglcontextlost', this.onContextLost);
    this.resize(); this.raf = requestAnimationFrame(this.frame);
    callbacks.onReady?.(this.rig.diagnostics);
  }

  private createStudio(): void {
    const ground = new Mesh(new PlaneGeometry(200, 200), new MeshStandardMaterial({ color: '#151b14', roughness: 1 }));
    ground.rotation.x = -Math.PI / 2; ground.position.y = -0.03; this.scene.add(ground);
    const disc = new Mesh(new CircleGeometry(1.05, 96), new MeshBasicMaterial({ color: '#050905', transparent: true, opacity: 0.30, depthWrite: false }));
    disc.rotation.x = -Math.PI / 2; disc.position.set(0, -0.015, 0); disc.scale.y = 0.45; this.scene.add(disc);
    // A large out-of-focus studio backdrop is represented by a subtle mesh,
    // avoiding heavyweight environment maps or external image requests.
    const backdrop = new Mesh(new CircleGeometry(1.80, 96), new MeshBasicMaterial({ color: '#20291d', transparent: true, opacity: 0.40, depthWrite: false }));
    backdrop.position.set(0, 1.50, -1.1); this.scene.add(backdrop);
  }

  setPlayback(state: PlaybackState): void {
    if (state.speechTiming !== this.timingInput) {
      this.timingInput = state.speechTiming; this.validatedTiming = undefined;
      try { if (state.speechTiming) this.validatedTiming = parseSpeechTiming(state.speechTiming); }
      catch (error) { this.callbacks.onError?.(error instanceof Error ? error : new Error('Invalid speech timing.')); }
    }
    if (state.performanceClip !== this.clipInput || state.allowGeometryPreview !== this.playback.allowGeometryPreview) {
      this.clipInput = state.performanceClip; this.validatedClip = undefined;
      try { if (state.performanceClip) this.validatedClip = parseAvatarClip(state.performanceClip, state.allowGeometryPreview); }
      catch (error) { this.callbacks.onError?.(error instanceof Error ? error : new Error('Invalid animation clip.')); }
    }
    this.playback = { ...state, speechTiming: this.validatedTiming, performanceClip: this.validatedClip };
  }

  setCalibrationPose(pose?: RigPose): void { this.calibrationPose = pose; this.pose = neutralPose(); }

  setFraming(framing: AvatarFraming): void {
    if (this.disposed || this.framing === framing) return;
    this.framing = framing; this.canvas.dataset.framing = framing; this.resetView();
  }

  async setModel(url?: string, mapping: RigMapping = {}): Promise<void> {
    if (this.disposed) return;
    const serial = ++this.modelSerial;
    this.modelAbort?.abort();
    if (!url) {
      if (this.rig.diagnostics.kind === 'glb') this.replaceRig(createProceduralRig());
      else this.callbacks.onReady?.(this.rig.diagnostics);
      this.callbacks.onModelState?.('ready');
      return;
    }
    const abort = new AbortController(); this.modelAbort = abort;
    this.callbacks.onModelState?.('loading');
    try {
      const rig = await loadGlbRig(url, mapping, abort.signal);
      if (this.disposed || serial !== this.modelSerial) { rig.dispose(); return; }
      this.replaceRig(rig);
      this.callbacks.onModelState?.('ready');
    } catch (error) {
      if (this.disposed || serial !== this.modelSerial || abort.signal.aborted) return;
      if (this.rig.diagnostics.kind !== 'procedural') this.replaceRig(createProceduralRig());
      this.callbacks.onModelState?.('fallback');
      this.callbacks.onError?.(error instanceof Error ? error : new Error('The model could not be loaded.'));
    }
  }

  private replaceRig(rig: AvatarRig): void {
    this.scene.remove(this.rig.root); this.rig.dispose(); this.rig = rig;
    this.pose = neutralPose(); rig.apply(this.pose); this.scene.add(rig.root);
    this.canvas.dataset.avatar = rig.diagnostics.kind;
    this.canvas.setAttribute('aria-label', rig.diagnostics.kind === 'glb'
      ? 'Interactive rigged 3D human avatar. Drag to change the viewing angle.'
      : 'Interactive stylized 3D avatar. Drag to change the viewing angle.');
    this.resetView();
    this.callbacks.onReady?.(rig.diagnostics);
  }

  private resize = (): void => {
    if (this.disposed) return;
    const width = Math.max(1, this.host.clientWidth); const height = Math.max(1, this.host.clientHeight);
    this.camera.aspect = width / height;
    // Preserve the face/shoulder framing on narrow mobile stages.
    this.camera.fov = avatarView(this.rig.diagnostics.kind, this.framing, this.camera.aspect).fov;
    this.camera.updateProjectionMatrix(); this.renderer.setSize(width, height, false);
  };

  private onVisibility = (): void => {
    this.visible = !document.hidden;
    if (this.visible) { this.previousFrame = 0; if (!this.raf && !this.contextLost) this.raf = requestAnimationFrame(this.frame); }
  };

  private onContextLost = (event: Event): void => {
    event.preventDefault(); this.contextLost = true;
    cancelAnimationFrame(this.raf); this.raf = 0;
    this.callbacks.onError?.(new Error('The graphics context was interrupted. Reload the avatar to continue.'));
  };

  private frame = (now: number): void => {
    this.raf = 0;
    if (this.disposed || this.contextLost || !this.visible) return;
    const delta = this.previousFrame ? (now - this.previousFrame) / 1000 : 1 / 60;
    this.previousFrame = now;
    const sample = samplePerformance({ ...this.playback, plan: this.playback.timeline, idleMs: now - this.start }, this.rig.diagnostics);
    const target = this.calibrationPose ?? sample.pose;
    const clock = this.playback.performanceClip ? this.playback.observationElapsedMs ?? this.playback.elapsedMs : this.playback.audioElapsedMs ?? this.playback.elapsedMs;
    const seek = clock < this.lastClock || clock - this.lastClock > 250;
    this.lastClock = clock;
    this.pose = seek || this.calibrationPose ? target : dampPose(this.pose, target, delta); this.rig.apply(this.pose);
    if (now - this.lastTelemetry > 250) {
      this.lastTelemetry = now;
      this.callbacks.onTelemetry?.(this.calibrationPose ? { ...sample.telemetry, faceSource: 'calibration', boneSource: 'calibration', speechSource: 'silent' } : sample.telemetry);
    }
    this.controls.update(); this.renderer.render(this.scene, this.camera);
    this.raf = requestAnimationFrame(this.frame);
  };

  resetView(): void {
    const view = avatarView(this.rig.diagnostics.kind, this.framing, this.camera.aspect);
    this.camera.position.fromArray(view.position); this.controls.target.fromArray(view.target);
    this.camera.fov = view.fov; this.camera.updateProjectionMatrix(); this.controls.update();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true; ++this.modelSerial; this.modelAbort?.abort();
    cancelAnimationFrame(this.raf); this.raf = 0;
    this.resizeObserver.disconnect(); this.controls.dispose();
    document.removeEventListener('visibilitychange', this.onVisibility);
    this.canvas.removeEventListener('webglcontextlost', this.onContextLost);
    // One scene traversal disposes both rig and stage, avoiding double disposal.
    disposeObject(this.scene);
    this.renderer.renderLists.dispose(); this.renderer.dispose(); this.renderer.forceContextLoss();
    this.canvas.remove();
  }
}
