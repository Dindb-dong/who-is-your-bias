import {
  ACESFilmicToneMapping, AmbientLight, CircleGeometry, DirectionalLight,
  FogExp2, HemisphereLight, Mesh, MeshBasicMaterial, MeshStandardMaterial,
  PerspectiveCamera, PlaneGeometry, Scene, SRGBColorSpace, Vector2, WebGLRenderer,
} from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { avatarView, type AvatarFraming } from './builtin';
import { disposeObject } from './dispose';
import { createProceduralRig, type AvatarRig, type PortraitStyle } from './procedural';
import { loadGlbRig } from './retarget';
import { dampPose, neutralPose } from './timeline';
import { samplePerformance } from './performance';
import { AttentionController, attentionCapabilities, composeAttentionPose, type AttentionTarget } from './attention';
import { parseAvatarClip, parseSpeechTiming } from './validation';
import { AUDITION_PROTOCOL, AuditionRecorder, canonicalJson, sealAudition, sha256Bytes } from './audition';
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
  attentionTarget?: AttentionTarget | null;
}

interface RendererCallbacks {
  onReady?: (diagnostics: RigDiagnostics) => void;
  onError?: (error: Error) => void;
  onModelState?: (state: 'loading' | 'ready' | 'fallback') => void;
  onTelemetry?: (telemetry: AnimationTelemetry) => void;
  onAuditionState?: (recording: boolean) => void;
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
  private audition?: AuditionRecorder;
  private attention = new AttentionController();
  private portraitStyle: PortraitStyle = 'reference';

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
    this.rig = createProceduralRig(this.portraitStyle); this.scene.add(this.rig.root);
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
    if (this.audition?.recording) {
      if (this.playback.speaking && !state.speaking) this.stopAudition('playback_inactive');
      else if (state.audioSha256 !== this.playback.audioSha256) this.stopAudition('audio_changed');
      else if (state.speechTiming !== this.timingInput) this.stopAudition('speech_timing_changed');
      else if (state.performanceClip !== this.clipInput || state.allowGeometryPreview !== this.playback.allowGeometryPreview
        || state.reducedMotion !== this.playback.reducedMotion) this.stopAudition('motion_settings_changed');
      else if (state.attentionTarget || this.playback.attentionTarget) this.stopAudition('attention_changed');
    }
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

  setCalibrationPose(pose?: RigPose): void {
    if (pose) this.stopAudition('manual_calibration');
    if (pose) this.attention.reset();
    this.calibrationPose = pose; this.pose = neutralPose();
  }

  async startAudition(): Promise<void> {
    if (this.disposed || this.contextLost || !this.visible) throw new Error('The avatar must be visible before recording.');
    if (this.calibrationPose) throw new Error('Turn off manual pose preview before recording.');
    if (this.playback.attentionTarget || this.attention.active) throw new Error('Stop attention tracking and wait for the avatar to settle before recording a fixed comparison audition.');
    const asset = this.rig.diagnostics.assetInspection;
    if (!asset?.sha256) throw new Error('Load a GLB before recording an audition.');
    if (!this.playback.speaking || !this.playback.audioSha256) throw new Error('Start a recorded audio sample, then record the audition.');
    this.stopAudition('restarted');
    const serial = this.modelSerial; const state = this.playback;
    const encodeHash = (value: unknown) => sha256Bytes(new TextEncoder().encode(canonicalJson(value)).buffer);
    const [speechTimingSha256, sourceClipSha256] = await Promise.all([
      state.speechTiming ? encodeHash(state.speechTiming) : null,
      state.performanceClip ? encodeHash(state.performanceClip) : null,
    ]);
    if (serial !== this.modelSerial || this.disposed || this.playback.audioSha256 !== state.audioSha256
      || !this.playback.speaking || this.playback.speechTiming !== state.speechTiming
      || this.playback.performanceClip !== state.performanceClip || this.playback.attentionTarget || this.attention.active) throw new Error('Playback changed; start the audition again.');
    // Consume pre-existing drag inertia, then reset the camera. Disabling input
    // alone does not stop OrbitControls' damping from moving a comparison view.
    const damping = this.controls.enableDamping; this.controls.enableDamping = false;
    this.controls.update(); this.resetView(); this.controls.enableDamping = damping; this.controls.enabled = false;
    this.start = performance.now(); this.previousFrame = 0; this.pose = neutralPose();
    const size = this.renderer.getSize(new Vector2());
    this.audition = new AuditionRecorder({ schema_version: 'avatar-audition-run.v1', protocol: AUDITION_PROTOCOL,
      asset: JSON.parse(canonicalJson(asset)), resolvedMapping: this.rig.diagnostics.resolvedMapping ?? {},
      resolvedTargets: this.rig.diagnostics.resolvedTargets ?? [],
      resolvedBones: this.rig.diagnostics.resolvedBones ?? [],
      audioSha256: state.audioSha256!, speechTimingSha256,
      camera: { framing: this.framing, position: this.camera.position.toArray(), target: this.controls.target.toArray(),
        fov: this.camera.fov, viewport: size.toArray(), pixelRatio: this.renderer.getPixelRatio() },
      settings: { reducedMotion: !!state.reducedMotion, geometryPreview: !!state.allowGeometryPreview, sourceClipSha256 },
      startedAt: new Date().toISOString(),
    }, performance.now());
    this.callbacks.onAuditionState?.(true);
  }

  stopAudition(reason = 'user_stopped'): void {
    if (!this.audition?.recording) return;
    this.audition.stop(reason, performance.now()); this.controls.enabled = true;
    this.callbacks.onAuditionState?.(false);
  }

  async exportAudition() {
    if (!this.audition) throw new Error('Record an audition before exporting it.');
    this.stopAudition('exported');
    return sealAudition(this.audition.report());
  }

  setFraming(framing: AvatarFraming): void {
    if (this.disposed || this.framing === framing) return;
    this.stopAudition('framing_changed');
    this.framing = framing; this.canvas.dataset.framing = framing; this.resetView();
  }

  setPortraitStyle(style: PortraitStyle): void {
    if (this.disposed || style === this.portraitStyle) return;
    this.portraitStyle = style;
    this.canvas.dataset.portraitStyle = style;
    if (this.rig.diagnostics.kind === 'procedural') this.replaceRig(createProceduralRig(style));
  }

  async setModel(url?: string, mapping: RigMapping = {}): Promise<void> {
    if (this.disposed) return;
    this.stopAudition('model_or_mapping_changed');
    const serial = ++this.modelSerial;
    this.modelAbort?.abort();
    if (!url) {
      if (this.rig.diagnostics.kind === 'glb') this.replaceRig(createProceduralRig(this.portraitStyle));
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
      if (this.rig.diagnostics.kind !== 'procedural') this.replaceRig(createProceduralRig(this.portraitStyle));
      this.callbacks.onModelState?.('fallback');
      this.callbacks.onError?.(error instanceof Error ? error : new Error('The model could not be loaded.'));
    }
  }

  private replaceRig(rig: AvatarRig): void {
    this.scene.remove(this.rig.root); this.rig.dispose(); this.rig = rig;
    this.pose = neutralPose(); this.attention.reset(); rig.apply(this.pose); this.scene.add(rig.root);
    this.canvas.dataset.avatar = rig.diagnostics.kind;
    this.canvas.dataset.portraitStyle = this.portraitStyle;
    this.canvas.setAttribute('aria-label', rig.diagnostics.kind === 'glb'
      ? 'Interactive rigged 3D human avatar. Drag to change the viewing angle.'
      : 'Interactive stylized 3D avatar. Drag to change the viewing angle.');
    this.resetView();
    this.callbacks.onReady?.(rig.diagnostics);
  }

  private resize = (): void => {
    if (this.disposed) return;
    const width = Math.max(1, this.host.clientWidth); const height = Math.max(1, this.host.clientHeight);
    const oldSize = this.renderer.getSize(new Vector2());
    if (oldSize.x !== width || oldSize.y !== height) this.stopAudition('viewport_changed');
    this.camera.aspect = width / height;
    // Preserve the face/shoulder framing on narrow mobile stages.
    this.camera.fov = avatarView(this.rig.diagnostics.kind, this.framing, this.camera.aspect).fov;
    this.camera.updateProjectionMatrix(); this.renderer.setSize(width, height, false);
  };

  private onVisibility = (): void => {
    this.visible = !document.hidden;
    if (!this.visible) { this.stopAudition('document_hidden'); this.attention.reset(); }
    if (this.visible) { this.previousFrame = 0; if (!this.raf && !this.contextLost) this.raf = requestAnimationFrame(this.frame); }
  };

  private onContextLost = (event: Event): void => {
    this.stopAudition('graphics_context_lost');
    this.attention.reset();
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
    this.pose = seek || this.calibrationPose ? target : dampPose(this.pose, target, delta);
    // RAF's timestamp can precede a pointer/capture event in the same frame.
    // Sample the input clock now so a freshly observed target is not discarded
    // as a future timestamp and the eyes-first delay is not restarted.
    const attentionFrame = this.attention.sample(this.calibrationPose ? null : this.playback.attentionTarget,
      performance.now(), this.playback.reducedMotion, attentionCapabilities(this.rig.diagnostics).head);
    const attended = composeAttentionPose(this.pose, attentionFrame, this.rig.diagnostics);
    this.rig.apply(this.calibrationPose ? this.pose : attended.pose);
    const telemetry = this.calibrationPose
      ? { ...sample.telemetry, faceSource: 'calibration' as const, boneSource: 'calibration' as const, speechSource: 'silent' as const, attention: attended.telemetry }
      : { ...sample.telemetry, attention: attended.telemetry };
    // Observable animation output, without retaining capture images or landmarks.
    this.canvas.dataset.attentionPhase = attended.telemetry.phase;
    this.canvas.dataset.attentionTarget = attended.telemetry.targetId ?? '';
    this.canvas.dataset.attentionEyeYaw = attended.telemetry.eyeYaw.toFixed(4);
    this.canvas.dataset.attentionEyePitch = attended.telemetry.eyePitch.toFixed(4);
    this.canvas.dataset.attentionHeadYaw = attended.telemetry.headYaw.toFixed(4);
    this.canvas.dataset.attentionHeadPitch = attended.telemetry.headPitch.toFixed(4);
    if (now - this.lastTelemetry > (attentionFrame.phase === 'idle' ? 250 : 80)) {
      this.lastTelemetry = now;
      this.callbacks.onTelemetry?.(telemetry);
    }
    this.controls.update(); this.renderer.render(this.scene, this.camera);
    if (this.audition?.recording) {
      this.audition.add({ monotonicMs: now, audioMs: this.playback.audioElapsedMs ?? null, speaking: this.playback.speaking,
        telemetry, drawCalls: this.renderer.info.render.calls, triangles: this.renderer.info.render.triangles,
        geometryCount: this.renderer.info.memory.geometries, textureCount: this.renderer.info.memory.textures }, this.playback.audioSha256);
      if (!this.audition.recording) { this.controls.enabled = true; this.callbacks.onAuditionState?.(false); }
    }
    this.raf = requestAnimationFrame(this.frame);
  };

  resetView(): void {
    this.stopAudition('view_reset');
    const view = avatarView(this.rig.diagnostics.kind, this.framing, this.camera.aspect);
    this.camera.position.fromArray(view.position); this.controls.target.fromArray(view.target);
    this.camera.fov = view.fov; this.camera.updateProjectionMatrix(); this.controls.update();
  }

  dispose(): void {
    if (this.disposed) return;
    this.stopAudition('renderer_disposed');
    this.attention.reset();
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

