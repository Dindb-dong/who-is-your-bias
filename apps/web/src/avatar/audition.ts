import { Mesh, MeshStandardMaterial, Object3D, SkinnedMesh, Texture } from 'three';
import type { AnimationTelemetry, RigDiagnostics, RigMapping } from './types';

export interface MorphInspection {
  mesh: string; target: string; index: number; vertices: number;
  maxDisplacement: number | null; movedVertices: number | null;
  status: 'deforms' | 'zero' | 'missing_position' | 'invalid';
}

export interface AssetInspection {
  schema_version: 'avatar-asset-inspection.v1';
  sha256: string | null; bytes: number | null; loadMs: number | null;
  geometryBasis: 'mesh_local_before_stage_normalization';
  meshes: { name: string; vertices: number; triangles: number; skinned: boolean; joints: number; materialNames: string[] }[];
  morphs: MorphInspection[];
  materials: { name: string; type: string; opacity: number; transparent: boolean; alphaTest: number; side: number;
    textures: { slot: string; colorSpace: string; width: number | null; height: number | null }[] }[];
  animations: { name: string; durationSeconds: number; tracks: number }[];
  warnings: string[];
  review: { likeness: 'ungraded'; mouthInterior: 'ungraded'; materialAppearance: 'ungraded' };
}

/** Reads decoded buffers, including Meshopt/sparse accessors resolved by GLTFLoader.
 * A matching target name alone never establishes usable deformation. */
export function inspectAvatarAsset(model: Object3D): AssetInspection {
  const meshes: AssetInspection['meshes'] = [];
  const morphs: MorphInspection[] = [];
  const materials = new Map<string, AssetInspection['materials'][number]>();
  const warnings: string[] = [];
  model.traverse((node) => {
    if (!(node instanceof Mesh)) return;
    const geometry = node.geometry; const positions = geometry.getAttribute('position');
    const materialList = Array.isArray(node.material) ? node.material : [node.material];
    meshes.push({ name: node.name, vertices: positions?.count ?? 0,
      triangles: (geometry.index?.count ?? positions?.count ?? 0) / 3,
      skinned: node instanceof SkinnedMesh, joints: node instanceof SkinnedMesh ? node.skeleton.bones.length : 0,
      materialNames: materialList.map((material) => material.name) });
    for (const [target, index] of Object.entries(node.morphTargetDictionary ?? {})) {
      const attribute = geometry.morphAttributes.position?.[index];
      const entry: MorphInspection = { mesh: node.name, target, index, vertices: attribute?.count ?? 0,
        maxDisplacement: null, movedVertices: null, status: 'missing_position' };
      if (attribute && positions && attribute.count === positions.count && attribute.itemSize === 3) {
        let max = 0; let moved = 0; let valid = true;
        for (let i = 0; i < attribute.count; i++) {
          const delta = Math.hypot(attribute.getX(i) - (geometry.morphTargetsRelative ? 0 : positions.getX(i)),
            attribute.getY(i) - (geometry.morphTargetsRelative ? 0 : positions.getY(i)),
            attribute.getZ(i) - (geometry.morphTargetsRelative ? 0 : positions.getZ(i)));
          if (!Number.isFinite(delta)) { valid = false; break; }
          max = Math.max(max, delta); if (delta > 1e-7) moved++;
        }
        entry.status = !valid ? 'invalid' : moved ? 'deforms' : 'zero';
        if (valid) { entry.maxDisplacement = max; entry.movedVertices = moved; }
      } else if (attribute) entry.status = 'invalid';
      morphs.push(entry);
    }
    for (const material of materialList) {
      if (materials.has(material.uuid)) continue;
      const textures: AssetInspection['materials'][number]['textures'] = [];
      // Only public texture slots are inspected. Embedded images are never copied
      // into a report, and file/object URLs are not published.
      for (const slot of ['map', 'normalMap', 'roughnessMap', 'metalnessMap', 'aoMap', 'emissiveMap', 'alphaMap'] as const) {
        const texture = (material as MeshStandardMaterial)[slot];
        if (!(texture instanceof Texture)) continue;
        const image = texture.image as { width?: number; height?: number } | undefined;
        textures.push({ slot, colorSpace: texture.colorSpace,
          width: Number.isFinite(image?.width) ? image!.width! : null,
          height: Number.isFinite(image?.height) ? image!.height! : null });
      }
      materials.set(material.uuid, { name: material.name, type: material.type, opacity: material.opacity,
        transparent: material.transparent, alphaTest: material.alphaTest, side: material.side, textures });
    }
  });
  if (!meshes.some((mesh) => mesh.skinned)) warnings.push('No skinned mesh was found; mapped nodes alone do not establish skin deformation.');
  if (new Set(meshes.map((mesh) => mesh.name)).size !== meshes.length) warnings.push('Duplicate mesh names need a manual mapping review; name-based reports may be ambiguous.');
  for (const entry of morphs) if (entry.status === 'invalid' || entry.status === 'missing_position') {
    warnings.push(`${entry.mesh}/${entry.target}: ${entry.status}.`);
  }
  return { schema_version: 'avatar-asset-inspection.v1', sha256: null, bytes: null, loadMs: null,
    geometryBasis: 'mesh_local_before_stage_normalization', meshes, morphs, materials: [...materials.values()],
    animations: [], warnings,
    review: { likeness: 'ungraded', mouthInterior: 'ungraded', materialAppearance: 'ungraded' } };
}

export function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value !== null && typeof value === 'object') return `{${Object.entries(value).filter(([, item]) => item !== undefined)
    .sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, item]) => `${JSON.stringify(key)}:${canonicalJson(item)}`).join(',')}}`;
  if (typeof value === 'number' && !Number.isFinite(value)) throw new Error('An audition record contains a non-finite number.');
  const encoded = JSON.stringify(value); if (encoded === undefined) throw new Error('An audition record contains an unsupported value.');
  return encoded;
}

export async function sha256Bytes(bytes: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

export const AUDITION_PROTOCOL = 'same-audio-fixed-camera.v1';
const SHA256 = /^[a-f0-9]{64}$/;

export interface AuditionFrame {
  monotonicMs: number; audioMs: number | null; speaking: boolean;
  telemetry: AnimationTelemetry; drawCalls: number; triangles: number;
  geometryCount: number; textureCount: number;
}

export interface AuditionRun {
  schema_version: 'avatar-audition-run.v1'; protocol: typeof AUDITION_PROTOCOL;
  asset: AssetInspection; resolvedMapping: RigMapping;
  resolvedTargets: NonNullable<RigDiagnostics['resolvedTargets']>;
  resolvedBones: NonNullable<RigDiagnostics['resolvedBones']>;
  audioSha256: string; speechTimingSha256: string | null;
  camera: { framing: string; position: number[]; target: number[]; fov: number; viewport: number[]; pixelRatio: number };
  settings: { reducedMotion: boolean; geometryPreview: boolean; sourceClipSha256: string | null };
  startedAt: string; stoppedReason: string;
  measurements: { durationMs: number; renderedFrames: number; frameIntervalP50Ms: number | null;
    frameIntervalP95Ms: number | null; firstAudioClockObservedMs: number | null;
    firstAudioOffsetMs: number | null; lastAudioOffsetMs: number | null;
    speakingFrames: number; maxDrawCalls: number; maxTriangles: number; maxGeometryCount: number; maxTextureCount: number };
  samples: AuditionFrame[];
  unmeasured: string[];
}

/** Bounded frame sampling. Wall-clock observations are kept separate from the
 * digital audio clock, acoustic latency, GPU memory, and human judgments. */
export class AuditionRecorder {
  private frames = 0; private speakingFrames = 0; private previous?: number;
  private intervals: number[] = []; private samples: AuditionFrame[] = [];
  private firstAudio: number | null = null; private lastSample = -Infinity;
  private firstAudioOffset: number | null = null; private lastAudioOffset: number | null = null;
  private maxima = { maxDrawCalls: 0, maxTriangles: 0, maxGeometryCount: 0, maxTextureCount: 0 };
  private reason = 'recording'; private ended: number;
  constructor(private base: Omit<AuditionRun, 'measurements' | 'samples' | 'unmeasured' | 'stoppedReason'>, private start: number) {
    if (!SHA256.test(base.audioSha256)) throw new Error('Play a hashed audio sample before starting an audition.');
    if (!base.asset.sha256 || !SHA256.test(base.asset.sha256)) throw new Error('Load a GLB with a verified file digest before recording.');
    this.ended = start;
  }
  get recording(): boolean { return this.reason === 'recording'; }
  add(frame: AuditionFrame, audioSha256: string | undefined): void {
    if (!this.recording) return;
    if (audioSha256 !== this.base.audioSha256) { this.stop('audio_changed', frame.monotonicMs); return; }
    if (frame.monotonicMs < this.start || !Number.isFinite(frame.monotonicMs)) return;
    if (frame.monotonicMs - this.start > 180_000 || this.frames >= 36_000) { this.stop('recording_limit', frame.monotonicMs); return; }
    if (this.previous !== undefined && frame.monotonicMs >= this.previous) this.intervals.push(frame.monotonicMs - this.previous);
    this.previous = frame.monotonicMs; this.ended = frame.monotonicMs; this.frames++;
    if (frame.speaking) this.speakingFrames++;
    if (frame.speaking && frame.audioMs !== null && Number.isFinite(frame.audioMs)) {
      if (this.firstAudio === null) { this.firstAudio = frame.monotonicMs - this.start; this.firstAudioOffset = frame.audioMs; }
      if (this.lastAudioOffset !== null && frame.audioMs < this.lastAudioOffset) { this.stop('audio_seek', frame.monotonicMs); return; }
      this.lastAudioOffset = frame.audioMs;
    }
    this.maxima.maxDrawCalls = Math.max(this.maxima.maxDrawCalls, frame.drawCalls);
    this.maxima.maxTriangles = Math.max(this.maxima.maxTriangles, frame.triangles);
    this.maxima.maxGeometryCount = Math.max(this.maxima.maxGeometryCount, frame.geometryCount);
    this.maxima.maxTextureCount = Math.max(this.maxima.maxTextureCount, frame.textureCount);
    if (frame.monotonicMs - this.lastSample >= 100) {
      this.samples.push({ ...frame, monotonicMs: frame.monotonicMs - this.start,
        telemetry: { ...frame.telemetry, unsupportedFace: [...frame.telemetry.unsupportedFace], unsupportedVisemes: [...frame.telemetry.unsupportedVisemes] } });
      this.lastSample = frame.monotonicMs;
    }
  }
  stop(reason: string, now: number): void {
    if (!this.recording) return;
    this.reason = reason; this.ended = Math.max(this.ended, Math.min(now, this.start + 180_000));
  }
  report(): AuditionRun {
    const sorted = [...this.intervals].sort((a, b) => a - b);
    const percentile = (q: number) => sorted.length ? sorted[Math.min(sorted.length - 1, Math.ceil(q * sorted.length) - 1)] : null;
    return { ...this.base, stoppedReason: this.reason,
      measurements: { durationMs: this.ended - this.start, renderedFrames: this.frames,
        frameIntervalP50Ms: percentile(0.5), frameIntervalP95Ms: percentile(0.95),
        firstAudioClockObservedMs: this.firstAudio, firstAudioOffsetMs: this.firstAudioOffset,
        lastAudioOffsetMs: this.lastAudioOffset, speakingFrames: this.speakingFrames, ...this.maxima }, samples: [...this.samples],
      unmeasured: ['human likeness and naturalness', 'mouth/teeth and material appearance review', 'acoustic loudspeaker latency',
        'GPU/total process memory bytes', 'completed spoken turn', 'simultaneous inference workload identity'] };
  }
}

export async function sealAudition(run: AuditionRun): Promise<{ report: AuditionRun; reportSha256: string }> {
  // The digest covers all measurements, exact mapping, camera and audio binding.
  return { report: run, reportSha256: await sha256Bytes(new TextEncoder().encode(canonicalJson(run)).buffer) };
}

export function auditionComparable(a: AuditionRun, b: AuditionRun): { comparable: boolean; reasons: string[] } {
  const reasons: string[] = [];
  if (a.protocol !== b.protocol) reasons.push('protocol differs');
  if (a.audioSha256 !== b.audioSha256) reasons.push('audio differs');
  if (a.speechTimingSha256 !== b.speechTimingSha256) reasons.push('speech timing differs');
  if (canonicalJson(a.camera) !== canonicalJson(b.camera)) reasons.push('camera/viewport differs');
  if (canonicalJson(a.settings) !== canonicalJson(b.settings)) reasons.push('motion/settings differ');
  if (!a.measurements.speakingFrames || !b.measurements.speakingFrames) reasons.push('speech was not sampled in both runs');
  for (const key of ['firstAudioOffsetMs', 'lastAudioOffsetMs'] as const) {
    const av = a.measurements[key]; const bv = b.measurements[key];
    if (av === null || bv === null || Math.abs(av - bv) > 100) reasons.push(`${key} differs by more than 100 ms or is unmeasured`);
  }
  if (a.stoppedReason === 'recording' || b.stoppedReason === 'recording') reasons.push('recording is still active');
  const ordinaryStops = new Set(['exported', 'user_stopped', 'playback_inactive']);
  if (!ordinaryStops.has(a.stoppedReason) || !ordinaryStops.has(b.stoppedReason)) reasons.push('a run was invalidated or reached a recording limit');
  return { comparable: !reasons.length, reasons };
}

export function effectiveFaceControls(diagnostics: RigDiagnostics): string[] {
  const inspection = diagnostics.assetInspection; if (!inspection) return [];
  const effective = new Set(inspection.morphs.filter((morph) => morph.status === 'deforms').map((morph) => `${morph.mesh}/${morph.target}`));
  const clamp = (value: number) => Math.max(0, Math.min(1, value));
  return (diagnostics.resolvedTargets ?? []).filter((entry) => entry.kind === 'face' && effective.has(`${entry.mesh}/${entry.target}`)
    && Math.abs(clamp((entry.outputNeutral ?? 0) + (entry.scale ?? 1)) - clamp(entry.outputNeutral ?? 0)) > 1e-7)
    .map((entry) => entry.control).filter((control, index, all) => all.indexOf(control) === index);
}
