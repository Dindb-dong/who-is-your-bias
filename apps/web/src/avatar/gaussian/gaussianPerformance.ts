import { Euler, Quaternion } from 'three';
import { FACE_SET, type FaceChannel, type FaceWeights } from '../channels';
import { clamp01 } from '../timeline';
import type { RigPose } from '../types';

/** FLAME expression coordinates are dense PCA coefficients, not ARKit names. */
export interface GaussianMotionHeader { n_expr: number; n_frames: number; fps: number }
export interface GaussianMotionBuffers { m_expr: ArrayLike<number>; m_pose: ArrayLike<number>; m_trans: ArrayLike<number> }
export interface GaussianParams { expr: Float32Array; pose: Float32Array; trans: Float32Array }
export interface GaussianPerformanceSample extends GaussianParams {
  expressionSource: 'calibrated' | 'observed_replay' | 'static_reference';
  /** Recorded PCA expressions can still contain lip motion without calibration. */
  sourceLipResidual: boolean;
}
export interface GaussianExpressionDriver {
  schema: 'gaussian-expression-driver.v1';
  subject_id: 'jensen-huang';
  bundle_sha256: string;
  n_expr: number;
  input_names: FaceChannel[];
  input_bounds: [number, number][];
  weights: number[][];
  bias: number[];
  output_bounds: [number, number][];
  neutral_expr: number[];
  jaw_open_radians: number;
  training: {
    method: 'ridge_regression';
    dataset_sha256: string;
    examples: number;
    frame_indices: number[];
    validation_frame_indices: number[];
  };
}

// The split is a documented property of these exact bytes, never a guessed
// fraction of an arbitrary new bundle. See docs/local-only-assets.md.
const DOCUMENTED_TRAIN_FRAMES: Record<string, { total: number; train: number }> = {
  '32d5bc15901a8673901c6234ae7af946d2bfcf2c0aa5e7ed9c816d1c3aff599c': { total: 3319, train: 2324 },
  '1a8c3bed73cdc93553257887bd682b3d87ea5c9dd1d9b2d6f4abed730e0605c7': { total: 3319, train: 2324 },
  '727ac52904b7a610b8b3bd59594ed85ef9990fd4b2105e8dc5a57079bc852c0d': { total: 3319, train: 2324 },
};
const digest = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const finiteArray = (value: unknown, length: number): value is number[] => Array.isArray(value) && value.length === length && value.every(finite);
const object = (value: unknown): Record<string, unknown> => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid expression driver object');
  return value as Record<string, unknown>;
};
function indices(value: unknown, count: number, allowEmpty = false): number[] {
  if (!Array.isArray(value) || (!allowEmpty && !value.length) || value.some((n, i) => !Number.isInteger(n) || n < 0 || n >= count || (i > 0 && n <= value[i - 1]))) {
    throw new Error('Expression frame indices must be ordered, unique, and inside the bundle');
  }
  return value as number[];
}
function bounds(value: unknown, length: number): value is [number, number][] {
  return Array.isArray(value) && value.length === length && value.every((v) => finiteArray(v, 2) && v[0] <= v[1]);
}

export function parseGaussianExpressionDriver(value: unknown, header: GaussianMotionHeader, bundleSha256: string): GaussianExpressionDriver {
  const d = object(value), training = object(d.training);
  if (d.schema !== 'gaussian-expression-driver.v1' || d.subject_id !== 'jensen-huang' || d.n_expr !== header.n_expr) throw new Error('Expression driver schema, subject, or FLAME dimensions do not match');
  if (!digest(bundleSha256) || d.bundle_sha256 !== bundleSha256) throw new Error('Expression driver does not match the exact head bundle');
  if (!Array.isArray(d.input_names) || !d.input_names.length || d.input_names.length > FACE_SET.size
      || d.input_names.some((n) => typeof n !== 'string' || !FACE_SET.has(n)) || new Set(d.input_names).size !== d.input_names.length) throw new Error('Invalid named expression inputs');
  const C = d.input_names.length, E = header.n_expr;
  if (!bounds(d.input_bounds, C) || d.input_bounds.some(([lo, hi]) => lo < 0 || hi > 1)
      || !Array.isArray(d.weights) || d.weights.length !== E || !d.weights.every((r) => finiteArray(r, C))
      || !finiteArray(d.bias, E) || !bounds(d.output_bounds, E) || !finiteArray(d.neutral_expr, E)
      || !finite(d.jaw_open_radians) || d.jaw_open_radians < 0 || d.jaw_open_radians > 0.8) throw new Error('Invalid expression calibration arrays');
  const train = indices(training.frame_indices, header.n_frames), validation = indices(training.validation_frame_indices, header.n_frames, true);
  if (training.method !== 'ridge_regression' || !digest(training.dataset_sha256) || training.examples !== train.length
      || validation.some((frame) => train.includes(frame))) throw new Error('Invalid or overlapping expression training partitions');
  const documented = DOCUMENTED_TRAIN_FRAMES[bundleSha256];
  if (documented && (header.n_frames !== documented.total || [...train, ...validation].some((frame) => frame >= documented.train))) throw new Error('Calibration includes held-out Gaussian frames or a mismatched documented split');
  return value as GaussianExpressionDriver;
}

/** Same legacy-to-ARKit bridge as the GLB rig, with explicit channels winning. */
export function gaussianFaceWeights(pose: RigPose, speaking: boolean): FaceWeights {
  const face: FaceWeights = {
    eyeBlinkLeft: pose.blink, eyeBlinkRight: pose.blink,
    browInnerUp: pose.browRaise, browOuterUpLeft: pose.browRaise, browOuterUpRight: pose.browRaise,
    browDownLeft: pose.browFurrow, browDownRight: pose.browFurrow,
    mouthSmileLeft: pose.mouthSmile, mouthSmileRight: pose.mouthSmile,
    jawOpen: pose.jawOpen, mouthFunnel: pose.mouthRound, ...pose.face,
  };
  for (const name of Object.keys(face) as FaceChannel[]) {
    // Brows, eyes, cheeks, smiles and frowns remain active while listening.
    const articulation = name.startsWith('jaw') || /^(mouth(Close|Funnel|Pucker|Roll|Stretch|Press|UpperUp|LowerDown)|tongueOut)/.test(name);
    face[name] = !speaking && articulation ? 0 : clamp01(face[name] ?? 0);
  }
  return face;
}

/** Attention is composed into canonical eye bone deltas or named look controls,
 * and deliberately zeroes legacy gaze. Preserve that selected route. */
function eyeRotation(pose: RigPose, side: 'Left' | 'Right'): [number, number, number] {
  const bone = pose.bones?.[side === 'Left' ? 'leftEye' : 'rightEye'];
  let q: Quaternion;
  if (bone && bone.length === 4 && bone.every(Number.isFinite) && Math.hypot(...bone) > 1e-8) {
    q = new Quaternion(...bone).normalize();
  } else {
    const horizontal = pose.face?.[`eyeLookOut${side}`] !== undefined || pose.face?.[`eyeLookIn${side}`] !== undefined;
    const vertical = pose.face?.[`eyeLookDown${side}`] !== undefined || pose.face?.[`eyeLookUp${side}`] !== undefined;
    const yaw = horizontal
      ? (clamp01(pose.face?.[`eyeLookOut${side}`] ?? 0) - clamp01(pose.face?.[`eyeLookIn${side}`] ?? 0)) * (side === 'Left' ? 0.35 : -0.35)
      : Math.max(-0.4, Math.min(0.4, finite(pose.gazeX) ? pose.gazeX : 0));
    const pitch = vertical
      ? (clamp01(pose.face?.[`eyeLookDown${side}`] ?? 0) - clamp01(pose.face?.[`eyeLookUp${side}`] ?? 0)) * 0.25
      : -Math.max(-0.3, Math.min(0.3, finite(pose.gazeY) ? pose.gazeY : 0));
    q = new Quaternion().setFromEuler(new Euler(pitch, yaw, 0));
  }
  // q and -q encode the same rotation. Use the shortest axis-angle branch.
  if (q.w < 0) q.set(-q.x, -q.y, -q.z, -q.w);
  const sine = Math.hypot(q.x, q.y, q.z);
  if (sine < 1e-8) return [0, 0, 0];
  const scale = 2 * Math.atan2(sine, q.w) / sine;
  return [q.x * scale, q.y * scale, q.z * scale];
}

export interface GaussianPerformanceInput {
  rigPose: RigPose;
  /** The caller passes its actual audio/media clock, including seeks and pauses. */
  elapsedMs: number;
  idleMs: number;
  speaking: boolean;
  reducedMotion?: boolean;
}
export interface GaussianPerformanceOptions {
  driver?: unknown;
  bundleSha256?: string;
  trainingFrameIndices?: number[];
}

export function createGaussianPerformance(header: GaussianMotionHeader, B: GaussianMotionBuffers, options: GaussianPerformanceOptions = {}) {
  const E = header.n_expr, T = header.n_frames;
  if (!Number.isInteger(E) || E < 1 || E > 512 || !Number.isInteger(T) || T < 1 || !finite(header.fps) || header.fps <= 0
      || B.m_expr.length !== E * T || B.m_pose.length !== T * 15 || B.m_trans.length !== T * 3) throw new Error('Invalid Gaussian motion dimensions');
  const driver = options.driver === undefined ? undefined : parseGaussianExpressionDriver(options.driver, header, options.bundleSha256 ?? '');
  const documented = DOCUMENTED_TRAIN_FRAMES[options.bundleSha256 ?? ''];
  const declared = options.trainingFrameIndices ?? driver?.training.frame_indices
    ?? (documented?.total === T ? Array.from({ length: documented.train }, (_, i) => i) : []);
  const frames = indices(declared, T, true);
  if (documented && (documented.total !== T || frames.some((frame) => frame >= documented.train))) throw new Error('Replay includes held-out Gaussian frames or a mismatched documented split');
  if (driver && frames.some((frame) => driver.training.validation_frame_indices.includes(frame))) throw new Error('Replay frames overlap the expression validation partition');
  const neutral = new Float32Array(E);
  for (let e = 0; e < E; e++) neutral[e] = driver?.neutral_expr[e] ?? B.m_expr[e];
  if (![...neutral].every(Number.isFinite)) throw new Error('The Gaussian reference expression is not finite');
  const diagnostics = {
    calibrated: !!driver,
    supportedChannels: driver?.input_names ?? [],
    replayFrameCount: frames.length,
    warning: driver ? undefined : frames.length
      ? 'Observed expression replay; recorded lip-expression residuals remain. A calibrated driver is required for semantic brows and mouth separation.'
      : 'No verified training split: using one static reference expression until a calibrated driver or explicit training frames are supplied.',
  };
  return {
    diagnostics,
    sample(input: GaussianPerformanceInput): GaussianPerformanceSample {
      const expr = new Float32Array(E), pose = new Float32Array(15), trans = new Float32Array(3);
      const face = gaussianFaceWeights(input.rigPose, input.speaking);
      let expressionSource: GaussianPerformanceSample['expressionSource'] = 'static_reference';
      if (driver) {
        const values = driver.input_names.map((name, i) => {
          const [lo, hi] = driver.input_bounds[i];
          return Math.max(lo, Math.min(hi, face[name] ?? 0));
        });
        for (let e = 0; e < E; e++) {
          let value = driver.bias[e];
          for (let c = 0; c < values.length; c++) value += driver.weights[e][c] * values[c];
          const [lo, hi] = driver.output_bounds[e]; expr[e] = Math.max(lo, Math.min(hi, value));
        }
        expressionSource = 'calibrated';
      } else if (frames.length && !input.reducedMotion) {
        const clock = input.speaking ? input.elapsedMs : input.idleMs;
        const time = Math.max(0, finite(clock) ? clock : 0) * header.fps / 1000;
        const i = Math.floor(time) % frames.length, blend = time - Math.floor(time);
        const a = frames[i], b = frames[(i + 1) % frames.length];
        for (let e = 0; e < E; e++) {
          const from = B.m_expr[a * E + e], to = B.m_expr[b * E + e];
          expr[e] = from + (to - from) * blend;
        }
        expressionSource = 'observed_replay';
      } else expr.set(neutral);
      // Body/attention owns global and neck rotations. Do not replay source
      // camera motion on top of its live skeleton. Lip amplitude owns the jaw.
      pose[6] = (face.jawOpen ?? 0) * (driver?.jaw_open_radians ?? 0.35);
      pose.set(eyeRotation(input.rigPose, 'Left'), 9);
      pose.set(eyeRotation(input.rigPose, 'Right'), 12);
      return { expr, pose, trans, expressionSource, sourceLipResidual: expressionSource === 'observed_replay' };
    },
  };
}
