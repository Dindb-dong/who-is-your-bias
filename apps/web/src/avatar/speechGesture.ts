import { Quaternion, Vector3 } from 'three';
import { FINGER_JOINTS, type BoneControl, type BoneRotations, type QuaternionTuple } from './channels';

/** Speech-generated SMPL-X body motion (pretrained EMAGE), bound to one audio clip.
 * It follows the rhythm of the synthesized speech; it is not an observation of the persona. */
export interface SpeechGestureMotion {
  fps: number;
  frames: number;
  /** frames x 165 SMPL-X axis-angle values (55 joints), after the calm filter. */
  poses: Float32Array;
  audioSha256: string;
  engine: string;
  /** Live voice: frames arrive while the turn plays (see appendStreamingGesture). */
  stream?: StreamingGestureState;
}

interface StreamingGestureState {
  complete: boolean;
  /** Audio time from which the motion fades in (first frames, or after the motion caught up again). */
  fadeFromMs: number;
  raw: Float32Array;
  sum: Float64Array;
  nextSequence: number;
  calm: number;
  smooth: number;
}

export const SMPLX_POSE_WIDTH = 165;
const MAX_FRAMES = 30 * 180;

/** Same filter and team defaults as tools/gaussian_head/smplx/preview.html (calm 0.7, smooth 3 frames):
 * a centred moving average, then every channel shrinks toward its clip mean. */
export function calmPoses(source: Float32Array, frames: number, calm = 0.7, smooth = 3): Float32Array {
  const width = SMPLX_POSE_WIDTH, c = Math.min(1, Math.max(0, calm)), s = Math.max(0, Math.round(smooth));
  if (c >= 1 && s === 0) return source;
  const mean = new Float64Array(width), out = new Float32Array(frames * width);
  for (let f = 0; f < frames; f++) for (let k = 0; k < width; k++) mean[k] += source[f * width + k] / frames;
  for (let f = 0; f < frames; f++) {
    const a = Math.max(0, f - s), b = Math.min(frames - 1, f + s);
    for (let k = 0; k < width; k++) {
      let v = 0; for (let g = a; g <= b; g++) v += source[g * width + k];
      out[f * width + k] = mean[k] + c * (v / (b - a + 1) - mean[k]);
    }
  }
  return out;
}

export function parseSpeechGestureMotion(bytes: ArrayBuffer, meta: { fps: number; frames: number; audioSha256: string; engine: string },
  filter: { calm?: number; smooth?: number } = {}): SpeechGestureMotion {
  const { fps, frames } = meta;
  if (fps !== 30 || !Number.isInteger(frames) || frames < 1 || frames > MAX_FRAMES || bytes.byteLength !== frames * SMPLX_POSE_WIDTH * 4
    || !/^[a-f0-9]{64}$/.test(meta.audioSha256)) throw new Error('The generated gesture motion has an unexpected shape.');
  const view = new DataView(bytes), poses = new Float32Array(frames * SMPLX_POSE_WIDTH);
  for (let i = 0; i < poses.length; i++) {
    const value = view.getFloat32(i * 4, true);
    if (!Number.isFinite(value) || Math.abs(value) > 4 * Math.PI) throw new Error('The generated gesture motion contains invalid rotations.');
    poses[i] = value;
  }
  return { fps, frames, audioSha256: meta.audioSha256, engine: meta.engine.slice(0, 120),
    poses: calmPoses(poses, frames, filter.calm, filter.smooth) };
}

/** Empty motion for one live turn; `key` binds it to that turn's audio clock (passed as audioSha256). */
export function createStreamingGesture(key: string, engine: string, filter: { calm?: number; smooth?: number } = {}): SpeechGestureMotion {
  return { fps: 30, frames: 0, poses: new Float32Array(0), audioSha256: key, engine: engine.slice(0, 120),
    stream: { complete: false, fadeFromMs: 0, raw: new Float32Array(0), sum: new Float64Array(SMPLX_POSE_WIDTH), nextSequence: 0,
      calm: Math.min(1, Math.max(0, filter.calm ?? 0.7)), smooth: Math.max(0, Math.round(filter.smooth ?? 3)) } };
}

/** Appends one server chunk (frames on the turn's PCM clock). The calm filter runs causally: it shrinks toward
 * the mean of the frames received so far, and frames near the end are re-smoothed when later frames arrive. */
export function appendStreamingGesture(motion: SpeechGestureMotion,
  chunk: { sequence: number; start_frame: number; frame_count: number; poses_base64: string }, audioElapsedMs = 0): void {
  const stream = motion.stream;
  if (!stream || stream.complete) throw new Error('The gesture stream is not open.');
  const width = SMPLX_POSE_WIDTH, start = motion.frames, count = chunk.frame_count;
  if (chunk.sequence !== stream.nextSequence || chunk.start_frame !== start || !Number.isInteger(count) || count < 1 || count > 64
    || start + count > MAX_FRAMES * 4) throw new Error('Gesture frames arrived out of order.');
  const raw = atob(chunk.poses_base64);
  if (raw.length !== count * width * 4) throw new Error('A gesture chunk has an unexpected size.');
  const view = new DataView(Uint8Array.from(raw, (c) => c.charCodeAt(0)).buffer);
  const end = start + count;
  if (stream.raw.length < end * width) {
    const capacity = Math.max(end, start * 2, 64) * width;
    const grownRaw = new Float32Array(capacity); grownRaw.set(stream.raw); stream.raw = grownRaw;
    const grownPoses = new Float32Array(capacity); grownPoses.set(motion.poses); motion.poses = grownPoses;
  }
  for (let i = 0; i < count * width; i++) {
    const value = view.getFloat32(i * 4, true);
    if (!Number.isFinite(value) || Math.abs(value) > 4 * Math.PI) throw new Error('The generated gesture motion contains invalid rotations.');
    stream.raw[start * width + i] = value; stream.sum[i % width] += value;
  }
  const { calm, smooth } = stream;
  for (let f = Math.max(0, start - smooth); f < end; f++) {
    const a = Math.max(0, f - smooth), b = Math.min(end - 1, f + smooth);
    for (let k = 0; k < width; k++) {
      let v = 0; for (let g = a; g <= b; g++) v += stream.raw[g * width + k];
      const mean = stream.sum[k] / end;
      motion.poses[f * width + k] = mean + calm * (v / (b - a + 1) - mean);
    }
  }
  // If playback had already run past the motion, fade in again from now instead of snapping.
  if (start === 0 || audioElapsedMs > start / motion.fps * 1000) stream.fadeFromMs = Math.max(0, audioElapsedMs);
  motion.frames = end; stream.nextSequence++;
}

export function completeStreamingGesture(motion: SpeechGestureMotion, frames: number): void {
  if (!motion.stream || frames !== motion.frames) throw new Error('The gesture stream ended with missing frames.');
  motion.stream.complete = true;
}

// SMPL-X kinematic tree (55 joints). Rest frames are world-aligned: Y up, facing +Z, subject's left at +X,
// which is also the runtime's subject-relative bone convention.
export const SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15,
  20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53];
// Per hand: index, middle, pinky, ring, thumb (3 joints each), starting at 25 (left) and 40 (right).
const SMPLX_FINGER_OFFSET: Record<string, number> = { Index: 0, Middle: 3, Little: 6, Ring: 9, Thumb: 12 };

/** Rig control <- SMPL-X joints (composed in order). Head, neck and jaw stay with attention and the
 * Gaussian head; the pelvis/legs are dropped so the speaker stays in place. */
const CHANNELS: [BoneControl, number[], number][] = [
  ['spine', [3, 6], 0.6], ['chest', [9], 0.6],
  ['leftShoulder', [13], 1], ['rightShoulder', [14], 1],
  ['leftUpperArm', [16], 1], ['rightUpperArm', [17], 1],
  ['leftForearm', [18], 1], ['rightForearm', [19], 1],
  ['leftHand', [20], 1], ['rightHand', [21], 1],
  ...(['left', 'right'] as const).flatMap((side) => FINGER_JOINTS.map((joint) => {
    const finger = Object.keys(SMPLX_FINGER_OFFSET).find((name) => joint.startsWith(name))!;
    const step = joint.endsWith('Distal') ? 2 : joint.endsWith('Intermediate') || (finger === 'Thumb' && joint.endsWith('Proximal')) ? 1 : 0;
    return [`${side}${joint}` as BoneControl, [(side === 'left' ? 25 : 40) + SMPLX_FINGER_OFFSET[finger] + step], 1] as [BoneControl, number[], number];
  })),
];

// EMAGE's average speaking posture (BEAT2 style: torso ~26° forward over three spine joints, collars ~16° raised,
// upper arms ~52° below the T-pose), axis-angle, measured over 48 s of generated motion on 2026-11-23. Torso,
// collars and upper arms are referenced to it, so that average maps to the rig's own relaxed neutral and only
// EMAGE's movement around it is applied (otherwise every reply would hunch the avatar forward with arms out).
const POSTURE_MEAN: Record<number, [number, number, number]> = {
  3: [0.2626, 0.0491, 0.0069], 6: [0.0839, -0.0354, -0.0186], 9: [0.0961, -0.0094, 0.0394],
  13: [0.0221, -0.1588, -0.2333], 14: [-0.0612, 0.2557, 0.1027],
  16: [0.0238, -0.2427, -0.873], 17: [0.0396, 0.3045, 0.8404],
};

// The rig's neutral relaxes the arms (builtin.ts createResearchHumanMapping), and its bone basis is taken
// from that relaxed pose, so a bone delta is a world-axis change from the neutral, composed down the chain.
// The reference pose is what leaves the rig at its neutral: the average posture above for torso, collars and upper
// arms, and an anatomical slight bend for the elbows (elbow bend is the gesture itself, so it stays absolute).
const reference = (() => {
  const local = Array.from({ length: 55 }, () => new Quaternion());
  for (const [side, sign] of [[0, 1], [1, -1]] as const) {
    const upper = new Vector3(sign * 0.16, -1, 0.025).normalize(), fore = new Vector3(sign * 0.10, -1, 0.14).normalize();
    const gUpper = new Quaternion().setFromUnitVectors(new Vector3(sign, 0, 0), upper);
    const gFore = new Quaternion().setFromUnitVectors(upper, fore).multiply(gUpper);
    local[18 + side] = gUpper.clone().invert().multiply(gFore);
  }
  for (const [joint, [x, y, z]] of Object.entries(POSTURE_MEAN)) {
    const angle = Math.hypot(x, y, z);
    local[Number(joint)].setFromAxisAngle(new Vector3(x / angle, y / angle, z / angle), angle);
  }
  const global: Quaternion[] = [];
  local.forEach((q, j) => { global[j] = j ? global[SMPLX_PARENTS[j]].clone().multiply(q) : q.clone(); });   // parents precede children
  return { global, local };
})();

/** The SMPL-X pose (local rotations) that leaves the rig at its neutral; exported for tests and tools. */
export function speechGestureReference(): { local: Quaternion[]; global: Quaternion[] } {
  return { local: reference.local.map((q) => q.clone()), global: reference.global.map((q) => q.clone()) };
}

// Control hierarchy used to hand a parent's rotation to its child when a rig lacks the parent
// (e.g. the procedural portrait has no collar or spine bones: EMAGE lowers the arm partly through the collar).
const CONTROL_PARENT = new Map<BoneControl, BoneControl>([['chest', 'spine']]);
for (const side of ['left', 'right'] as const) {
  CONTROL_PARENT.set(`${side}Shoulder`, 'chest');
  CONTROL_PARENT.set(`${side}UpperArm`, `${side}Shoulder`);
  CONTROL_PARENT.set(`${side}Forearm`, `${side}UpperArm`);
  CONTROL_PARENT.set(`${side}Hand`, `${side}Forearm`);
  FINGER_JOINTS.forEach((joint, index) => {
    const first = index % 3 === 0;
    CONTROL_PARENT.set(`${side}${joint}` as BoneControl, first ? `${side}Hand` : `${side}${FINGER_JOINTS[index - 1]}` as BoneControl);
  });
}

const smoothstep = (t: number) => { const x = Math.max(0, Math.min(1, t)); return x * x * (3 - 2 * x); };
const axisAngle = (poses: Float32Array, frame: number, joint: number, target: Quaternion): Quaternion => {
  const o = frame * SMPLX_POSE_WIDTH + joint * 3, x = poses[o], y = poses[o + 1], z = poses[o + 2], angle = Math.hypot(x, y, z);
  if (angle < 1e-8) return target.identity();
  const s = Math.sin(angle / 2) / angle;
  return target.set(x * s, y * s, z * s, Math.cos(angle / 2));
};

/** Bone deltas for the audio clock; null outside the clip. Deterministic for seeks and restarts.
 * `supported` lists the rig's bones: a missing bone's rotation is passed down to its nearest supported
 * descendant so the limbs still point where the generated motion puts them. */
export function sampleSpeechGesture(motion: SpeechGestureMotion, audioElapsedMs: number,
  options: { reducedMotion?: boolean; supported?: readonly BoneControl[]; layer?: { weight: number } } = {}): BoneRotations | null {
  const duration = motion.frames / motion.fps * 1000;
  const stream = motion.stream;
  if (!Number.isFinite(audioElapsedMs) || audioElapsedMs < 0 || motion.frames < 1) return null;
  let envelope: number;
  if (stream && !stream.complete) {
    // Still arriving: if playback reaches the last frame, hold it and fade out until more frames come.
    const last = (motion.frames - 1) / motion.fps * 1000;
    const stall = smoothstep(1 - Math.max(0, audioElapsedMs - last) / 350);
    if (stall <= 0) return null;
    envelope = smoothstep((audioElapsedMs - stream.fadeFromMs) / 250) * stall;
  } else {
    if (audioElapsedMs >= duration) return null;
    envelope = smoothstep((audioElapsedMs - (stream?.fadeFromMs ?? 0)) / 250) * smoothstep((duration - audioElapsedMs) / 350);
  }
  envelope *= options.reducedMotion ? 0.15 : 1;
  const position = Math.min(motion.frames - 1, audioElapsedMs / 1000 * motion.fps);
  const a = Math.floor(position), b = Math.min(motion.frames - 1, a + 1), mix = position - a;
  const identity = new Quaternion();
  const qa = new Quaternion(), qb = new Quaternion(), local = new Quaternion(), refLocal = new Quaternion();
  const supported = options.supported ? new Set(options.supported) : null;
  const world = new Map<BoneControl, Quaternion>();   // accumulated world-axis change, rig deltas compose parent-first
  const bones: BoneRotations = {};
  for (const [control, joints, gain] of CHANNELS) {
    local.identity(); refLocal.identity();
    for (const joint of joints) {
      local.multiply(axisAngle(motion.poses, a, joint, qa).slerp(axisAngle(motion.poses, b, joint, qb), mix));
      refLocal.multiply(reference.local[joint]);
    }
    // Change relative to the reference pose, expressed in world axes at the reference parent: T_p (r a⁻¹) T_p⁻¹.
    const parent = reference.global[SMPLX_PARENTS[joints[0]]];
    const change = parent.clone().multiply(local).multiply(refLocal.invert()).multiply(parent.clone().invert());
    if (change.w < 0) change.set(-change.x, -change.y, -change.z, -change.w);
    let delta = identity.clone().slerp(change, gain);
    if (supported) {
      // world change of this control, then relative to the nearest ancestor the rig actually has
      let up = CONTROL_PARENT.get(control);
      const accumulated = (up && world.get(up)?.clone().multiply(delta)) || delta.clone();
      world.set(control, accumulated);
      if (!supported.has(control)) continue;
      while (up && !supported.has(up)) up = CONTROL_PARENT.get(up);
      delta = (up ? world.get(up)!.clone().invert() : new Quaternion()).multiply(accumulated);
      if (delta.w < 0) delta.set(-delta.x, -delta.y, -delta.z, -delta.w);
    }
    // With `layer`, the caller blends full-strength bones over the designed pose by layer.weight instead.
    if (!options.layer) delta = identity.clone().slerp(delta, envelope);
    bones[control] = delta.normalize().toArray() as QuaternionTuple;
  }
  if (options.layer) options.layer.weight = envelope;
  return bones;
}
