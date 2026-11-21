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

// SMPL-X kinematic tree (55 joints). Rest frames are world-aligned: Y up, facing +Z, subject's left at +X,
// which is also the runtime's subject-relative bone convention.
const SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15,
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

// The rig's neutral relaxes the arms (builtin.ts createResearchHumanMapping), and its bone basis is taken
// from that relaxed pose, so a bone delta is a world-axis change from the neutral, composed down the chain.
// The SMPL-X reference pose points the same limbs the same way: EMAGE arms hanging down leave the rig at
// its neutral, and changes from the reference map one to one.
const reference = (() => {
  const global = Array.from({ length: 55 }, () => new Quaternion());
  for (const [side, sign] of [[0, 1], [1, -1]] as const) {
    const upper = new Vector3(sign * 0.16, -1, 0.025).normalize(), fore = new Vector3(sign * 0.10, -1, 0.14).normalize();
    const shoulder = 16 + side, elbow = 18 + side, wrist = 20 + side;
    global[shoulder].setFromUnitVectors(new Vector3(sign, 0, 0), upper);
    global[elbow].setFromUnitVectors(upper, fore).multiply(global[shoulder]);
    global[wrist].copy(global[elbow]);
    for (let j = 25 + side * 15; j < 40 + side * 15; j++) global[j].copy(global[wrist]);
  }
  const local = global.map((q, j) => j ? global[SMPLX_PARENTS[j]].clone().invert().multiply(q) : q.clone());
  return { global, local };
})();

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
  options: { reducedMotion?: boolean; supported?: readonly BoneControl[] } = {}): BoneRotations | null {
  const duration = motion.frames / motion.fps * 1000;
  if (!Number.isFinite(audioElapsedMs) || audioElapsedMs < 0 || audioElapsedMs >= duration) return null;
  const position = Math.min(motion.frames - 1, audioElapsedMs / 1000 * motion.fps);
  const a = Math.floor(position), b = Math.min(motion.frames - 1, a + 1), mix = position - a;
  const envelope = smoothstep(audioElapsedMs / 250) * smoothstep((duration - audioElapsedMs) / 350) * (options.reducedMotion ? 0.15 : 1);
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
    delta = identity.clone().slerp(delta, envelope);
    bones[control] = delta.normalize().toArray() as QuaternionTuple;
  }
  return bones;
}
