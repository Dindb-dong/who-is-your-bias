import { Quaternion } from 'three';
import type { AvatarClip, BoneRotations, FaceWeights, QuaternionTuple } from './types';
import { clamp01 } from './timeline';

export interface MotionSample { face?: FaceWeights; bones?: BoneRotations; }

export function interpolateRotation(left: QuaternionTuple, right: QuaternionTuple, amount: number): QuaternionTuple {
  return new Quaternion().slerpQuaternions(new Quaternion(...left), new Quaternion(...right), clamp01(amount)).normalize().toArray() as QuaternionTuple;
}

/** Missing observations and cuts stay missing; they are never held indefinitely. */
export function sampleMotion(clip: AvatarClip, elapsedMs: number, allowGeometryPreview = false): MotionSample | null {
  if (clip.review.status === 'geometry_preview' && !allowGeometryPreview) return null;
  if (!Number.isFinite(elapsedMs) || elapsedMs < 0 || elapsedMs > clip.duration_ms) return null;
  const frames = clip.frames;
  let low = 0; let high = frames.length;
  while (low < high) { const middle = (low + high) >>> 1; if (frames[middle].time_ms <= elapsedMs) low = middle + 1; else high = middle; }
  const left = frames[low - 1]; const right = frames[low];
  if (!left || (!right && elapsedMs - left.time_ms > clip.max_gap_ms)) return null;
  if (right && right.time_ms - left.time_ms > clip.max_gap_ms && elapsedMs !== left.time_ms) return null;
  const following = right && elapsedMs !== left.time_ms ? right : left;
  const amount = following === left ? 0 : clamp01((elapsedMs - left.time_ms) / (following.time_ms - left.time_ms));
  const sample: MotionSample = {};
  if (left.face && following.face) {
    const face: FaceWeights = {};
    for (const channel of Object.keys(left.face) as (keyof FaceWeights)[]) {
      if (following.face[channel] === undefined) continue;
      const raw = left.face[channel]! + (following.face[channel]! - left.face[channel]!) * amount;
      const neutral = clip.calibration.face_neutral[channel] ?? 0;
      face[channel] = clamp01((raw - neutral) / Math.max(0.01, 1 - neutral));
    }
    sample.face = face;
  }
  if (left.bones && following.bones) {
    const bones: BoneRotations = {};
    for (const channel of Object.keys(left.bones) as (keyof BoneRotations)[]) {
      if (following.bones[channel]) bones[channel] = interpolateRotation(left.bones[channel]!, following.bones[channel]!, amount);
    }
    sample.bones = bones;
  }
  return sample.face || sample.bones ? sample : null;
}
