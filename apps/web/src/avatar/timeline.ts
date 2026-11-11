import { Quaternion } from 'three';
import type { BehaviorCue, BoneRotations, FaceWeights, QuaternionTuple, RigPose, TimelineInput, VisemeWeights } from './types';

export const clamp01 = (value: number): number =>
  Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;

const finite = (value: number): number => Number.isFinite(value) ? value : 0;
const smoothstep = (t: number): number => { const x = clamp01(t); return x * x * (3 - 2 * x); };

export function neutralPose(): RigPose {
  return {
    blink: 0, jawOpen: 0, mouthSmile: 0, mouthRound: 0,
    browRaise: 0, browFurrow: 0, gazeX: 0, gazeY: 0,
    headPitch: 0, headYaw: 0, headRoll: 0,
    openPalm: 0, emphasis: 0, shrug: 0, wave: 0, breath: 0,
  };
}

/** Smooth, bounded envelopes make overlapping cues safe to blend. */
export function cueWeight(cue: BehaviorCue, elapsedMs: number): number {
  if (!Number.isFinite(cue.start_ms) || !Number.isFinite(cue.duration_ms)
    || cue.duration_ms <= 0 || cue.start_ms < 0 || !Number.isFinite(elapsedMs)) return 0;
  const position = (elapsedMs - cue.start_ms) / cue.duration_ms;
  if (position <= 0 || position >= 1) return 0;
  const edge = Math.min(0.3, 180 / cue.duration_ms);
  return clamp01(cue.intensity) * smoothstep(position / edge) * smoothstep((1 - position) / edge);
}

/**
 * This is designed animation, not a learned model of Musk's mannerisms.
 * Only explicitly observed cues should ever carry observed provenance.
 */
export function samplePose(input: TimelineInput): RigPose {
  const pose = neutralPose();
  const time = Math.max(0, finite(input.idleMs)) / 1000;
  const elapsed = Math.max(0, finite(input.elapsedMs));
  const motion = input.reducedMotion ? 0.15 : 1;

  // Stable procedural idle: no Math.random(), so pauses/seeks are reproducible.
  if (!input.reducedMotion) {
    const cycle = time % 5.3;
    pose.blink = Math.exp(-Math.pow((cycle - 4.7) / 0.085, 2));
    pose.breath = Math.sin(time * 1.65) * 0.006;
    pose.headYaw = Math.sin(time * 0.47) * 0.015;
    pose.headRoll = Math.sin(time * 0.31 + 1.8) * 0.009;
    pose.gazeX = Math.sin(time * 0.38) * 0.025;
  }

  // An absent amplitude uses a modest designed articulation cycle. It must
  // never be represented as phoneme-level lip synchronization.
  if (input.speaking) {
    const amp = input.amplitude === undefined
      ? 0.17 + 0.28 * Math.pow(Math.sin(elapsed * 0.008), 2)
      : clamp01(input.amplitude);
    pose.jawOpen = clamp01(amp * 0.85);
    pose.mouthRound = Math.pow(Math.sin(elapsed * 0.0041 + 0.9), 4) * amp * 0.45;
  }

  const planDuration = input.plan?.duration_ms;
  if (planDuration !== undefined && (!Number.isFinite(planDuration) || elapsed > planDuration)) return pose;
  for (const cue of input.plan?.cues ?? []) {
    const weight = cueWeight(cue, elapsed);
    if (weight === 0) continue;
    const phase = (elapsed - cue.start_ms) / cue.duration_ms;
    const w = weight * motion;
    const name = cue.name.toLowerCase().replaceAll('-', '_');
    if (cue.channel === 'face') {
      if (name === 'smile' || name === 'warm') pose.mouthSmile += weight * 0.65;
      if (name === 'thoughtful') { pose.browFurrow += weight * 0.25; pose.gazeY -= w * 0.05; }
      if (name === 'brow_raise' || name === 'surprise') pose.browRaise += weight * 0.65;
    } else if (cue.channel === 'gaze') {
      if (name === 'aversion' || name === 'left') { pose.gazeX -= w * 0.3; pose.headYaw -= w * 0.08; }
      if (name === 'right') { pose.gazeX += w * 0.3; pose.headYaw += w * 0.08; }
      if (name === 'down') pose.gazeY -= w * 0.22;
      if (name === 'center') { pose.gazeX *= 1 - weight; pose.gazeY *= 1 - weight; }
    } else if (cue.channel === 'head') {
      if (name === 'nod') pose.headPitch += Math.sin(phase * Math.PI * 2) * w * 0.13;
      if (name === 'tilt') pose.headRoll += w * 0.09;
    } else if (cue.channel === 'gesture') {
      if (name === 'open_palm') pose.openPalm += w;
      if (name === 'emphasis') pose.emphasis += w * (0.65 + Math.sin(phase * Math.PI * 2) * 0.35);
      if (name === 'shrug') pose.shrug += w;
      if (name === 'wave') pose.wave += w;
    }
  }
  for (const key of ['blink', 'jawOpen', 'mouthSmile', 'mouthRound', 'browRaise',
    'browFurrow', 'openPalm', 'emphasis', 'shrug', 'wave'] as const) pose[key] = clamp01(pose[key]);
  pose.gazeX = Math.max(-0.4, Math.min(0.4, pose.gazeX));
  pose.gazeY = Math.max(-0.3, Math.min(0.3, pose.gazeY));
  pose.headPitch = Math.max(-0.2, Math.min(0.2, pose.headPitch));
  pose.headYaw = Math.max(-0.3, Math.min(0.3, pose.headYaw));
  pose.headRoll = Math.max(-0.2, Math.min(0.2, pose.headRoll));
  return pose;
}

/** Frame-rate-independent damping prevents a late audio chunk from snapping. */
export function dampPose(current: RigPose, target: RigPose, deltaSeconds: number): RigPose {
  const t = 1 - Math.exp(-18 * Math.max(0, Math.min(0.1, finite(deltaSeconds))));
  const output = { ...current };
  for (const key of Object.keys(neutralPose()) as (keyof Omit<RigPose, 'face' | 'bones' | 'visemes'>)[]) output[key] += (target[key] - output[key]) * t;
  if (current.face || target.face) {
    const face: FaceWeights = {};
    for (const channel of new Set([...Object.keys(current.face ?? {}), ...Object.keys(target.face ?? {})]) as Set<keyof FaceWeights>) {
      const from = current.face?.[channel] ?? 0; const to = target.face?.[channel] ?? 0;
      const value = from + (to - from) * t;
      if (target.face?.[channel] !== undefined || value > 0.001) face[channel] = clamp01(value);
    }
    output.face = Object.keys(face).length ? face : undefined;
  }
  if (current.visemes || target.visemes) {
    const visemes: VisemeWeights = {};
    for (const channel of new Set([...Object.keys(current.visemes ?? {}), ...Object.keys(target.visemes ?? {})]) as Set<keyof VisemeWeights>) {
      const from = current.visemes?.[channel] ?? 0; const to = target.visemes?.[channel] ?? 0;
      const value = from + (to - from) * t;
      if (target.visemes?.[channel] !== undefined || value > 0.001) visemes[channel] = clamp01(value);
    }
    output.visemes = Object.keys(visemes).length ? visemes : undefined;
  }
  if (current.bones || target.bones) {
    const bones: BoneRotations = {}; const identity: QuaternionTuple = [0, 0, 0, 1];
    for (const channel of new Set([...Object.keys(current.bones ?? {}), ...Object.keys(target.bones ?? {})]) as Set<keyof BoneRotations>) {
      const from = new Quaternion(...(current.bones?.[channel] ?? identity));
      const to = new Quaternion(...(target.bones?.[channel] ?? identity));
      const value = from.slerp(to, t).normalize();
      if (target.bones?.[channel] || value.angleTo(new Quaternion()) > 0.001) bones[channel] = value.toArray() as QuaternionTuple;
    }
    output.bones = Object.keys(bones).length ? bones : undefined;
  }
  return output;
}
