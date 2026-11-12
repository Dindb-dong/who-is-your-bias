import { Euler, Quaternion } from 'three';
import type { AttentionTelemetry, FaceChannel, FaceWeights, QuaternionTuple, RigDiagnostics, RigPose } from './types';

/** Display coordinates, after any preview mirroring. No identity or demographic data. */
export interface AttentionTarget {
  id: string;
  x: number;
  y: number;
  /** The same monotonic performance.now() clock used by the renderer. */
  observedAtMs: number;
}

/** Artistic animation settings, not measurements of a person's behavior. */
export const ATTENTION_SETTINGS = Object.freeze({
  headDelayMs: 220,
  maxTargetAgeMs: 500,
  targetYaw: 0.60,
  targetPitch: 0.36,
  maxEyeYaw: 0.35,
  maxEyePitch: 0.25,
  maxHeadYaw: 0.42,
  maxHeadPitch: 0.22,
  reducedEyeAngle: 0.08,
  eyeTimeConstantMs: 55,
  headTimeConstantMs: 280,
  returnTimeConstantMs: 240,
});

export type AttentionPhase = 'idle' | 'eyes' | 'following' | 'returning' | 'reduced';

export interface AttentionFrame {
  phase: AttentionPhase;
  targetId: string | null;
  /** Desired world-facing direction. Positive pitch points down; yaw points right. */
  lookYaw: number;
  lookPitch: number;
  /** Total head + neck contribution; the compositor divides it across mapped joints. */
  headYaw: number;
  headPitch: number;
  eyeWeight: number;
  headWeight: number;
  reducedMotion: boolean;
}

const clamp = (value: number, limit: number): number => Math.max(-limit, Math.min(limit, value));
const blend = (from: number, to: number, dtMs: number, timeConstantMs: number): number =>
  from + (to - from) * (1 - Math.exp(-dtMs / timeConstantMs));
const idleFrame = (): AttentionFrame => ({ phase: 'idle', targetId: null, lookYaw: 0, lookPitch: 0,
  headYaw: 0, headPitch: 0, eyeWeight: 0, headWeight: 0, reducedMotion: false });

function validTarget(target: AttentionTarget | null | undefined, now: number): target is AttentionTarget {
  return !!target && typeof target.id === 'string' && target.id.length > 0 && target.id.length <= 256
    && Number.isFinite(target.x) && target.x >= 0 && target.x <= 1
    && Number.isFinite(target.y) && target.y >= 0 && target.y <= 1
    && Number.isFinite(target.observedAtMs) && target.observedAtMs >= 0 && target.observedAtMs <= now
    && now - target.observedAtMs <= ATTENTION_SETTINGS.maxTargetAgeMs;
}

/** Persistent wall-clock state: new detections never restart a selected target's head delay. */
export class AttentionController {
  private frame = idleFrame();
  private previousNow: number | null = null;
  private activeId: string | null = null;
  private headStartsAt = 0;
  private lastObservation = -Infinity;

  get active(): boolean { return this.frame.phase !== 'idle'; }

  reset(): void {
    this.frame = idleFrame(); this.previousNow = null; this.activeId = null;
    this.headStartsAt = 0; this.lastObservation = -Infinity;
  }

  sample(target: AttentionTarget | null | undefined, now: number, reducedMotion = false, headSupported = true): AttentionFrame {
    if (!Number.isFinite(now) || now < 0) { this.reset(); return { ...this.frame }; }
    if (this.previousNow !== null && now < this.previousNow) this.reset();
    const dtMs = this.previousNow === null ? 1000 / 60 : Math.min(100, Math.max(0, now - this.previousNow));
    this.previousNow = now;
    const fresh = validTarget(target, now) && target.observedAtMs >= this.lastObservation;
    if (fresh) {
      if (target.id !== this.activeId) {
        this.activeId = target.id; this.headStartsAt = now + ATTENTION_SETTINGS.headDelayMs;
      }
      this.lastObservation = target.observedAtMs;
    } else this.activeId = null;

    const turnHead = fresh && now >= this.headStartsAt && !reducedMotion && headSupported;
    const goalYaw = fresh ? (target.x * 2 - 1) * ATTENTION_SETTINGS.targetYaw : 0;
    const goalPitch = fresh ? (target.y * 2 - 1) * ATTENTION_SETTINGS.targetPitch : 0;
    const eyeLimit = reducedMotion ? ATTENTION_SETTINGS.reducedEyeAngle : Infinity;
    const timeConstant = fresh ? ATTENTION_SETTINGS.eyeTimeConstantMs : ATTENTION_SETTINGS.returnTimeConstantMs;
    this.frame.lookYaw = blend(this.frame.lookYaw, clamp(goalYaw, eyeLimit), dtMs, timeConstant);
    this.frame.lookPitch = blend(this.frame.lookPitch, clamp(goalPitch, eyeLimit), dtMs, timeConstant);
    // Hold the existing head angle during the eye lead after a target switch.
    if (turnHead || !fresh || reducedMotion || !headSupported) {
      this.frame.headYaw = blend(this.frame.headYaw,
        turnHead ? clamp(goalYaw * 0.7, ATTENTION_SETTINGS.maxHeadYaw) : 0,
        dtMs, fresh ? ATTENTION_SETTINGS.headTimeConstantMs : ATTENTION_SETTINGS.returnTimeConstantMs);
      this.frame.headPitch = blend(this.frame.headPitch,
        turnHead ? clamp(goalPitch * 0.65, ATTENTION_SETTINGS.maxHeadPitch) : 0,
        dtMs, fresh ? ATTENTION_SETTINGS.headTimeConstantMs : ATTENTION_SETTINGS.returnTimeConstantMs);
    }
    this.frame.eyeWeight = blend(this.frame.eyeWeight, fresh ? 1 : 0, dtMs, fresh ? 65 : ATTENTION_SETTINGS.returnTimeConstantMs);
    // Preserve the original head animation until the first delayed head turn.
    if (turnHead || !fresh || reducedMotion || !headSupported) {
      this.frame.headWeight = blend(this.frame.headWeight, turnHead ? 1 : 0, dtMs,
        fresh ? ATTENTION_SETTINGS.headTimeConstantMs : ATTENTION_SETTINGS.returnTimeConstantMs);
    }
    this.frame.targetId = fresh ? target.id : null;
    this.frame.reducedMotion = reducedMotion;
    this.frame.phase = fresh ? reducedMotion ? 'reduced' : now < this.headStartsAt ? 'eyes' : 'following' : 'returning';
    if (!fresh && Math.max(this.frame.eyeWeight, this.frame.headWeight, Math.abs(this.frame.lookYaw),
      Math.abs(this.frame.lookPitch), Math.abs(this.frame.headYaw), Math.abs(this.frame.headPitch)) < 0.0005) {
      this.frame = idleFrame();
    }
    return { ...this.frame };
  }
}

const rotation = (pitch = 0, yaw = 0, roll = 0): Quaternion => new Quaternion().setFromEuler(new Euler(pitch, yaw, roll));
const tuple = (value: Quaternion): QuaternionTuple => value.toArray() as QuaternionTuple;
const angles = (value: Quaternion): Euler => new Euler().setFromQuaternion(value);
const sides = ['Left', 'Right'] as const;
const eyeChannels = (side: typeof sides[number]): FaceChannel[] =>
  (['In', 'Out', 'Up', 'Down'] as const).map((direction) => `eyeLook${direction}${side}` as FaceChannel);

export function attentionCapabilities(diagnostics: RigDiagnostics): { eyes: boolean; head: boolean; unsupported: string[] } {
  const unsupported: string[] = [];
  for (const side of sides) {
    const bone = side === 'Left' ? 'leftEye' : 'rightEye';
    if (!diagnostics.bones.includes(bone) && !eyeChannels(side).every((channel) => diagnostics.morphs.includes(channel))) {
      unsupported.push(`${side.toLowerCase()} eye gaze`);
    }
  }
  const head = diagnostics.bones.includes('head') || diagnostics.bones.includes('neck');
  if (!head) unsupported.push('head/neck turn');
  return { eyes: !unsupported.some((entry) => entry.includes('eye')), head, unsupported };
}

function eyeAngles(face: FaceWeights | undefined, side: typeof sides[number], base: RigPose): { yaw: number; pitch: number } {
  const horizontal = eyeChannels(side).slice(0, 2).some((channel) => face?.[channel] !== undefined);
  const vertical = eyeChannels(side).slice(2).some((channel) => face?.[channel] !== undefined);
  const out = face?.[`eyeLookOut${side}`] ?? 0; const inward = face?.[`eyeLookIn${side}`] ?? 0;
  return {
    yaw: horizontal ? (out - inward) * (side === 'Left' ? 1 : -1) * ATTENTION_SETTINGS.maxEyeYaw : base.gazeX,
    pitch: vertical ? ((face?.[`eyeLookDown${side}`] ?? 0) - (face?.[`eyeLookUp${side}`] ?? 0)) * ATTENTION_SETTINGS.maxEyePitch : -base.gazeY,
  };
}

/**
 * Apply after normal playback damping. Attention owns only eye/head/neck direction;
 * lips, speech, blink and all remaining face/body channels retain their source pose.
 * This creates new maps and never mutates a recorded clip or its sampled pose.
 */
export function composeAttentionPose(base: RigPose, frame: AttentionFrame, diagnostics: RigDiagnostics): { pose: RigPose; telemetry: AttentionTelemetry } {
  const capability = attentionCapabilities(diagnostics);
  const telemetry: AttentionTelemetry = { phase: frame.phase, targetId: frame.targetId, eyeYaw: 0, eyePitch: 0,
    headYaw: 0, headPitch: 0, eyeWeight: frame.eyeWeight, headWeight: frame.headWeight, unsupported: capability.unsupported };
  if (frame.phase === 'idle') return { pose: base, telemetry };
  const pose: RigPose = { ...base, face: { ...base.face }, bones: { ...base.bones } };
  const hasHead = diagnostics.bones.includes('head'); const hasNeck = diagnostics.bones.includes('neck');
  const neckShare = hasNeck ? hasHead ? 0.2 : 1 : 0;
  const headShare = hasHead ? 1 - neckShare : 0;
  const baseHead = base.bones?.head ? new Quaternion(...base.bones.head) : rotation(base.headPitch, base.headYaw, base.headRoll);
  const baseNeck = base.bones?.neck ? new Quaternion(...base.bones.neck)
    : rotation(base.headPitch * 0.2, base.headYaw * (diagnostics.kind === 'procedural' ? 0.15 : 0.2), base.headRoll * 0.2);
  const head = hasHead ? baseHead.clone().slerp(rotation(frame.headPitch * headShare, frame.headYaw * headShare, angles(baseHead).z), frame.headWeight) : new Quaternion();
  const neck = hasNeck ? baseNeck.clone().slerp(rotation(frame.headPitch * neckShare, frame.headYaw * neckShare, angles(baseNeck).z), frame.headWeight) : new Quaternion();
  if (hasHead && frame.headWeight > 0) pose.bones!.head = tuple(head);
  if (hasNeck && frame.headWeight > 0) pose.bones!.neck = tuple(neck);
  const total = angles(neck.clone().multiply(head));
  telemetry.headYaw = total.y; telemetry.headPitch = total.x;
  const yawLimit = frame.reducedMotion ? ATTENTION_SETTINGS.reducedEyeAngle : ATTENTION_SETTINGS.maxEyeYaw;
  const pitchLimit = frame.reducedMotion ? ATTENTION_SETTINGS.reducedEyeAngle : ATTENTION_SETTINGS.maxEyePitch;
  const desiredEyeYaw = clamp(frame.lookYaw - total.y, yawLimit);
  const desiredEyePitch = clamp(frame.lookPitch - total.x, pitchLimit);
  // All supported eye paths below receive one route: either bones or morphs.
  pose.gazeX = 0; pose.gazeY = 0;
  for (const side of sides) {
    const eye = side === 'Left' ? 'leftEye' : 'rightEye';
    const original = eyeAngles(base.face, side, base);
    const eyeYaw = original.yaw + (desiredEyeYaw - original.yaw) * frame.eyeWeight;
    const eyePitch = original.pitch + (desiredEyePitch - original.pitch) * frame.eyeWeight;
    let reportedYaw = eyeYaw; let reportedPitch = eyePitch;
    const bone = diagnostics.bones.includes(eye);
    for (const channel of eyeChannels(side)) delete pose.face![channel];
    delete pose.bones![eye];
    if (bone) {
      const originalRotation = base.bones?.[eye] ? new Quaternion(...base.bones[eye]!) : rotation(original.pitch, original.yaw);
      const composed = originalRotation.slerp(rotation(desiredEyePitch, desiredEyeYaw), frame.eyeWeight);
      pose.bones![eye] = tuple(composed);
      const composedAngles = angles(composed);
      reportedYaw = composedAngles.y; reportedPitch = composedAngles.x;
    } else {
      const horizontal = clamp(eyeYaw / ATTENTION_SETTINGS.maxEyeYaw, 1) * (side === 'Left' ? 1 : -1);
      const vertical = clamp(eyePitch / ATTENTION_SETTINGS.maxEyePitch, 1);
      const controls: FaceWeights = { [`eyeLookOut${side}`]: Math.max(0, horizontal), [`eyeLookIn${side}`]: Math.max(0, -horizontal),
        [`eyeLookDown${side}`]: Math.max(0, vertical), [`eyeLookUp${side}`]: Math.max(0, -vertical) };
      for (const channel of eyeChannels(side)) if (diagnostics.morphs.includes(channel)) pose.face![channel] = controls[channel];
    }
    telemetry.eyeYaw += reportedYaw / 2; telemetry.eyePitch += reportedPitch / 2;
  }
  return { pose, telemetry };
}
