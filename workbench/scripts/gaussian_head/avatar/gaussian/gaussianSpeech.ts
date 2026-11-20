import { sampleSpeech } from '../speech';
import type { SpeechTiming, Viseme } from '../types';

// Weights for [closed, slightly open, wide open, narrow lips]. The poses are
// matched to the rendered 240k bundle. In that bundle a larger jaw X opens
// the mouth. Each example keeps its jaw and expression coefficients together.
type MouthWeights = readonly [number, number, number, number];
type HeadFrame = { expr: Float32Array; pose: Float32Array };

const mouthFrames = [942, 1215, 2190, 1200];
export const restingMouth: MouthWeights = [1, 0, 0, 0];
const targets: Record<Viseme, MouthWeights> = {
  sil: restingMouth, PP: restingMouth,
  FF: [0.72, 0.28, 0, 0],
  TH: [0.52, 0.48, 0, 0],
  DD: [0.57, 0.43, 0, 0],
  kk: [0.52, 0.48, 0, 0],
  CH: [0.35, 0.35, 0, 0.30],
  SS: [0.66, 0.34, 0, 0],
  nn: [0.4, 0.6, 0, 0],
  RR: [0.38, 0.32, 0, 0.30],
  aa: [0.10, 0, 0.90, 0],
  E: [0.12, 0.88, 0, 0],
  I: [0.45, 0.55, 0, 0],
  O: [0, 0, 0.25, 0.75],
  U: [0.12, 0, 0, 0.88],
};

export function createTrackedMouth(frameAt: (index: number) => HeadFrame) {
  const poses = mouthFrames.map(frameAt);
  const weighted = (channel: 'expr' | 'pose', i: number, [closed, narrow, wide, rounded]: MouthWeights) =>
    poses[0][channel][i] * closed + poses[1][channel][i] * narrow +
    poses[2][channel][i] * wide + poses[3][channel][i] * rounded;
  return (params: HeadFrame, weights: MouthWeights) => {
    for (let i = 0; i < params.expr.length; i++) params.expr[i] = weighted('expr', i, weights);
    // Leave the head pose, eyes and translation controlled by the stage.
    for (let i = 6; i < 9; i++) params.pose[i] = weighted('pose', i, weights);
  };
}

export function mouthTarget(timing: SpeechTiming | undefined, hash: string | undefined, elapsedMs: number) {
  if (!timing) return null;
  const sample = sampleSpeech(timing, elapsedMs + 12, hash);
  if (!sample?.usable) return null;
  const weights: [number, number, number, number] = [0, 0, 0, 0];
  let label: Viseme = 'sil';
  let strongest = 0;
  for (const [viseme, weight] of Object.entries(sample.visemes) as [Viseme, number][]) {
    const target = targets[viseme];
    for (let i = 0; i < weights.length; i++) weights[i] += weight * target[i];
    if (weight > strongest) {
      label = viseme;
      strongest = weight;
    }
  }
  return { weights, label };
}

export function blendMouth(current: MouthWeights, target: MouthWeights, elapsedSeconds: number): MouthWeights {
  const timeConstant = target[0] > current[0] ? 0.014 : 0.024;
  const alpha = 1 - Math.exp(-elapsedSeconds / timeConstant);
  const blend = (i: number) => current[i] + (target[i] - current[i]) * alpha;
  return [blend(0), blend(1), blend(2), blend(3)];
}

export function energyMouth(amplitude: number): MouthWeights {
  const energy = Math.min(1, Math.max(0, (amplitude - 0.04) * 1.45));
  return [1 - energy, energy * 0.68, energy * 0.32, 0];
}

/** At most one small nod, during a substantive internal pause in the verified audio. */
export function speechNod(timing: SpeechTiming | undefined, hash: string | undefined) {
  if (!timing || timing.audio_sha256 !== hash || timing.label_set !== 'rhubarb') return null;
  const pause = timing.segments.find(cue => cue.label === 'X' && cue.start_ms >= 1500 &&
    cue.end_ms <= timing.audio_duration_ms - 650 && cue.end_ms - cue.start_ms >= 260);
  return pause ? { start: pause.start_ms, duration: Math.min(420, pause.end_ms - pause.start_ms) } : null;
}

export function nodPitch(nod: ReturnType<typeof speechNod>, elapsedMs: number) {
  if (!nod) return 0;
  const phase = (elapsedMs - nod.start) / nod.duration;
  return phase > 0 && phase < 1 ? 0.023 * Math.sin(Math.PI * phase) ** 2 : 0;
}
