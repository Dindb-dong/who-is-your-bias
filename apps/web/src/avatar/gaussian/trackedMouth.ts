import type { Viseme, VisemeWeights } from '../types';

export const TRACKED_MOUTH_BUNDLE = '32d5bc15901a8673901c6234ae7af946d2bfcf2c0aa5e7ed9c816d1c3aff599c';

// Weights for [closed, slightly open, wide open, narrow lips]. The poses are
// matched to the rendered 240k bundle. In that bundle a larger jaw X opens
// the mouth. Each example keeps its jaw and expression coefficients together.
type MouthWeights = readonly [number, number, number, number];
type MouthFrame = { expr: Float32Array; pose: Float32Array };

const mouthFrames = [942, 1215, 2190, 1200];
const restingMouth: MouthWeights = [1, 0, 0, 0];
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

function timedMouth(visemes: VisemeWeights): MouthWeights {
  const weights: [number, number, number, number] = [0, 0, 0, 0];
  for (const [name, weight] of Object.entries(visemes) as [Viseme, number][]) {
    const target = targets[name];
    for (let i = 0; i < weights.length; i++) weights[i] += target[i] * weight;
  }
  return weights;
}

function blendMouth(current: MouthWeights, target: MouthWeights, elapsedSeconds: number): MouthWeights {
  const timeConstant = target[0] > current[0] ? 0.014 : 0.024;
  const alpha = 1 - Math.exp(-elapsedSeconds / timeConstant);
  const blend = (i: number) => current[i] + (target[i] - current[i]) * alpha;
  return [blend(0), blend(1), blend(2), blend(3)];
}

function energyMouth(amplitude: number): MouthWeights {
  const energy = Math.min(1, Math.max(0, (amplitude - 0.04) * 1.45));
  return [1 - energy, energy * 0.68, energy * 0.32, 0];
}

export function createTrackedMouth(frameAt: (index: number) => MouthFrame) {
  const poses = mouthFrames.map(frameAt);
  let weights = restingMouth;
  const mix = (channel: 'expr' | 'pose', i: number) =>
    poses[0][channel][i] * weights[0] + poses[1][channel][i] * weights[1]
    + poses[2][channel][i] * weights[2] + poses[3][channel][i] * weights[3];
  return (params: MouthFrame, speaking: boolean, visemes: VisemeWeights | undefined,
          amplitude: number, delta: number | undefined) => {
    const target = speaking ? visemes ? timedMouth(visemes) : energyMouth(amplitude) : restingMouth;
    weights = delta === undefined ? target : blendMouth(weights, target, Math.max(0, Math.min(delta, 0.1)));
    for (let i = 0; i < params.expr.length; i++) params.expr[i] = mix('expr', i);
    for (let i = 6; i < 9; i++) params.pose[i] = mix('pose', i);
  };
}
