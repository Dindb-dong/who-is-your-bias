import { VISEME_SET } from './channels';
import type { FaceWeights, SpeechTiming, Viseme, VisemeWeights } from './types';
import { clamp01 } from './timeline';

const RHUBARB: Record<string, Viseme> = { A: 'PP', B: 'kk', C: 'E', D: 'aa', E: 'O', F: 'U', G: 'FF', H: 'nn', X: 'sil' };
const ARPABET: Record<string, Viseme> = {
  SIL: 'sil', SP: 'sil', SPN: 'sil', PAU: 'sil', '<SIL>': 'sil',
  P: 'PP', B: 'PP', M: 'PP', F: 'FF', V: 'FF', TH: 'TH', DH: 'TH',
  T: 'DD', D: 'DD', K: 'kk', G: 'kk', NG: 'kk', HH: 'kk',
  CH: 'CH', JH: 'CH', SH: 'CH', ZH: 'CH', S: 'SS', Z: 'SS',
  N: 'nn', L: 'nn', R: 'RR', ER: 'RR',
  AA: 'aa', AH: 'aa', AE: 'E', EH: 'E', EY: 'E',
  IH: 'I', IY: 'I', Y: 'I', AO: 'O', OW: 'O', OY: 'O', AW: 'aa', AY: 'aa',
  UH: 'U', UW: 'U', W: 'U',
};
const IPA: Record<string, Viseme> = {
  sil: 'sil', sp: 'sil', pau: 'sil', p: 'PP', b: 'PP', m: 'PP', f: 'FF', v: 'FF',
  θ: 'TH', ð: 'TH', t: 'DD', d: 'DD', k: 'kk', g: 'kk', ɡ: 'kk', ŋ: 'kk', h: 'kk',
  tʃ: 'CH', dʒ: 'CH', ʃ: 'CH', ʒ: 'CH', s: 'SS', z: 'SS', n: 'nn', l: 'nn',
  r: 'RR', ɹ: 'RR', ɝ: 'RR', ɚ: 'RR', ɑ: 'aa', a: 'aa', ʌ: 'aa', ə: 'aa',
  æ: 'E', ɛ: 'E', e: 'E', eɪ: 'E', ɪ: 'I', i: 'I', j: 'I',
  ɔ: 'O', o: 'O', oʊ: 'O', ɔɪ: 'O', ʊ: 'U', u: 'U', w: 'U', aɪ: 'aa', aʊ: 'aa',
};

/** These are authored articulatory approximations, not measured facial coefficients. */
const SHAPES: Record<Viseme, FaceWeights> = {
  sil: {}, PP: { mouthClose: 1, mouthPressLeft: 0.65, mouthPressRight: 0.65 },
  FF: { jawOpen: 0.12, mouthRollLower: 0.5, mouthUpperUpLeft: 0.2, mouthUpperUpRight: 0.2 },
  TH: { jawOpen: 0.16, tongueOut: 0.45 }, DD: { jawOpen: 0.22, tongueOut: 0.08 },
  kk: { jawOpen: 0.18, mouthStretchLeft: 0.12, mouthStretchRight: 0.12 },
  CH: { jawOpen: 0.22, mouthFunnel: 0.3 }, SS: { jawOpen: 0.1, mouthStretchLeft: 0.3, mouthStretchRight: 0.3 },
  nn: { jawOpen: 0.18, tongueOut: 0.08 }, RR: { jawOpen: 0.24, mouthFunnel: 0.25 },
  aa: { jawOpen: 0.68, mouthLowerDownLeft: 0.25, mouthLowerDownRight: 0.25 },
  E: { jawOpen: 0.34, mouthStretchLeft: 0.4, mouthStretchRight: 0.4 },
  I: { jawOpen: 0.17, mouthStretchLeft: 0.65, mouthStretchRight: 0.65 },
  O: { jawOpen: 0.48, mouthFunnel: 0.65 }, U: { jawOpen: 0.17, mouthPucker: 0.75, mouthFunnel: 0.3 },
};
const SPEECH_CHANNELS = [...new Set(Object.values(SHAPES).flatMap((shape) => Object.keys(shape)))] as (keyof FaceWeights)[];

export function labelToViseme(label: string, set: SpeechTiming['label_set']): Viseme | undefined {
  if (set === 'rhubarb') return RHUBARB[label.toUpperCase()];
  if (set === 'viseme_15') return VISEME_SET.has(label) ? label as Viseme : undefined;
  if (set === 'arpabet') return ARPABET[label.toUpperCase().replace(/[012]$/, '')];
  return IPA[label.normalize('NFC').replace(/[ˈˌːˑ]/g, '')];
}

export function visemesToFace(weights: VisemeWeights): FaceWeights {
  const face: FaceWeights = Object.fromEntries(SPEECH_CHANNELS.map((channel) => [channel, 0]));
  for (const [name, weight] of Object.entries(weights) as [Viseme, number][]) {
    for (const [channel, value] of Object.entries(SHAPES[name]) as [keyof FaceWeights, number][]) face[channel] = clamp01((face[channel] ?? 0) + value * clamp01(weight));
  }
  return face;
}

export interface SpeechSample { face: FaceWeights; visemes: VisemeWeights; unsupportedLabels: string[]; usable: boolean; }

/** Uses actual audio seconds, never the estimated behavior-plan clock. */
export function sampleSpeech(timing: SpeechTiming, elapsedMs: number, audioSha256?: string): SpeechSample | null {
  if (audioSha256 !== timing.audio_sha256 || !Number.isFinite(elapsedMs) || elapsedMs < 0 || !timing.segments.length) return null;
  const unsupportedLabels: string[] = [];
  const lookup = (label: string | undefined): Viseme | undefined => {
    if (!label) return 'sil';
    const viseme = labelToViseme(label, timing.label_set);
    if (!viseme) unsupportedLabels.push(label);
    return viseme;
  };
  const segments = timing.segments;
  let low = 0; let high = segments.length;
  while (low < high) { const middle = (low + high) >>> 1; if (segments[middle].start_ms <= elapsedMs) low = middle + 1; else high = middle; }
  const index = low - 1; const current = segments[index];
  if (!current || elapsedMs >= current.end_ms || elapsedMs >= timing.audio_duration_ms) {
    const weights: VisemeWeights = { sil: 1 };
    return { face: visemesToFace(weights), visemes: weights, unsupportedLabels, usable: true };
  }
  const active = lookup(current.label);
  if (!active) return { face: {}, visemes: {}, unsupportedLabels, usable: false };
  let left: Viseme = active; let right: Viseme = active; let amount = 0;
  const previous = segments[index - 1]; const next = segments[index + 1];
  // A short, explicit coarticulation approximation around adjacent cue boundaries.
  const width = Math.min(35, (current.end_ms - current.start_ms) / 3);
  if (previous && Math.abs(previous.end_ms - current.start_ms) < 1 && elapsedMs < current.start_ms + width) {
    const preceding = lookup(previous.label);
    if (preceding) { left = preceding; right = active; amount = clamp01((elapsedMs - current.start_ms + width) / (2 * width)); }
  } else if (next && Math.abs(next.start_ms - current.end_ms) < 1 && elapsedMs > current.end_ms - width) {
    const following = lookup(next.label);
    if (following) { left = active; right = following; amount = clamp01((elapsedMs - current.end_ms + width) / (2 * width)); }
  }
  amount = amount * amount * (3 - 2 * amount);
  const weights: VisemeWeights = { [left]: 1 - amount };
  weights[right] = (weights[right] ?? 0) + amount;
  return { face: visemesToFace(weights), visemes: weights, unsupportedLabels: [...new Set(unsupportedLabels)], usable: true };
}
