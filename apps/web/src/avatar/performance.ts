import { Quaternion } from 'three';
import { sampleMotion } from './motion';
import { sampleSpeech } from './speech';
import { samplePose } from './timeline';
import type { AnimationTelemetry, AvatarClip, QuaternionTuple, RigDiagnostics, RigPose, SpeechTiming, TimelineInput } from './types';

export interface PerformanceInput extends TimelineInput {
  audioElapsedMs?: number;
  audioSha256?: string;
  speechTiming?: SpeechTiming;
  performanceClip?: AvatarClip;
  observationElapsedMs?: number;
  allowGeometryPreview?: boolean;
}

export function samplePerformance(input: PerformanceInput, diagnostics: RigDiagnostics): { pose: RigPose; telemetry: AnimationTelemetry } {
  const pose = samplePose(input);
  const telemetry: AnimationTelemetry = {
    faceSource: 'designed', boneSource: 'designed', speechSource: input.speaking ? input.amplitude === undefined ? 'estimated' : 'amplitude' : 'silent',
    unsupportedFace: [], unsupportedVisemes: [],
  };
  if (input.performanceClip) {
    const sample = sampleMotion(input.performanceClip, input.observationElapsedMs ?? input.elapsedMs, input.allowGeometryPreview);
    if (sample) {
      pose.face = sample.face; pose.bones = sample.bones;
      const source = input.performanceClip.review.status === 'geometry_preview' ? 'geometry_preview' : 'observed';
      if (Object.keys(sample.face ?? {}).length) telemetry.faceSource = source;
      if (Object.keys(sample.bones ?? {}).length) telemetry.boneSource = source;
      if (input.reducedMotion && pose.bones) {
        for (const name of Object.keys(pose.bones) as (keyof NonNullable<RigPose['bones']>)[]) {
          pose.bones[name] = new Quaternion().slerp(new Quaternion(...pose.bones[name]!), 0.15).toArray() as QuaternionTuple;
        }
      }
    }
    if (input.performanceClip.review.status === 'geometry_preview') telemetry.warning = 'Geometry preview: speaker association is unreviewed.';
  }
  if (input.speaking && input.speechTiming) {
    const sample = sampleSpeech(input.speechTiming, input.audioElapsedMs ?? NaN, input.audioSha256);
    if (sample?.usable) {
      pose.face = { ...pose.face, ...sample.face }; pose.visemes = sample.visemes;
      pose.jawOpen = sample.face.jawOpen ?? 0; pose.mouthRound = sample.face.mouthFunnel ?? 0;
      telemetry.speechSource = input.speechTiming.timing_basis;
      if (sample.unsupportedLabels.length) telemetry.warning = `Unsupported speech labels: ${sample.unsupportedLabels.join(', ')}.`;
      telemetry.unsupportedVisemes = Object.keys(sample.visemes).filter((name) => !diagnostics.visemes?.includes(name));
    } else telemetry.warning = sample?.unsupportedLabels.length
      ? `Unsupported speech labels: ${sample.unsupportedLabels.join(', ')}; using the amplitude or estimated fallback.`
      : 'Speech timing requires the matching audio digest and its actual playback clock.';
  }
  telemetry.unsupportedFace = Object.keys(pose.face ?? {}).filter((name) => !diagnostics.morphs.includes(name));
  return { pose, telemetry };
}
