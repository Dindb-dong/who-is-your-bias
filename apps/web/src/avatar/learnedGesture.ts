import { FINGER_JOINTS, type BoneControl, type BoneRotations, type QuaternionTuple } from './channels';

/** Fitted geometry, with the original speaker-review gate intact. */
export interface GestureBasisGroup {
  channels: BoneControl[];
  rank: number;
  mean: number[];
  basis: number[][];
  latent_examples: number[][];
  latent_ids: string[];
  training_source_events: string[];
  support_counts: number[];
  max_angle_rad: number;
  iterations_completed: number;
}
export interface GestureBasisModel {
  schema_version: 'gesture-basis.v1';
  persona_id: 'jensen-huang';
  review_status: 'geometry_preview';
  coordinate_basis: 'right_handed_y_up';
  rotation_space: 'local_delta';
  flattening: 'time_channel_xyz';
  id: string;
  dataset_sha256: string;
  sample_count: number;
  duration_ms: number;
  groups: Record<string, GestureBasisGroup>;
  limitations: string[];
}

const GROUP_CHANNELS: Record<string, string[]> = {
  left_arm: ['leftUpperArm', 'leftForearm', 'leftHand'],
  right_arm: ['rightUpperArm', 'rightForearm', 'rightHand'],
  left_fingers: FINGER_JOINTS.map((name) => `left${name}`).sort(),
  right_fingers: FINGER_JOINTS.map((name) => `right${name}`).sort(),
};
const finite = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n);
const vector = (v: unknown, length: number): v is number[] => Array.isArray(v) && v.length === length && v.every((n) => finite(n) && Math.abs(n) <= 1e4);
const record = (v: unknown): Record<string, unknown> => {
  if (!v || typeof v !== 'object' || Array.isArray(v)) throw new Error('Expected a gesture model object.');
  return v as Record<string, unknown>;
};

export function parseGestureBasis(value: unknown): GestureBasisModel {
  const model = record(value);
  if (model.schema_version !== 'gesture-basis.v1' || model.persona_id !== 'jensen-huang'
      || model.review_status !== 'geometry_preview' || model.coordinate_basis !== 'right_handed_y_up'
      || model.rotation_space !== 'local_delta' || model.flattening !== 'time_channel_xyz') {
    throw new Error('The learned gesture model has incompatible subject, review or coordinate metadata.');
  }
  if (typeof model.id !== 'string' || !model.id.length || model.id.length > 200
      || typeof model.dataset_sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(model.dataset_sha256)
      || !Number.isInteger(model.sample_count) || Number(model.sample_count) < 2 || Number(model.sample_count) > 300
      || !finite(model.duration_ms) || model.duration_ms < 500 || model.duration_ms > 30_000
      || !Array.isArray(model.limitations) || !model.limitations.every((v) => typeof v === 'string')) {
    throw new Error('The learned gesture model metadata is invalid.');
  }
  const groups = record(model.groups);
  if (Object.keys(groups).length < 1 || Object.keys(groups).length > 4) throw new Error('Invalid gesture groups.');
  for (const [name, raw] of Object.entries(groups)) {
    const group = record(raw), expected = GROUP_CHANNELS[name];
    if (!expected || !Array.isArray(group.channels) || JSON.stringify(group.channels) !== JSON.stringify(expected)) throw new Error('Unsupported learned gesture channels.');
    const dimensions = Number(model.sample_count) * expected.length * 3;
    const exampleCount = Array.isArray(group.latent_examples) ? group.latent_examples.length : 0;
    if (!Number.isInteger(group.rank) || Number(group.rank) < 1 || Number(group.rank) > 16
        || !vector(group.mean, dimensions) || !Array.isArray(group.basis) || group.basis.length !== group.rank
        || !group.basis.every((v) => vector(v, dimensions))
        || !Array.isArray(group.latent_examples) || group.latent_examples.length < 1 || group.latent_examples.length > 1000
        || !group.latent_examples.every((v) => vector(v, Number(group.rank)))
        || !Array.isArray(group.latent_ids) || group.latent_ids.length !== group.latent_examples.length
        || group.latent_ids.some((id) => typeof id !== 'string' || !id.length || id.length > 200)
        || new Set(group.latent_ids).size !== group.latent_ids.length
        || !Array.isArray(group.training_source_events) || !group.training_source_events.length
        || group.training_source_events.some((v) => typeof v !== 'string' || !v.length)
        || !vector(group.support_counts, dimensions) || group.support_counts.some((v) => !Number.isInteger(v) || v < 0 || v > exampleCount)
        || !finite(group.max_angle_rad) || group.max_angle_rad <= 0 || group.max_angle_rad > 1.22
        || !Number.isInteger(group.iterations_completed) || Number(group.iterations_completed) < 1) {
      throw new Error(`Invalid learned gesture group: ${name}.`);
    }
  }
  return value as GestureBasisModel;
}

function rotation(vector: number[], maxAngle: number, weight: number): QuaternionTuple {
  const raw = Math.hypot(...vector), angle = Math.min(raw, maxAngle) * weight;
  if (raw < 1e-8 || angle < 1e-8) return [0, 0, 0, 1];
  const scale = Math.sin(angle / 2) / raw;
  return [vector[0] * scale, vector[1] * scale, vector[2] * scale, Math.cos(angle / 2)];
}
const smoothstep = (t: number) => { const x = Math.max(0, Math.min(1, t)); return x * x * (3 - 2 * x); };

/** No random clock: seeks, pause and restart reproduce the same pose. */
export function sampleLearnedGesture(model: GestureBasisModel, elapsedMs: number): BoneRotations | null {
  if (!Number.isFinite(elapsedMs) || elapsedMs < 0) return null;
  const groups = Object.values(model.groups);
  // A shared source-window ID keeps both arms and fingers coordinated.
  const anchors = [model.groups.left_arm, model.groups.right_arm].filter(Boolean);
  const anchorGroups = anchors.length ? anchors : groups;
  const common = anchorGroups[0].latent_ids.filter((id) => anchorGroups.every((group) => group.latent_ids.includes(id)));
  if (!common.length) return null;
  const period = model.duration_ms + 500, cycle = Math.floor(elapsedMs / period), local = elapsedMs % period;
  if (local >= model.duration_ms) return null;
  const id = common[cycle % common.length];
  const frame = Math.min(model.sample_count - 1, local / model.duration_ms * model.sample_count);
  const a = Math.floor(frame), b = Math.min(model.sample_count - 1, a + 1), mix = frame - a;
  const weight = smoothstep(local / 220) * smoothstep((model.duration_ms - local) / 350);
  const bones: BoneRotations = {};
  for (const group of groups) {
    const example = group.latent_ids.indexOf(id);
    if (example < 0) continue;
    const latent = group.latent_examples[example], width = group.channels.length * 3;
    for (let channel = 0; channel < group.channels.length; channel++) {
      const start = channel * 3;
      if ([0, 1, 2].some((axis) => group.support_counts[a * width + start + axis] < 2 || group.support_counts[b * width + start + axis] < 2)) continue;
      const coordinates = [0, 1, 2].map((axis) => {
        const i = a * width + start + axis, j = b * width + start + axis;
        let from = group.mean[i], to = group.mean[j];
        for (let k = 0; k < group.rank; k++) { from += latent[k] * group.basis[k][i]; to += latent[k] * group.basis[k][j]; }
        return from + (to - from) * mix;
      });
      bones[group.channels[channel]] = rotation(coordinates, group.max_angle_rad, weight);
    }
  }
  return Object.keys(bones).length ? bones : null;
}
