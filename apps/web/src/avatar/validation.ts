import { BONE_SET, FACE_SET, VISEME_SET } from './channels';
import type { AvatarClip, BoneBinding, BoneRotations, FaceWeights, MorphBinding, MotionSelection, QuaternionTuple, RigMapping, SpeechTiming, VideoCrop } from './types';

type RecordValue = Record<string, unknown>;
const object = (value: unknown, name: string): RecordValue => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${name} must be an object.`);
  return value as RecordValue;
};
const text = (value: unknown, name: string, max = 160): string => {
  if (typeof value !== 'string' || !value.trim() || value.length > max || /[\u0000-\u001f]/.test(value)) throw new Error(`${name} must be a nonempty bounded string.`);
  return value;
};
const number = (value: unknown, name: string, min: number, max: number): number => {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) throw new Error(`${name} must be between ${min} and ${max}.`);
  return value;
};
const hash = (value: unknown, name: string): string => {
  if (typeof value !== 'string' || !/^[a-f0-9]{64}$/.test(value)) throw new Error(`${name} must be a SHA-256 digest.`);
  return value;
};
const boolean = (value: unknown, name: string): boolean => {
  if (typeof value !== 'boolean') throw new Error(`${name} must be a boolean.`);
  return value;
};
const choice = <T extends string>(value: unknown, values: readonly T[], name: string): T => {
  if (!values.includes(value as T)) throw new Error(`${name} is unsupported.`);
  return value as T;
};

export function parseQuaternion(value: unknown, name = 'rotation'): QuaternionTuple {
  if (!Array.isArray(value) || value.length !== 4) throw new Error(`${name} must contain XYZW components.`);
  const values = value.map((item) => number(item, name, -1, 1));
  const length = Math.hypot(...values);
  if (Math.abs(length - 1) > 0.05) throw new Error(`${name} must be a unit quaternion.`);
  return values.map((item) => item / length) as QuaternionTuple;
}

function faceWeights(value: unknown, name: string): FaceWeights {
  const result: FaceWeights = {};
  for (const [key, item] of Object.entries(object(value, name))) {
    if (!FACE_SET.has(key)) throw new Error(`${name} contains an unsupported face channel: ${key}.`);
    result[key as keyof FaceWeights] = number(item, `${name}.${key}`, 0, 1);
  }
  return result;
}

function boneRotations(value: unknown, name: string): BoneRotations {
  const result: BoneRotations = {};
  for (const [key, item] of Object.entries(object(value, name))) {
    if (!BONE_SET.has(key)) throw new Error(`${name} contains an unsupported bone channel: ${key}.`);
    result[key as keyof BoneRotations] = parseQuaternion(item, `${name}.${key}`);
  }
  return result;
}

function videoCrop(value: unknown): VideoCrop {
  const data = object(value, 'video_crop'); const result: Record<string, number> = {};
  for (const key of ['x', 'y', 'width', 'height', 'input_width', 'input_height']) {
    const coordinate = number(data[key], `video_crop.${key}`, key === 'x' || key === 'y' ? 0 : 1, 32768);
    if (!Number.isInteger(coordinate)) throw new Error('Video crop coordinates must be integers.');
    result[key] = coordinate;
  }
  if (result.x + result.width > result.input_width || result.y + result.height > result.input_height
    || data.coordinate_basis !== 'input_pixels_before_scale' || data.mirrored !== false) throw new Error('Invalid video crop transform.');
  return { x: result.x, y: result.y, width: result.width, height: result.height, input_width: result.input_width, input_height: result.input_height,
    coordinate_basis: 'input_pixels_before_scale', mirrored: false };
}

export function parseSpeechTiming(value: unknown): SpeechTiming {
  const data = object(value, 'Speech timing');
  if (data.schema_version !== 'speech-timing.v1') throw new Error('Unsupported speech timing schema.');
  const duration = number(data.audio_duration_ms, 'audio_duration_ms', 1, 180000);
  const labelSet = choice(data.label_set, ['rhubarb', 'arpabet', 'ipa', 'viseme_15'] as const, 'label_set');
  const provenance = object(data.provenance, 'provenance');
  if (!Array.isArray(data.segments) || !data.segments.length || data.segments.length > 10000) throw new Error('Speech timing requires 1–10000 segments.');
  let previousEnd = 0;
  const segments = data.segments.map((item, index) => {
    const segment = object(item, `segment ${index}`);
    const start = number(segment.start_ms, 'start_ms', previousEnd, duration);
    const end = number(segment.end_ms, 'end_ms', start, duration);
    if (end <= start) throw new Error('Speech segments must have positive duration and cannot overlap.');
    previousEnd = end;
    const label = text(segment.label, 'label', 32);
    if (labelSet === 'rhubarb' && !/^[A-HX]$/.test(label)) throw new Error('Rhubarb labels must be A–H or X.');
    if (labelSet === 'viseme_15' && !VISEME_SET.has(label)) throw new Error('The native viseme label is unsupported.');
    return { start_ms: start, end_ms: end, label,
      ...(segment.confidence === undefined ? {} : { confidence: number(segment.confidence, 'confidence', 0, 1) }) };
  });
  return {
    schema_version: 'speech-timing.v1', audio_sha256: hash(data.audio_sha256, 'audio_sha256'), audio_duration_ms: duration,
    timing_basis: choice(data.timing_basis, ['audio_recognition', 'forced_alignment', 'tts_model', 'manual'], 'timing_basis'),
    label_set: labelSet, segments,
    provenance: { engine: text(provenance.engine, 'engine'),
      ...(provenance.version === undefined ? {} : { version: text(provenance.version, 'version') }),
      ...(provenance.model === undefined ? {} : { model: text(provenance.model, 'model') }) },
  };
}

export function parseMotionSelection(value: unknown): MotionSelection {
  const data = object(value, 'Motion selection');
  if (data.schema_version !== 'motion-selection.v1') throw new Error('Unsupported motion selection schema.');
  const identifier = (value: unknown, name: string) => {
    const result = text(value, name, 120);
    if (!/^[a-z0-9][a-z0-9_.-]{0,119}$/.test(result)) throw new Error(`Invalid ${name}.`);
    return result;
  };
  const primitive = identifier(data.primitive_id, 'primitive_id');
  const duration = number(data.duration_ms, 'duration_ms', 1, 120000);
  const start = number(data.source_start_seconds, 'source_start_seconds', 0, 86400);
  const end = number(data.source_end_seconds, 'source_end_seconds', start, 86400);
  if (Math.abs((end - start) * 1000 - duration) > 0.01) throw new Error('Motion source interval does not match its duration.');
  if (data.clip_url !== `/api/motion/clip/${primitive}`) throw new Error('Motion selection has an unexpected clip URL.');
  return {
    schema_version: 'motion-selection.v1', primitive_id: primitive,
    library_id: identifier(data.library_id, 'library_id'), library_version: text(data.library_version, 'library_version', 60),
    family: choice(data.family, ['face', 'upper_body'], 'family'),
    source_id: identifier(data.source_id, 'source_id'), source_event_id: identifier(data.source_event_id, 'source_event_id'),
    source_timeline: text(data.source_timeline, 'source_timeline'), source_start_seconds: start, source_end_seconds: end,
    duration_ms: duration, review_status: choice(data.review_status, ['geometry_preview', 'speaker_reviewed'], 'review_status'),
    selection_basis: choice(data.selection_basis, ['source_context_retrieval', 'nearby_source_context_retrieval'], 'selection_basis'),
    context_summary: text(data.context_summary, 'context_summary', 1000),
    context_alignment: choice(data.context_alignment, ['reviewed_word_anchors', 'publisher_metadata_mapping', 'asr_word_matches', 'unverified_nearby_context'], 'context_alignment'),
    score: number(data.score, 'score', 0, 1e9),
    geometry_cluster_id: data.geometry_cluster_id == null ? null : identifier(data.geometry_cluster_id, 'geometry_cluster_id'),
    clip_url: data.clip_url, clip_sha256: hash(data.clip_sha256, 'clip_sha256'), parent_clip_sha256: hash(data.parent_clip_sha256, 'parent_clip_sha256'),
  };
}

/** Unreviewed geometry requires the separate source-geometry preview opt-in. */
export function parseAvatarClip(value: unknown, allowGeometryPreview = false): AvatarClip {
  const data = object(value, 'Animation clip');
  if (data.schema_version !== 'avatar-clip.v1') throw new Error('Unsupported animation clip schema.');
  const duration = number(data.duration_ms, 'duration_ms', 1, 120000);
  const source = object(data.source, 'source'); const review = object(data.review, 'review');
  const calibration = object(data.calibration, 'calibration');
  const status = choice(review.status, ['geometry_preview', 'speaker_reviewed'], 'review.status');
  const continuity = boolean(review.track_continuity_reviewed, 'track_continuity_reviewed');
  const clock = boolean(review.source_clock_reviewed, 'source_clock_reviewed');
  const speaker = boolean(review.speaker_association_reviewed, 'speaker_association_reviewed');
  if (status === 'geometry_preview' && !allowGeometryPreview) throw new Error('This unreviewed geometry clip is restricted to an explicit calibration preview.');
  if (status === 'speaker_reviewed' && (!continuity || !clock || !speaker)) throw new Error('Speaker replay requires reviewed track continuity, source clock, and speaker association.');
  if (calibration.coordinate_basis !== 'right_handed_y_up' || calibration.bone_rotation_space !== 'local_delta') throw new Error('Animation rotations must use a right-handed Y-up local-delta basis.');
  let parentReference: AvatarClip['calibration']['parent_reference'];
  if (calibration.parent_reference !== undefined) {
    const parent = object(calibration.parent_reference, 'parent_reference');
    parentReference = {
      clip_sha256: hash(parent.clip_sha256, 'parent_reference.clip_sha256'),
      neutral_frame_ms: number(parent.neutral_frame_ms, 'parent_reference.neutral_frame_ms', 0, 120000),
      window_start_ms: number(parent.window_start_ms, 'parent_reference.window_start_ms', 0, 120000),
      rotation_operation: choice(parent.rotation_operation, ['retained_parent_local_delta'], 'parent_reference.rotation_operation'),
    };
  }
  const neutralFrame = number(calibration.neutral_frame_ms, 'neutral_frame_ms', parentReference ? -120000 : 0, parentReference ? 120000 : duration);
  if (parentReference && Math.abs(neutralFrame - (parentReference.neutral_frame_ms - parentReference.window_start_ms)) > 0.01) {
    throw new Error('Neutral reference does not match the retained parent calibration.');
  }
  if (!Array.isArray(data.frames) || !data.frames.length || data.frames.length > 3601) throw new Error('Animation clips require 1–3601 frames.');
  let previousTime = -1;
  const frames = data.frames.map((value, index) => {
    const frame = object(value, `frame ${index}`);
    const time = number(frame.time_ms, 'frame.time_ms', 0, duration);
    if (time <= previousTime) throw new Error('Animation frame timestamps must strictly increase.');
    previousTime = time;
    return { time_ms: time, ...(frame.face === undefined ? {} : { face: faceWeights(frame.face, 'face') }),
      ...(frame.bones === undefined ? {} : { bones: boneRotations(frame.bones, 'bones') }) };
  });
  if (!Array.isArray(data.limitations) || data.limitations.length > 30) throw new Error('Clip limitations must be a bounded list.');
  const measurement = data.measurement === undefined ? undefined : object(data.measurement, 'measurement');
  return {
    schema_version: 'avatar-clip.v1', ...(data.id === undefined ? {} : { id: text(data.id, 'id', 120) }),
    duration_ms: duration, max_gap_ms: number(data.max_gap_ms, 'max_gap_ms', 1, 1000),
    source: { source_id: text(source.source_id, 'source_id'), source_timeline: text(source.source_timeline, 'source_timeline'),
      source_start_seconds: number(source.source_start_seconds, 'source_start_seconds', 0, 1e7),
      video_sha256: hash(source.video_sha256, 'video_sha256'), observations_sha256: hash(source.observations_sha256, 'observations_sha256'),
      ...(source.observation_offset_ms === undefined ? {} : { observation_offset_ms: number(source.observation_offset_ms, 'observation_offset_ms', 0, 120000) }),
      ...(source.review_sha256 === undefined ? {} : { review_sha256: hash(source.review_sha256, 'review_sha256') }),
      ...(source.input_sha256 === undefined ? {} : { input_sha256: hash(source.input_sha256, 'input_sha256') }),
      ...(source.preparation_manifest_sha256 === undefined ? {} : { preparation_manifest_sha256: hash(source.preparation_manifest_sha256, 'preparation_manifest_sha256') }),
      ...(source.video_crop === undefined ? {} : { video_crop: videoCrop(source.video_crop) }) },
    review: { status, reviewer: text(review.reviewer, 'reviewer'), track_id: text(review.track_id, 'track_id'),
      track_continuity_reviewed: continuity, source_clock_reviewed: clock, speaker_association_reviewed: speaker,
      attribution_basis: text(review.attribution_basis, 'attribution_basis', 1000) },
    calibration: { neutral_frame_ms: neutralFrame,
      ...(parentReference ? { parent_reference: parentReference } : {}),
      face_neutral: faceWeights(calibration.face_neutral, 'face_neutral'),
      ...(calibration.bone_neutral === undefined ? {} : { bone_neutral: boneRotations(calibration.bone_neutral, 'bone_neutral') }),
      coordinate_basis: 'right_handed_y_up', bone_rotation_space: 'local_delta' },
    ...(measurement ? { measurement: { engine: text(measurement.engine, 'measurement.engine'), version: text(measurement.version, 'measurement.version'),
      face_model_sha256: hash(measurement.face_model_sha256, 'measurement.face_model_sha256'),
      ...(measurement.pose_model_sha256 === undefined ? {} : { pose_model_sha256: hash(measurement.pose_model_sha256, 'measurement.pose_model_sha256') }),
      ...(measurement.hand_model_sha256 === undefined ? {} : { hand_model_sha256: hash(measurement.hand_model_sha256, 'measurement.hand_model_sha256') }) } } : {}),
    frames, limitations: data.limitations.map((value) => text(value, 'limitation', 1000)),
  };
}

function morphBinding(value: unknown): string | MorphBinding {
  if (typeof value === 'string') return text(value, 'target');
  const data = object(value, 'Morph binding');
  const binding: MorphBinding = { target: text(data.target, 'target') };
  if (data.mesh !== undefined) binding.mesh = text(data.mesh, 'mesh');
  if (data.scale !== undefined) binding.scale = number(data.scale, 'scale', -2, 2);
  if (data.inputNeutral !== undefined) binding.inputNeutral = number(data.inputNeutral, 'inputNeutral', 0, 0.99);
  if (data.inputMax !== undefined) binding.inputMax = number(data.inputMax, 'inputMax', 0.01, 1);
  if ((binding.inputMax ?? 1) <= (binding.inputNeutral ?? 0)) throw new Error('Morph inputMax must exceed inputNeutral.');
  if (data.outputNeutral !== undefined) binding.outputNeutral = number(data.outputNeutral, 'outputNeutral', 0, 1);
  return binding;
}

export function parseRigMapping(value: unknown): RigMapping {
  const data = object(value, 'Rig mapping'); const result: RigMapping = {};
  for (const [group, allowed] of [['morphs', FACE_SET], ['visemes', VISEME_SET]] as const) {
    if (data[group] === undefined) continue;
    const bindings: Record<string, string | MorphBinding> = {};
    for (const [key, binding] of Object.entries(object(data[group], group))) {
      if (!allowed.has(key)) throw new Error(`Unsupported ${group} channel: ${key}.`);
      bindings[key] = morphBinding(binding);
    }
    result[group] = bindings;
  }
  if (data.bones !== undefined) {
    const bindings: NonNullable<RigMapping['bones']> = {};
    for (const [key, value] of Object.entries(object(data.bones, 'bones'))) {
      if (!BONE_SET.has(key)) throw new Error(`Unsupported bone channel: ${key}.`);
      if (typeof value === 'string') { bindings[key as keyof typeof bindings] = text(value, 'bone name'); continue; }
      const data = object(value, 'Bone binding'); const binding: BoneBinding = { name: text(data.name, 'bone name') };
      if (data.axis !== undefined) binding.axis = choice(data.axis, ['x', 'y', 'z'] as const, 'axis');
      if (data.scale !== undefined) binding.scale = number(data.scale, 'scale', -2, 2);
      if (data.neutral !== undefined) binding.neutral = parseQuaternion(data.neutral, 'neutral');
      if (data.basis !== undefined) binding.basis = parseQuaternion(data.basis, 'basis');
      if (data.maxAngle !== undefined) binding.maxAngle = number(data.maxAngle, 'maxAngle', 0, Math.PI);
      bindings[key as keyof typeof bindings] = binding;
    }
    result.bones = bindings;
  }
  if (data.yaw !== undefined) result.yaw = number(data.yaw, 'yaw', -Math.PI * 2, Math.PI * 2);
  return result;
}
