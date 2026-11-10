import type { BoneControl, BoneRotations, FaceChannel, FaceWeights, QuaternionTuple, Viseme, VisemeWeights } from './channels';
export type { BoneControl, BoneRotations, FaceChannel, FaceWeights, QuaternionTuple, Viseme, VisemeWeights } from './channels';

export type CueChannel = 'face' | 'gaze' | 'head' | 'gesture';

export interface BehaviorCue {
  start_ms: number;
  duration_ms: number;
  channel: CueChannel;
  name: string;
  intensity: number;
  provenance: 'designed' | 'observed';
  evidence_ids?: string[];
}

export interface BehaviorPlan {
  cues: BehaviorCue[];
  duration_ms: number;
  timing_basis?: string;
  speech?: { rate: number; pitch: number; pause_ms: number };
}

/** Controls are normalized, except gaze/head angles which are in radians. */
export interface RigPose {
  blink: number;
  jawOpen: number;
  mouthSmile: number;
  mouthRound: number;
  browRaise: number;
  browFurrow: number;
  gazeX: number;
  gazeY: number;
  headPitch: number;
  headYaw: number;
  headRoll: number;
  openPalm: number;
  emphasis: number;
  shrug: number;
  wave: number;
  breath: number;
  /** Per-channel measurements override the corresponding designed controls. */
  face?: FaceWeights;
  /** Source-local delta rotations in XYZW order, relative to source neutral. */
  bones?: BoneRotations;
  visemes?: VisemeWeights;
}

export interface TimelineInput {
  /** Elapsed playback time supplied by the actual audio transport. */
  elapsedMs: number;
  /** Separate wall-clock used only for nonverbal idle motion. */
  idleMs: number;
  speaking: boolean;
  amplitude?: number;
  plan?: BehaviorPlan;
  reducedMotion?: boolean;
}

export type MorphControl = FaceChannel;

export interface MorphBinding {
  target: string;
  /** Optional mesh name for models containing duplicate morph names. */
  mesh?: string;
  scale?: number;
  /** Subtract measured neutral, then normalize by inputMax - inputNeutral. */
  inputNeutral?: number;
  inputMax?: number;
  /** Defaults to the authored asset's initial morph weight. */
  outputNeutral?: number;
}

export interface BoneBinding {
  name: string;
  /** Retargeting axes depend on how the original rig was authored. */
  axis?: 'x' | 'y' | 'z';
  scale?: number;
  /** Authored local rotation followed by this optional neutral offset. */
  neutral?: QuaternionTuple;
  /** Source-to-target local basis; applied as B * delta * inverse(B). */
  basis?: QuaternionTuple;
  maxAngle?: number;
}

export interface RigMapping {
  morphs?: Partial<Record<MorphControl, string | MorphBinding>>;
  bones?: Partial<Record<BoneControl, string | BoneBinding>>;
  visemes?: Partial<Record<Viseme, string | MorphBinding>>;
  /** Rotation around Y, in radians. Models are never guessed to face the camera. */
  yaw?: number;
}

export interface RigDiagnostics {
  kind: 'procedural' | 'glb';
  morphs: string[];
  bones: string[];
  missing: string[];
  visemes?: string[];
  unsupportedFace?: FaceChannel[];
  unsupportedBones?: BoneControl[];
  availableMorphs?: { mesh: string; target: string; initial: number }[];
  availableNodes?: string[];
  warnings?: string[];
}

export interface SpeechTiming {
  schema_version: 'speech-timing.v1';
  audio_sha256: string;
  audio_duration_ms: number;
  timing_basis: 'audio_recognition' | 'forced_alignment' | 'tts_model' | 'manual';
  label_set: 'rhubarb' | 'arpabet' | 'ipa' | 'viseme_15';
  segments: { start_ms: number; end_ms: number; label: string; confidence?: number }[];
  provenance: { engine: string; version?: string; model?: string };
}

export interface AvatarClip {
  schema_version: 'avatar-clip.v1';
  duration_ms: number;
  max_gap_ms: number;
  source: {
    source_id: string;
    source_timeline: string;
    source_start_seconds: number;
    video_sha256: string;
    observations_sha256: string;
    observation_offset_ms?: number;
    review_sha256?: string;
    input_sha256?: string;
    preparation_manifest_sha256?: string;
    video_crop?: VideoCrop;
  };
  review: {
    status: 'geometry_preview' | 'speaker_reviewed';
    reviewer: string;
    track_id: string;
    track_continuity_reviewed: boolean;
    source_clock_reviewed: boolean;
    speaker_association_reviewed: boolean;
    /** Attribution is supplied by a reviewer, never inferred from geometry. */
    attribution_basis: string;
  };
  calibration: {
    neutral_frame_ms: number;
    face_neutral: FaceWeights;
    bone_neutral?: BoneRotations;
    coordinate_basis: 'right_handed_y_up';
    bone_rotation_space: 'local_delta';
  };
  measurement?: { engine: string; version: string; face_model_sha256: string; pose_model_sha256?: string };
  frames: { time_ms: number; face?: FaceWeights; bones?: BoneRotations }[];
  limitations: string[];
}

export interface VideoCrop {
  x: number; y: number; width: number; height: number; input_width: number; input_height: number;
  coordinate_basis: 'input_pixels_before_scale'; mirrored: false;
}

export interface AnimationTelemetry {
  faceSource: 'designed' | 'observed' | 'geometry_preview' | 'calibration';
  boneSource: 'designed' | 'observed' | 'geometry_preview' | 'calibration';
  speechSource: SpeechTiming['timing_basis'] | 'amplitude' | 'estimated' | 'silent';
  unsupportedFace: string[];
  unsupportedVisemes: string[];
  warning?: string;
}
