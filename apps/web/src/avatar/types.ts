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

export type MorphControl =
  | 'eyeBlinkLeft' | 'eyeBlinkRight' | 'jawOpen'
  | 'mouthSmileLeft' | 'mouthSmileRight' | 'mouthFunnel'
  | 'browInnerUp' | 'browDownLeft' | 'browDownRight';

export type BoneControl = 'head' | 'neck' | 'jaw' | 'leftUpperArm' | 'rightUpperArm'
  | 'leftForearm' | 'rightForearm' | 'leftHand' | 'rightHand';

export interface MorphBinding {
  target: string;
  /** Optional mesh name for models containing duplicate morph names. */
  mesh?: string;
  scale?: number;
}

export interface BoneBinding {
  name: string;
  /** Retargeting axes depend on how the original rig was authored. */
  axis?: 'x' | 'y' | 'z';
  scale?: number;
}

export interface RigMapping {
  morphs?: Partial<Record<MorphControl, string | MorphBinding>>;
  bones?: Partial<Record<BoneControl, string | BoneBinding>>;
  /** Rotation around Y, in radians. Models are never guessed to face the camera. */
  yaw?: number;
}

export interface RigDiagnostics {
  kind: 'procedural' | 'glb';
  morphs: string[];
  bones: string[];
  missing: string[];
}
