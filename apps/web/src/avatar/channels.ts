/** ARKit location names. Left/right are the subject's, never screen directions. */
export const FACE_CHANNELS = [
  'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
  'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
  'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight',
  'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
  'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight',
  'jawForward', 'jawLeft', 'jawOpen', 'jawRight',
  'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight',
  'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight',
  'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper',
  'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight',
  'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight',
  'noseSneerLeft', 'noseSneerRight', 'tongueOut',
] as const;

export type FaceChannel = typeof FACE_CHANNELS[number];
export const BONE_CHANNELS = [
  'hips', 'spine', 'chest', 'neck', 'head', 'jaw', 'leftEye', 'rightEye',
  'leftShoulder', 'rightShoulder', 'leftUpperArm', 'rightUpperArm',
  'leftForearm', 'rightForearm', 'leftHand', 'rightHand',
] as const;
export type BoneControl = typeof BONE_CHANNELS[number];

/** Common 15-target viseme vocabulary; a loaded asset must actually supply it. */
export const VISEMES = ['sil', 'PP', 'FF', 'TH', 'DD', 'kk', 'CH', 'SS', 'nn', 'RR', 'aa', 'E', 'I', 'O', 'U'] as const;
export type Viseme = typeof VISEMES[number];
export type FaceWeights = Partial<Record<FaceChannel, number>>;
export type VisemeWeights = Partial<Record<Viseme, number>>;
export type QuaternionTuple = [number, number, number, number];
export type BoneRotations = Partial<Record<BoneControl, QuaternionTuple>>;

export const FACE_SET = new Set<string>(FACE_CHANNELS);
export const BONE_SET = new Set<string>(BONE_CHANNELS);
export const VISEME_SET = new Set<string>(VISEMES);
