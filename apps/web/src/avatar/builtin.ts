import { Object3D, Quaternion, Vector3 } from 'three';
import { FINGER_JOINTS } from './channels';
import type { BoneControl, QuaternionTuple, RigMapping } from './types';

export const RESEARCH_HUMAN_AVATAR = {
  id: 'research-human-v2',
  label: 'Research human',
  modelUrl: '/models/research-human-v2.glb',
} as const;

export const JENSEN_SOURCE_FITTED_AVATAR = {
  id: 'jensen-source-fitted-v3',
  label: 'Jensen source-fitted candidate',
  modelUrl: '/models/jensen-source-fitted-v3.glb',
} as const;

export type BuiltinAvatar = 'human' | 'portrait' | 'jensen';

export type AvatarFraming = 'portrait' | 'upper_body' | 'full_body';

const JOINTS = {
  hips: 'root', spine: 'spine03', chest: 'spine01', neck: 'neck01', head: 'head', jaw: 'jaw',
  leftEye: 'eye.L', rightEye: 'eye.R', leftShoulder: 'clavicle.L', rightShoulder: 'clavicle.R',
  leftUpperArm: 'upperarm01.L', rightUpperArm: 'upperarm01.R',
  leftForearm: 'lowerarm01.L', rightForearm: 'lowerarm01.R', leftHand: 'wrist.L', rightHand: 'wrist.R',
  ...Object.fromEntries((['left', 'right'] as const).flatMap((side) => FINGER_JOINTS.map((joint, index) =>
    [`${side}${joint}`, `finger${Math.floor(index / 3) + 1}-${index % 3 + 1}.${side === 'left' ? 'L' : 'R'}`]))),
} as Record<BoneControl, string>;
const canonicalName = (name: string): string => name.toLowerCase().replace(/[^a-z0-9]/g, '');
const tuple = (rotation: Quaternion): QuaternionTuple => rotation.normalize().toArray() as QuaternionTuple;

/** Explicit calibration for bundled MPFB-derived assets, never guessed for an upload. */
export function createResearchHumanMapping(model: Object3D): RigMapping {
  const nodes = new Map<string, Object3D>();
  model.traverse((node) => { nodes.set(canonicalName(node.name), node); });
  const joints = Object.fromEntries(Object.entries(JOINTS).map(([control, name]) => {
    const node = nodes.get(canonicalName(name));
    if (!node) throw new Error(`The research human asset is missing its ${name} joint.`);
    return [control, node];
  })) as Record<BoneControl, Object3D>;
  const originals = new Map(Object.values(joints).map((node) => [node, node.quaternion.clone()]));
  // Its exported face points toward -Z. This explicit correction also puts
  // anatomical left at +X, matching the runtime's subject-relative convention.
  const yaw = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), Math.PI);
  const worldRotation = (node: Object3D): Quaternion => yaw.clone().multiply(node.getWorldQuaternion(new Quaternion())).normalize();
  const position = (node: Object3D): Vector3 => node.getWorldPosition(new Vector3()).applyQuaternion(yaw);
  const neutrals: Partial<Record<BoneControl, QuaternionTuple>> = {};
  const relax = (control: BoneControl, child: BoneControl, direction: Vector3): void => {
    const node = joints[control]; const current = position(joints[child]).sub(position(node));
    if (current.lengthSq() < 1e-10) throw new Error(`The research human ${control} segment has no usable length.`);
    const change = new Quaternion().setFromUnitVectors(current.normalize(), direction.normalize());
    const orientation = worldRotation(node);
    const local = orientation.clone().invert().multiply(change).multiply(orientation).normalize();
    neutrals[control] = tuple(local);
    node.quaternion.multiply(local).normalize(); model.updateMatrixWorld(true);
  };
  try {
    model.updateMatrixWorld(true);
    // Keep a slight clearance from the jacket and a small natural elbow bend.
    // No geometry or skin bind matrices are rewritten by this calibration.
    for (const [side, sign] of [['left', 1], ['right', -1]] as const) {
      relax(`${side}UpperArm`, `${side}Forearm`, new Vector3(sign * 0.16, -1, 0.025));
      relax(`${side}Forearm`, `${side}Hand`, new Vector3(sign * 0.10, -1, 0.14));
    }
    const bones: NonNullable<RigMapping['bones']> = {};
    for (const [control, node] of Object.entries(joints) as [BoneControl, Object3D][]) {
      bones[control] = { name: node.name, basis: tuple(worldRotation(node).invert()),
        maxAngle: control.endsWith('Eye') ? 0.4 : control === 'jaw' ? 0.4 : control === 'neck' ? 0.65 : 1.22,
        ...(neutrals[control] ? { neutral: neutrals[control] } : {}) };
    }
    return { yaw: Math.PI, bones };
  } finally {
    // Calibration inspection must not change the model's authored rest pose.
    for (const [node, rotation] of originals) node.quaternion.copy(rotation);
    model.updateMatrixWorld(true);
  }
}

export function mergeRigMappings(base: RigMapping, override: RigMapping): RigMapping {
  return { ...base, ...override, morphs: { ...base.morphs, ...override.morphs },
    bones: { ...base.bones, ...override.bones }, visemes: { ...base.visemes, ...override.visemes } };
}

/** Both rig kinds are normalized near 2.4 stage units; the procedural rig is a bust. */
export function avatarView(kind: 'procedural' | 'glb', framing: AvatarFraming, aspect: number) {
  if (kind === 'procedural') return { position: [0.07, 1.58, 4.45] as const, target: [0, 1.40, 0] as const, fov: aspect < 0.8 ? 41 : 34 };
  if (framing === 'upper_body') {
    // Fit a conservative complete arm/hand envelope using the horizontal FOV
    // as well as the vertical FOV. Narrow displays move back instead of clipping.
    const fov = 36; const halfVertical = Math.tan(fov * Math.PI / 360);
    const distance = Math.max(1.05 / halfVertical, 1.30 / (halfVertical * Math.max(0.2, aspect)));
    return { position: [0, 1.65, distance] as const, target: [0, 1.65, 0] as const, fov };
  }
  if (framing === 'full_body') return { position: [0.04, 1.32, 4.55] as const, target: [0, 1.20, 0] as const, fov: aspect < 0.65 ? 40 : 34 };
  return { position: [0.035, 2.06, 2.20] as const, target: [0, 1.99, 0] as const, fov: aspect < 0.7 ? 39 : 34 };
}
