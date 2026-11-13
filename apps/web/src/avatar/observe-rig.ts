import { Mesh, Quaternion } from 'three';
import type { AvatarRig } from './procedural';

/** Read the mapped GLB outputs after apply(), without changing its animation. */
export function createRigOutputReader(rig: AvatarRig) {
  const face = ['jawOpen', 'eyeBlinkLeft', 'eyeBlinkRight', 'mouthSmileLeft'] as const;
  const bones = ['leftUpperArm', 'rightUpperArm', 'leftHand', 'leftIndexProximal'] as const;
  const targets = (rig.diagnostics.resolvedTargets ?? []).flatMap((binding) => {
    if (!face.some((name) => name === binding.control)) return [];
    const mesh = rig.root.getObjectByName(binding.mesh);
    const index = mesh instanceof Mesh ? mesh.morphTargetDictionary?.[binding.target] : undefined;
    return mesh instanceof Mesh && index !== undefined ? [{ control: binding.control, mesh, index }] : [];
  });
  const joints = (rig.diagnostics.resolvedBones ?? []).flatMap((binding) => {
    if (!bones.some((name) => name === binding.control)) return [];
    const node = rig.root.getObjectByName(binding.node);
    const rest = new Quaternion(...binding.authoredRest).multiply(new Quaternion(...binding.neutral)).normalize();
    return node ? [{ control: binding.control, node, rest }] : [];
  });
  return () => {
    const values = { jawOpen: 0, eyeBlinkLeft: 0, eyeBlinkRight: 0, mouthSmileLeft: 0,
      leftUpperArm: 0, rightUpperArm: 0, leftHand: 0, leftIndexProximal: 0 };
    for (const target of targets) {
      const control = target.control as typeof face[number];
      values[control] = Math.max(values[control], target.mesh.morphTargetInfluences?.[target.index] ?? 0);
    }
    for (const joint of joints) values[joint.control as typeof bones[number]] = joint.node.quaternion.angleTo(joint.rest);
    return values;
  };
}
