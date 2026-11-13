import { Object3D, Vector3 } from 'three';
import type { RigDiagnostics } from './types';

export interface FramingPoint { position: Vector3; radius: number }

/** Fit the visible upper-body joints in the current orbit direction, with padding.
 * This bounds geometry, not source landmarks or camera-capture participants.
 */
export function fitUpperBody(points: readonly FramingPoint[], aspect: number, direction = new Vector3(0, 0, 1)) {
  const target = new Vector3(0, 1.65, 0); const outward = direction.clone().normalize();
  const right = new Vector3(0, 1, 0).cross(outward).normalize();
  const up = outward.clone().cross(right).normalize();
  const tangent = Math.tan(36 * Math.PI / 360);
  let distance = 2.3;
  const torso = [new Vector3(-.40, 1.05, 0), new Vector3(.40, 1.05, 0), new Vector3(0, 2.4, 0)]
    .map((position) => ({ position, radius: .06 }));
  for (const point of [...torso, ...points]) {
    const offset = point.position.clone().sub(target); const depth = offset.dot(outward) + point.radius;
    distance = Math.max(distance,
      depth + (Math.abs(offset.dot(right)) + point.radius) / (tangent * Math.max(.2, aspect) * .90),
      depth + (Math.abs(offset.dot(up)) + point.radius) / (tangent * .90));
  }
  return { position: target.clone().addScaledVector(outward, distance).toArray() as [number, number, number],
    target: target.toArray() as [number, number, number], fov: 36 };
}

/** Resolve mapped bones once. Missing hands retain the conservative fallback. */
export function createUpperBodyFramer(root: Object3D, diagnostics: RigDiagnostics) {
  const bindings = diagnostics.resolvedBones ?? [];
  if (!['head', 'leftHand', 'rightHand'].every((control) => bindings.some((binding) => binding.control === control))) return null;
  const tracked = bindings.filter(({ control }) => control === 'head' || control === 'chest'
    || (/^(left|right)/.test(control) && !control.endsWith('Eye'))).map(({ control, node }) => ({
    object: root.getObjectByName(node), position: new Vector3(),
    radius: control === 'head' ? .27 : control === 'chest' ? .22 : /Shoulder|UpperArm|Forearm/.test(control) ? .12 : .08,
  }));
  if (tracked.some((point) => !point.object)) return null;
  return (aspect: number, direction?: Vector3) => {
    root.updateMatrixWorld(true);
    for (const point of tracked) point.object!.getWorldPosition(point.position);
    return fitUpperBody(tracked, aspect, direction);
  };
}
