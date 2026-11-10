import { Box3, BoxGeometry, Group, Mesh, MeshStandardMaterial, Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import { createProceduralRig } from './procedural';
import { createRetargetedRig, validateModelUrl } from './retarget';
import { neutralPose } from './timeline';

function model() {
  const root = new Group();
  const head = new Group(); head.name = 'Head'; head.rotation.y = 0.1; root.add(head);
  const face = new Mesh(new BoxGeometry(1, 2, 1), new MeshStandardMaterial());
  face.name = 'face'; face.morphTargetDictionary = { JawOpen: 0, eyeBlinkLeft: 1 };
  face.morphTargetInfluences = [0.1, 0]; head.add(face);
  return { root, head, face };
}

describe('retargeted avatar', () => {
  it('finds morphs and bones, reports unsupported controls, and preserves authored baselines', () => {
    const { root, face, head } = model(); const rig = createRetargetedRig(root);
    expect(rig.diagnostics.morphs).toContain('jawOpen');
    expect(rig.diagnostics.missing).toContain('eyeBlinkRight');
    const pose = { ...neutralPose(), jawOpen: 0.4, headYaw: 0.2 };
    rig.apply(pose); const first = head.quaternion.clone(); rig.apply(pose);
    expect(head.quaternion.equals(first)).toBe(true);
    expect(face.morphTargetInfluences?.[0]).toBeCloseTo(0.5);
    rig.apply(neutralPose());
    expect(face.morphTargetInfluences?.[0]).toBeCloseTo(0.1);
    expect(head.rotation.y).toBeCloseTo(0.1);
    rig.dispose();
  });

  it('supports explicit per-mesh mapping and bounded weights', () => {
    const { root, face } = model();
    face.morphTargetDictionary = { CustomJaw: 0 }; face.morphTargetInfluences = [0];
    const rig = createRetargetedRig(root, { morphs: { jawOpen: { target: 'CustomJaw', mesh: 'face', scale: 2 } } });
    rig.apply({ ...neutralPose(), jawOpen: 0.9 });
    expect(face.morphTargetInfluences?.[0]).toBe(1); rig.dispose();
  });

  it('normalizes models with transformed roots to the stage floor', () => {
    const { root } = model(); root.position.set(8, 5, -3); root.scale.setScalar(3);
    const rig = createRetargetedRig(root);
    const bounds = new Box3().setFromObject(rig.root); const center = bounds.getCenter(new Vector3());
    expect(bounds.min.y).toBeCloseTo(0); expect(center.x).toBeCloseTo(0); expect(center.z).toBeCloseTo(0);
    expect(bounds.getSize(new Vector3()).y).toBeCloseTo(2.4); rig.dispose();
  });

  it('rejects external models while accepting user uploads and app-hosted assets', () => {
    expect(validateModelUrl('/models/portrait.glb', 'https://app.example').origin).toBe('https://app.example');
    expect(validateModelUrl('blob:https://app.example/123', 'https://app.example').protocol).toBe('blob:');
    expect(() => validateModelUrl('https://other.example/model.glb', 'https://app.example')).toThrow();
    expect(() => validateModelUrl('blob:https://other.example/123', 'https://app.example')).toThrow();
    expect(() => validateModelUrl('javascript:alert(1)', 'https://app.example')).toThrow();
  });
});

describe('procedural portrait rig', () => {
  it('has finite geometry and genuinely articulated face and body controls', () => {
    const rig = createProceduralRig(); let meshes = 0;
    rig.root.traverse((object) => {
      if (!(object instanceof Mesh)) return;
      meshes++;
      for (const component of object.geometry.attributes.position.array) expect(Number.isFinite(component)).toBe(true);
    });
    expect(meshes).toBeGreaterThan(40);
    const hand = rig.root.getObjectByName('leftHand')!;
    const jaw = rig.root.getObjectByName('jaw')!;
    rig.apply({ ...neutralPose(), openPalm: 0.8, jawOpen: 0.5 });
    expect(hand.rotation.y).not.toBe(0); expect(jaw.position.y).toBeLessThan(0);
    rig.apply(neutralPose()); expect(hand.rotation.y).toBeCloseTo(0); expect(jaw.position.y).toBeCloseTo(0);
    rig.dispose();
  });
});
