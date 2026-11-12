import { Box3, BoxGeometry, Group, Mesh, MeshStandardMaterial, Quaternion, Vector3 } from 'three';
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

  it('keeps head direction independent from eye gaze and matches upward gaze across rigs', () => {
    const { root, head } = model();
    const eye = new Group(); eye.name = 'LeftEye'; head.add(eye);
    const rig = createRetargetedRig(root);
    rig.apply({ ...neutralPose(), gazeX: 0.25, gazeY: 0.2 });
    expect(head.rotation.y).toBeCloseTo(0.1); expect(head.rotation.x).toBeCloseTo(0);
    expect(eye.rotation.y).toBeCloseTo(0.25); expect(eye.rotation.x).toBeCloseTo(-0.2);
    rig.dispose();
  });

  it('calibrates neutral/range and rotates measured deltas through an explicit local basis without drift', () => {
    const { root, face, head } = model();
    const basis = new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), Math.PI / 2);
    const rig = createRetargetedRig(root, { morphs: { jawOpen: { target: 'JawOpen', inputNeutral: 0.2, inputMax: 0.8, outputNeutral: 0.05 } },
      bones: { head: { name: 'Head', basis: basis.toArray() as [number, number, number, number], maxAngle: 0.25 } } });
    const measured = new Quaternion().setFromAxisAngle(new Vector3(1, 0, 0), 0.8);
    const pose = { ...neutralPose(), face: { jawOpen: 0.5 }, bones: { head: measured.toArray() as [number, number, number, number] } };
    rig.apply(pose);
    expect(face.morphTargetInfluences?.[0]).toBeCloseTo(0.55);
    expect(head.rotation.y).toBeCloseTo(0.35);
    const first = head.quaternion.clone(); for (let index = 0; index < 100; index++) rig.apply(pose);
    expect(head.quaternion.angleTo(first)).toBeLessThan(1e-7);
    rig.apply(neutralPose()); expect(head.rotation.y).toBeCloseTo(0.1); rig.dispose();
  });

  it('does not double jaw deformation when a target supplies native visemes', () => {
    const { root, face } = model(); face.morphTargetDictionary!.viseme_aa = 2; face.morphTargetInfluences!.push(0);
    const rig = createRetargetedRig(root);
    rig.apply({ ...neutralPose(), face: { jawOpen: 0.8 }, visemes: { aa: 0.6 } });
    expect(face.morphTargetInfluences?.[0]).toBeCloseTo(0.1);
    expect(face.morphTargetInfluences?.[2]).toBeCloseTo(0.6);
    expect(rig.diagnostics.unsupportedFace).toContain('tongueOut'); rig.dispose();
  });

  it('keeps complete ARKit articulation when only part of a native viseme blend is mapped', () => {
    const { root, face } = model(); face.morphTargetDictionary!.viseme_sil = 2; face.morphTargetInfluences!.push(0);
    const rig = createRetargetedRig(root);
    rig.apply({ ...neutralPose(), face: { jawOpen: 0.612 }, visemes: { sil: 0.1, aa: 0.9 } });
    expect(face.morphTargetInfluences?.[0]).toBeCloseTo(0.712);
    expect(face.morphTargetInfluences?.[2]).toBe(0);
    rig.apply({ ...neutralPose(), face: { jawOpen: 0.4 }, visemes: { sil: 1 } });
    expect(face.morphTargetInfluences?.[0]).toBeCloseTo(0.1);
    expect(face.morphTargetInfluences?.[2]).toBe(1); rig.dispose();
  });

  it('uses an aa-only mouth as a fallback without opening the jaw bone a second time', () => {
    const { root, face } = model();
    face.morphTargetDictionary = { viseme_aa: 0 }; face.morphTargetInfluences = [0];
    const jaw = new Group(); jaw.name = 'Jaw'; jaw.rotation.x = 0.1; root.add(jaw);
    const rig = createRetargetedRig(root);
    rig.apply({ ...neutralPose(), face: { jawOpen: 0.6 }, visemes: { kk: 1 } });
    expect(face.morphTargetInfluences?.[0]).toBeCloseTo(0.6);
    expect(jaw.rotation.x).toBeCloseTo(0.1);
    rig.apply(neutralPose()); expect(face.morphTargetInfluences?.[0]).toBe(0); rig.dispose();
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

  it('applies independent measured face channels and quaternion joints on the procedural fallback', () => {
    const rig = createProceduralRig();
    rig.apply({ ...neutralPose(), face: { eyeBlinkLeft: 1, eyeBlinkRight: 0, jawOpen: 0.7 }, bones: { head: [0, Math.sin(0.2), 0, Math.cos(0.2)] } });
    expect(rig.root.getObjectByName('eyeLeft')!.scale.y).toBeLessThan(0.1);
    expect(rig.root.getObjectByName('eyeRight')!.scale.y).toBeCloseTo(1);
    expect(rig.root.getObjectByName('head')!.rotation.y).toBeCloseTo(0.4);
    expect(rig.root.getObjectByName('head')!.parent?.name).toBe('neck');
    expect(rig.diagnostics.unsupportedFace).toContain('noseSneerLeft'); rig.dispose();
  });
});

