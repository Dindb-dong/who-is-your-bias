import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { Mesh, MeshStandardMaterial, Object3D, Quaternion, SkinnedMesh, Vector3 } from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { describe, expect, it } from 'vitest';
import { createResearchHumanMapping, mergeRigMappings } from './builtin';
import { FACE_CHANNELS, VISEMES } from './channels';
import { sampleMotion } from './motion';
import { createRetargetedRig } from './retarget';
import { neutralPose } from './timeline';
import type { BoneControl, BoneBinding, RigMapping } from './types';
import { parseAvatarClip } from './validation';

const asset = new URL('../../public/models/research-human-v2.glb', import.meta.url);

async function loadHuman() {
  const bytes = await readFile(asset);
  const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
  // This Node test decodes the real compressed geometry, morphs, and skin.
  // Only material/image creation is replaced: actual WebP/GPU rendering is
  // covered by the browser test with the unmodified same-origin GLB.
  loader.register((parser) => ({ name: 'CPU_GEOMETRY_INSPECTION', loadMaterial: async (index) =>
    new MeshStandardMaterial({ name: parser.json.materials[index].name }) }));
  const imported = await loader.parseAsync(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), '');
  const before = new Map<Object3D, Quaternion>();
  imported.scene.traverse((node) => before.set(node, node.quaternion.clone()));
  const mapping = createResearchHumanMapping(imported.scene);
  for (const [node, orientation] of before) expect(node.quaternion.equals(orientation)).toBe(true);
  const rig = createRetargetedRig(imported.scene, mapping);
  const meshes: SkinnedMesh[] = [];
  rig.root.traverse((node) => { if (node instanceof SkinnedMesh) meshes.push(node); });
  const node = (control: BoneControl) => rig.root.getObjectByName((mapping.bones![control] as BoneBinding).name)!;
  const update = () => { rig.root.updateMatrixWorld(true); meshes.forEach((mesh) => mesh.skeleton.update()); };
  rig.apply(neutralPose()); update();
  return { rig, mapping, meshes, node, update };
}

function positions(mesh: Mesh): Vector3[] {
  return Array.from({ length: mesh.geometry.attributes.position.count }, (_, index) => mesh.getVertexPosition(index, new Vector3()).applyMatrix4(mesh.matrixWorld));
}
function largestChange(mesh: Mesh, before: Vector3[]): number {
  let maximum = 0; const value = new Vector3();
  before.forEach((position, index) => { mesh.getVertexPosition(index, value).applyMatrix4(mesh.matrixWorld); maximum = Math.max(maximum, value.distanceTo(position)); });
  return maximum;
}

describe('shipped Meshopt research human', () => {
  it('decodes the actual compressed asset, preserves all native channels, and binds every anatomical control', async () => {
    const bytes = await readFile(asset);
    const json = JSON.parse(bytes.subarray(20, 20 + bytes.readUInt32LE(12)).toString());
    const receipt = JSON.parse(await readFile(new URL('../../../../tools/avatar/asset-report.json', import.meta.url), 'utf8'));
    expect(createHash('sha256').update(bytes).digest('hex')).toBe(receipt.output_sha256);
    expect(json.extensionsRequired).toContain('EXT_meshopt_compression');
    expect(json.extensionsUsed).toContain('EXT_texture_webp');
    expect(json.images.every((image: { mimeType: string }) => image.mimeType === 'image/webp')).toBe(true);
    // The eye asset contains an alpha layer over the iris. Making it opaque
    // produces black eyes even though every geometry/rig assertion passes.
    for (const name of ['Eyes', 'Hair', 'Brows']) {
      expect(json.materials.find((material: { name: string }) => material.name === name))
        .toMatchObject({ alphaMode: 'MASK', alphaCutoff: 0.35, doubleSided: true });
    }
    expect(receipt.before.target_names).toHaveLength(67);
    expect(receipt.after.target_names).toEqual(receipt.before.target_names);
    expect(receipt.after.nonzero_target_names).toHaveLength(66); // sil is deliberately neutral.
    const { rig, meshes } = await loadHuman();
    expect(rig.diagnostics.morphs.sort()).toEqual([...FACE_CHANNELS].sort());
    expect(rig.diagnostics.visemes!.sort()).toEqual([...VISEMES].sort());
    expect(rig.diagnostics.bones).toHaveLength(16);
    expect(rig.diagnostics.missing).toEqual([]);
    expect(meshes).toHaveLength(7);
    expect(meshes[0].skeleton.bones).toHaveLength(163);
    expect(meshes.reduce((sum, mesh) => sum + mesh.geometry.attributes.position.count, 0)).toBeGreaterThan(25_000);
    rig.dispose();
  });

  it('deforms actual skinned vertices for face shapes and native speech while silence returns to neutral', async () => {
    const { rig, meshes, node, update } = await loadHuman();
    const skin = meshes.reduce((largest, mesh) => mesh.geometry.attributes.position.count > largest.geometry.attributes.position.count ? mesh : largest);
    const tongue = meshes.find((mesh) => mesh.morphTargetDictionary?.tongueOut !== undefined)!;
    const baseline = positions(skin); const tongueBaseline = positions(tongue); const jawRest = node('jaw').quaternion.clone();
    for (const channel of ['jawOpen', 'eyeBlinkLeft', 'mouthSmileLeft'] as const) {
      rig.apply({ ...neutralPose(), face: { [channel]: 0.8 } }); update();
      expect(largestChange(skin, baseline), `${channel} must move the shipped geometry`).toBeGreaterThan(0.002);
    }
    rig.apply({ ...neutralPose(), face: { tongueOut: 0.8 } }); update();
    expect(largestChange(tongue, tongueBaseline)).toBeGreaterThan(0.002);
    rig.apply({ ...neutralPose(), face: { jawOpen: 0.8 }, visemes: { aa: 0.8 } }); update();
    expect(largestChange(skin, baseline)).toBeGreaterThan(0.01);
    expect(skin.morphTargetInfluences![skin.morphTargetDictionary!.jawOpen]).toBe(0);
    expect(node('jaw').quaternion.angleTo(jawRest)).toBeLessThan(1e-6);
    rig.apply({ ...neutralPose(), visemes: { sil: 1 } }); update();
    expect(largestChange(skin, baseline)).toBeLessThan(1e-6);
    rig.dispose();
  });

  it('faces the viewer with relaxed arms and transfers actual measured body deltas to skin and joints', async () => {
    const { rig, meshes, node, update } = await loadHuman();
    const point = (control: BoneControl) => node(control).getWorldPosition(new Vector3());
    expect(point('leftEye').z).toBeGreaterThan(point('head').z);
    expect(point('leftUpperArm').x).toBeGreaterThan(0);
    expect(point('rightUpperArm').x).toBeLessThan(0);
    for (const side of ['left', 'right'] as const) {
      const arm = point(`${side}Forearm`).sub(point(`${side}UpperArm`)).normalize();
      const forearm = point(`${side}Hand`).sub(point(`${side}Forearm`)).normalize();
      expect(arm.dot(new Vector3(0, -1, 0))).toBeGreaterThan(0.97);
      expect(forearm.dot(new Vector3(0, -1, 0))).toBeGreaterThan(0.97);
    }
    const clip = parseAvatarClip(JSON.parse(await readFile(new URL('../../public/samples/ted-2013-body-geometry-preview.json', import.meta.url), 'utf8')), true);
    const clothing = meshes.find((mesh) => mesh.name.includes('casualsuit'))!;
    const baseline = positions(clothing); const wrist = point('leftHand');
    let greatestWristChange = 0; let greatestSkinChange = 0;
    for (const frame of clip.frames) {
      rig.apply({ ...neutralPose(), ...sampleMotion(clip, frame.time_ms, true)! }); update();
      greatestWristChange = Math.max(greatestWristChange, point('leftHand').distanceTo(wrist));
    }
    // The largest forearm deviation is used for a real mesh deformation check.
    const expressive = clip.frames.reduce((a, b) => Math.abs(a.bones!.leftForearm![3]) < Math.abs(b.bones!.leftForearm![3]) ? a : b);
    rig.apply({ ...neutralPose(), ...sampleMotion(clip, expressive.time_ms, true)! }); update();
    greatestSkinChange = largestChange(clothing, baseline);
    expect(greatestWristChange).toBeGreaterThan(0.1);
    expect(greatestSkinChange).toBeGreaterThan(0.03);
    rig.apply(neutralPose()); update();
    expect(point('leftHand').distanceTo(wrist)).toBeLessThan(1e-6);
    rig.dispose();
  });

  it('avoids duplicate eye rotation and allows explicit uploaded mappings to override built-in calibration', async () => {
    const { rig, meshes, node, update, mapping } = await loadHuman();
    const leftRest = node('leftEye').quaternion.clone();
    rig.apply({ ...neutralPose(), gazeX: 0.2, gazeY: 0.1, face: {
      eyeLookInLeft: 0, eyeLookOutLeft: 0.7, eyeLookUpLeft: 0, eyeLookDownLeft: 0,
    } }); update();
    expect(node('leftEye').quaternion.angleTo(leftRest)).toBeLessThan(1e-6);
    const eyes = meshes.find((mesh) => mesh.name.includes('high-poly'))!;
    expect(eyes.morphTargetInfluences![eyes.morphTargetDictionary!.eyeLookOutLeft]).toBeCloseTo(0.7);
    rig.apply({ ...neutralPose(), gazeX: 0.2, gazeY: 0.1 }); update();
    expect(node('leftEye').quaternion.angleTo(leftRest)).toBeGreaterThan(0.1);
    expect(eyes.morphTargetInfluences![eyes.morphTargetDictionary!.eyeLookOutLeft]).toBe(0);
    const override: RigMapping = { bones: { head: { name: 'CustomHead', scale: 0.5 } }, yaw: 0 };
    const merged = mergeRigMappings(mapping, override);
    expect(merged.bones!.head).toEqual(override.bones!.head);
    expect(merged.bones!.leftForearm).toEqual(mapping.bones!.leftForearm);
    expect(merged.yaw).toBe(0); rig.dispose();
  });
});
