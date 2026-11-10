import { Box3, Euler, Group, LoadingManager, Mesh, Object3D, Quaternion, Vector3 } from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { disposeObject } from './dispose';
import type { AvatarRig } from './procedural';
import type { BoneBinding, BoneControl, MorphBinding, MorphControl, RigMapping, RigPose } from './types';

const MORPHS: Record<MorphControl, string[]> = {
  eyeBlinkLeft: ['eyeBlinkLeft', 'EyeBlinkLeft', 'eyeBlink_L', 'blinkLeft'],
  eyeBlinkRight: ['eyeBlinkRight', 'EyeBlinkRight', 'eyeBlink_R', 'blinkRight'],
  jawOpen: ['jawOpen', 'JawOpen', 'mouthOpen', 'viseme_aa'],
  mouthSmileLeft: ['mouthSmileLeft', 'MouthSmileLeft', 'mouthSmile_L'],
  mouthSmileRight: ['mouthSmileRight', 'MouthSmileRight', 'mouthSmile_R'],
  mouthFunnel: ['mouthFunnel', 'MouthFunnel', 'viseme_O'],
  browInnerUp: ['browInnerUp', 'BrowInnerUp'],
  browDownLeft: ['browDownLeft', 'BrowDownLeft', 'browDown_L'],
  browDownRight: ['browDownRight', 'BrowDownRight', 'browDown_R'],
};

const BONES: Record<BoneControl, string[]> = {
  head: ['Head', 'head', 'mixamorigHead', 'J_Bip_C_Head'],
  neck: ['Neck', 'neck', 'mixamorigNeck', 'J_Bip_C_Neck'],
  jaw: ['Jaw', 'jaw', 'CC_Base_JawRoot'],
  leftUpperArm: ['LeftArm', 'LeftUpperArm', 'mixamorigLeftArm', 'J_Bip_L_UpperArm'],
  rightUpperArm: ['RightArm', 'RightUpperArm', 'mixamorigRightArm', 'J_Bip_R_UpperArm'],
  leftForearm: ['LeftForeArm', 'LeftLowerArm', 'mixamorigLeftForeArm', 'J_Bip_L_LowerArm'],
  rightForearm: ['RightForeArm', 'RightLowerArm', 'mixamorigRightForeArm', 'J_Bip_R_LowerArm'],
  leftHand: ['LeftHand', 'mixamorigLeftHand', 'J_Bip_L_Hand'],
  rightHand: ['RightHand', 'mixamorigRightHand', 'J_Bip_R_Hand'],
};

const normalized = (value: string): string => value.toLowerCase().replace(/[^a-z0-9]/g, '');
const bounded = (value: number, min: number, max: number): number =>
  Number.isFinite(value) ? Math.max(min, Math.min(max, value)) : 0;

interface ResolvedMorph { mesh: Mesh; index: number; scale: number; baseline: number; }
interface ResolvedBone { object: Object3D; baseline: Quaternion; axis?: 'x' | 'y' | 'z'; scale: number; }

export function createRetargetedRig(model: Group, mapping: RigMapping = {}): AvatarRig {
  const root = new Group(); root.name = 'retargetedPortrait'; root.add(model);
  const morphs = new Map<MorphControl, ResolvedMorph[]>();
  const bones = new Map<BoneControl, ResolvedBone>();
  const meshes: Mesh[] = []; const nodes: Object3D[] = [];
  model.traverse((object) => {
    nodes.push(object);
    if (object instanceof Mesh) {
      meshes.push(object); object.castShadow = true; object.receiveShadow = true;
    }
  });

  for (const [control, aliases] of Object.entries(MORPHS) as [MorphControl, string[]][]) {
    const user = mapping.morphs?.[control];
    const binding: MorphBinding | undefined = typeof user === 'string' ? { target: user } : user;
    const names = binding ? [binding.target] : aliases;
    const found: ResolvedMorph[] = [];
    for (const mesh of meshes) {
      if (binding?.mesh && mesh.name !== binding.mesh) continue;
      if (!mesh.morphTargetDictionary || !mesh.morphTargetInfluences) continue;
      const entry = Object.entries(mesh.morphTargetDictionary).find(([name]) => names.some((alias) => normalized(alias) === normalized(name)));
      if (entry) found.push({ mesh, index: entry[1], scale: bounded(binding?.scale ?? 1, -2, 2), baseline: mesh.morphTargetInfluences[entry[1]] });
    }
    if (found.length) morphs.set(control, found);
  }
  for (const [control, aliases] of Object.entries(BONES) as [BoneControl, string[]][]) {
    const user = mapping.bones?.[control];
    const binding: BoneBinding | undefined = typeof user === 'string' ? { name: user } : user;
    const names = binding ? [binding.name] : aliases;
    const object = nodes.find((node) => names.some((name) => normalized(node.name) === normalized(name)));
    if (object) bones.set(control, { object, baseline: object.quaternion.clone(), axis: binding?.axis, scale: bounded(binding?.scale ?? 1, -2, 2) });
  }

  // Normalize arbitrary source units while preserving the authored proportions.
  model.rotation.y += bounded(mapping.yaw ?? 0, -Math.PI * 2, Math.PI * 2);
  model.updateMatrixWorld(true);
  const box = new Box3().setFromObject(model); const size = box.getSize(new Vector3()); const center = box.getCenter(new Vector3());
  const maximum = Math.max(size.x, size.y, size.z);
  if (!Number.isFinite(maximum) || maximum <= 0) throw new Error('The GLB has no visible geometry.');
  const scale = 2.40 / size.y;
  if (!Number.isFinite(scale) || scale <= 0) throw new Error('The GLB has invalid bounds.');
  model.scale.multiplyScalar(scale);
  model.position.multiplyScalar(scale).sub(new Vector3(center.x * scale, box.min.y * scale, center.z * scale));

  const rotation = new Quaternion(); const euler = new Euler();
  const rotate = (name: BoneControl, x: number, y = 0, z = 0): void => {
    const binding = bones.get(name); if (!binding) return;
    if (binding.axis) {
      const angle = (x || y || z) * binding.scale;
      euler.set(binding.axis === 'x' ? angle : 0, binding.axis === 'y' ? angle : 0, binding.axis === 'z' ? angle : 0);
    } else euler.set(x * binding.scale, y * binding.scale, z * binding.scale);
    rotation.setFromEuler(euler);
    binding.object.quaternion.copy(binding.baseline).multiply(rotation);
  };
  const setMorph = (name: MorphControl, value: number): void => {
    for (const binding of morphs.get(name) ?? []) {
      if (binding.mesh.morphTargetInfluences) binding.mesh.morphTargetInfluences[binding.index] = bounded(binding.baseline + value * binding.scale, 0, 1);
    }
  };
  const required: MorphControl[] = ['eyeBlinkLeft', 'eyeBlinkRight', 'jawOpen'];
  return {
    root,
    diagnostics: {
      kind: 'glb', morphs: [...morphs.keys()], bones: [...bones.keys()],
      missing: [...required.filter((name) => !morphs.has(name)), ...(!bones.has('head') ? ['head'] : [])],
    },
    apply(pose: RigPose) {
      root.position.y = pose.breath;
      setMorph('eyeBlinkLeft', pose.blink); setMorph('eyeBlinkRight', pose.blink);
      setMorph('jawOpen', pose.jawOpen);
      setMorph('mouthSmileLeft', pose.mouthSmile); setMorph('mouthSmileRight', pose.mouthSmile);
      setMorph('mouthFunnel', pose.mouthRound);
      setMorph('browInnerUp', pose.browRaise);
      setMorph('browDownLeft', pose.browFurrow); setMorph('browDownRight', pose.browFurrow);
      rotate('head', pose.headPitch + pose.gazeY * 0.2, pose.headYaw + pose.gazeX * 0.2, pose.headRoll);
      rotate('neck', pose.headPitch * 0.2, pose.headYaw * 0.2, pose.headRoll * 0.2);
      // Avoid applying both a jaw morph and jaw bone unless the asset has no
      // jaw morph; double deformation is a common retargeting artifact.
      rotate('jaw', morphs.has('jawOpen') ? 0 : pose.jawOpen * 0.2);
      rotate('leftUpperArm', -pose.openPalm * 0.35 - pose.emphasis * 0.25, 0, pose.openPalm * 0.13 + pose.shrug * 0.08);
      rotate('rightUpperArm', -pose.openPalm * 0.35 - pose.wave * 0.6, 0, -pose.openPalm * 0.13 - pose.shrug * 0.08);
      rotate('leftForearm', -pose.openPalm * 0.7 - pose.emphasis * 0.45);
      rotate('rightForearm', -pose.openPalm * 0.7 - pose.wave * 0.7);
      rotate('leftHand', 0, pose.openPalm * 0.4, pose.emphasis * 0.1);
      rotate('rightHand', 0, -pose.openPalm * 0.4, -pose.wave * 0.2);
    },
    dispose() { disposeObject(root); },
  };
}

export const MAX_MODEL_BYTES = 64 * 1024 * 1024;

/** Only self-contained GLB assets are supported: no arbitrary subresource fetch. */
export function validateModelUrl(url: string, origin: string): URL {
  const parsed = new URL(url, origin);
  if (parsed.protocol === 'blob:' && parsed.origin === new URL(origin).origin) return parsed;
  if ((parsed.protocol === 'http:' || parsed.protocol === 'https:') && parsed.origin === new URL(origin).origin) return parsed;
  throw new Error('Choose a local GLB file or a model hosted by this application.');
}

async function readBoundedBody(response: Response, signal?: AbortSignal): Promise<ArrayBuffer> {
  if (Number(response.headers.get('content-length')) > MAX_MODEL_BYTES) throw new Error('The model exceeds the 64 MiB limit.');
  if (!response.body) throw new Error('The model response was empty.');
  const reader = response.body.getReader(); const chunks: Uint8Array[] = []; let length = 0;
  try {
    while (true) {
      signal?.throwIfAborted();
      const result = await reader.read(); if (result.done) break;
      length += result.value.byteLength;
      if (length > MAX_MODEL_BYTES) throw new Error('The model exceeds the 64 MiB limit.');
      chunks.push(result.value);
    }
  } catch (error) { await reader.cancel().catch(() => undefined); throw error; }
  finally { reader.releaseLock(); }
  const all = new Uint8Array(length); let cursor = 0;
  for (const chunk of chunks) { all.set(chunk, cursor); cursor += chunk.length; }
  return all.buffer;
}

export async function loadGlbRig(url: string, mapping: RigMapping, signal?: AbortSignal): Promise<AvatarRig> {
  const target = validateModelUrl(url, window.location.origin);
  const response = await fetch(target, { signal, credentials: 'same-origin' });
  if (!response.ok) throw new Error(`The model could not be loaded (${response.status}).`);
  const bytes = await readBoundedBody(response, signal);
  if (bytes.byteLength < 12 || new DataView(bytes).getUint32(0, true) !== 0x46546c67) throw new Error('Please upload a binary glTF (.glb) model.');
  signal?.throwIfAborted();
  const manager = new LoadingManager();
  manager.setURLModifier((resource) => {
    if (resource.startsWith('blob:') || resource.startsWith('data:')) return resource;
    throw new Error('Use a self-contained GLB with embedded textures and buffers.');
  });
  const loader = new GLTFLoader(manager);
  const gltf = await loader.parseAsync(bytes, '');
  if (signal?.aborted) { disposeObject(gltf.scene); signal.throwIfAborted(); }
  try { return createRetargetedRig(gltf.scene, mapping); }
  catch (error) { disposeObject(gltf.scene); throw error; }
}
