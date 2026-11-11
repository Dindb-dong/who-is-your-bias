import { Box3, Euler, Group, LoadingManager, Mesh, Object3D, Quaternion, Vector3 } from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { BONE_CHANNELS, FACE_CHANNELS, VISEMES } from './channels';
import { disposeObject } from './dispose';
import type { AvatarRig } from './procedural';
import type { BoneBinding, BoneControl, FaceWeights, MorphBinding, MorphControl, RigMapping, RigPose, Viseme } from './types';
import { parseRigMapping } from './validation';

const MORPH_ALIASES: Partial<Record<MorphControl, string[]>> = {
  eyeBlinkLeft: ['eyeBlink_L', 'blinkLeft'], eyeBlinkRight: ['eyeBlink_R', 'blinkRight'],
  jawOpen: ['mouthOpen'], mouthSmileLeft: ['mouthSmile_L'], mouthSmileRight: ['mouthSmile_R'],
  browDownLeft: ['browDown_L'], browDownRight: ['browDown_R'],
};
const BONES: Record<BoneControl, string[]> = {
  hips: ['Hips', 'mixamorigHips', 'J_Bip_C_Hips'],
  spine: ['Spine', 'mixamorigSpine', 'J_Bip_C_Spine'],
  chest: ['Chest', 'Spine2', 'mixamorigSpine2', 'J_Bip_C_Chest'],
  head: ['Head', 'mixamorigHead', 'J_Bip_C_Head'],
  neck: ['Neck', 'mixamorigNeck', 'J_Bip_C_Neck'],
  jaw: ['Jaw', 'CC_Base_JawRoot'],
  leftEye: ['LeftEye', 'Eye_L', 'eyeLeft', 'J_Adj_L_FaceEye'],
  rightEye: ['RightEye', 'Eye_R', 'eyeRight', 'J_Adj_R_FaceEye'],
  leftShoulder: ['LeftShoulder', 'mixamorigLeftShoulder', 'J_Bip_L_Shoulder'],
  rightShoulder: ['RightShoulder', 'mixamorigRightShoulder', 'J_Bip_R_Shoulder'],
  leftUpperArm: ['LeftArm', 'LeftUpperArm', 'mixamorigLeftArm', 'J_Bip_L_UpperArm'],
  rightUpperArm: ['RightArm', 'RightUpperArm', 'mixamorigRightArm', 'J_Bip_R_UpperArm'],
  leftForearm: ['LeftForeArm', 'LeftLowerArm', 'mixamorigLeftForeArm', 'J_Bip_L_LowerArm'],
  rightForearm: ['RightForeArm', 'RightLowerArm', 'mixamorigRightForeArm', 'J_Bip_R_LowerArm'],
  leftHand: ['LeftHand', 'mixamorigLeftHand', 'J_Bip_L_Hand'],
  rightHand: ['RightHand', 'mixamorigRightHand', 'J_Bip_R_Hand'],
};
const normalized = (value: string): string => value.toLowerCase().replace(/[^a-z0-9]/g, '');
const bounded = (value: number, min: number, max: number): number => Number.isFinite(value) ? Math.max(min, Math.min(max, value)) : 0;

/** Aggregate designed controls become named channels; measurements override them individually. */
export function poseFace(pose: RigPose): FaceWeights {
  return {
    eyeBlinkLeft: pose.blink, eyeBlinkRight: pose.blink, jawOpen: pose.jawOpen,
    mouthSmileLeft: pose.mouthSmile, mouthSmileRight: pose.mouthSmile, mouthFunnel: pose.mouthRound,
    browInnerUp: pose.browRaise, browDownLeft: pose.browFurrow, browDownRight: pose.browFurrow,
    ...pose.face,
  };
}
interface ResolvedMorph { mesh: Mesh; index: number; scale: number; neutral: number; inputNeutral: number; inputMax: number; key: string; }
interface ResolvedBone { object: Object3D; baseline: Quaternion; neutral: Quaternion; basis: Quaternion; inverseBasis: Quaternion; axis?: 'x' | 'y' | 'z'; scale: number; maxAngle: number; }

export function createRetargetedRig(model: Group, suppliedMapping: RigMapping = {}): AvatarRig {
  const mapping = parseRigMapping(suppliedMapping);
  const root = new Group(); root.name = 'retargetedPortrait'; root.add(model);
  const morphs = new Map<MorphControl, ResolvedMorph[]>();
  const visemes = new Map<Viseme, ResolvedMorph[]>();
  const bones = new Map<BoneControl, ResolvedBone>();
  const meshes: Mesh[] = []; const nodes: Object3D[] = []; const warnings: string[] = [];
  const availableMorphs: { mesh: string; target: string; initial: number }[] = [];
  const targetOwners = new Map<string, string[]>();
  const targets = new Map<string, ResolvedMorph>();
  model.traverse((object) => {
    if (object !== model) nodes.push(object);
    if (object instanceof Mesh) {
      meshes.push(object); object.castShadow = true; object.receiveShadow = true;
      for (const [name, index] of Object.entries(object.morphTargetDictionary ?? {})) {
        availableMorphs.push({ mesh: object.name, target: name, initial: object.morphTargetInfluences?.[index] ?? 0 });
      }
    }
  });
  const resolve = (control: string, aliases: string[], user?: string | MorphBinding): ResolvedMorph[] => {
    const binding = typeof user === 'string' ? { target: user } : user;
    const names = binding ? [binding.target] : aliases;
    const found: ResolvedMorph[] = [];
    for (const mesh of meshes) {
      if (binding?.mesh && mesh.name !== binding.mesh) continue;
      if (!mesh.morphTargetDictionary || !mesh.morphTargetInfluences) continue;
      const entry = Object.entries(mesh.morphTargetDictionary).find(([name]) => names.some((alias) => normalized(alias) === normalized(name)));
      if (!entry) continue;
      const key = `${mesh.uuid}:${entry[1]}`;
      const resolved = { mesh, index: entry[1], scale: binding?.scale ?? 1,
        neutral: binding?.outputNeutral ?? mesh.morphTargetInfluences[entry[1]],
        inputNeutral: binding?.inputNeutral ?? 0, inputMax: binding?.inputMax ?? 1, key };
      found.push(resolved); targets.set(key, resolved);
      targetOwners.set(key, [...(targetOwners.get(key) ?? []), control]);
    }
    if (binding && !found.length) warnings.push(`Mapped target for ${control} was not found.`);
    return found;
  };
  for (const control of FACE_CHANNELS) {
    const found = resolve(control, [control, ...(MORPH_ALIASES[control] ?? [])], mapping.morphs?.[control]);
    if (found.length) morphs.set(control, found);
  }
  for (const control of VISEMES) {
    const aliases = [`viseme_${control}`, `viseme${control}`];
    if (control === 'I') aliases.push('viseme_ih');
    if (control === 'O') aliases.push('viseme_oh');
    if (control === 'U') aliases.push('viseme_ou');
    const found = resolve(control, aliases, mapping.visemes?.[control]);
    if (found.length) visemes.set(control, found);
  }
  for (const owners of targetOwners.values()) if (owners.length > 1) warnings.push(`Shared morph target: ${owners.join(', ')}; the largest absolute contribution wins.`);
  for (const control of BONE_CHANNELS) {
    const user = mapping.bones?.[control];
    const binding: BoneBinding | undefined = typeof user === 'string' ? { name: user } : user;
    const names = binding ? [binding.name] : BONES[control];
    const object = nodes.find((node) => names.some((name) => normalized(node.name) === normalized(name)));
    if (object) {
      const basis = new Quaternion(...(binding?.basis ?? [0, 0, 0, 1]));
      bones.set(control, { object, baseline: object.quaternion.clone(), neutral: new Quaternion(...(binding?.neutral ?? [0, 0, 0, 1])),
        basis, inverseBasis: basis.clone().invert(), axis: binding?.axis, scale: binding?.scale ?? 1, maxAngle: binding?.maxAngle ?? Math.PI });
    } else if (binding) warnings.push(`Mapped bone for ${control} was not found.`);
  }
  model.rotation.y += mapping.yaw ?? 0;
  model.updateMatrixWorld(true);
  const box = new Box3().setFromObject(model); const size = box.getSize(new Vector3()); const center = box.getCenter(new Vector3());
  if (![size.x, size.y, size.z].every(Number.isFinite) || size.y <= 0) throw new Error('The GLB has invalid visible geometry bounds.');
  const scale = 2.40 / size.y;
  model.scale.multiplyScalar(scale);
  model.position.multiplyScalar(scale).sub(new Vector3(center.x * scale, box.min.y * scale, center.z * scale));

  const rotation = new Quaternion(); const scaled = new Quaternion(); const identity = new Quaternion(); const euler = new Euler();
  const rotate = (pose: RigPose, name: BoneControl, x = 0, y = 0, z = 0, allowObserved = true): void => {
    const binding = bones.get(name); if (!binding) return;
    const observed = allowObserved ? pose.bones?.[name] : undefined;
    if (observed) rotation.fromArray(observed).normalize();
    else {
      if (binding.axis) {
        const angle = x || y || z;
        euler.set(binding.axis === 'x' ? angle : 0, binding.axis === 'y' ? angle : 0, binding.axis === 'z' ? angle : 0);
      } else euler.set(x, y, z);
      rotation.setFromEuler(euler);
    }
    // q and -q are the same orientation. Canonicalizing avoids long-path scaling.
    if (rotation.w < 0) rotation.set(-rotation.x, -rotation.y, -rotation.z, -rotation.w);
    const angle = 2 * Math.acos(bounded(rotation.w, -1, 1));
    const scaledAngle = Math.min(binding.maxAngle, angle * Math.abs(binding.scale));
    if (angle > 1e-8) rotation.copy(scaled.slerpQuaternions(identity, rotation, scaledAngle / angle));
    else rotation.identity();
    if (binding.scale < 0) rotation.invert();
    rotation.premultiply(binding.basis).multiply(binding.inverseBasis);
    binding.object.quaternion.copy(binding.baseline).multiply(binding.neutral).multiply(rotation).normalize();
  };
  const contributions = new Map<string, number>();
  const setMorph = (bindings: ResolvedMorph[] | undefined, value: number): void => {
    for (const binding of bindings ?? []) {
      const calibrated = bounded((value - binding.inputNeutral) / (binding.inputMax - binding.inputNeutral), 0, 1) * binding.scale;
      const previous = contributions.get(binding.key) ?? 0;
      if (Math.abs(calibrated) > Math.abs(previous)) contributions.set(binding.key, calibrated);
    }
  };
  return {
    root,
    diagnostics: {
      kind: 'glb', morphs: [...morphs.keys()], bones: [...bones.keys()], visemes: [...visemes.keys()],
      missing: [...(['eyeBlinkLeft', 'eyeBlinkRight', 'jawOpen'] as const).filter((name) => !morphs.has(name)), ...(!bones.has('head') ? ['head'] : [])],
      unsupportedFace: FACE_CHANNELS.filter((name) => !morphs.has(name)), unsupportedBones: BONE_CHANNELS.filter((name) => !bones.has(name)),
      availableMorphs, availableNodes: nodes.map((node) => node.name).filter(Boolean), warnings,
    },
    apply(pose: RigPose) {
      root.position.y = pose.breath;
      contributions.clear(); const face = poseFace(pose);
      // Native visemes and ARKit speech deformations are alternatives; applying
      // both to a mouth doubles the opening. Non-mouth observed channels remain.
      const activeVisemes = Object.entries(pose.visemes ?? {}).filter(([, value]) => (value ?? 0) > 0.001);
      // A partially mapped blend cannot replace the full ARKit articulation.
      // For example, a rig with only sil must not erase an unsupported aa.
      const nativeSpeech = activeVisemes.length > 0 && activeVisemes.every(([key]) => visemes.has(key as Viseme));
      const fallbackAa = !nativeSpeech && !morphs.has('jawOpen') && visemes.has('aa') ? face.jawOpen ?? pose.jawOpen : 0;
      const nativeMouth = nativeSpeech || fallbackAa > 0.001;
      for (const name of FACE_CHANNELS) {
        const mouth = name.startsWith('mouth') || name.startsWith('jaw') || name === 'tongueOut';
        setMorph(morphs.get(name), nativeMouth && mouth ? 0 : face[name] ?? 0);
      }
      for (const name of VISEMES) setMorph(visemes.get(name), nativeSpeech ? pose.visemes?.[name] ?? 0 : name === 'aa' ? fallbackAa : 0);
      for (const [key, binding] of targets) {
        if (binding.mesh.morphTargetInfluences) binding.mesh.morphTargetInfluences[binding.index] = bounded(binding.neutral + (contributions.get(key) ?? 0), 0, 1);
      }
      rotate(pose, 'hips'); rotate(pose, 'spine'); rotate(pose, 'chest');
      rotate(pose, 'head', pose.headPitch + pose.gazeY * 0.2, pose.headYaw + pose.gazeX * 0.2, pose.headRoll);
      rotate(pose, 'neck', pose.headPitch * 0.2, pose.headYaw * 0.2, pose.headRoll * 0.2);
      rotate(pose, 'leftEye', pose.gazeY, pose.gazeX); rotate(pose, 'rightEye', pose.gazeY, pose.gazeX);
      rotate(pose, 'jaw', morphs.has('jawOpen') || nativeMouth ? 0 : (face.jawOpen ?? pose.jawOpen) * 0.2, 0, 0, !morphs.has('jawOpen') && !nativeMouth);
      rotate(pose, 'leftShoulder', 0, 0, pose.shrug * 0.08); rotate(pose, 'rightShoulder', 0, 0, -pose.shrug * 0.08);
      rotate(pose, 'leftUpperArm', -pose.openPalm * 0.35 - pose.emphasis * 0.25, 0, pose.openPalm * 0.13 + pose.shrug * 0.08);
      rotate(pose, 'rightUpperArm', -pose.openPalm * 0.35 - pose.wave * 0.6, 0, -pose.openPalm * 0.13 - pose.shrug * 0.08);
      rotate(pose, 'leftForearm', -pose.openPalm * 0.7 - pose.emphasis * 0.45);
      rotate(pose, 'rightForearm', -pose.openPalm * 0.7 - pose.wave * 0.7);
      rotate(pose, 'leftHand', 0, pose.openPalm * 0.4, pose.emphasis * 0.1);
      rotate(pose, 'rightHand', 0, -pose.openPalm * 0.4, -pose.wave * 0.2);
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
