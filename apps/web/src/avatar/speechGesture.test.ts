import { readFile } from 'node:fs/promises';
import { MeshStandardMaterial, Quaternion, Vector3 } from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { describe, expect, it } from 'vitest';
import { createResearchHumanMapping } from './builtin';
import { createProceduralRig } from './procedural';
import { samplePerformance } from './performance';
import { createRetargetedRig } from './retarget';
import { appendStreamingGesture, calmPoses, completeStreamingGesture, createStreamingGesture, parseSpeechGestureMotion, sampleSpeechGesture, SMPLX_POSE_WIDTH, type SpeechGestureMotion } from './speechGesture';
import { neutralPose } from './timeline';
import type { BoneBinding, BoneControl, RigDiagnostics } from './types';

const sha = 'a'.repeat(64);
const upperL = new Vector3(0.16, -1, 0.025).normalize(), foreL = new Vector3(0.10, -1, 0.14).normalize();
const upperR = new Vector3(-0.16, -1, 0.025).normalize();

function setJoint(poses: Float32Array, frame: number, joint: number, q: Quaternion) {
  const angle = 2 * Math.acos(Math.min(1, Math.abs(q.w))), s = Math.sqrt(Math.max(0, 1 - q.w * q.w)), sign = q.w < 0 ? -1 : 1;
  const o = frame * SMPLX_POSE_WIDTH + joint * 3;
  if (s < 1e-9) { poses[o] = poses[o + 1] = poses[o + 2] = 0; return; }
  poses[o] = sign * q.x / s * angle; poses[o + 1] = sign * q.y / s * angle; poses[o + 2] = sign * q.z / s * angle;
}

/** An SMPL-X clip whose arms point along the given world directions (collars at rest). */
function clip(leftUpper: Vector3, leftFore: Vector3, frames = 60, rightUpper = upperR): SpeechGestureMotion {
  const poses = new Float32Array(frames * SMPLX_POSE_WIDTH);
  const gU = new Quaternion().setFromUnitVectors(new Vector3(1, 0, 0), leftUpper);
  const gF = new Quaternion().setFromUnitVectors(leftUpper, leftFore).multiply(gU);
  const gUR = new Quaternion().setFromUnitVectors(new Vector3(-1, 0, 0), rightUpper);
  const gFR = new Quaternion().setFromUnitVectors(rightUpper, rightUpper === upperR ? new Vector3(-0.10, -1, 0.14).normalize() : rightUpper).multiply(gUR);
  for (let f = 0; f < frames; f++) {
    setJoint(poses, f, 16, gU); setJoint(poses, f, 18, gU.clone().invert().multiply(gF));
    setJoint(poses, f, 17, gUR); setJoint(poses, f, 19, gUR.clone().invert().multiply(gFR));
  }
  return { fps: 30, frames, poses, audioSha256: sha, engine: 'test' };
}

const rotated = (q: [number, number, number, number] | undefined, v: Vector3) => v.clone().applyQuaternion(new Quaternion(...(q ?? [0, 0, 0, 1])));

describe('speech-driven gesture conversion', () => {
  it('maps the arms-down reference pose to the rig neutral and follows raised arms', () => {
    const rest = sampleSpeechGesture(clip(upperL, foreL), 1000)!;
    for (const [control, q] of Object.entries(rest)) expect(Math.abs(q![3]), control).toBeGreaterThan(0.9999);
    const forward = new Vector3(0, 0, 1);
    const raised = sampleSpeechGesture(clip(forward, forward), 1000)!;
    expect(rotated(raised.leftUpperArm, upperL).dot(forward)).toBeGreaterThan(0.999);
    expect(Math.abs(raised.rightUpperArm![3])).toBeGreaterThan(0.9999);
  });

  it('fades in and out, is silent outside the clip, and damps for reduced motion', () => {
    const motion = clip(new Vector3(0, 0, 1), new Vector3(0, 0, 1));
    const angle = (ms: number, reducedMotion = false) => 2 * Math.acos(Math.abs(sampleSpeechGesture(motion, ms, { reducedMotion })!.leftUpperArm![3]));
    expect(angle(0)).toBeLessThan(1e-6);
    expect(angle(100)).toBeLessThan(angle(1000));
    expect(angle(1000, true)).toBeLessThan(angle(1000) * 0.2);
    expect(sampleSpeechGesture(motion, 2000)).toBeNull();
    expect(sampleSpeechGesture(motion, -1)).toBeNull();
  });

  it('applies the team calm filter and validates the binary', () => {
    const constant = new Float32Array(10 * SMPLX_POSE_WIDTH).fill(0.3);
    expect(Array.from(calmPoses(constant, 10)).every((v) => Math.abs(v - 0.3) < 1e-6)).toBe(true);
    const spike = new Float32Array(10 * SMPLX_POSE_WIDTH); spike[5 * SMPLX_POSE_WIDTH] = 1;
    expect(calmPoses(spike, 10)[5 * SMPLX_POSE_WIDTH]).toBeLessThan(0.2);
    expect(calmPoses(spike, 10, 1, 0)).toBe(spike);
    const bytes = new ArrayBuffer(2 * SMPLX_POSE_WIDTH * 4);
    expect(parseSpeechGestureMotion(bytes, { fps: 30, frames: 2, audioSha256: sha, engine: 'x' }).frames).toBe(2);
    expect(() => parseSpeechGestureMotion(bytes, { fps: 30, frames: 3, audioSha256: sha, engine: 'x' })).toThrow();
    new DataView(bytes).setFloat32(0, Number.NaN, true);
    expect(() => parseSpeechGestureMotion(bytes, { fps: 30, frames: 2, audioSha256: sha, engine: 'x' })).toThrow();
  });

  it('hands a missing collar rotation to the upper arm so the arm still hangs where EMAGE puts it', () => {
    const motion = clip(upperL, foreL);
    const collar = new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), -0.35);           // collar lowers the arm…
    const shoulder = new Quaternion().setFromUnitVectors(new Vector3(1, 0, 0), upperL);
    for (let f = 0; f < motion.frames; f++) {
      setJoint(motion.poses, f, 13, collar);
      setJoint(motion.poses, f, 16, collar.clone().invert().multiply(shoulder));            // …and the shoulder compensates
    }
    const full = sampleSpeechGesture(motion, 1000)!;
    expect(2 * Math.acos(Math.abs(full.leftUpperArm![3]))).toBeGreaterThan(0.3);            // alone it would raise the arm
    const noCollar = sampleSpeechGesture(motion, 1000, { supported: ['chest', 'leftUpperArm', 'leftForearm', 'leftHand'] })!;
    expect(Math.abs(noCollar.leftUpperArm![3])).toBeGreaterThan(0.9999);
    expect(Math.abs(noCollar.leftForearm![3])).toBeGreaterThan(0.9999);
    expect(noCollar.leftShoulder).toBeUndefined();
    const withCollar = sampleSpeechGesture(motion, 1000, { supported: ['chest', 'leftShoulder', 'leftUpperArm', 'leftForearm'] })!;
    withCollar.leftUpperArm!.forEach((v, i) => expect(v).toBeCloseTo(full.leftUpperArm![i], 6));
  });

  it('plays live chunks on the PCM clock, holds briefly when playback outruns them, and fades back in', () => {
    const chunk = (sequence: number, start: number, count: number, value = 0.4) => {
      const bytes = new Uint8Array(count * SMPLX_POSE_WIDTH * 4); const view = new DataView(bytes.buffer);
      for (let i = 0; i < count * SMPLX_POSE_WIDTH; i++) view.setFloat32(i * 4, (i % SMPLX_POSE_WIDTH) === 16 * 3 + 2 ? value : 0, true);
      return { sequence, start_frame: start, frame_count: count, poses_base64: btoa(String.fromCharCode(...bytes)) };
    };
    const live = createStreamingGesture('live:turn', 'emage');
    expect(sampleSpeechGesture(live, 100)).toBeNull();                       // nothing yet: designed movement
    appendStreamingGesture(live, chunk(0, 0, 60));
    expect(live.frames).toBe(60);
    expect(live.poses[30 * SMPLX_POSE_WIDTH + 16 * 3 + 2]).toBeCloseTo(0.4, 5);   // constant input survives the calm filter
    const lifted = (ms: number) => 2 * Math.acos(Math.min(1, Math.abs(sampleSpeechGesture(live, ms)?.leftUpperArm?.[3] ?? 1)));
    expect(lifted(1000)).toBeGreaterThan(0.2);
    expect(lifted(2100)).toBeGreaterThan(0);                                 // past the last frame: held…
    expect(lifted(2100)).toBeLessThan(lifted(1000));                        // …while fading
    expect(sampleSpeechGesture(live, 2500)).toBeNull();
    expect(() => appendStreamingGesture(live, chunk(1, 61, 10))).toThrow();   // gap in frames
    appendStreamingGesture(live, chunk(1, 60, 60), 2500);                    // arrives after playback passed it
    expect(lifted(2510)).toBeLessThan(lifted(3200) * 0.2);                   // fades in again instead of snapping
    expect(() => completeStreamingGesture(live, 100)).toThrow();
    completeStreamingGesture(live, 120);
    expect(sampleSpeechGesture(live, 4100)).toBeNull();
    const diagnostics = { kind: 'procedural', morphs: [], bones: [], missing: [] } as unknown as RigDiagnostics;
    const base = { elapsedMs: 3000, idleMs: 3000, speaking: true, audioElapsedMs: 3000, speechGesture: live };
    expect(samplePerformance({ ...base, audioSha256: 'live:turn' }, diagnostics).telemetry.boneSource).toBe('speech_generated');
    expect(samplePerformance({ ...base, audioSha256: 'live:other' }, diagnostics).telemetry.boneSource).toBe('designed');
  });

  it('hands over from the designed arm pose without passing through neutral', () => {
    const rig = createProceduralRig('jensen');
    const forearm = () => rig.root.getObjectByName('leftForearm')!.quaternion.clone();
    const designed = { ...neutralPose(), openPalm: 1 };
    const observed: [number, number, number, number] = [0, 0, Math.sin(0.3), Math.cos(0.3)];
    rig.apply(designed); const designedQ = forearm();
    rig.apply({ ...designed, bones: { leftForearm: observed }, boneWeight: 0 });
    expect(forearm().angleTo(designedQ)).toBeLessThan(1e-6);                 // weight 0 = designed pose exactly
    rig.apply({ ...designed, bones: { leftForearm: observed }, boneWeight: 1 }); const observedQ = forearm();
    rig.apply({ ...designed, bones: { leftForearm: observed }, boneWeight: 0.5 }); const half = forearm();
    expect(half.angleTo(designedQ)).toBeCloseTo(half.angleTo(observedQ), 4);  // halfway between the two, not toward neutral
    rig.apply(designed);
    expect(forearm().angleTo(designedQ)).toBeLessThan(1e-6);                 // nothing leaks into the next designed frame
    const live = clip(new Vector3(0, 0, 1), new Vector3(0, 0, 1));
    const diagnostics = { kind: 'procedural', morphs: [], bones: [], missing: [] } as unknown as RigDiagnostics;
    const early = samplePerformance({ elapsedMs: 100, idleMs: 100, speaking: true, audioElapsedMs: 100, speechGesture: live, audioSha256: sha }, diagnostics);
    expect(early.pose.boneWeight).toBeGreaterThan(0);
    expect(early.pose.boneWeight).toBeLessThan(1);
    expect(2 * Math.acos(Math.abs(early.pose.bones!.leftUpperArm![3]))).toBeGreaterThan(1);   // full-strength bones, faded by weight
  });

  it('is used only for the audio it was generated from', () => {
    const motion = clip(new Vector3(0, 0, 1), new Vector3(0, 0, 1));
    const diagnostics = { kind: 'procedural', morphs: [], bones: [], missing: [] } as unknown as RigDiagnostics;
    const base = { elapsedMs: 1000, idleMs: 1000, speaking: true, audioElapsedMs: 1000, speechGesture: motion };
    expect(samplePerformance({ ...base, audioSha256: sha }, diagnostics).telemetry.boneSource).toBe('speech_generated');
    expect(samplePerformance({ ...base, audioSha256: 'b'.repeat(64) }, diagnostics).telemetry.boneSource).toBe('designed');
  });
});

describe('speech-driven gesture on the shipped research human', () => {
  it('drives the real rig limbs in the generated directions through its calibrated neutral', async () => {
    const bytes = await readFile(new URL('../../public/models/research-human-v2.glb', import.meta.url));
    const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
    loader.register((parser) => ({ name: 'CPU_GEOMETRY_INSPECTION', loadMaterial: async (index) =>
      new MeshStandardMaterial({ name: parser.json.materials[index].name }) }));
    const imported = await loader.parseAsync(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), '');
    const mapping = createResearchHumanMapping(imported.scene);
    const rig = createRetargetedRig(imported.scene, mapping);
    const node = (control: BoneControl) => rig.root.getObjectByName((mapping.bones![control] as BoneBinding).name)!;
    const direction = (from: BoneControl, to: BoneControl) => {
      rig.root.updateMatrixWorld(true);
      return node(to).getWorldPosition(new Vector3()).sub(node(from).getWorldPosition(new Vector3())).normalize();
    };
    const apply = (motion: SpeechGestureMotion) => rig.apply({ ...neutralPose(), bones: sampleSpeechGesture(motion, 1000)! });

    rig.apply(neutralPose());
    const neutralUpper = direction('leftUpperArm', 'leftForearm'), neutralFore = direction('leftForearm', 'leftHand');
    apply(clip(upperL, foreL));
    expect(direction('leftUpperArm', 'leftForearm').dot(neutralUpper)).toBeGreaterThan(0.999);
    expect(direction('leftForearm', 'leftHand').dot(neutralFore)).toBeGreaterThan(0.995);

    // Within the rig's 70° joint limit: upper arm 60° forward, then the elbow bends the forearm 60° further up.
    const forward = upperL.clone().applyAxisAngle(new Vector3(1, 0, 0), -Math.PI / 3);
    const bent = forward.clone().applyAxisAngle(new Vector3(1, 0, 0), -Math.PI / 3);
    apply(clip(forward, bent));
    expect(direction('leftUpperArm', 'leftForearm').dot(forward)).toBeGreaterThan(0.995);
    expect(direction('leftForearm', 'leftHand').dot(bent)).toBeGreaterThan(0.99);
    // Abduction toward the subject's left (+X), 50°.
    const out = upperL.clone().applyAxisAngle(new Vector3(0, 0, 1), 50 * Math.PI / 180);
    apply(clip(out, out));
    expect(out.x).toBeGreaterThan(0.8);
    expect(direction('leftUpperArm', 'leftForearm').dot(out)).toBeGreaterThan(0.995);
    // Right arm 50° toward the subject's right (-X) and 60° forward.
    const rightOut = upperR.clone().applyAxisAngle(new Vector3(0, 0, 1), -50 * Math.PI / 180);
    apply(clip(upperL, foreL, 60, rightOut));
    expect(rightOut.x).toBeLessThan(-0.8);
    expect(direction('rightUpperArm', 'rightForearm').dot(rightOut)).toBeGreaterThan(0.995);
    const rightForward = upperR.clone().applyAxisAngle(new Vector3(1, 0, 0), -Math.PI / 3);
    apply(clip(upperL, foreL, 60, rightForward));
    expect(direction('rightUpperArm', 'rightForearm').dot(rightForward)).toBeGreaterThan(0.995);
  }, 30_000);
});
