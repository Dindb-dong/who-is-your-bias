import { readFile } from 'node:fs/promises';
import { MeshStandardMaterial, Quaternion, Vector3 } from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { describe, expect, it } from 'vitest';
import { createResearchHumanMapping } from './builtin';
import { createProceduralRig } from './procedural';
import { samplePerformance } from './performance';
import { createRetargetedRig } from './retarget';
import { appendStreamingGesture, calmPoses, completeStreamingGesture, createStreamingGesture, parseSpeechGestureMotion, sampleSpeechGesture,
  SMPLX_PARENTS, SMPLX_POSE_WIDTH, speechGestureReference, type SpeechGestureMotion } from './speechGesture';
import { neutralPose } from './timeline';
import type { BoneBinding, BoneControl, RigDiagnostics } from './types';

const sha = 'a'.repeat(64);
const reference = speechGestureReference();
const X = new Vector3(1, 0, 0), Z = new Vector3(0, 0, 1);
const turn = (axis: Vector3, degrees: number) => new Quaternion().setFromAxisAngle(axis, degrees * Math.PI / 180);

function setJoint(poses: Float32Array, frame: number, joint: number, q: Quaternion) {
  const angle = 2 * Math.acos(Math.min(1, Math.abs(q.w))), s = Math.sqrt(Math.max(0, 1 - q.w * q.w)), sign = q.w < 0 ? -1 : 1;
  const o = frame * SMPLX_POSE_WIDTH + joint * 3;
  if (s < 1e-9) { poses[o] = poses[o + 1] = poses[o + 2] = 0; return; }
  poses[o] = sign * q.x / s * angle; poses[o + 1] = sign * q.y / s * angle; poses[o + 2] = sign * q.z / s * angle;
}

/** A clip at the reference posture (what leaves the rig at its neutral), with world-axis turns applied at the
 * given joints; children follow their parents, as in SMPL-X. */
function clip(changes: Record<number, Quaternion> = {}, frames = 60): SpeechGestureMotion {
  const poses = new Float32Array(frames * SMPLX_POSE_WIDTH), global: Quaternion[] = [], local: Quaternion[] = [];
  for (let j = 0; j < 55; j++) {
    const parent = j ? global[SMPLX_PARENTS[j]] : new Quaternion();
    const g = parent.clone().multiply(reference.local[j]);
    if (changes[j]) g.premultiply(changes[j]);
    global[j] = g; local[j] = parent.clone().invert().multiply(g);
  }
  for (let f = 0; f < frames; f++) for (let j = 0; j < 55; j++) setJoint(poses, f, j, local[j]);
  return { fps: 30, frames, poses, audioSha256: sha, engine: 'test' };
}

const asQuaternion = (q: [number, number, number, number] | undefined) => new Quaternion(...(q ?? [0, 0, 0, 1]));

describe('speech-driven gesture conversion', () => {
  it('maps the EMAGE average posture to the rig neutral and follows changes around it', () => {
    const rest = sampleSpeechGesture(clip(), 1000)!;
    for (const [control, q] of Object.entries(rest)) expect(Math.abs(q![3]), control).toBeGreaterThan(0.9999);
    // The average posture leans the torso forward and lifts the arms: none of that reaches the rig.
    expect(2 * Math.acos(Math.abs(reference.local[3].w))).toBeGreaterThan(0.2);
    const raise = turn(X, -60);
    const raised = sampleSpeechGesture(clip({ 16: raise }), 1000)!;
    expect(asQuaternion(raised.leftUpperArm).angleTo(raise)).toBeLessThan(1e-4);
    expect(Math.abs(raised.rightUpperArm![3])).toBeGreaterThan(0.9999);
  });

  it('fades in and out, is silent outside the clip, and damps for reduced motion', () => {
    const motion = clip({ 16: turn(X, -60) });
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
    const collar = turn(Z, -20);                                                // collar lowers the arm…
    const motion = clip({ 13: collar, 16: collar.clone().invert() });             // …and the shoulder compensates
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
    const live = clip({ 16: turn(X, -65) });
    const diagnostics = { kind: 'procedural', morphs: [], bones: [], missing: [] } as unknown as RigDiagnostics;
    const early = samplePerformance({ elapsedMs: 100, idleMs: 100, speaking: true, audioElapsedMs: 100, speechGesture: live, audioSha256: sha }, diagnostics);
    expect(early.pose.boneWeight).toBeGreaterThan(0);
    expect(early.pose.boneWeight).toBeLessThan(1);
    expect(2 * Math.acos(Math.abs(early.pose.bones!.leftUpperArm![3]))).toBeGreaterThan(1);   // full-strength bones, faded by weight
  });

  it('is used only for the audio it was generated from', () => {
    const motion = clip({ 16: turn(X, -60) });
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
    const neutral = { lu: direction('leftUpperArm', 'leftForearm'), lf: direction('leftForearm', 'leftHand'),
      ru: direction('rightUpperArm', 'rightForearm') };
    apply(clip());
    expect(direction('leftUpperArm', 'leftForearm').dot(neutral.lu)).toBeGreaterThan(0.999);
    expect(direction('leftForearm', 'leftHand').dot(neutral.lf)).toBeGreaterThan(0.995);

    // Within the rig's 70° joint limit: upper arm 60° forward, then the elbow lifts the forearm 60° further.
    const forward = turn(X, -60), lift = turn(X, -60);
    apply(clip({ 16: forward, 18: lift }));
    expect(direction('leftUpperArm', 'leftForearm').dot(neutral.lu.clone().applyQuaternion(forward))).toBeGreaterThan(0.995);
    expect(direction('leftForearm', 'leftHand').dot(neutral.lf.clone().applyQuaternion(lift.clone().multiply(forward)))).toBeGreaterThan(0.99);
    // Abduction toward the subject's left (+X) and, for the right arm, toward the subject's right (-X), 50°.
    const outLeft = turn(Z, 50), outRight = turn(Z, -50);
    apply(clip({ 16: outLeft, 17: outRight }));
    expect(direction('leftUpperArm', 'leftForearm').dot(neutral.lu.clone().applyQuaternion(outLeft))).toBeGreaterThan(0.995);
    expect(direction('rightUpperArm', 'rightForearm').dot(neutral.ru.clone().applyQuaternion(outRight))).toBeGreaterThan(0.995);
    expect(direction('rightUpperArm', 'rightForearm').x).toBeLessThan(neutral.ru.x);
    apply(clip({ 17: forward }));
    expect(direction('rightUpperArm', 'rightForearm').dot(neutral.ru.clone().applyQuaternion(forward))).toBeGreaterThan(0.995);
  }, 30_000);
});
