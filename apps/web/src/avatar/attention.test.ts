import { BoxGeometry, Euler, Group, Mesh, MeshStandardMaterial, Quaternion, Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import { AttentionController, ATTENTION_SETTINGS, attentionCapabilities, composeAttentionPose, type AttentionTarget } from './attention';
import { createProceduralRig } from './procedural';
import { createRetargetedRig } from './retarget';
import { neutralPose } from './timeline';
import type { QuaternionTuple, RigDiagnostics } from './types';

const target = (now: number, x = 1, y = 0.5, id = 'selected-participant'): AttentionTarget => ({ id, x, y, observedAtMs: now });
const boneDiagnostics: RigDiagnostics = { kind: 'glb', morphs: [], bones: ['head', 'neck', 'leftEye', 'rightEye'], missing: [] };

function trackedFrame(controller: AttentionController, end: number, x = 1, y = 0.5, reduced = false) {
  let frame = controller.sample(target(0, x, y), 0, reduced);
  for (let now = 16; now <= end; now += 16) frame = controller.sample(target(now, x, y), now, reduced);
  return frame;
}

function eyeRig() {
  const model = new Group();
  const neck = new Group(); neck.name = 'Neck'; model.add(neck);
  const head = new Group(); head.name = 'Head'; neck.add(head);
  const face = new Mesh(new BoxGeometry(1, 2, 1), new MeshStandardMaterial()); head.add(face);
  const left = new Group(); left.name = 'LeftEye'; head.add(left);
  const right = new Group(); right.name = 'RightEye'; head.add(right);
  return { rig: createRetargetedRig(model), left, right, head, neck, face };
}

describe('wall-clock attention controller', () => {
  it('moves the eyes first and keeps the attention head and neck still for 220 ms', () => {
    const controller = new AttentionController();
    let frame = controller.sample(target(0), 0);
    for (let now = 16; now < ATTENTION_SETTINGS.headDelayMs; now += 16) {
      frame = controller.sample(target(now), now);
      const { pose, telemetry } = composeAttentionPose(neutralPose(), frame, boneDiagnostics);
      expect(frame.phase).toBe('eyes'); expect(frame.headYaw).toBe(0); expect(frame.headWeight).toBe(0);
      expect(telemetry.headYaw).toBe(0); expect(pose.bones?.head).toBeUndefined(); expect(pose.bones?.neck).toBeUndefined();
    }
    expect(frame.lookYaw).toBeGreaterThan(0.5);
    const eyes = composeAttentionPose(neutralPose(), frame, boneDiagnostics).pose.bones!.leftEye!;
    expect(new Quaternion(...eyes).angleTo(new Quaternion())).toBeGreaterThan(0.25);
    frame = controller.sample(target(224), 224);
    expect(frame.phase).toBe('following'); expect(frame.headYaw).toBeGreaterThan(0);
  });

  it('compensates the eye direction as the head catches up and clamps the gesture envelope', () => {
    const frame = trackedFrame(new AttentionController(), 3000);
    const { telemetry } = composeAttentionPose(neutralPose(), frame, boneDiagnostics);
    expect(telemetry.headYaw).toBeGreaterThan(0.40);
    expect(telemetry.headYaw).toBeLessThanOrEqual(ATTENTION_SETTINGS.maxHeadYaw);
    expect(telemetry.eyeYaw).toBeLessThan(0.20);
    expect(telemetry.eyeYaw + telemetry.headYaw).toBeCloseTo(ATTENTION_SETTINGS.targetYaw, 3);
    expect(Math.abs(frame.headPitch)).toBeLessThanOrEqual(ATTENTION_SETTINGS.maxHeadPitch);
  });

  it('does not restart the head delay for new detections of the selected target', () => {
    const controller = new AttentionController();
    controller.sample(target(0), 0);
    for (let now = 10; now < 500; now += 10) controller.sample(target(now, 0.8 + Math.sin(now) * 0.01), now);
    expect(controller.sample(target(500), 500).headYaw).toBeGreaterThan(0.1);
  });

  it('gives a newly selected person an eye lead while holding the previous head position', () => {
    const controller = new AttentionController(); const before = trackedFrame(controller, 1008);
    const switched = controller.sample(target(1024, 0, 0.5, 'new-participant'), 1024);
    expect(switched.phase).toBe('eyes'); expect(switched.headYaw).toBe(before.headYaw);
    let frame = switched;
    for (let now = 1040; now < 1244; now += 16) frame = controller.sample(target(now, 0, 0.5, 'new-participant'), now);
    expect(frame.lookYaw).toBeLessThan(0); expect(frame.headYaw).toBe(before.headYaw);
    frame = controller.sample(target(1260, 0, 0.5, 'new-participant'), 1260);
    expect(frame.headYaw).toBeLessThan(before.headYaw);
  });

  it('returns smoothly after stop or stale input, then relinquishes channel ownership', () => {
    const controller = new AttentionController(); const before = trackedFrame(controller, 1008);
    const first = controller.sample(null, 1024);
    expect(first.phase).toBe('returning'); expect(first.targetId).toBeNull();
    expect(first.headYaw).toBeGreaterThan(0); expect(first.headYaw).toBeLessThan(before.headYaw);
    let frame = first;
    for (let now = 1040; now < 4000; now += 16) frame = controller.sample(null, now);
    expect(frame.phase).toBe('idle'); expect(frame.eyeWeight).toBe(0);
    const original = { ...neutralPose(), gazeX: 0.11 };
    expect(composeAttentionPose(original, frame, boneDiagnostics).pose).toBe(original);
    const stale = new AttentionController(); stale.sample(target(0), 0);
    expect(stale.sample(target(0), ATTENTION_SETTINGS.maxTargetAgeMs + 1).phase).toBe('returning');
  });

  it('rejects future, nonfinite, out-of-range and old observations without producing NaN', () => {
    for (const invalid of [{ ...target(100), observedAtMs: 101 }, { ...target(100), x: NaN },
      { ...target(100), y: -0.1 }, { ...target(100), id: '' }, { ...target(100), x: Infinity }]) {
      const frame = new AttentionController().sample(invalid, 100);
      expect(frame.phase).toBe('idle'); expect(frame.headYaw).toBe(0);
    }
    const controller = new AttentionController(); trackedFrame(controller, 1008);
    expect(controller.sample(target(900, 0), 1024).phase).toBe('returning');
    expect(controller.sample(target(1024), NaN).phase).toBe('idle');
  });

  it('uses bounded eye movement and no new head motion in reduced-motion mode', () => {
    const frame = trackedFrame(new AttentionController(), 2000, 1, 0, true);
    const { telemetry } = composeAttentionPose(neutralPose(), frame, boneDiagnostics);
    expect(frame.phase).toBe('reduced'); expect(frame.headYaw).toBe(0); expect(frame.headPitch).toBe(0);
    expect(Math.abs(telemetry.eyeYaw)).toBeLessThanOrEqual(ATTENTION_SETTINGS.reducedEyeAngle);
    expect(Math.abs(telemetry.eyePitch)).toBeLessThanOrEqual(ATTENTION_SETTINGS.reducedEyeAngle);
  });

  it('handles clock resets and resumes at a bounded wall-clock pose after a long gap', () => {
    const controller = new AttentionController(); trackedFrame(controller, 1008);
    const reset = controller.sample(target(0, 0), 0);
    expect(reset.phase).toBe('eyes'); expect(reset.headYaw).toBe(0);
    const resumed = controller.sample(target(60000, 1, 1), 60000);
    expect(resumed.phase).toBe('following');
    expect(resumed.headYaw).toBe(ATTENTION_SETTINGS.maxHeadYaw);
    expect(resumed.headPitch).toBe(ATTENTION_SETTINGS.maxHeadPitch);
    expect(resumed.eyeWeight).toBe(1); expect(resumed.headWeight).toBe(1);
    const switched = controller.sample(target(120000, 0, 0, 'new-participant'), 120000);
    expect(switched.phase).toBe('eyes'); expect(switched.headYaw).toBe(resumed.headYaw);
    // A long render gap cannot keep an expired observation active.
    expect(controller.sample(target(120000, 0, 0, 'new-participant'), 180000).phase).toBe('idle');
    controller.reset(); expect(controller.sample(null, 180001).phase).toBe('idle');
  });

  it('uses only time after the eye-lead deadline when a sparse frame crosses it', () => {
    const sparse = new AttentionController(); sparse.sample(target(0), 0);
    const dense = new AttentionController();
    for (const now of [0, 100, 200, 220]) dense.sample(target(now), now);
    const expected = dense.sample(target(250), 250);
    const actual = sparse.sample(target(250), 250);
    expect(actual.phase).toBe('following');
    expect(actual.headYaw).toBeGreaterThan(0); expect(actual.headYaw).toBeLessThan(0.05);
    expect(actual.headYaw).toBeCloseTo(expected.headYaw, 10);
    expect(actual.headWeight).toBeCloseTo(expected.headWeight, 10);
  });

  it('follows and returns on the same wall clock at low, high and irregular render rates', () => {
    const run = (times: number[]) => {
      const controller = new AttentionController(); controller.sample(target(0), 0);
      let frame = controller.sample(target(0), 0);
      for (const now of times) frame = controller.sample(target(now), now);
      const followed = composeAttentionPose(neutralPose(), frame, boneDiagnostics).telemetry;
      const firstReturn = controller.sample(null, 2000);
      for (const now of times) frame = controller.sample(null, now + 2000);
      return { followed, firstReturn, returned: frame };
    };
    const times = (hz: number) => Array.from({ length: hz * 2 }, (_, i) => (i + 1) * 1000 / hz);
    const sixty = run(times(60));
    for (const schedule of [...[1, 2, 3, 5, 15, 30, 120].map(times), [180, 750, 940, 1500, 2000]]) {
      const actual = run(schedule);
      expect(actual.followed.headYaw).toBeCloseTo(sixty.followed.headYaw, 8);
      expect(actual.followed.eyeYaw).toBeCloseTo(sixty.followed.eyeYaw, 8);
      expect(actual.firstReturn.phase).toBe('returning');
      expect(actual.firstReturn.headYaw).toBeGreaterThan(0.4);
      expect(actual.returned.phase).toBe('idle');
      expect(actual.returned.eyeWeight).toBe(0); expect(actual.returned.headWeight).toBe(0);
    }
  });

  it('turns toward the selected screen face at one rendered frame per second', () => {
    const controller = new AttentionController(); controller.sample(target(0, 0.2, 0.375), 0);
    const frame = controller.sample(target(1000, 0.2, 0.375), 1000);
    const { telemetry } = composeAttentionPose(neutralPose(), frame, boneDiagnostics);
    expect(telemetry.headYaw).toBeLessThan(-0.18);
    expect(telemetry.eyeYaw).toBeLessThan(0);
    expect(Math.abs(telemetry.headYaw)).toBeLessThanOrEqual(ATTENTION_SETTINGS.maxHeadYaw);
  });
});

describe('attention and rig composition', () => {
  it('reports the actual composed eye bones while blending from recorded gaze', () => {
    const recorded = new Quaternion().setFromEuler(new Euler(-0.12, -0.25, 0.03)).toArray() as QuaternionTuple;
    const base = { ...neutralPose(), bones: { leftEye: recorded, rightEye: recorded } };
    const frame = new AttentionController().sample(target(0), 0);
    const { pose, telemetry } = composeAttentionPose(base, frame, boneDiagnostics);
    const actual = new Euler().setFromQuaternion(new Quaternion(...pose.bones!.leftEye!));
    expect(telemetry.eyeYaw).toBeCloseTo(actual.y, 8);
    expect(telemetry.eyePitch).toBeCloseTo(actual.x, 8);
    expect(telemetry.eyeYaw).toBeLessThan(0);
  });

  it('preserves speech, blinks, expressions and arm motion while overriding recorded gaze/head', () => {
    const base = { ...neutralPose(), jawOpen: 0.7, blink: 0.4, mouthSmile: 0.3, openPalm: 0.8,
      face: { jawOpen: 0.7, eyeBlinkLeft: 0.8, browInnerUp: 0.2, eyeLookOutLeft: 0.9 },
      bones: { head: [0, -0.2, 0, Math.sqrt(0.96)] as QuaternionTuple,
        leftEye: [0, -0.1, 0, Math.sqrt(0.99)] as QuaternionTuple,
        leftUpperArm: [0.1, 0, 0, Math.sqrt(0.99)] as QuaternionTuple }, visemes: { aa: 0.7 } };
    const snapshot = JSON.stringify(base);
    const frame = trackedFrame(new AttentionController(), 3000);
    const { pose } = composeAttentionPose(base, frame, { ...boneDiagnostics, morphs: ['eyeLookOutLeft'] });
    expect(pose.visemes).toEqual(base.visemes); expect(pose.jawOpen).toBe(0.7); expect(pose.blink).toBe(0.4);
    expect(pose.face?.eyeBlinkLeft).toBe(0.8); expect(pose.face?.browInnerUp).toBe(0.2);
    expect(pose.openPalm).toBe(0.8); expect(pose.bones?.leftUpperArm).toEqual(base.bones.leftUpperArm);
    expect(pose.bones?.head?.[1]).toBeGreaterThan(0); expect(pose.bones?.leftEye?.[1]).toBeGreaterThan(0);
    expect(pose.face?.eyeLookOutLeft).toBeUndefined(); expect(JSON.stringify(base)).toBe(snapshot);
  });

  it('moves actual GLB eye bones right and up without an early head rotation', () => {
    const { rig, left, right, head, neck } = eyeRig();
    const frame = trackedFrame(new AttentionController(), 160, 1, 0);
    rig.apply(composeAttentionPose(neutralPose(), frame, rig.diagnostics).pose);
    rig.root.updateMatrixWorld(true);
    for (const eye of [left, right]) {
      const direction = new Vector3(0, 0, 1).applyQuaternion(eye.getWorldQuaternion(new Quaternion()));
      expect(direction.x).toBeGreaterThan(0.1); expect(direction.y).toBeGreaterThan(0.1);
    }
    expect(head.quaternion.angleTo(new Quaternion())).toBe(0);
    expect(neck.quaternion.angleTo(new Quaternion())).toBe(0); rig.dispose();
  });

  it('points GLB eyes at the target after the head turns, without double counting the neck', () => {
    const { rig, left, right, head } = eyeRig();
    const frame = trackedFrame(new AttentionController(), 3000);
    rig.apply(composeAttentionPose(neutralPose(), frame, rig.diagnostics).pose); rig.root.updateMatrixWorld(true);
    expect(head.rotation.y).toBeGreaterThan(0.3);
    for (const eye of [left, right]) {
      const direction = new Vector3(0, 0, 1).applyQuaternion(eye.getWorldQuaternion(new Quaternion()));
      expect(Math.atan2(direction.x, direction.z)).toBeCloseTo(ATTENTION_SETTINGS.targetYaw, 3);
    }
    rig.dispose();
  });

  it.each([[1, 0, 1, 1], [0, 1, -1, -1]])('moves procedural irises consistently at target (%s, %s)', (x, y, signX, signY) => {
    const rig = createProceduralRig();
    const frame = trackedFrame(new AttentionController(), 160, x, y);
    rig.apply(composeAttentionPose(neutralPose(), frame, rig.diagnostics).pose);
    for (const side of ['Left', 'Right']) {
      const iris = rig.root.getObjectByName(`iris${side}`)!;
      expect(iris.position.x * signX).toBeGreaterThan(0.005);
      expect(iris.position.y * signY).toBeGreaterThan(0.004);
    }
    expect(rig.root.getObjectByName('head')!.quaternion.angleTo(new Quaternion())).toBe(0); rig.dispose();
  });

  it('uses morph-only eyes and reports rigs missing part of the required controls', () => {
    const allEyes = ['Left', 'Right'].flatMap((side) => ['In', 'Out', 'Up', 'Down'].map((direction) => `eyeLook${direction}${side}`));
    const diagnostics: RigDiagnostics = { kind: 'glb', bones: [], morphs: allEyes, missing: [] };
    expect(attentionCapabilities(diagnostics)).toEqual({ eyes: true, head: false, unsupported: ['head/neck turn'] });
    const controller = new AttentionController(); let frame = controller.sample(target(0), 0, false, false);
    for (let now = 16; now <= 2000; now += 16) frame = controller.sample(target(now), now, false, false);
    const { pose, telemetry } = composeAttentionPose(neutralPose(), frame, diagnostics);
    expect(frame.headYaw).toBe(0); expect(pose.face?.eyeLookOutLeft).toBeGreaterThan(0.9);
    expect(pose.face?.eyeLookInRight).toBeGreaterThan(0.9); expect(pose.face?.eyeLookInLeft).toBe(0);
    expect(telemetry.unsupported).toContain('head/neck turn');
    expect(attentionCapabilities({ ...diagnostics, morphs: [] }).unsupported).toEqual(['left eye gaze', 'right eye gaze', 'head/neck turn']);
  });
});
