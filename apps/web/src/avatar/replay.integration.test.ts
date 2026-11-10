import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { Mesh, Quaternion } from 'three';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { sampleMotion } from './motion';
import { samplePerformance } from './performance';
import { createProceduralRig } from './procedural';
import { createRetargetedRig } from './retarget';
import { neutralPose } from './timeline';
import { parseAvatarClip } from './validation';

class NodeFileReader {
  result: ArrayBuffer | null = null;
  onloadend: (() => void) | null = null;
  readAsArrayBuffer(blob: Blob): void { void blob.arrayBuffer().then((value) => { this.result = value; this.onloadend?.(); }); }
}
afterEach(() => vi.unstubAllGlobals());

describe('actual measured sample replay through binary glTF', () => {
  it('replays the fixed-ROI body measurements on GLB joints while preserving crop lineage and separate facial provenance', async () => {
    const clip = parseAvatarClip(JSON.parse(await readFile(new URL('../../public/samples/ted-2013-body-geometry-preview.json', import.meta.url), 'utf8')), true);
    expect(clip.source.video_crop).toEqual({ x: 170, y: 50, width: 280, height: 350, input_width: 854, input_height: 480, coordinate_basis: 'input_pixels_before_scale', mirrored: false });
    expect(clip.source.input_sha256).toHaveLength(64);
    expect(clip.source.preparation_manifest_sha256).toHaveLength(64);
    expect(clip.frames).toHaveLength(40);
    expect(clip.frames.every((frame) => !frame.face && frame.bones?.leftForearm && frame.bones?.rightForearm)).toBe(true);
    vi.stubGlobal('FileReader', NodeFileReader);
    const original = createProceduralRig(); original.apply(neutralPose());
    const binary = await new GLTFExporter().parseAsync(original.root, { binary: true }) as ArrayBuffer;
    const rig = createRetargetedRig((await new GLTFLoader().parseAsync(binary, '')).scene);
    const names = ['chest', 'leftUpperArm', 'leftForearm', 'leftHand', 'rightUpperArm', 'rightForearm', 'rightHand'] as const;
    const reference = Object.fromEntries(names.map((name) => [name, rig.root.getObjectByName(name)!.quaternion.clone()]));
    const maximumRotation = Object.fromEntries(names.map((name) => [name, 0]));
    for (const frame of clip.frames) {
      const sample = samplePerformance({ speaking: false, elapsedMs: 0, idleMs: 0, observationElapsedMs: frame.time_ms, performanceClip: clip, allowGeometryPreview: true }, rig.diagnostics);
      expect(sample.telemetry.faceSource).toBe('designed');
      expect(sample.telemetry.boneSource).toBe('geometry_preview');
      rig.apply(sample.pose);
      for (const name of names) {
        const rotation = rig.root.getObjectByName(name)!.quaternion;
        expect(rotation.length()).toBeCloseTo(1);
        maximumRotation[name] = Math.max(maximumRotation[name], rotation.angleTo(reference[name]));
      }
    }
    expect(maximumRotation.leftForearm).toBeGreaterThan(0.8);
    expect(maximumRotation.rightForearm).toBeGreaterThan(0.4);
    expect(Object.values(maximumRotation).every((angle) => angle < 70 * Math.PI / 180)).toBe(true);
    const output = process.env.BIAS_AVATAR_QA_DIR;
    if (output) {
      await mkdir(output, { recursive: true });
      const largest = clip.frames.reduce((previous, frame) => Math.abs(frame.bones!.leftForearm![3]) < Math.abs(previous.bones!.leftForearm![3]) ? frame : previous);
      rig.apply({ ...neutralPose(), ...sampleMotion(clip, largest.time_ms, true)! });
      await writeFile(join(output, 'body-measured.glb'), new Uint8Array(await new GLTFExporter().parseAsync(rig.root, { binary: true }) as ArrayBuffer));
      await writeFile(join(output, 'body-replay-report.json'), JSON.stringify({ source: clip.source, measurement: clip.measurement, frames: clip.frames.length,
        review: clip.review.status, maximumRotationRadians: maximumRotation, faceFrames: 0, rendering: 'GLB joint roundtrip; no identity, browser or likeness validation' }, null, 2));
    }
    rig.dispose(); original.dispose();
  });

  it('imports the measured preview and applies its face and head tracks to a real GLB roundtrip', async () => {
    const source = JSON.parse(await readFile(new URL('../../public/samples/ted-2013-geometry-preview.json', import.meta.url), 'utf8'));
    expect(() => parseAvatarClip(source)).toThrow(/preview/);
    const clip = parseAvatarClip(source, true);
    expect(clip.measurement?.engine).toBe('mediapipe');
    expect(clip.review.speaker_association_reviewed).toBe(false);
    expect(clip.frames.length).toBeGreaterThan(30);
    expect(clip.frames.every((frame) => !frame.bones?.leftForearm && !frame.bones?.rightForearm)).toBe(true);

    vi.stubGlobal('FileReader', NodeFileReader);
    const original = createProceduralRig(); original.apply(neutralPose());
    const binary = await new GLTFExporter().parseAsync(original.root, { binary: true }) as ArrayBuffer;
    const loaded = await new GLTFLoader().parseAsync(binary, '');
    const rig = createRetargetedRig(loaded.scene);
    expect(rig.diagnostics.morphs).toContain('jawOpen');
    // Procedural eye scaling is not a GLB morph target; honest diagnostics must say so.
    expect(rig.diagnostics.unsupportedFace).toContain('eyeBlinkLeft');
    const head = rig.root.getObjectByName('head')!;
    const face = rig.root.getObjectsByProperty('isMesh', true).find((object) => (object as Mesh).morphTargetDictionary?.jawOpen !== undefined) as Mesh;
    const jawIndex = face.morphTargetDictionary!.jawOpen;
    const reference = sampleMotion(clip, clip.calibration.neutral_frame_ms, true)!;
    rig.apply({ ...neutralPose(), ...reference });
    const referenceRotation = head.quaternion.clone();
    let greatestRotation = 0; let minimumJaw = 1; let maximumJaw = 0;
    for (const frame of clip.frames) {
      const sample = sampleMotion(clip, frame.time_ms, true)!;
      rig.apply({ ...neutralPose(), ...sample });
      greatestRotation = Math.max(greatestRotation, head.quaternion.angleTo(referenceRotation));
      minimumJaw = Math.min(minimumJaw, face.morphTargetInfluences![jawIndex]);
      maximumJaw = Math.max(maximumJaw, face.morphTargetInfluences![jawIndex]);
      expect(head.quaternion.length()).toBeCloseTo(1);
    }
    expect(greatestRotation).toBeGreaterThan(0.4);
    expect(maximumJaw - minimumJaw).toBeGreaterThan(0.01);
    expect(sampleMotion(clip, 3000)).toBeNull(); // Preview cannot escape its gate.

    // Optional offline geometry artifacts. Normal CI writes no files.
    const output = process.env.BIAS_AVATAR_QA_DIR;
    if (output) {
      await mkdir(output, { recursive: true });
      await writeFile(join(output, 'neutral.glb'), new Uint8Array(binary));
      rig.apply({ ...neutralPose(), ...sampleMotion(clip, clip.frames[clip.frames.length - 1].time_ms, true)! });
      const measuredBinary = await new GLTFExporter().parseAsync(rig.root, { binary: true }) as ArrayBuffer;
      await writeFile(join(output, 'measured.glb'), new Uint8Array(measuredBinary));
      original.apply({ ...neutralPose(), face: { jawOpen: 0.55, mouthFunnel: 0.15, mouthSmileLeft: 0.15 } });
      const speechBinary = await new GLTFExporter().parseAsync(original.root, { binary: true }) as ArrayBuffer;
      await writeFile(join(output, 'articulation.glb'), new Uint8Array(speechBinary));
      await writeFile(join(output, 'replay-report.json'), JSON.stringify({ source: clip.source, measurement: clip.measurement,
        frames: clip.frames.length, review: clip.review.status, greatestRotationRadians: greatestRotation,
        jawRange: [minimumJaw, maximumJaw], diagnostics: rig.diagnostics, rendering: 'GLB data roundtrip; no browser rendering or likeness evaluation' }, null, 2));
    }
    expect(new Quaternion(...clip.frames[clip.frames.length - 1].bones!.head!).length()).toBeCloseTo(1);
    rig.dispose(); original.dispose();
  });
});
