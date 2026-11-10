import { Mesh } from 'three';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { disposeObject } from './dispose';
import { createProceduralRig } from './procedural';
import { neutralPose } from './timeline';

// The exporter needs FileReader for its binary Blob, not a browser renderer.
class NodeFileReader {
  result: ArrayBuffer | null = null;
  onloadend: (() => void) | null = null;
  readAsArrayBuffer(blob: Blob): void {
    void blob.arrayBuffer().then((result) => { this.result = result; this.onloadend?.(); });
  }
}

afterEach(() => vi.unstubAllGlobals());

describe('procedural GLB export roundtrip', () => {
  it('preserves geometry, facial morph targets and articulated node hierarchy', async () => {
    vi.stubGlobal('FileReader', NodeFileReader);
    const rig = createProceduralRig(); rig.apply(neutralPose());
    const bytes = await new GLTFExporter().parseAsync(rig.root, { binary: true });
    expect(bytes).toBeInstanceOf(ArrayBuffer);
    const binary = bytes as ArrayBuffer;
    expect(new DataView(binary).getUint32(0, true)).toBe(0x46546c67);
    const imported = await new GLTFLoader().parseAsync(binary, '');
    expect(imported.scene.getObjectByName('head')).toBeDefined();
    expect(imported.scene.getObjectByName('leftHand')?.parent?.name).toBe('leftForearm');
    expect(imported.scene.getObjectByName('jaw')?.parent?.name).toBe('mouth');
    let faceMorphs = 0;
    imported.scene.traverse((object) => {
      if (!(object instanceof Mesh)) return;
      if (object.morphTargetDictionary?.jawOpen !== undefined) {
        faceMorphs++;
        expect(object.morphTargetDictionary.mouthSmile).toBeDefined();
        expect(object.morphTargetDictionary.browRaise).toBeDefined();
      }
    });
    expect(faceMorphs).toBe(1);
    disposeObject(imported.scene); rig.dispose();
  });
});
