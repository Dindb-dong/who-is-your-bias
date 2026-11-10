# Avatar runtime

The built-in portrait is a **stylized procedural prototype**, not a scan or a
photorealistic reconstruction of Elon Musk. Its expressions and body movements
are designed animation. Nothing in this renderer infers a person's mental state
or establishes that a movement is characteristic of that person.

## Runtime contract

`AvatarStage` accepts `speaking`, `timeline`, `elapsedMs`, `amplitude`, optional
`modelUrl`, `rigMapping`, and `reducedMotion` props. `onReady` returns the controls
actually found on the active rig; `onError` receives an `Error`.

- `elapsedMs` is the transport clock, not a timer started when the server replies.
- `amplitude` is normalized audio energy in `[0, 1]`. Zero means silence. Omitting
  it while speaking enables designed articulation; this is not phoneme timing.
- A plan marked `timing_basis: "estimated_text"` uses estimated cue times. If the
  speech duration is known, the transport may map actual audio time into the
  plan's duration. Such scaling still does not make the plan an observed motion.
- Each cue has `start_ms`, `duration_ms`, `channel`, `name`, `intensity`, and
  `provenance`. The renderer's smooth envelopes do not alter provenance.
- The user/system reduced-motion preference disables idle motion and reduces
  the amplitude of nonverbal cues. Mouth articulation remains available.

Supported cue names:

| Channel | Names |
| --- | --- |
| `face` | `smile`, `thoughtful`, `brow_raise` |
| `gaze` | `center`, `aversion`, `left`, `right`, `down` |
| `head` | `nod`, `tilt` |
| `gesture` | `open_palm`, `emphasis`, `shrug`, `wave` |

## Importing a finished rig

Use a self-contained, uncompressed binary glTF 2.0 (`.glb`) file with embedded
textures and buffers. Draco/KTX2 compressed assets need decoders not bundled
here. There are no automatic external model or decoder downloads. Models can be
selected locally through an object URL or served from the application's origin.
The runtime rejects files over 64 MiB; the app may impose a smaller upload limit.

The retargeter detects common ARKit-style morph names and conventional head,
neck, jaw, arm, forearm, and hand node names. It applies rotations relative to
the authored rest pose. Missing controls are returned in the diagnostics; they
are not synthesized. A rig without mapped facial targets can display successfully
while remaining unable to animate its face. Asset authoring and calibration are
still necessary for a high-fidelity portrait.

For a custom asset, pass an explicit mapping:

```ts
const mapping = {
  morphs: {
    jawOpen: { target: 'Jaw_Open', mesh: 'Face', scale: 0.8 },
    eyeBlinkLeft: 'Blink_L',
    eyeBlinkRight: 'Blink_R',
    mouthSmileLeft: 'Smile_L',
    mouthSmileRight: 'Smile_R',
  },
  bones: {
    head: 'HeadJoint',
    leftForearm: { name: 'Elbow_L', axis: 'z', scale: -1 },
  },
  yaw: Math.PI,
};
```

A jaw morph takes precedence over a jaw bone to avoid double deformation.
Retargeting conventions differ between assets; inspect the result before
claiming correct facial or anatomical motion. The built-in procedural rig
implements equivalent normalized controls through articulated mesh groups and
authored face/lip morph targets; its geometry is not a face scan.

## Lifecycle and verification

The stage owns one renderer per mount, supports React StrictMode cleanup, aborts
obsolete model downloads, disposes decoded images and GPU resources, stops
rendering while the page is hidden, and reports WebGL context interruption with
a reload action. Pixel ratio is capped to limit rendering cost. Dragging changes
the view within a narrow portrait range; wheel scrolling is not intercepted.

Tests cover cue boundaries, silence, replay determinism, reduced motion, finite
geometry, real joint movement, rest-pose preservation, source-unit normalization,
explicit mapping, and model URL restrictions. A GLTFExporter/GLTFLoader
roundtrip test also checks facial morph targets and the articulated node
hierarchy. The same geometry was inspected with an offline Blender CPU render;
that does not verify browser appearance, interactive rendering, or likeness.
GPU appearance still requires a browser capable of WebGL 2.

Implementation references:

- [Three.js GLTFLoader](https://threejs.org/docs/pages/GLTFLoader.html)
- [Three.js WebGLRenderer](https://threejs.org/docs/pages/WebGLRenderer.html)
- [Three.js mesh morph targets](https://threejs.org/docs/pages/Mesh.html)
