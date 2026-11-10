import {
  BufferGeometry, CatmullRomCurve3, Color, Float32BufferAttribute,
  Group, LatheGeometry, Mesh, MeshStandardMaterial, SphereGeometry, TorusGeometry, TubeGeometry,
  Vector2, Vector3,
} from 'three';
import { disposeObject } from './dispose';
import type { RigDiagnostics, RigPose } from './types';

export interface AvatarRig {
  root: Group;
  diagnostics: RigDiagnostics;
  apply: (pose: RigPose) => void;
  dispose: () => void;
}

type Material = MeshStandardMaterial;

const ellipsoid = (material: Material, scale: [number, number, number], at: [number, number, number], segments = 32): Mesh => {
  const mesh = new Mesh(new SphereGeometry(1, segments, Math.max(12, segments / 2)), material);
  mesh.scale.set(...scale); mesh.position.set(...at); mesh.castShadow = true; mesh.receiveShadow = true;
  return mesh;
};

function curveMesh(points: [number, number, number][], radius: number, material: Material, segments = 20): Mesh {
  const curve = new CatmullRomCurve3(points.map((p) => new Vector3(...p)));
  const mesh = new Mesh(new TubeGeometry(curve, segments, radius, 6, false), material);
  mesh.castShadow = true;
  return mesh;
}

function faceGeometry(): BufferGeometry {
  const geometry = new SphereGeometry(1, 64, 48);
  const positions = geometry.attributes.position;
  const colors: number[] = [];
  const skin = new Color('#d6ae95');
  const blush = new Color('#b97a70');
  for (let i = 0; i < positions.count; i++) {
    const rawX = positions.getX(i); const rawY = positions.getY(i); const rawZ = positions.getZ(i);
    // A broad forehead, gently flattened cheeks, tapered lower jaw and chin.
    const taper = rawY < -0.12 ? 1 - 0.09 * Math.pow((-rawY - 0.12) / 0.88, 1.3) : 1;
    const x = rawX * 0.365 * taper;
    const y = rawY * 0.48;
    let z = rawZ * 0.327;
    if (rawZ > 0) {
      const eyeSocket = Math.exp(-Math.pow((Math.abs(x) - 0.137) / 0.071, 2)
        - Math.pow((y - 0.085) / 0.044, 2));
      z -= eyeSocket * 0.022;
      z += Math.exp(-Math.pow(x / 0.22, 2) - Math.pow((y + 0.21) / 0.11, 2)) * 0.018;
    }
    positions.setXYZ(i, x, y, z);
    const cheek = rawZ > 0 ? Math.exp(-Math.pow((Math.abs(x) - 0.22) / 0.11, 2)
      - Math.pow((y + 0.05) / 0.11, 2)) * 0.15 : 0;
    const color = skin.clone().lerp(blush, cheek);
    colors.push(color.r, color.g, color.b);
  }
  geometry.setAttribute('color', new Float32BufferAttribute(colors, 3));
  const smileDelta: number[] = []; const jawDelta: number[] = []; const browDelta: number[] = [];
  for (let i = 0; i < positions.count; i++) {
    const x = positions.getX(i); const y = positions.getY(i); const z = positions.getZ(i);
    const front = Math.max(0, z / 0.327);
    const cheek = Math.exp(-Math.pow((Math.abs(x) - 0.20) / 0.10, 2) - Math.pow((y + 0.06) / 0.15, 2)) * front;
    const lower = Math.max(0, Math.min(1, (-y - 0.16) / 0.25));
    const forehead = Math.exp(-Math.pow((y - 0.18) / 0.10, 2)) * front;
    smileDelta.push(0, cheek * 0.022, cheek * 0.004);
    jawDelta.push(0, -lower * 0.018, -lower * 0.004);
    browDelta.push(0, forehead * 0.012, 0);
  }
  const faceMorphs = [
    new Float32BufferAttribute(smileDelta, 3), new Float32BufferAttribute(jawDelta, 3), new Float32BufferAttribute(browDelta, 3),
  ];
  ['mouthSmile', 'jawOpen', 'browRaise'].forEach((name, index) => { faceMorphs[index].name = name; });
  geometry.morphAttributes.position = faceMorphs;
  geometry.morphTargetsRelative = true;
  geometry.computeVertexNormals();
  return geometry;
}

function addLipSmile(mesh: Mesh): void {
  const position = mesh.geometry.attributes.position; const delta: number[] = [];
  for (let i = 0; i < position.count; i++) {
    const weight = Math.min(1, Math.pow(Math.abs(position.getX(i)) / 0.101, 2));
    delta.push(0, weight * 0.036, 0);
  }
  const target = new Float32BufferAttribute(delta, 3); target.name = 'mouthSmile';
  mesh.geometry.morphAttributes.position = [target]; mesh.geometry.morphTargetsRelative = true;
  mesh.updateMorphTargets();
}

function hairGeometry(): BufferGeometry {
  const vertices: number[] = []; const indices: number[] = [];
  const segments = 64; const rows = 20;
  for (let row = 0; row <= rows; row++) {
    for (let segment = 0; segment <= segments; segment++) {
      const azimuth = segment / segments * Math.PI * 2;
      const front = Math.cos(azimuth);
      const hairline = 1.45 - front * 0.49 + Math.exp(-Math.pow(Math.sin(azimuth) / 0.25, 2)) * Math.max(front, 0) * 0.12;
      const polar = row / rows * hairline;
      const sweep = Math.sin(polar) * Math.max(front, 0) * 0.018;
      vertices.push(Math.sin(polar) * Math.sin(azimuth) * 0.374 + sweep,
        Math.cos(polar) * 0.49 + 0.012,
        Math.sin(polar) * Math.cos(azimuth) * 0.339 - 0.008);
      if (row < rows && segment < segments) {
        const a = row * (segments + 1) + segment; const b = a + segments + 1;
        indices.push(a, b, a + 1, a + 1, b, b + 1);
      }
    }
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new Float32BufferAttribute(vertices, 3));
  geometry.setIndex(indices); geometry.computeVertexNormals();
  return geometry;
}

function makeHand(skin: Material, side: number): Group {
  const hand = new Group();
  hand.name = side > 0 ? 'leftHand' : 'rightHand';
  hand.add(ellipsoid(skin, [0.075, 0.095, 0.035], [0, -0.06, 0]));
  for (let finger = 0; finger < 4; finger++) {
    const length = [0.105, 0.12, 0.11, 0.085][finger];
    const x = (finger - 1.5) * 0.033;
    const digit = ellipsoid(skin, [0.016, length / 2, 0.018], [x, -0.13 - length / 2, 0.009], 16);
    digit.rotation.z = (finger - 1.5) * 0.045;
    hand.add(digit);
  }
  const thumb = ellipsoid(skin, [0.027, 0.068, 0.023], [side * 0.068, -0.075, 0.026], 16);
  thumb.rotation.z = side * 0.55; hand.add(thumb);
  return hand;
}

function shirtGeometry(): BufferGeometry {
  // One continuous cloth surface avoids a visible seam where overlapping
  // shoulder/torso primitives would intersect under directional light.
  const profile = new CatmullRomCurve3([
    new Vector3(0, 0.02, 0), new Vector3(0.38, 0.03, 0),
    new Vector3(0.49, 0.15, 0), new Vector3(0.55, 0.52, 0),
    new Vector3(0.59, 0.94, 0), new Vector3(0.65, 1.12, 0),
    new Vector3(0.54, 1.26, 0), new Vector3(0.27, 1.34, 0),
    new Vector3(0.18, 1.34, 0), new Vector3(0, 1.34, 0),
  ]);
  const geometry = new LatheGeometry(profile.getPoints(80).map((point) => new Vector2(Math.max(0, point.x), point.y)), 64);
  geometry.scale(1, 1, 0.43); return geometry;
}

/** Procedural portrait rig. This is a stylized stand-in, not a face scan. */
export function createProceduralRig(): AvatarRig {
  const root = new Group(); root.name = 'proceduralPortrait';
  const skin = new MeshStandardMaterial({ color: '#d0a58c', roughness: 0.7, metalness: 0 });
  const faceMaterial = new MeshStandardMaterial({ vertexColors: true, roughness: 0.72, metalness: 0 });
  const ears = new MeshStandardMaterial({ color: '#bd8d7a', roughness: 0.75 });
  const lips = new MeshStandardMaterial({ color: '#a86f66', roughness: 0.69 });
  const crease = new MeshStandardMaterial({ color: '#5f3f38', roughness: 0.85 });
  const hair = new MeshStandardMaterial({ color: '#171511', roughness: 0.69, metalness: 0.03 });
  const strandMaterial = new MeshStandardMaterial({ color: '#28221e', roughness: 0.78 });
  const shirt = new MeshStandardMaterial({ color: '#202320', roughness: 0.94 });
  const shirtSeam = new MeshStandardMaterial({ color: '#30342e', roughness: 0.96 });
  const eyeWhite = new MeshStandardMaterial({ color: '#dedbd0', roughness: 0.38 });
  const irisMaterial = new MeshStandardMaterial({ color: '#78877c', roughness: 0.36 });
  const pupilMaterial = new MeshStandardMaterial({ color: '#111715', roughness: 0.26 });
  const highlight = new MeshStandardMaterial({ color: '#ffffff', roughness: 0.2, emissive: '#bfcdbf', emissiveIntensity: 0.15 });

  const torso = new Group(); torso.name = 'chest'; root.add(torso);
  const cloth = new Mesh(shirtGeometry(), shirt); cloth.position.z = -0.02;
  cloth.castShadow = true; cloth.receiveShadow = true; torso.add(cloth);
  const neck = new Group(); neck.name = 'neck'; neck.position.set(0, 1.29, 0); torso.add(neck);
  neck.add(ellipsoid(skin, [0.166, 0.28, 0.17], [0, 0.13, 0], 32));
  const collar = new Mesh(new TorusGeometry(0.19, 0.026, 10, 48, Math.PI * 2), shirtSeam);
  collar.rotation.x = Math.PI / 2; collar.position.set(0, 1.34, 0.012); collar.scale.set(1, 0.88, 1); torso.add(collar);

  const head = new Group(); head.name = 'head'; head.position.set(0, 1.92, 0); torso.add(head);
  const face = new Mesh(faceGeometry(), faceMaterial); face.castShadow = true; face.receiveShadow = true; head.add(face);
  for (const side of [-1, 1]) {
    head.add(ellipsoid(skin, [0.06, 0.103, 0.05], [side * 0.355, -0.015, -0.008], 24));
    head.add(ellipsoid(ears, [0.028, 0.067, 0.017], [side * 0.384, -0.01, 0.027], 20));
  }
  const hairCap = new Mesh(hairGeometry(), hair); hairCap.castShadow = true; head.add(hairCap);
  for (let strand = 0; strand < 15; strand++) {
    const normalizedX = (strand / 14 - 0.5) * 1.60;
    const section = Math.sqrt(1 - normalizedX * normalizedX);
    const start = 0.8 + 0.25 * (1 - Math.abs(normalizedX));
    const points: [number, number, number][] = [];
    for (let point = 0; point < 10; point++) {
      const angle = start - point / 9 * (start + 1.0);
      points.push([normalizedX * 0.374, Math.cos(angle) * section * 0.49 + 0.015,
        Math.sin(angle) * section * 0.339 - 0.008]);
    }
    head.add(curveMesh(points, 0.0025, strandMaterial, 28));
  }

  const eyeGroups: Group[] = []; const irises: Group[] = []; const brows: Group[] = [];
  for (const side of [-1, 1]) {
    const eye = new Group(); eye.name = side < 0 ? 'eyeRight' : 'eyeLeft';
    eye.position.set(side * 0.138, 0.083, 0.302); head.add(eye); eyeGroups.push(eye);
    eye.add(ellipsoid(eyeWhite, [0.065, 0.022, 0.025], [0, 0, 0], 32));
    const iris = new Group(); iris.position.set(0, 0, 0.026); eye.add(iris); irises.push(iris);
    iris.add(ellipsoid(irisMaterial, [0.019, 0.02, 0.005], [0, 0, 0], 24));
    iris.add(ellipsoid(pupilMaterial, [0.009, 0.010, 0.005], [0, 0, 0.004], 20));
    iris.add(ellipsoid(highlight, [0.004, 0.004, 0.002], [-0.005, 0.006, 0.009], 12));
    eye.add(curveMesh([[-0.062, 0, 0.006], [-0.035, 0.023, 0.021], [0.009, 0.025, 0.026], [0.061, 0, 0.006]], 0.0055, crease));
    eye.add(curveMesh([[-0.061, -0.002, 0.006], [-0.025, -0.023, 0.021], [0.026, -0.021, 0.021], [0.061, 0, 0.006]], 0.006, skin));
    const brow = new Group(); brow.name = `brow${side < 0 ? 'Right' : 'Left'}`;
    brow.position.set(side * 0.138, 0.15, 0.295); head.add(brow); brows.push(brow);
    brow.add(curveMesh([[-0.066, -0.005 * side, -0.01], [-0.025, 0.015, 0.016], [0.029, 0.016, 0.015], [0.069, -0.007 * -side, -0.012]], 0.012, hair));
  }

  // Nose bridge, tip and nostril wings remain attached to the face mesh.
  const bridge = ellipsoid(skin, [0.041, 0.112, 0.047], [0, 0.015, 0.323], 32);
  bridge.rotation.x = -0.1; head.add(bridge);
  head.add(ellipsoid(skin, [0.053, 0.039, 0.049], [0, -0.063, 0.365], 32));
  for (const side of [-1, 1]) {
    head.add(ellipsoid(skin, [0.031, 0.027, 0.028], [side * 0.041, -0.076, 0.34], 24));
    const nostril = ellipsoid(crease, [0.014, 0.007, 0.009], [side * 0.031, -0.091, 0.362], 16);
    nostril.rotation.z = side * -0.2; head.add(nostril);
  }

  const mouth = new Group(); mouth.name = 'mouth'; mouth.position.set(0, -0.185, 0.291); head.add(mouth);
  const cavity = ellipsoid(crease, [0.099, 0.006, 0.012], [0, -0.006, 0.012], 24); mouth.add(cavity);
  const upperLip = curveMesh([[-0.101, 0, 0], [-0.048, 0.009, 0.018], [-0.02, 0.015, 0.023], [0, 0.008, 0.024], [0.022, 0.015, 0.023], [0.054, 0.008, 0.018], [0.101, 0, 0]], 0.008, lips, 28);
  addLipSmile(upperLip);
  mouth.add(upperLip);
  const jaw = new Group(); jaw.name = 'jaw'; mouth.add(jaw);
  const lowerLip = curveMesh([[-0.101, -0.001, 0], [-0.052, -0.012, 0.022], [0, -0.015, 0.029], [0.055, -0.011, 0.021], [0.101, -0.001, 0]], 0.011, lips, 24);
  addLipSmile(lowerLip);
  jaw.add(lowerLip);

  const shoulders: Group[] = []; const forearms: Group[] = []; const hands: Group[] = [];
  for (const side of [-1, 1]) {
    const shoulder = new Group(); shoulder.name = side > 0 ? 'leftUpperArm' : 'rightUpperArm';
    shoulder.position.set(side * 0.59, 1.12, -0.025); shoulder.rotation.z = side * 0.13;
    torso.add(shoulder); shoulders.push(shoulder);
    shoulder.add(ellipsoid(shirt, [0.19, 0.28, 0.22], [side * 0.035, -0.11, 0], 32));
    shoulder.add(ellipsoid(skin, [0.12, 0.27, 0.13], [side * 0.04, -0.40, 0.012], 32));
    const forearm = new Group(); forearm.name = side > 0 ? 'leftForearm' : 'rightForearm';
    forearm.position.set(side * 0.04, -0.61, 0.01); shoulder.add(forearm); forearms.push(forearm);
    forearm.add(ellipsoid(skin, [0.10, 0.255, 0.105], [0, -0.19, 0], 32));
    const hand = makeHand(skin, side); hand.position.set(0, -0.41, 0); forearm.add(hand); hands.push(hand);
  }
  root.rotation.y = -0.075;
  const diagnostics: RigDiagnostics = {
    kind: 'procedural',
    morphs: ['eyeBlinkLeft', 'eyeBlinkRight', 'jawOpen', 'mouthSmileLeft', 'mouthSmileRight', 'mouthFunnel', 'browInnerUp', 'browDownLeft', 'browDownRight'],
    bones: ['head', 'neck', 'jaw', 'leftUpperArm', 'rightUpperArm', 'leftForearm', 'rightForearm', 'leftHand', 'rightHand'],
    missing: [],
  };
  return {
    root, diagnostics,
    apply(pose) {
      torso.position.y = pose.breath;
      head.rotation.set(pose.headPitch, pose.headYaw, pose.headRoll);
      neck.rotation.set(pose.headPitch * 0.2, pose.headYaw * 0.15, pose.headRoll * 0.2);
      eyeGroups.forEach((eye) => { eye.scale.y = Math.max(0.05, 1 - pose.blink * 0.95 - pose.mouthSmile * 0.12); });
      irises.forEach((iris) => { iris.position.x = pose.gazeX * 0.031; iris.position.y = pose.gazeY * 0.03; });
      brows.forEach((brow, index) => {
        brow.position.y = 0.15 + pose.browRaise * 0.035 - pose.browFurrow * 0.017;
        brow.rotation.z = (index === 0 ? 1 : -1) * (pose.browFurrow * 0.2 + pose.browRaise * 0.05);
      });
      jaw.position.y = -pose.jawOpen * 0.060;
      if (face.morphTargetInfluences) {
        face.morphTargetInfluences[0] = pose.mouthSmile;
        face.morphTargetInfluences[1] = pose.jawOpen;
        face.morphTargetInfluences[2] = pose.browRaise;
      }
      if (upperLip.morphTargetInfluences) upperLip.morphTargetInfluences[0] = pose.mouthSmile;
      if (lowerLip.morphTargetInfluences) lowerLip.morphTargetInfluences[0] = pose.mouthSmile;
      cavity.scale.y = 0.006 + pose.jawOpen * 0.045;
      cavity.position.y = -0.006 - pose.jawOpen * 0.022;
      mouth.scale.x = 1 + pose.mouthSmile * 0.17 - pose.mouthRound * 0.20;
      mouth.rotation.z = pose.mouthSmile * -0.018;
      upperLip.scale.y = 1 + pose.mouthSmile * 0.5;
      lowerLip.scale.y = 1 - pose.mouthSmile * 0.25;
      shoulders.forEach((shoulder, index) => {
        const side = index === 0 ? -1 : 1;
        shoulder.rotation.z = side * (0.13 + pose.openPalm * 0.19 + pose.shrug * 0.1);
        shoulder.rotation.x = -(pose.openPalm * 0.4 + pose.emphasis * (index === 1 ? 0.4 : 0.08) + pose.wave * (index === 0 ? 0.8 : 0));
        shoulder.position.y = 1.12 + pose.shrug * 0.055;
        forearms[index].rotation.x = -(pose.openPalm * 1.05 + pose.emphasis * (index === 1 ? 1.1 : 0.1) + pose.wave * (index === 0 ? 1.1 : 0));
        forearms[index].rotation.z = side * pose.openPalm * 0.1;
        hands[index].rotation.y = side * pose.openPalm * 0.7;
        hands[index].rotation.z = side * (pose.emphasis * 0.13 + pose.wave * 0.35);
      });
    },
    dispose() { disposeObject(root); },
  };
}
