import {
  Box3, BufferAttribute, BufferGeometry, DataTexture, DynamicDrawUsage, FloatType,
  GLSL3, Group, InstancedBufferAttribute, InstancedBufferGeometry, Matrix4, Mesh,
  MeshStandardMaterial, NearestFilter, NormalBlending, RawShaderMaterial, RGBAFormat, SkinnedMesh, Vector2, Vector3,
} from 'three';
import type { Camera, Material, Object3D } from 'three';
import type { AvatarRig } from '../procedural';
import { FlameHead, gaussianCenters } from './gaussianHeadCore';
import { cleanOpacity } from './gaussianMatte';

export interface GaussianHeadConfig {
  /** Basename under /models/gaussian-head; the private bundle never enters Git. */
  bundle: string;
  driverUrl?: string;
  trainingFrameIndices?: number[];
  /** Uniform correction after the fitted head/body scale. */
  scale?: number;
  /** Stage-space adjustment relative to the body's neutral head pivot. */
  offset?: [number, number, number];
  /** Rest-space band above the FLAME mesh's lower boundary, in metres. Negative keeps gaussians below it (neck and
   * collar re-bound from an upper-body model). */
  collarCutMeters?: number;
  /** Transition from neck motion at the collar to head motion above it. */
  seamBlendMeters?: number;
}

export interface GaussianBodyHeader {
  n_verts: number; n_faces: number; n_expr: number; n_gauss: number;
  n_frames: number; fps: number; parents: number[];
}
export interface GaussianBodyArrays extends Record<string, Float32Array | Uint32Array> {
  m_expr: Float32Array; m_pose: Float32Array; m_trans: Float32Array;
}
export interface GaussianBodyParams { expr: Float32Array; pose: Float32Array; trans: Float32Array }

/** Exact positive-depth ordering for a transformed Gaussian cloud. Reused by
 * tests; the counting-sort workspace avoids allocations in the rendering loop. */
export function sortGaussianDepth(centers: Float32Array, view: Matrix4, order: Float32Array,
  depth: Float32Array = new Float32Array(order.length), keys: Uint16Array = new Uint16Array(order.length), counts: Uint32Array = new Uint32Array(65536)): void {
  const v = view.elements; let min = Infinity, max = -Infinity;
  for (let i = 0; i < order.length; i++) {
    const d = -(v[2] * centers[i * 3] + v[6] * centers[i * 3 + 1] + v[10] * centers[i * 3 + 2] + v[14]);
    depth[i] = d; min = Math.min(min, d); max = Math.max(max, d);
  }
  const scale = 65535 / Math.max(1e-6, max - min); counts.fill(0);
  for (let i = 0; i < order.length; i++) { keys[i] = Math.max(0, Math.min(65535, Math.floor((depth[i] - min) * scale))); counts[keys[i]]++; }
  // Far -> near: normal alpha blending can now share the opaque body's depth.
  for (let i = 65534; i >= 0; i--) counts[i] += counts[i + 1];
  for (let i = order.length - 1; i >= 0; i--) order[--counts[keys[i]]] = i;
}

const vertexShader = `
precision highp float;
precision highp int;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform sampler2D uLocal;
uniform sampler2D uRotation;
uniform sampler2D uBinding;
uniform sampler2D uFaces;
uniform sampler2D uColor;
uniform vec2 uViewport;
uniform mat4 uNeckCorrection;
in vec3 position;
in float gaussianIndex;
out vec2 vPosition;
out vec4 vColor;
ivec2 at(int i, int w) { return ivec2(i % w, i / w); }
vec3 qrot(vec4 q, vec3 v) { return v + 2.0 * cross(q.yzw, cross(q.yzw, v) + q.x * v); }
vec4 qmul(vec4 a, vec4 b) { return vec4(a.x*b.x-dot(a.yzw,b.yzw), a.x*b.yzw+b.x*a.yzw+cross(a.yzw,b.yzw)); }
mat3 qmat(vec4 q) {
  float w=q.x,x=q.y,y=q.z,z=q.w;
  return mat3(1.-2.*(y*y+z*z), 2.*(x*y+w*z), 2.*(x*z-w*y),
    2.*(x*y-w*z), 1.-2.*(x*x+z*z), 2.*(y*z+w*x),
    2.*(x*z+w*y), 2.*(y*z-w*x), 1.-2.*(x*x+y*y));
}
void main() {
  int i=int(gaussianIndex), W=textureSize(uLocal,0).x;
  vec4 L=texelFetch(uLocal,at(i,W),0);
  vec4 binding=texelFetch(uBinding,at(i,W),0);
  int f=int(binding.r);
  int fw=textureSize(uFaces,0).x/2;
  ivec2 fp=ivec2((f%fw)*2,f/fw);
  vec4 F0=texelFetch(uFaces,fp,0), F1=normalize(texelFetch(uFaces,fp+ivec2(1,0),0));
  int rw=textureSize(uRotation,0).x/2;
  ivec2 rp=ivec2((i%rw)*2,i/rw);
  vec4 q=normalize(texelFetch(uRotation,rp,0));
  vec3 scale=texelFetch(uRotation,rp+ivec2(1,0),0).xyz*F0.w;
  mat4 seam=mat4(1.)*(1.-binding.g)+uNeckCorrection*binding.g;
  vec3 center=(seam*vec4(qrot(F1,L.xyz)*F0.w+F0.xyz,1.)).xyz;
  vec4 cam=modelViewMatrix*vec4(center,1.);
  float z=-cam.z;
  if(z<0.01 || L.w<=0.) { gl_Position=vec4(0.,0.,2.,1.); vColor=vec4(0.); vPosition=vec2(0.); return; }
  mat3 M=mat3(modelViewMatrix)*mat3(seam)*qmat(qmul(F1,q))*mat3(scale.x,0.,0.,0.,scale.y,0.,0.,0.,scale.z);
  vec2 focal=0.5*uViewport*vec2(projectionMatrix[0][0],projectionMatrix[1][1]);
  mat3 J=mat3(focal.x/z,0.,0.,0.,focal.y/z,0.,focal.x*cam.x/(z*z),focal.y*cam.y/(z*z),0.);
  mat3 covariance=J*(M*transpose(M))*transpose(J);
  float a=covariance[0][0]+0.3, b=covariance[0][1], c=covariance[1][1]+0.3;
  float middle=.5*(a+c), radius=length(vec2(.5*(a-c),b));
  float l1=middle+radius, l2=max(middle-radius,.1);
  vec2 d1=abs(b)<1e-9 ? (a>=c?vec2(1.,0.):vec2(0.,1.)) : normalize(vec2(b,l1-a));
  vec2 d2=vec2(-d1.y,d1.x);
  vPosition=position.xy*3.;
  vec2 offset=vPosition.x*sqrt(l1)*d1+vPosition.y*sqrt(l2)*d2;
  vec4 clip=projectionMatrix*cam;
  // Retain real projected depth: a raised opaque hand can occlude the face.
  clip.xy+=2.*offset/uViewport*clip.w;
  gl_Position=clip;
  vColor=vec4(texelFetch(uColor,at(i,W),0).rgb,L.w);
}`;

const fragmentShader = `
precision highp float;
in vec2 vPosition;
in vec4 vColor;
out vec4 outColor;
void main() {
  float r2=dot(vPosition,vPosition);
  if(r2>9.) discard;
  float a=min(.99,vColor.a*exp(-.5*r2));
  if(a<1./255.) discard;
  // Stored colour is already baked for display. No second lighting/tone map.
  outColor=vec4(vColor.rgb,a);
}`;

function texture(data: Float32Array, width: number, height: number): DataTexture {
  const t = new DataTexture(data, width, height, RGBAFormat, FloatType);
  t.minFilter = t.magFilter = NearestFilter; t.generateMipmaps = false; t.needsUpdate = true;
  return t;
}

export interface GaussianAttachment {
  head: Object3D;
  neck?: Object3D;
  /** Matrix from FLAME metres to the current head's local frame. */
  matrix: Matrix4;
  sourceCutY: number;
  /** Where the neck seam starts (rest FLAME space); below it gaussians follow the neck joint fully. */
  seamStartY?: number;
  targetCutY: number;
  scale: number;
  seamBlendMeters?: number;
}

export function gaussianSeamWeight(height: number, cutY: number, band: number): number {
  if (band <= 0) return 0;
  const t = Math.max(0, Math.min(1, (height - cutY) / band));
  return 1 - t * t * (3 - 2 * t);
}

/** Fit one uniform similarity transform without altering the learned identity.
 * Call on a neutral rig. Unknown uploaded rigs need their own explicit mapping. */
export function fitGaussianAttachment(rig: AvatarRig, B: GaussianBodyArrays, config: GaussianHeadConfig): GaussianAttachment {
  const mapped = rig.diagnostics.resolvedBones?.find((bone) => bone.control === 'head')?.node;
  const head = rig.root.getObjectByName(mapped ?? 'head');
  if (!head) throw new Error('The body has no mapped head joint for the Gaussian face.');
  rig.root.updateMatrixWorld(true);
  const source = B.v_shaped, pivot = new Vector3().fromArray(B.j_base, 3);
  let sourceMinY = Infinity, sourceMaxY = -Infinity;
  for (let i = 1; i < source.length; i += 3) { sourceMinY = Math.min(sourceMinY, source[i]); sourceMaxY = Math.max(sourceMaxY, source[i]); }
  const bodyBounds = new Box3().setFromObject(rig.root), targetPivot = head.getWorldPosition(new Vector3());
  const ratio = (bodyBounds.max.y - targetPivot.y) / (sourceMaxY - pivot.y);
  const correction = config.scale ?? 1;
  const cut = config.collarCutMeters ?? 0.05;
  const band = config.seamBlendMeters ?? 0.06;
  const offset = config.offset ?? [0, 0, 0];
  if (!Number.isFinite(ratio) || ratio <= 0 || ratio > 20 || !Number.isFinite(correction) || correction < 0.5 || correction > 2
    || !Number.isFinite(cut) || cut < -0.15 || cut > 0.15 || !Number.isFinite(band) || band < 0 || band > 0.3
    || offset.some((v) => !Number.isFinite(v) || Math.abs(v) > 0.5)) {
    throw new Error('The Gaussian/body alignment is invalid; inspect the neutral head and collar calibration.');
  }
  const scale = ratio * correction;
  const target = targetPivot.clone().add(new Vector3(...offset));
  const world = new Matrix4().makeTranslation(target.x, target.y, target.z)
    .multiply(new Matrix4().makeScale(scale, scale, scale))
    .multiply(new Matrix4().makeTranslation(-pivot.x, -pivot.y, -pivot.z));
  const sourceCutY = sourceMinY + cut;
  const neckName = rig.diagnostics.resolvedBones?.find((bone) => bone.control === 'neck')?.node;
  const neck = rig.root.getObjectByName(neckName ?? 'neck');
  return { head, neck, matrix: head.matrixWorld.clone().invert().multiply(world), sourceCutY, seamStartY: Math.max(sourceCutY, sourceMinY),
    targetCutY: target.y + (sourceCutY - pivot.y) * scale, scale, seamBlendMeters: band };
}

interface CutMesh { mesh: Mesh; geometry: BufferGeometry; material: Material | Material[]; clonedGeometry: BufferGeometry; clonedMaterials: Material[] }

/** Cut only the original head's rest-space geometry. A world-space clip plane
 * would also erase raised hands. Geometry/material clones make disabling exact. */
export function hideOriginalHead(rig: AvatarRig, targetCutY: number): () => void {
  const changed: CutMesh[] = []; const point = new Vector3();
  rig.root.updateMatrixWorld(true);
  rig.root.traverse((node) => { if (node instanceof SkinnedMesh) node.skeleton.update(); });
  try {
    rig.root.traverse((node) => {
      if (!(node instanceof Mesh) || /suit|jacket|shirt|clothes/i.test(node.name)) return;
      const positions = node.geometry.getAttribute('position'); if (!positions) return;
      const cut = new Float32Array(positions.count); let any = false;
      for (let i = 0; i < positions.count; i++) {
        node.getVertexPosition(i, point).applyMatrix4(node.matrixWorld);
        cut[i] = point.y - targetCutY; any ||= cut[i] > 0;
      }
      if (!any) return;
      const originalGeometry = node.geometry, originalMaterial = node.material;
      const clonedGeometry = originalGeometry.clone();
      clonedGeometry.setAttribute('gaussianHeadCut', new BufferAttribute(cut, 1));
      const originals: Material[] = Array.isArray(originalMaterial) ? originalMaterial : [originalMaterial];
      const clonedMaterials = originals.map((original) => {
        const material = original.clone();
        material.onBeforeCompile = (shader, renderer) => {
          original.onBeforeCompile(shader, renderer);
          shader.vertexShader = `attribute float gaussianHeadCut;\nvarying float vGaussianHeadCut;\n${shader.vertexShader}`
            .replace('#include <begin_vertex>', '#include <begin_vertex>\nvGaussianHeadCut = gaussianHeadCut;');
          shader.fragmentShader = `varying float vGaussianHeadCut;\n${shader.fragmentShader}`
            .replace('#include <clipping_planes_fragment>', '#include <clipping_planes_fragment>\nif (vGaussianHeadCut > 0.0) discard;');
        };
        material.customProgramCacheKey = () => `${original.customProgramCacheKey()}|gaussian-head-cut-v1`;
        return material;
      });
      changed.push({ mesh: node, geometry: originalGeometry, material: originalMaterial, clonedGeometry, clonedMaterials });
      node.geometry = clonedGeometry; node.material = Array.isArray(originalMaterial) ? clonedMaterials : clonedMaterials[0];
    });
  } catch (error) {
    for (const entry of changed) {
      entry.mesh.geometry = entry.geometry; entry.mesh.material = entry.material;
      entry.clonedGeometry.dispose(); for (const material of entry.clonedMaterials) material.dispose();
    }
    throw error;
  }
  let restored = false;
  return () => {
    if (restored) return; restored = true;
    for (const entry of changed) {
      entry.mesh.geometry = entry.geometry; entry.mesh.material = entry.material;
      entry.clonedGeometry.dispose(); for (const material of entry.clonedMaterials) material.dispose();
    }
  };
}

/** A Gaussian head inside the same Three.js scene as its skinned body. */
export class GaussianBodyHead {
  readonly root = new Group();
  readonly mesh: Mesh<InstancedBufferGeometry, RawShaderMaterial>;
  private readonly head: FlameHead;
  private readonly faceData: Float32Array;
  private readonly faceTexture: DataTexture;
  private readonly textures: DataTexture[];
  private readonly centers: Float32Array;
  private readonly order: Float32Array;
  private readonly depth: Float32Array;
  private readonly keys: Uint16Array;
  private readonly counts = new Uint32Array(65536);
  private readonly view = new Matrix4();
  private readonly neckCorrection = new Matrix4();
  private readonly neckReference?: Matrix4;
  private readonly neck?: Object3D;
  private readonly neckWeights: Float32Array;
  private disposed = false;

  constructor(private readonly header: GaussianBodyHeader, private readonly B: GaussianBodyArrays, attachment: GaussianAttachment, cleanWhiteMatte = false) {
    this.head = new FlameHead(header, B);
    this.root.name = 'GaussianHead'; this.root.matrixAutoUpdate = false; this.root.matrix.copy(attachment.matrix); this.root.matrixWorldNeedsUpdate = true;
    const n = header.n_gauss, width = 1024, height = Math.max(1, Math.ceil(n / width));
    this.neck = attachment.neck;
    if (this.neck) this.neckReference = this.neck.matrixWorld.clone().invert()
      .multiply(attachment.head.matrixWorld).multiply(attachment.matrix);
    this.neckWeights = new Float32Array(n);
    const local = new Float32Array(width * height * 4), color = new Float32Array(width * height * 4);
    const binding = new Float32Array(width * height * 4), rotation = new Float32Array(width * height * 8);
    const zero = { expr: new Float32Array(header.n_expr), pose: new Float32Array(15), trans: new Float32Array(3) };
    this.head.forward(zero.expr, zero.pose, zero.trans);
    const restFrames = this.head.updateFaceFrames();
    const restCenters = gaussianCenters(B, restFrames, new Float32Array(n * 3)) as Float32Array;
    for (let i = 0; i < n; i++) {
      const kept = restCenters[i * 3 + 1] >= attachment.sourceCutY;
      local.set([B.g_xyz[i * 3], B.g_xyz[i * 3 + 1], B.g_xyz[i * 3 + 2], kept ? 1 / (1 + Math.exp(-B.g_opacity[i])) : 0], i * 4);
      for (let c = 0; c < 3; c++) color[i * 4 + c] = Math.min(1, Math.max(0, .5 + .28209479 * B.g_color[i * 3 + c]));
      if (cleanWhiteMatte) local[i * 4 + 3] *= cleanOpacity(restCenters[i * 3], restCenters[i * 3 + 1], restCenters[i * 3 + 2], color[i * 4], color[i * 4 + 1], color[i * 4 + 2]);
      binding[i * 4] = B.g_binding[i];
      this.neckWeights[i] = this.neck ? gaussianSeamWeight(restCenters[i * 3 + 1], attachment.seamStartY ?? attachment.sourceCutY, attachment.seamBlendMeters ?? .06) : 0;
      binding[i * 4 + 1] = this.neckWeights[i];
      rotation.set([B.g_rot[i * 4], B.g_rot[i * 4 + 1], B.g_rot[i * 4 + 2], B.g_rot[i * 4 + 3],
        Math.exp(B.g_scale[i * 3]), Math.exp(B.g_scale[i * 3 + 1]), Math.exp(B.g_scale[i * 3 + 2]), 0], i * 8);
    }
    const faceHeight = Math.max(1, Math.ceil(header.n_faces / width));
    this.faceData = new Float32Array(width * faceHeight * 8);
    this.faceTexture = texture(this.faceData, width * 2, faceHeight);
    this.textures = [texture(local, width, height), texture(rotation, width * 2, height), texture(binding, width, height), this.faceTexture, texture(color, width, height)];
    const geometry = new InstancedBufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]), 3));
    geometry.setIndex([0, 1, 2, 0, 2, 3]); geometry.instanceCount = n;
    this.order = Float32Array.from({ length: n }, (_, i) => i);
    geometry.setAttribute('gaussianIndex', new InstancedBufferAttribute(this.order, 1).setUsage(DynamicDrawUsage));
    const material = new RawShaderMaterial({ glslVersion: GLSL3, vertexShader, fragmentShader,
      uniforms: { uLocal: { value: this.textures[0] }, uRotation: { value: this.textures[1] }, uBinding: { value: this.textures[2] },
        uFaces: { value: this.faceTexture }, uColor: { value: this.textures[4] }, uViewport: { value: new Vector2(1, 1) }, uNeckCorrection: { value: this.neckCorrection } },
      transparent: true, blending: NormalBlending, depthTest: true, depthWrite: false, toneMapped: false });
    this.mesh = new Mesh(geometry, material); this.mesh.frustumCulled = false; this.mesh.renderOrder = 5;
    this.root.add(this.mesh);
    this.centers = new Float32Array(n * 3); this.depth = new Float32Array(n); this.keys = new Uint16Array(n);
  }

  update(params: GaussianBodyParams, camera: Camera, width: number, height: number): void {
    if (this.disposed) return;
    this.head.forward(params.expr, params.pose, params.trans);
    const frames = this.head.updateFaceFrames();
    for (let f = 0; f < this.header.n_faces; f++) {
      const o = f * 8;
      this.faceData[o] = frames[o]; this.faceData[o + 1] = frames[o + 1]; this.faceData[o + 2] = frames[o + 2]; this.faceData[o + 3] = frames[o + 7];
      this.faceData[o + 4] = frames[o + 3]; this.faceData[o + 5] = frames[o + 4]; this.faceData[o + 6] = frames[o + 5]; this.faceData[o + 7] = frames[o + 6];
    }
    this.faceTexture.needsUpdate = true;
    gaussianCenters(this.B, frames, this.centers);
    this.root.parent?.updateWorldMatrix(true, false);
    // The calibrated local matrix is manual: force propagation after a parent
    // bone moves, before sorting rather than waiting for renderer.render().
    this.root.updateMatrixWorld(true); camera.updateMatrixWorld();
    if (this.neck && this.neckReference) {
      this.neck.updateWorldMatrix(true, false);
      this.neckCorrection.copy(this.mesh.matrixWorld).invert().multiply(this.neck.matrixWorld).multiply(this.neckReference);
      // Sorting must use the same skinned centres as the vertex shader.
      const m = this.neckCorrection.elements;
      for (let i = 0; i < this.neckWeights.length; i++) {
        const weight = this.neckWeights[i]; if (weight === 0) continue;
        const k = i * 3, x = this.centers[k], y = this.centers[k + 1], z = this.centers[k + 2];
        this.centers[k] += weight * (m[0] * x + m[4] * y + m[8] * z + m[12] - x);
        this.centers[k + 1] += weight * (m[1] * x + m[5] * y + m[9] * z + m[13] - y);
        this.centers[k + 2] += weight * (m[2] * x + m[6] * y + m[10] * z + m[14] - z);
      }
    }
    this.view.multiplyMatrices(camera.matrixWorldInverse, this.mesh.matrixWorld);
    sortGaussianDepth(this.centers, this.view, this.order, this.depth, this.keys, this.counts);
    this.mesh.geometry.getAttribute('gaussianIndex').needsUpdate = true;
    this.mesh.material.uniforms.uViewport.value.set(Math.max(1, width), Math.max(1, height));
  }

  dispose(): void {
    if (this.disposed) return; this.disposed = true;
    this.root.removeFromParent(); this.mesh.geometry.dispose(); this.mesh.material.dispose();
    for (const entry of this.textures) entry.dispose();
  }
}

/** Clone materials so leaving the composite restores the authored body exactly. */
export function softenCompositeMaterials(rig: AvatarRig): () => void {
  const changed: { mesh: Mesh; original: Material | Material[]; clones: Material[] }[] = [];
  rig.root.traverse((node) => {
    if (!(node instanceof Mesh)) return;
    const originals = Array.isArray(node.material) ? node.material : [node.material];
    const clones = originals.map((original) => {
      const material = original.clone();
      material.onBeforeCompile = original.onBeforeCompile;
      material.customProgramCacheKey = original.customProgramCacheKey;
      if (material instanceof MeshStandardMaterial) {
        if (/suit|jacket/i.test(`${node.name} ${material.name}`)) {
          material.roughness = .62;
          material.metalness = 0;
        } else if (material.map) {
          // Keep the skin texture; reduce the pale, smooth neck/hand highlights.
          material.roughness = Math.max(.78, material.roughness);
          material.color.multiplyScalar(.84);
        }
      }
      return material;
    });
    changed.push({ mesh: node, original: node.material, clones });
    node.material = Array.isArray(node.material) ? clones : clones[0];
  });
  let restored = false;
  return () => {
    if (restored) return;
    restored = true;
    for (const entry of changed) {
      entry.mesh.material = entry.original;
      entry.clones.forEach((material) => material.dispose());
    }
  };
}
