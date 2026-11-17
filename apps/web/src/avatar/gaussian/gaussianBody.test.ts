import { describe, expect, it } from 'vitest';
import { BufferAttribute, BufferGeometry, Group, Matrix4, Mesh, MeshStandardMaterial, PerspectiveCamera, Vector3 } from 'three';
import type { AvatarRig } from '../procedural';
import { neutralPose } from '../timeline';
import { fitGaussianAttachment, GaussianBodyHead, hideOriginalHead, sortGaussianDepth } from './gaussianBody';
import type { GaussianBodyArrays, GaussianBodyHeader } from './gaussianBody';

function body(): AvatarRig {
  const root = new Group(); const head = new Group(); head.name = 'head'; head.position.set(0, 2, 0); root.add(head);
  const geometry = new BufferGeometry();
  // Two head vertices and a low rest-space hand vertex in one skinned-like mesh.
  geometry.setAttribute('position', new BufferAttribute(new Float32Array([0, 2.4, 0, .1, 2.1, 0, .7, 1.2, .1]), 3));
  root.add(new Mesh(geometry, new MeshStandardMaterial()));
  return { root, diagnostics: { kind: 'glb', morphs: [], bones: ['head'], missing: [] }, apply: () => undefined, dispose: () => undefined };
}

function bundle(): { header: GaussianBodyHeader; B: GaussianBodyArrays } {
  const header = { n_verts: 3, n_faces: 1, n_expr: 1, n_gauss: 2, n_frames: 1, fps: 15, parents: [-1, 0, 1, 1, 1] };
  const B: GaussianBodyArrays = { v_shaped: new Float32Array([-.1, -.15, 0, .1, -.15, 0, 0, .2, .02]),
    expr_dirs: new Float32Array(9), pose_dirs: new Float32Array(36 * 9), j_base: new Float32Array(15),
    j_expr_dirs: new Float32Array(15), lbs_weights: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    faces: new Uint32Array([0, 1, 2]), g_binding: new Uint32Array([0, 0]),
    g_xyz: new Float32Array([0, 0, 0, 0, 0, .01]), g_rot: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0]),
    g_opacity: new Float32Array([2, 2]), g_color: new Float32Array(6), g_scale: new Float32Array(6).fill(-4),
    m_expr: new Float32Array(1), m_pose: new Float32Array(15), m_trans: new Float32Array(3) };
  return { header, B };
}

describe('Gaussian body composition', () => {
  it('sorts splats back to front using the transformed shared camera', () => {
    const centers = new Float32Array([0, 0, -1, 0, 0, -4, 0, 0, -2]); const order = new Float32Array(3);
    sortGaussianDepth(centers, new Matrix4(), order); expect([...order]).toEqual([1, 2, 0]);
    sortGaussianDepth(centers, new Matrix4().makeRotationY(Math.PI), order); expect([...order]).toEqual([0, 2, 1]);
  });

  it('attaches the FLAME head pivot exactly and inherits the body head rotation', () => {
    const rig = body(); const { B } = bundle();
    const fit = fitGaussianAttachment(rig, B, { bundle: 'test', collarCutMeters: .05 });
    const world = fit.head.matrixWorld.clone().multiply(fit.matrix);
    const pivot = new Vector3().fromArray(B.j_base, 3).applyMatrix4(world);
    expect(pivot.distanceTo(fit.head.getWorldPosition(new Vector3()))).toBeLessThan(1e-7);
    expect(fit.scale).toBeCloseTo(2, 6);
    expect(fit.sourceCutY).toBeCloseTo(-.1, 6);
    const point = new Vector3(0, .1, .1).applyMatrix4(world);
    fit.head.rotation.y = .4; rig.root.updateMatrixWorld(true);
    const moved = new Vector3(0, .1, .1).applyMatrix4(fit.head.matrixWorld.clone().multiply(fit.matrix));
    expect(moved.distanceTo(point)).toBeGreaterThan(.02);
  });

  it('rejects a missing joint and invalid alignment before hiding original geometry', () => {
    const rig = body(); const { B } = bundle(); rig.root.remove(rig.root.getObjectByName('head')!);
    expect(() => fitGaussianAttachment(rig, B, { bundle: 'test' })).toThrow('head joint');
    expect(() => fitGaussianAttachment(body(), B, { bundle: 'test', scale: NaN })).toThrow('invalid');
  });

  it('clips the original head in rest space without hiding a raised hand or editing the source geometry', () => {
    const rig = body(); const skin = rig.root.children.find((node) => node instanceof Mesh) as Mesh;
    const geometry = skin.geometry, material = skin.material;
    const restore = hideOriginalHead(rig, 2.0);
    expect(skin.geometry).not.toBe(geometry); expect(geometry.getAttribute('gaussianHeadCut')).toBeUndefined();
    const cutoff = skin.geometry.getAttribute('gaussianHeadCut');
    expect(cutoff.getX(0)).toBeGreaterThan(0); expect(cutoff.getX(2)).toBeLessThan(0);
    // Animation moves the hand above the neck later; its stored rest cut remains negative.
    skin.geometry.getAttribute('position').setY(2, 2.5);
    expect(cutoff.getX(2)).toBeLessThan(0);
    restore(); restore(); expect(skin.geometry).toBe(geometry); expect(skin.material).toBe(material);
  });

  it('uses one scene depth buffer, leaves the learned arrays intact and disposes idempotently', () => {
    const rig = body(); const { header, B } = bundle();
    const sourceCopy = Array.from(B.v_shaped); const fit = fitGaussianAttachment(rig, B, { bundle: 'test', collarCutMeters: 0 });
    const head = new GaussianBodyHead(header, B, fit); fit.head.add(head.root);
    const camera = new PerspectiveCamera(36, 1, .1, 10); camera.position.set(0, 2, 3); camera.lookAt(0, 2, 0);
    head.update({ expr: new Float32Array(1), pose: new Float32Array(15), trans: new Float32Array(3) }, camera, 800, 800);
    expect(head.mesh.material.depthTest).toBe(true); expect(head.mesh.material.depthWrite).toBe(false);
    expect(head.mesh.material.toneMapped).toBe(false); expect(head.mesh.material.vertexShader).toContain('projectionMatrix*cam');
    expect(head.mesh.material.uniforms.uViewport.value.toArray()).toEqual([800, 800]);
    expect([...B.v_shaped]).toEqual(sourceCopy);
    head.dispose(); head.dispose(); expect(head.root.parent).toBeNull(); rig.apply(neutralPose());
  });
});
