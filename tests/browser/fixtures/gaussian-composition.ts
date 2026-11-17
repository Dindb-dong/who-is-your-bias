/** Synthetic rendering fixture. No person-specific geometry, motion or weights.
 * Built only by Playwright into memory, never into the production application. */
import { Group, Matrix4, Mesh, MeshBasicMaterial, PerspectiveCamera, PlaneGeometry, Scene, WebGLRenderer } from 'three';
import { GaussianBodyHead } from '../../../apps/web/src/avatar/gaussian/gaussianBody';
import type { GaussianBodyArrays } from '../../../apps/web/src/avatar/gaussian/gaussianBody';

export function runCompositionFixture() {
  const renderer = new WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
  renderer.setSize(128, 128); renderer.setClearColor(0x000000, 1); document.body.appendChild(renderer.domElement);
  const camera = new PerspectiveCamera(36, 1, .01, 10); camera.position.set(0, 0, 1); camera.lookAt(0, 0, 0);
  const scene = new Scene();
  const header = { n_verts: 3, n_faces: 1, n_expr: 1, n_gauss: 1, n_frames: 1, fps: 15, parents: [-1, 0, 1, 1, 1] };
  const B: GaussianBodyArrays = { v_shaped: new Float32Array([-.15, -.1, 0, .15, -.1, 0, 0, .2, 0]),
    expr_dirs: new Float32Array([0, .1, 0, 0, .1, 0, 0, .1, 0]), pose_dirs: new Float32Array(36 * 9), j_base: new Float32Array(15),
    j_expr_dirs: new Float32Array(15), lbs_weights: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    faces: new Uint32Array([0, 1, 2]), g_binding: new Uint32Array([0]), g_xyz: new Float32Array(3),
    g_rot: new Float32Array([1, 0, 0, 0]), g_opacity: new Float32Array([8]),
    g_color: new Float32Array([(1 - .5) / .28209479, -.5 / .28209479, -.5 / .28209479]),
    g_scale: new Float32Array([-1, -1, -1]), m_expr: new Float32Array(1), m_pose: new Float32Array(15), m_trans: new Float32Array(3) };
  const head = new GaussianBodyHead(header, B, { head: new Group(), matrix: new Matrix4(), sourceCutY: -1, targetCutY: -1, scale: 1 });
  scene.add(head.root);
  const parameters = { expr: new Float32Array(1), pose: new Float32Array(15), trans: new Float32Array(3) };
  const pixel = () => {
    head.update(parameters, camera, 128, 128); renderer.render(scene, camera);
    const values = new Uint8Array(4); const gl = renderer.getContext();
    gl.readPixels(64, 64, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, values);
    return [...values];
  };
  const face = pixel();
  const centroid = () => {
    const gl = renderer.getContext(); const rgba = new Uint8Array(128 * 128 * 4); gl.readPixels(0, 0, 128, 128, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
    let total = 0, y = 0;
    for (let i = 0; i < 128 * 128; i++) { const red = Math.max(0, rgba[i * 4] - rgba[i * 4 + 1]); total += red; y += Math.floor(i / 128) * red; }
    return y / total;
  };
  const restY = centroid(); parameters.expr[0] = 1; pixel(); const movedY = centroid(); parameters.expr[0] = 0;
  const hand = new Mesh(new PlaneGeometry(.4, .4), new MeshBasicMaterial({ color: 0x00ff00 }));
  hand.position.z = .4; scene.add(hand); const occluded = pixel();
  hand.position.z = -.4; const behind = pixel();
  head.dispose(); const disposedGeometries = renderer.info.memory.geometries;
  hand.geometry.dispose(); hand.material.dispose(); renderer.dispose(); renderer.forceContextLoss();
  const result = { face, occluded, behind, restY, movedY, disposedGeometries, canvasCount: document.querySelectorAll('canvas').length };
  renderer.domElement.remove(); return result;
}
