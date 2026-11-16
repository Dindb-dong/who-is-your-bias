// Preview renderer: opaque SMPL-X body mesh + transplanted gaussian head, depth-composited.
// Pass 1 writes body depth only; pass 2 draws splats front-to-back ("under" blending) tested against that depth;
// pass 3 adds the body colour underneath whatever the splats already covered.
import { gaussianCenters } from '../bundle/gaussian-head-core.js';
import { VS as SPLAT_VS, FS as SPLAT_FS } from './_splat_shaders.js';

const MESH_VS = `#version 300 es
precision highp float;
uniform mat4 uView; uniform mat4 uProj;
in vec3 aPos; in vec3 aNrm; out vec3 vN;
void main() { vN = mat3(uView) * aNrm; gl_Position = uProj * uView * vec4(aPos, 1.0); }`;
const MESH_FS = `#version 300 es
precision highp float;
uniform vec3 uColor; in vec3 vN; out vec4 o;
void main() {
  vec3 n = normalize(vN); if (!gl_FrontFacing) n = -n;
  float key = max(dot(n, normalize(vec3(0.35, 0.55, 0.75))), 0.0), rim = pow(1.0 - abs(n.z), 3.0);
  o = vec4(uColor * (0.32 + 0.68 * key) + 0.08 * rim, 1.0);
}`;

function program(gl, vs, fs) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    gl.attachShader(p, s);
  }
  gl.linkProgram(p); if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}
function tex(gl, internal, format, type, w, h, data) {
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, data);
  for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
  return t;
}
export function perspective(fovY, aspect, n, f) {
  const t = 1 / Math.tan(fovY / 2);
  return new Float32Array([t / aspect, 0, 0, 0, 0, t, 0, 0, 0, 0, (f + n) / (n - f), -1, 0, 0, 2 * f * n / (n - f), 0]);
}

export class SmplxPreviewRenderer {
  constructor(canvas, header, B, head) {
    const gl = canvas.getContext('webgl2', { premultipliedAlpha: true, alpha: true, antialias: false, depth: true });
    if (!gl) throw new Error('WebGL2 is required');
    Object.assign(this, { gl, canvas, header, B, head, near: 0.05, far: 50 });
    this.stats = { cpuMs: 0 };
    // ---- splats (same static layout as GaussianHeadRenderer)
    const N = header.n_gauss, F = header.n_faces_g; this.N = N;
    this.splat = program(gl, SPLAT_VS, SPLAT_FS); gl.useProgram(this.splat);
    const W = 2048, H = Math.ceil(N / W), sig = x => 1 / (1 + Math.exp(-x));
    const local = new Float32Array(W * H * 4), color = new Float32Array(W * H * 4), bind = new Uint32Array(W * H);
    const WR = 1024, HR = Math.ceil(N / WR), rots = new Float32Array(WR * 2 * HR * 4);
    for (let n = 0; n < N; n++) {
      local.set([B.g_xyz[3 * n], B.g_xyz[3 * n + 1], B.g_xyz[3 * n + 2], sig(B.g_opacity[n])], 4 * n);
      for (let c = 0; c < 3; c++) color[4 * n + c] = Math.min(1, Math.max(0, 0.5 + 0.28209479 * B.g_color[3 * n + c]));
      bind[n] = B.g_binding[n];
      rots.set([B.g_rot[4 * n], B.g_rot[4 * n + 1], B.g_rot[4 * n + 2], B.g_rot[4 * n + 3]], 8 * n);
      rots.set([Math.exp(B.g_scale[3 * n]), Math.exp(B.g_scale[3 * n + 1]), Math.exp(B.g_scale[3 * n + 2]), 0], 8 * n + 4);
    }
    this.tLocal = tex(gl, gl.RGBA32F, gl.RGBA, gl.FLOAT, W, H, local);
    this.tColor = tex(gl, gl.RGBA32F, gl.RGBA, gl.FLOAT, W, H, color);
    this.tBind = tex(gl, gl.R32UI, gl.RED_INTEGER, gl.UNSIGNED_INT, W, H, bind);
    this.tRotS = tex(gl, gl.RGBA32F, gl.RGBA, gl.FLOAT, WR * 2, HR, rots);
    this.FW = 1024; this.FH = Math.ceil(F / this.FW);
    this.faceTexData = new Float32Array(this.FW * 2 * this.FH * 4);
    this.tFace = tex(gl, gl.RGBA32F, gl.RGBA, gl.FLOAT, this.FW * 2, this.FH, null);
    this.vaoS = gl.createVertexArray(); gl.bindVertexArray(this.vaoS);
    const quad = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, 1, 1, -1, 1]), gl.STATIC_DRAW);
    const aC = gl.getAttribLocation(this.splat, 'aCorner'); gl.enableVertexAttribArray(aC); gl.vertexAttribPointer(aC, 2, gl.FLOAT, false, 0, 0);
    this.indexBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuf); gl.bufferData(gl.ARRAY_BUFFER, N * 4, gl.DYNAMIC_DRAW);
    const aI = gl.getAttribLocation(this.splat, 'aIndex'); gl.enableVertexAttribArray(aI); gl.vertexAttribIPointer(aI, 1, gl.UNSIGNED_INT, 0, 0); gl.vertexAttribDivisor(aI, 1);
    const ebo = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
    this.uS = Object.fromEntries(['uLocal', 'uRotS', 'uBind', 'uFace', 'uColor', 'uView', 'uFocal', 'uViewport', 'uNear', 'uFar'].map(k => [k, gl.getUniformLocation(this.splat, k)]));
    this.centers = new Float32Array(N * 3); this.depth = new Float32Array(N); this.order = new Uint32Array(N);
    this.counts = new Uint32Array(65536); this.keys = new Uint16Array(N);
    // ---- body mesh
    this.mesh = program(gl, MESH_VS, MESH_FS);
    this.faces = B.mesh_faces; this.nrm = new Float32Array(header.n_verts * 3);
    this.vaoM = gl.createVertexArray(); gl.bindVertexArray(this.vaoM);
    this.posBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf); gl.bufferData(gl.ARRAY_BUFFER, header.n_verts * 12, gl.DYNAMIC_DRAW);
    const aP = gl.getAttribLocation(this.mesh, 'aPos'); gl.enableVertexAttribArray(aP); gl.vertexAttribPointer(aP, 3, gl.FLOAT, false, 0, 0);
    this.nrmBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.nrmBuf); gl.bufferData(gl.ARRAY_BUFFER, header.n_verts * 12, gl.DYNAMIC_DRAW);
    const aN = gl.getAttribLocation(this.mesh, 'aNrm'); gl.enableVertexAttribArray(aN); gl.vertexAttribPointer(aN, 3, gl.FLOAT, false, 0, 0);
    const mebo = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mebo); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, this.faces, gl.STATIC_DRAW);
    this.uM = Object.fromEntries(['uView', 'uProj', 'uColor'].map(k => [k, gl.getUniformLocation(this.mesh, k)]));
    gl.bindVertexArray(null);
  }

  sortByDepth(view) {
    const { N, centers, depth, order, counts, keys } = this;
    const v2 = view[2], v6 = view[6], v10 = view[10], v14 = view[14];
    let mn = Infinity, mx = -Infinity;
    for (let n = 0; n < N; n++) {
      const d = -(v2 * centers[3 * n] + v6 * centers[3 * n + 1] + v10 * centers[3 * n + 2] + v14);
      depth[n] = d; if (d < mn) mn = d; if (d > mx) mx = d;
    }
    const k = 65535 / Math.max(1e-6, mx - mn); counts.fill(0);
    for (let n = 0; n < N; n++) { keys[n] = ((depth[n] - mn) * k) | 0; counts[keys[n]]++; }
    for (let i = 1; i < 65536; i++) counts[i] += counts[i - 1];
    for (let n = N - 1; n >= 0; n--) order[--counts[keys[n]]] = n;   // ascending depth = front to back
  }

  normals(v) {
    const f = this.faces, nr = this.nrm; nr.fill(0);
    for (let k = 0; k < f.length; k += 3) {
      const a = 3 * f[k], b = 3 * f[k + 1], c = 3 * f[k + 2];
      const ex = v[b] - v[a], ey = v[b + 1] - v[a + 1], ez = v[b + 2] - v[a + 2], fx = v[c] - v[a], fy = v[c + 1] - v[a + 1], fz = v[c + 2] - v[a + 2];
      const nx = ey * fz - ez * fy, ny = ez * fx - ex * fz, nz = ex * fy - ey * fx;
      for (const i of [a, b, c]) { nr[i] += nx; nr[i + 1] += ny; nr[i + 2] += nz; }
    }
    return nr;
  }

  render(pose, expr, trans, camera, opts = {}) {
    const gl = this.gl, t0 = performance.now();
    const verts = this.head.forward(pose, expr, trans);
    const fd = this.head.updateFaceFrames();
    const F = this.header.n_faces_g, T = this.faceTexData;
    for (let k = 0; k < F; k++) { const o = 8 * k; T[o] = fd[o]; T[o + 1] = fd[o + 1]; T[o + 2] = fd[o + 2]; T[o + 3] = fd[o + 7]; T[o + 4] = fd[o + 3]; T[o + 5] = fd[o + 4]; T[o + 6] = fd[o + 5]; T[o + 7] = fd[o + 6]; }
    gaussianCenters(this.B, fd, this.centers);
    this.sortByDepth(camera.view);
    const nrm = this.normals(verts);
    this.stats.cpuMs = performance.now() - t0;
    const w = this.canvas.width, h = this.canvas.height;
    const proj = perspective(camera.fovY, w / h, this.near, this.far);
    gl.viewport(0, 0, w, h); gl.clearColor(0, 0, 0, 0); gl.clearDepth(1); gl.depthMask(true);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const showBody = opts.showBody !== false;
    // pass 1: body depth only
    gl.useProgram(this.mesh); gl.bindVertexArray(this.vaoM);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, verts);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.nrmBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, nrm);
    gl.uniformMatrix4fv(this.uM.uView, false, camera.view); gl.uniformMatrix4fv(this.uM.uProj, false, proj);
    gl.uniform3fv(this.uM.uColor, opts.bodyColor || [0.42, 0.47, 0.55]);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.disable(gl.BLEND); gl.disable(gl.CULL_FACE);
    if (showBody) { gl.colorMask(false, false, false, false); gl.drawElements(gl.TRIANGLES, this.faces.length, gl.UNSIGNED_INT, 0); gl.colorMask(true, true, true, true); }
    // pass 2: splats, tested against body depth, no depth writes, front-to-back under-blending
    gl.useProgram(this.splat); gl.bindVertexArray(this.vaoS);
    gl.bindTexture(gl.TEXTURE_2D, this.tFace); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.FW * 2, this.FH, gl.RGBA, gl.FLOAT, T);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.order);
    [['uLocal', this.tLocal], ['uRotS', this.tRotS], ['uBind', this.tBind], ['uFace', this.tFace], ['uColor', this.tColor]]
      .forEach(([name, t], i) => { gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, t); gl.uniform1i(this.uS[name], i); });
    gl.activeTexture(gl.TEXTURE0);
    gl.uniformMatrix4fv(this.uS.uView, false, camera.view);
    const fy = 0.5 * h / Math.tan(camera.fovY / 2);
    gl.uniform2f(this.uS.uFocal, fy, fy); gl.uniform2f(this.uS.uViewport, w, h);
    gl.uniform1f(this.uS.uNear, this.near); gl.uniform1f(this.uS.uFar, this.far);
    gl.depthMask(false); gl.enable(gl.BLEND); gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
    gl.drawElementsInstanced(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0, this.N);
    // pass 3: body colour underneath
    if (showBody) {
      gl.useProgram(this.mesh); gl.bindVertexArray(this.vaoM);
      gl.drawElements(gl.TRIANGLES, this.faces.length, gl.UNSIGNED_INT, 0);
    }
    gl.depthMask(true); gl.bindVertexArray(null);
  }
}
