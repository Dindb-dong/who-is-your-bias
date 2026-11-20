// @ts-nocheck -- WebGL2 gaussian splat renderer
// Source: who-is-your-bias PR #21, 61385492e18c47a49104ba26cd9cbab46e6068f2.
// WebGL2 splat renderer for a FLAME-bound Gaussian head.
// Static per-gaussian data lives in textures; per frame only the face frames (8 floats x F)
// and a depth-sorted index buffer are uploaded. World transforms are applied in the vertex shader.
import { FlameHead, gaussianCenters } from './gaussianHeadCore';
import { cleanOpacity } from './gaussianMatte';

const VS = `#version 300 es
precision highp float; precision highp int; precision highp usampler2D;
uniform sampler2D uLocal;   // N x 4: local xyz, opacity(sigmoid)
uniform sampler2D uRotS;    // N x 2 texels: (qw qx qy qz), (sx sy sz, -)
uniform usampler2D uBind;   // N x 1: face index
uniform sampler2D uFace;    // F x 2 texels: (cx cy cz s), (qw qx qy qz)
uniform sampler2D uColor;   // N x 1: rgb
uniform mat4 uView; uniform vec2 uFocal; uniform vec2 uViewport;
in vec2 aCorner; in uint aIndex;
out vec4 vColor; out vec2 vPos;
ivec2 at(uint i, int w) { return ivec2(int(i) % w, int(i) / w); }
vec3 qrot(vec4 q, vec3 v) { vec3 u = q.yzw; return v + 2.0 * cross(u, cross(u, v) + q.x * v); }
vec4 qmul(vec4 a, vec4 b) { return vec4(a.x*b.x - dot(a.yzw,b.yzw), a.x*b.yzw + b.x*a.yzw + cross(a.yzw,b.yzw)); }
mat3 qmat(vec4 q) {
  float w=q.x,x=q.y,y=q.z,z=q.w;
  return mat3(1.-2.*(y*y+z*z), 2.*(x*y+w*z), 2.*(x*z-w*y),
              2.*(x*y-w*z), 1.-2.*(x*x+z*z), 2.*(y*z+w*x),
              2.*(x*z+w*y), 2.*(y*z-w*x), 1.-2.*(x*x+y*y));
}
void main() {
  int W = textureSize(uLocal, 0).x; int WF = textureSize(uFace, 0).x / 2 * 2;
  vec4 L = texelFetch(uLocal, at(aIndex, W), 0);
  uint f = texelFetch(uBind, at(aIndex, W), 0).r;
  int fw = textureSize(uFace, 0).x / 2;
  ivec2 fp = ivec2(int(f) % fw * 2, int(f) / fw);
  vec4 F0 = texelFetch(uFace, fp, 0), F1 = texelFetch(uFace, fp + ivec2(1, 0), 0);
  int WR = textureSize(uRotS, 0).x / 2;
  ivec2 rp = ivec2(int(aIndex) % WR * 2, int(aIndex) / WR);
  vec4 q = normalize(texelFetch(uRotS, rp, 0)); vec3 sc = texelFetch(uRotS, rp + ivec2(1, 0), 0).xyz;
  vec4 qf = normalize(F1);
  vec3 world = qrot(qf, L.xyz) * F0.w + F0.xyz;
  mat3 R = qmat(qmul(qf, q)); vec3 S = sc * F0.w;
  mat3 M = R * mat3(S.x,0.,0., 0.,S.y,0., 0.,0.,S.z);
  mat3 Sigma = M * transpose(M);
  vec4 cam = uView * vec4(world, 1.0);
  float z = -cam.z;
  if (z < 0.01) { gl_Position = vec4(0., 0., 2., 1.); return; }
  mat3 Wv = mat3(uView);
  mat3 J = mat3(uFocal.x / z, 0., 0.,  0., uFocal.y / z, 0.,  uFocal.x * cam.x / (z*z), uFocal.y * cam.y / (z*z), 0.);
  mat3 T = J * Wv;
  mat3 cov = T * Sigma * transpose(T);
  float a = cov[0][0] + 0.3, b = cov[0][1], c = cov[1][1] + 0.3;
  float mid = 0.5 * (a + c), rad = length(vec2(0.5 * (a - c), b));
  float l1 = mid + rad, l2 = max(mid - rad, 0.1);
  if (l2 < 0.0) { gl_Position = vec4(0., 0., 2., 1.); return; }
  vec2 d1 = normalize(vec2(b, l1 - a)); if (abs(b) < 1e-9) d1 = a >= c ? vec2(1., 0.) : vec2(0., 1.);
  vec2 d2 = vec2(-d1.y, d1.x);
  vec2 centerPx = vec2(uFocal.x * cam.x / z, uFocal.y * cam.y / z);
  vec2 off = aCorner * 3.0;
  vec2 px = centerPx + off.x * sqrt(l1) * d1 + off.y * sqrt(l2) * d2;
  gl_Position = vec4(px / (0.5 * uViewport), 0.0, 1.0);
  vPos = off;
  vColor = vec4(texelFetch(uColor, at(aIndex, W), 0).rgb, L.w);
}`;
const FS = `#version 300 es
precision highp float;
in vec4 vColor; in vec2 vPos; out vec4 o;
void main() {
  float r2 = dot(vPos, vPos); if (r2 > 9.0) discard;
  float al = min(0.99, vColor.a * exp(-0.5 * r2)); if (al < 1.0/255.0) discard;
  o = vec4(vColor.rgb * al, al);
}`;

function tex(gl, internal, format, type, w, h, data) {
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, data);
  for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
  return t;
}

export class GaussianHeadRenderer {
  constructor(canvas, header, B, cleanMatte) {
    const gl = canvas.getContext('webgl2', { premultipliedAlpha: true, alpha: true, antialias: false });
    if (!gl) throw new Error('WebGL2 is required');
    this.gl = gl; this.canvas = canvas; this.header = header; this.B = B;
    this.head = new FlameHead(header, B);
    const N = header.n_gauss, F = header.n_faces; this.N = N;
    this.head.verts.set(B.v_shaped);
    const restCenters = cleanMatte ? gaussianCenters(B, this.head.updateFaceFrames()) : null;
    const prog = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, VS], [gl.FRAGMENT_SHADER, FS]]) {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      gl.attachShader(prog, s); gl.deleteShader(s);
    }
    gl.linkProgram(prog); if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    this.prog = prog; gl.useProgram(prog);
    // static textures
    const W = 2048, H = Math.ceil(N / W), sig = x => 1 / (1 + Math.exp(-x));
    const local = new Float32Array(W * H * 4), color = new Float32Array(W * H * 4), bind = new Uint32Array(W * H);
    const WR = 1024, HR = Math.ceil(N / WR), rots = new Float32Array(WR * 2 * HR * 4);
    for (let n = 0; n < N; n++) {
      local.set([B.g_xyz[3 * n], B.g_xyz[3 * n + 1], B.g_xyz[3 * n + 2], sig(B.g_opacity[n])], 4 * n);
      for (let c = 0; c < 3; c++) color[4 * n + c] = Math.min(1, Math.max(0, 0.5 + 0.28209479 * B.g_color[3 * n + c]));
      if (restCenters) {
        const o = 3 * n, c = 4 * n;
        local[c + 3] *= cleanOpacity(restCenters[o], restCenters[o + 1], restCenters[o + 2], color[c], color[c + 1], color[c + 2]);
      }
      bind[n] = B.g_binding[n];
      const r = 8 * n;
      rots.set([B.g_rot[4 * n], B.g_rot[4 * n + 1], B.g_rot[4 * n + 2], B.g_rot[4 * n + 3]], r);
      rots.set([Math.exp(B.g_scale[3 * n]), Math.exp(B.g_scale[3 * n + 1]), Math.exp(B.g_scale[3 * n + 2]), 0], r + 4);
    }
    this.tLocal = tex(gl, gl.RGBA32F, gl.RGBA, gl.FLOAT, W, H, local);
    this.tColor = tex(gl, gl.RGBA32F, gl.RGBA, gl.FLOAT, W, H, color);
    this.tBind = tex(gl, gl.R32UI, gl.RED_INTEGER, gl.UNSIGNED_INT, W, H, bind);
    this.tRotS = tex(gl, gl.RGBA32F, gl.RGBA, gl.FLOAT, WR * 2, HR, rots);
    this.FW = 1024; this.FH = Math.ceil(F / this.FW);
    this.faceTexData = new Float32Array(this.FW * 2 * this.FH * 4);
    this.tFace = tex(gl, gl.RGBA32F, gl.RGBA, gl.FLOAT, this.FW * 2, this.FH, null);
    // geometry
    this.vao = gl.createVertexArray(); gl.bindVertexArray(this.vao);
    this.quad = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, 1, 1, -1, 1]), gl.STATIC_DRAW);
    const aC = gl.getAttribLocation(prog, 'aCorner'); gl.enableVertexAttribArray(aC); gl.vertexAttribPointer(aC, 2, gl.FLOAT, false, 0, 0);
    this.indexBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuf);
    gl.bufferData(gl.ARRAY_BUFFER, N * 4, gl.DYNAMIC_DRAW);
    const aI = gl.getAttribLocation(prog, 'aIndex'); gl.enableVertexAttribArray(aI); gl.vertexAttribIPointer(aI, 1, gl.UNSIGNED_INT, 0, 0); gl.vertexAttribDivisor(aI, 1);
    this.ebo = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ebo); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
    this.u = Object.fromEntries(['uLocal', 'uRotS', 'uBind', 'uFace', 'uColor', 'uView', 'uFocal', 'uViewport'].map(k => [k, gl.getUniformLocation(prog, k)]));
    this.centers = new Float32Array(N * 3); this.depth = new Float32Array(N);
    this.order = new Uint32Array(N); this.counts = new Uint32Array(65536);
    this.stats = { cpuMs: 0 };
  }

  dispose() {
    const gl = this.gl;
    gl.bindVertexArray(null);
    gl.useProgram(null);
    for (const texture of [this.tLocal, this.tRotS, this.tBind, this.tFace, this.tColor]) gl.deleteTexture(texture);
    for (const buffer of [this.quad, this.indexBuf, this.ebo]) gl.deleteBuffer(buffer);
    gl.deleteVertexArray(this.vao);
    gl.deleteProgram(this.prog);
  }

  sortByDepth(view) {
    const { N, centers, depth, order, counts } = this;
    const v2 = view[2], v6 = view[6], v10 = view[10], v14 = view[14];
    let mn = Infinity, mx = -Infinity;
    for (let n = 0; n < N; n++) {
      const d = -(v2 * centers[3 * n] + v6 * centers[3 * n + 1] + v10 * centers[3 * n + 2] + v14); // distance along view dir
      depth[n] = d; if (d < mn) mn = d; if (d > mx) mx = d;
    }
    const k = 65535 / Math.max(1e-6, mx - mn); counts.fill(0);
    const key = new Uint16Array(N);
    for (let n = 0; n < N; n++) { key[n] = ((depth[n] - mn) * k) | 0; counts[key[n]]++; }
    for (let i = 1; i < 65536; i++) counts[i] += counts[i - 1];
    for (let n = N - 1; n >= 0; n--) order[--counts[key[n]]] = n; // ascending depth = front to back
  }

  render(params, camera) {
    const gl = this.gl, t0 = performance.now();
    this.head.forward(params.expr, params.pose, params.trans);
    const fd = this.head.updateFaceFrames();
    const F = this.header.n_faces, T = this.faceTexData;
    for (let k = 0; k < F; k++) { const o = 8 * k, r = 8 * k; T[r] = fd[o]; T[r + 1] = fd[o + 1]; T[r + 2] = fd[o + 2]; T[r + 3] = fd[o + 7]; T[r + 4] = fd[o + 3]; T[r + 5] = fd[o + 4]; T[r + 6] = fd[o + 5]; T[r + 7] = fd[o + 6]; }
    gaussianCenters(this.B, fd, this.centers);
    this.sortByDepth(camera.view);
    this.stats.cpuMs = performance.now() - t0;
    const w = this.canvas.width, h = this.canvas.height;
    gl.viewport(0, 0, w, h); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.prog); gl.bindVertexArray(this.vao);
    gl.bindTexture(gl.TEXTURE_2D, this.tFace); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.FW * 2, this.FH, gl.RGBA, gl.FLOAT, T);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.order);
    const units = [['uLocal', this.tLocal], ['uRotS', this.tRotS], ['uBind', this.tBind], ['uFace', this.tFace], ['uColor', this.tColor]];
    units.forEach(([name, t], i) => { gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, t); gl.uniform1i(this.u[name], i); });
    gl.uniformMatrix4fv(this.u.uView, false, camera.view);
    const fy = 0.5 * h / Math.tan(camera.fovY / 2);
    gl.uniform2f(this.u.uFocal, fy, fy); gl.uniform2f(this.u.uViewport, w, h);
    gl.enable(gl.BLEND); gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
    gl.disable(gl.DEPTH_TEST);
    gl.drawElementsInstanced(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0, this.N);
  }
}

export function lookAt(eye, target, up = [0, 1, 0]) {
  const zx = eye[0] - target[0], zy = eye[1] - target[1], zz = eye[2] - target[2];
  const zl = Math.hypot(zx, zy, zz), z = [zx / zl, zy / zl, zz / zl];
  let x = [up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0]];
  const xl = Math.hypot(...x); x = x.map(c => c / xl);
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  const d = (a) => -(a[0] * eye[0] + a[1] * eye[1] + a[2] * eye[2]);
  // column-major 4x4
  return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, d(x), d(y), d(z), 1]);
}
