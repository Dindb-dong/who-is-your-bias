// @ts-nocheck -- numeric WebGL/FLAME core validated against the numpy reference
// FLAME-bound Gaussian head: bundle loading + per-frame mesh/face-frame math (no WebGL here).
// Mirrors head_bundle.py (numpy reference). Works in browsers and Node.

function f16ToF32(src) {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const h = src[i], s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    out[i] = e === 0 ? s * m * 5.960464477539063e-8 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * Math.pow(2, e - 15);
  }
  return out;
}

export function parseBundle(header, buffer) {
  if (header.schema !== 'gaussian-head.v1' || !(buffer instanceof ArrayBuffer) || !header.buffers || typeof header.buffers !== 'object') throw new Error('Invalid Gaussian head bundle');
  const B: Record<string, Float32Array | Uint32Array> = {};
  for (const [name, b] of Object.entries(header.buffers)) {
    if (!Array.isArray(b.shape) || !b.shape.length || b.shape.some((size) => !Number.isSafeInteger(size) || size < 0)) throw new Error('Invalid buffer shape: ' + name);
    const n = b.shape.reduce((a, c) => a * c, 1);
    const width = b.dtype === 'f16' ? 2 : b.dtype === 'f32' || b.dtype === 'u32' ? 4 : 0;
    if (!width || !Number.isSafeInteger(n) || !Number.isSafeInteger(b.offset) || b.offset < 0 || b.offset % width !== 0
        || b.bytes !== n * width || b.offset + b.bytes > buffer.byteLength) throw new Error('Invalid or truncated buffer: ' + name);
    if (b.dtype === 'f32') B[name] = new Float32Array(buffer, b.offset, n);
    else if (b.dtype === 'u32') B[name] = new Uint32Array(buffer, b.offset, n);
    else if (b.dtype === 'f16') B[name] = f16ToF32(new Uint16Array(buffer, b.offset, n));
    else throw new Error('dtype ' + b.dtype);
  }
  return B;
}

/** Reject incomplete geometry before hiding the body's original face. Buffer
 * offsets alone cannot detect a wrong vertex count or dangling face binding. */
export function validateGaussianBundle(header, B) {
  for (const [field, limit] of [['n_verts', 1000000], ['n_faces', 2000000], ['n_gauss', 2000000], ['n_expr', 512], ['n_frames', 1000000]]) {
    if (!Number.isSafeInteger(header[field]) || header[field] < 1 || header[field] > limit) throw new Error('Invalid Gaussian dimension: ' + field);
  }
  if (!Number.isFinite(header.fps) || header.fps <= 0 || header.fps > 240
      || !Array.isArray(header.parents) || header.parents.join(',') !== '-1,0,1,1,1') throw new Error('Invalid FLAME motion or joint hierarchy');
  const V = header.n_verts, F = header.n_faces, E = header.n_expr, N = header.n_gauss, T = header.n_frames;
  const shapes = { v_shaped: [V, 3], expr_dirs: [E, V * 3], pose_dirs: [36, V * 3], lbs_weights: [V, 5],
    j_base: [5, 3], j_expr_dirs: [E, 15], faces: [F, 3], g_xyz: [N, 3], g_rot: [N, 4], g_scale: [N, 3],
    g_opacity: [N], g_color: [N, 3], g_binding: [N], m_expr: [T, E], m_pose: [T, 15], m_trans: [T, 3] };
  for (const [name, shape] of Object.entries(shapes)) {
    const data = B[name], declared = header.buffers[name];
    if (!data || !declared || declared.shape.join(',') !== shape.join(',') || data.length !== shape.reduce((a, n) => a * n, 1)) throw new Error('Missing or mismatched Gaussian array: ' + name);
    if ((name === 'faces' || name === 'g_binding') !== (data instanceof Uint32Array)) throw new Error('Invalid Gaussian array dtype: ' + name);
    for (let i = 0; i < data.length; i++) if (!Number.isFinite(data[i])) throw new Error('Nonfinite Gaussian array: ' + name);
  }
  for (const index of B.faces) if (index >= V) throw new Error('Gaussian face refers to an unavailable vertex');
  for (const index of B.g_binding) if (index >= F) throw new Error('Gaussian binding refers to an unavailable face');
  for (let i = 0; i < N; i++) if (Math.hypot(...B.g_rot.subarray(i * 4, i * 4 + 4)) < 1e-12) throw new Error('A Gaussian has no valid rotation');
  for (let i = 0; i < V; i++) {
    let sum = 0;
    for (let j = 0; j < 5; j++) {
      const w = B.lbs_weights[i * 5 + j];
      if (w < 0 || w > 1.00001) throw new Error('Invalid FLAME skinning weight');
      sum += w;
    }
    if (Math.abs(sum - 1) > 0.001) throw new Error('FLAME skinning weights do not sum to one');
  }
}

export async function loadBundle(stemUrl, signal: AbortSignal | undefined = undefined) {
  const manifest = await fetch(stemUrl + '.json', { signal });
  if (!manifest.ok) throw new Error('Could not load head manifest (HTTP ' + manifest.status + ')');
  const header = await manifest.json();
  if (header.schema !== 'gaussian-head.v1') throw new Error('Unsupported head bundle ' + header.schema);
  const response = await fetch(stemUrl + '.bin', { signal });
  if (!response.ok) throw new Error('Could not load head data (HTTP ' + response.status + ')');
  const buffer = await response.arrayBuffer();
  const B = parseBundle(header, buffer);
  validateGaussianBundle(header, B);
  const hash = await crypto.subtle.digest('SHA-256', buffer);
  signal?.throwIfAborted();
  const sha256 = [...new Uint8Array(hash)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
  return { header, B, sha256 };
}

function rodrigues(x, y, z, out, o) {
  const th = Math.hypot(x, y, z) + 1e-8, kx = x / th, ky = y / th, kz = z / th;
  const s = Math.sin(th), c1 = 1 - Math.cos(th);
  // R = I + s K + (1-c) K^2, row-major
  out[o] = 1 + c1 * (-ky * ky - kz * kz); out[o + 1] = -s * kz + c1 * kx * ky; out[o + 2] = s * ky + c1 * kx * kz;
  out[o + 3] = s * kz + c1 * kx * ky; out[o + 4] = 1 + c1 * (-kx * kx - kz * kz); out[o + 5] = -s * kx + c1 * ky * kz;
  out[o + 6] = -s * ky + c1 * kx * kz; out[o + 7] = s * kx + c1 * ky * kz; out[o + 8] = 1 + c1 * (-kx * kx - ky * ky);
}

export class FlameHead {
  constructor(header, B) {
    this.V = header.n_verts; this.F = header.n_faces; this.E = header.n_expr; this.parents = header.parents;
    this.B = B;
    this.v = new Float32Array(this.V * 3);
    this.verts = new Float32Array(this.V * 3);
    this.J = new Float32Array(15);
    this.R = new Float32Array(45);
    this.A = new Float32Array(5 * 12);   // 3x4 row-major per joint
    this.faceData = new Float32Array(this.F * 8); // cx cy cz qw qx qy qz s
  }

  forward(expr, pose, trans) {
    const { V, E, B, v, J, R, A } = this, n3 = V * 3;
    v.set(B.v_shaped); J.set(B.j_base);
    for (let e = 0; e < E; e++) {
      const w = expr[e]; if (w === 0) continue;
      const d = B.expr_dirs, o = e * n3;
      for (let i = 0; i < n3; i++) v[i] += w * d[o + i];
      const jd = B.j_expr_dirs, jo = e * 15;
      for (let i = 0; i < 15; i++) J[i] += w * jd[jo + i];
    }
    for (let j = 0; j < 5; j++) rodrigues(pose[3 * j], pose[3 * j + 1], pose[3 * j + 2], R, 9 * j);
    const pd = B.pose_dirs;
    for (let k = 0; k < 36; k++) {
      const j = 1 + Math.floor(k / 9), r = k % 9;
      const w = R[9 * j + r] - (r === 0 || r === 4 || r === 8 ? 1 : 0);
      if (Math.abs(w) < 1e-7) continue;
      const o = k * n3;
      for (let i = 0; i < n3; i++) v[i] += w * pd[o + i];
    }
    // rigid chain: G_i = G_parent * [R_i | J_i - J_parent]; A_i = G_i with t -= R_world J_i
    const G = new Float32Array(60);
    for (let j = 0; j < 5; j++) {
      const p = this.parents[j], r = 9 * j;
      const tx = J[3 * j] - (p >= 0 ? J[3 * p] : 0), ty = J[3 * j + 1] - (p >= 0 ? J[3 * p + 1] : 0), tz = J[3 * j + 2] - (p >= 0 ? J[3 * p + 2] : 0);
      const g = 12 * j;
      if (p < 0) {
        for (let a = 0; a < 3; a++) { G[g + 4 * a] = R[r + 3 * a]; G[g + 4 * a + 1] = R[r + 3 * a + 1]; G[g + 4 * a + 2] = R[r + 3 * a + 2]; }
        G[g + 3] = tx; G[g + 7] = ty; G[g + 11] = tz;
      } else {
        const q = 12 * p;
        for (let a = 0; a < 3; a++) {
          for (let b = 0; b < 3; b++) G[g + 4 * a + b] = G[q + 4 * a] * R[r + b] + G[q + 4 * a + 1] * R[r + 3 + b] + G[q + 4 * a + 2] * R[r + 6 + b];
          G[g + 4 * a + 3] = G[q + 4 * a] * tx + G[q + 4 * a + 1] * ty + G[q + 4 * a + 2] * tz + G[q + 4 * a + 3];
        }
      }
    }
    for (let j = 0; j < 5; j++) {
      const g = 12 * j;
      for (let a = 0; a < 3; a++) {
        A[g + 4 * a] = G[g + 4 * a]; A[g + 4 * a + 1] = G[g + 4 * a + 1]; A[g + 4 * a + 2] = G[g + 4 * a + 2];
        A[g + 4 * a + 3] = G[g + 4 * a + 3] - (G[g + 4 * a] * J[3 * j] + G[g + 4 * a + 1] * J[3 * j + 1] + G[g + 4 * a + 2] * J[3 * j + 2]);
      }
    }
    const W = B.lbs_weights, out = this.verts, T = new Float32Array(12);
    for (let i = 0; i < V; i++) {
      T.fill(0);
      for (let j = 0; j < 5; j++) {
        const w = W[5 * i + j]; if (w === 0) continue;
        for (let k = 0; k < 12; k++) T[k] += w * A[12 * j + k];
      }
      const x = v[3 * i], y = v[3 * i + 1], z = v[3 * i + 2];
      out[3 * i] = T[0] * x + T[1] * y + T[2] * z + T[3] + trans[0];
      out[3 * i + 1] = T[4] * x + T[5] * y + T[6] * z + T[7] + trans[1];
      out[3 * i + 2] = T[8] * x + T[9] * y + T[10] * z + T[11] + trans[2];
    }
    return out;
  }

  // per-face centre, orientation quaternion (wxyz) and scale, matching GaussianAvatars compute_face_orientation
  updateFaceFrames() {
    const f = this.B.faces, v = this.verts, D = this.faceData;
    for (let k = 0; k < this.F; k++) {
      const i0 = 3 * f[3 * k], i1 = 3 * f[3 * k + 1], i2 = 3 * f[3 * k + 2];
      const ex = v[i1] - v[i0], ey = v[i1 + 1] - v[i0 + 1], ez = v[i1 + 2] - v[i0 + 2];
      const fx = v[i2] - v[i0], fy = v[i2 + 1] - v[i0 + 1], fz = v[i2 + 2] - v[i0 + 2];
      const le = Math.hypot(ex, ey, ez) + 1e-12;
      const a0x = ex / le, a0y = ey / le, a0z = ez / le;
      let a1x = a0y * fz - a0z * fy, a1y = a0z * fx - a0x * fz, a1z = a0x * fy - a0y * fx;
      const l1 = Math.hypot(a1x, a1y, a1z) + 1e-12; a1x /= l1; a1y /= l1; a1z /= l1;
      // a2 = -normalize(a1 x a0)
      let a2x = -(a1y * a0z - a1z * a0y), a2y = -(a1z * a0x - a1x * a0z), a2z = -(a1x * a0y - a1y * a0x);
      const l2 = Math.hypot(a2x, a2y, a2z) + 1e-12; a2x /= l2; a2y /= l2; a2z /= l2;
      const s = (le + Math.abs(a2x * fx + a2y * fy + a2z * fz)) / 2;
      // rotation matrix columns a0 a1 a2 -> quaternion
      const m00 = a0x, m01 = a1x, m02 = a2x, m10 = a0y, m11 = a1y, m12 = a2y, m20 = a0z, m21 = a1z, m22 = a2z;
      const tr = m00 + m11 + m22; let qw, qx, qy, qz;
      if (tr > 0) { const S = Math.sqrt(tr + 1) * 2; qw = 0.25 * S; qx = (m21 - m12) / S; qy = (m02 - m20) / S; qz = (m10 - m01) / S; }
      else if (m00 > m11 && m00 > m22) { const S = Math.sqrt(1 + m00 - m11 - m22) * 2; qw = (m21 - m12) / S; qx = 0.25 * S; qy = (m01 + m10) / S; qz = (m02 + m20) / S; }
      else if (m11 > m22) { const S = Math.sqrt(1 + m11 - m00 - m22) * 2; qw = (m02 - m20) / S; qx = (m01 + m10) / S; qy = 0.25 * S; qz = (m12 + m21) / S; }
      else { const S = Math.sqrt(1 + m22 - m00 - m11) * 2; qw = (m10 - m01) / S; qx = (m02 + m20) / S; qy = (m12 + m21) / S; qz = 0.25 * S; }
      const o = 8 * k;
      D[o] = (v[i0] + v[i1] + v[i2]) / 3; D[o + 1] = (v[i0 + 1] + v[i1 + 1] + v[i2 + 1]) / 3; D[o + 2] = (v[i0 + 2] + v[i1 + 2] + v[i2 + 2]) / 3;
      D[o + 3] = qw; D[o + 4] = qx; D[o + 5] = qy; D[o + 6] = qz; D[o + 7] = s;
    }
    return D;
  }
}

// world-space gaussian centres (used for depth sorting and tests)
export function gaussianCenters(B, faceData, out) {
  const N = B.g_binding.length, X = B.g_xyz, b = B.g_binding;
  out = out || new Float32Array(N * 3);
  for (let n = 0; n < N; n++) {
    const o = 8 * b[n], w = faceData[o + 3], x = faceData[o + 4], y = faceData[o + 5], z = faceData[o + 6], s = faceData[o + 7];
    const px = X[3 * n], py = X[3 * n + 1], pz = X[3 * n + 2];
    // rotate p by q: p + 2w(q x p) + 2 q x (q x p)
    const cx = y * pz - z * py, cy = z * px - x * pz, cz = x * py - y * px;
    const rx = px + 2 * (w * cx + y * cz - z * cy), ry = py + 2 * (w * cy + z * cx - x * cz), rz = pz + 2 * (w * cz + x * cy - y * cx);
    out[3 * n] = rx * s + faceData[o]; out[3 * n + 1] = ry * s + faceData[o + 1]; out[3 * n + 2] = rz * s + faceData[o + 2];
  }
  return out;
}

export function frameParams(B, header, t) {
  const E = header.n_expr, T = header.n_frames, f = ((t % T) + T) % T;
  return { expr: B.m_expr.subarray(f * E, f * E + E), pose: B.m_pose.subarray(f * 15, f * 15 + 15), trans: B.m_trans.subarray(f * 3, f * 3 + 3) };
}
