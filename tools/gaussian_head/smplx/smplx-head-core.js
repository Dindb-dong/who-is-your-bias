// SMPL-X body with a transplanted FLAME gaussian head (schema smplx-head-preview.v1).
// Head/teeth vertices: FLAME expression basis + FLAME pose correctives (joints 15, 22, 23, 24) + their skinning.
// Body vertices: SMPL-X skinning only (no body pose correctives in the preview).
import { parseBundle } from '../bundle/gaussian-head-core.js';

export async function loadPreview(stemUrl) {
  const header = await (await fetch(stemUrl + '.json')).json();
  if (header.schema !== 'smplx-head-preview.v1') throw new Error('Unsupported preview bundle ' + header.schema);
  const buffer = await (await fetch(stemUrl + '.bin')).arrayBuffer();
  return { header, B: parseBundle(header, buffer) };
}

const HEAD_POSE_JOINTS = [15, 22, 23, 24];

function rodrigues(x, y, z, out, o) {
  const th = Math.hypot(x, y, z);
  if (th < 1e-12) { out.set([1, 0, 0, 0, 1, 0, 0, 0, 1], o); return; }
  const kx = x / th, ky = y / th, kz = z / th, s = Math.sin(th), c1 = 1 - Math.cos(th);
  out[o] = 1 + c1 * (-ky * ky - kz * kz); out[o + 1] = -s * kz + c1 * kx * ky; out[o + 2] = s * ky + c1 * kx * kz;
  out[o + 3] = s * kz + c1 * kx * ky; out[o + 4] = 1 + c1 * (-kx * kx - kz * kz); out[o + 5] = -s * kx + c1 * ky * kz;
  out[o + 6] = -s * ky + c1 * kx * kz; out[o + 7] = s * kx + c1 * ky * kz; out[o + 8] = 1 + c1 * (-kx * kx - ky * ky);
}

export class SmplxHead {
  constructor(header, B) {
    this.header = header; this.B = B;
    this.N = header.n_verts; this.NH = header.n_head_verts; this.NJ = header.n_joints; this.E = header.n_expr; this.K = header.influences;
    this.F = header.n_faces_g; this.mapped = header.mapped_joints;
    this.parents = Array.from(B.parents, p => (p === 0xFFFFFFFF ? -1 : p));
    this.vh = new Float32Array(this.NH * 3);          // deformed head rest positions
    this.verts = new Float32Array(this.N * 3);
    this.J = new Float32Array(this.NJ * 3); this.R = new Float32Array(this.NJ * 9);
    this.G = new Float32Array(this.NJ * 12); this.A = new Float32Array(this.NJ * 12);
    this.faceData = new Float32Array(this.F * 8);
    this.isHead = new Uint8Array(this.N); for (const v of B.vmap) this.isHead[v] = 1;
  }

  /** pose: Float32Array(NJ*3) axis-angle per SMPL-X joint; expr: FLAME expression (E); trans: root translation (3) */
  forward(pose, expr, trans) {
    const { B, NH, NJ, E, vh, J, R, G, A } = this, n3 = NH * 3;
    J.set(B.j_rest);
    for (let k = 0; k < NH; k++) { const v = B.vmap[k]; vh[3 * k] = B.v_rest[3 * v]; vh[3 * k + 1] = B.v_rest[3 * v + 1]; vh[3 * k + 2] = B.v_rest[3 * v + 2]; }
    for (let e = 0; e < E; e++) {
      const w = expr[e]; if (w === 0) continue;
      const d = B.expr_head, o = e * n3;
      for (let i = 0; i < n3; i++) vh[i] += w * d[o + i];
      for (let m = 0; m < 5; m++) { const j = this.mapped[m]; for (let c = 0; c < 3; c++) J[3 * j + c] += w * B.j_expr[e * 15 + 3 * m + c]; }
    }
    for (let j = 0; j < NJ; j++) rodrigues(pose[3 * j], pose[3 * j + 1], pose[3 * j + 2], R, 9 * j);
    for (let k = 0; k < 36; k++) {
      const j = HEAD_POSE_JOINTS[Math.floor(k / 9)], r = k % 9;
      const w = R[9 * j + r] - (r === 0 || r === 4 || r === 8 ? 1 : 0);
      if (Math.abs(w) < 1e-7) continue;
      const pd = B.pose_head, o = k * n3;
      for (let i = 0; i < n3; i++) vh[i] += w * pd[o + i];
    }
    for (let j = 0; j < NJ; j++) {
      const p = this.parents[j], r = 9 * j, g = 12 * j;
      const tx = J[3 * j] - (p >= 0 ? J[3 * p] : 0), ty = J[3 * j + 1] - (p >= 0 ? J[3 * p + 1] : 0), tz = J[3 * j + 2] - (p >= 0 ? J[3 * p + 2] : 0);
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
    for (let j = 0; j < NJ; j++) {
      const g = 12 * j;
      for (let a = 0; a < 3; a++) {
        A[g + 4 * a] = G[g + 4 * a]; A[g + 4 * a + 1] = G[g + 4 * a + 1]; A[g + 4 * a + 2] = G[g + 4 * a + 2];
        A[g + 4 * a + 3] = G[g + 4 * a + 3] - (G[g + 4 * a] * J[3 * j] + G[g + 4 * a + 1] * J[3 * j + 1] + G[g + 4 * a + 2] * J[3 * j + 2]);
      }
    }
    // rest positions: body from v_rest, head from vh
    const out = this.verts, K = this.K, wi = B.w_idx, wv = B.w_val, T = new Float32Array(12);
    out.set(B.v_rest);
    for (let k = 0; k < NH; k++) { const v = B.vmap[k]; out[3 * v] = vh[3 * k]; out[3 * v + 1] = vh[3 * k + 1]; out[3 * v + 2] = vh[3 * k + 2]; }
    for (let i = 0; i < this.N; i++) {
      T.fill(0);
      for (let k = 0; k < K; k++) {
        const w = wv[K * i + k]; if (w === 0) continue;
        const a = 12 * wi[K * i + k];
        for (let c = 0; c < 12; c++) T[c] += w * A[a + c];
      }
      const x = out[3 * i], y = out[3 * i + 1], z = out[3 * i + 2];
      out[3 * i] = T[0] * x + T[1] * y + T[2] * z + T[3] + trans[0];
      out[3 * i + 1] = T[4] * x + T[5] * y + T[6] * z + T[7] + trans[1];
      out[3 * i + 2] = T[8] * x + T[9] * y + T[10] * z + T[11] + trans[2];
    }
    return out;
  }

  // per gaussian-binding triangle: centre, orientation quaternion (wxyz), scale (same maths as FlameHead)
  updateFaceFrames() {
    const f = this.B.faces_g, v = this.verts, D = this.faceData;
    for (let k = 0; k < this.F; k++) {
      const i0 = 3 * f[3 * k], i1 = 3 * f[3 * k + 1], i2 = 3 * f[3 * k + 2];
      const ex = v[i1] - v[i0], ey = v[i1 + 1] - v[i0 + 1], ez = v[i1 + 2] - v[i0 + 2];
      const fx = v[i2] - v[i0], fy = v[i2 + 1] - v[i0 + 1], fz = v[i2 + 2] - v[i0 + 2];
      const le = Math.hypot(ex, ey, ez) + 1e-12;
      const a0x = ex / le, a0y = ey / le, a0z = ez / le;
      let a1x = a0y * fz - a0z * fy, a1y = a0z * fx - a0x * fz, a1z = a0x * fy - a0y * fx;
      const l1 = Math.hypot(a1x, a1y, a1z) + 1e-12; a1x /= l1; a1y /= l1; a1z /= l1;
      let a2x = -(a1y * a0z - a1z * a0y), a2y = -(a1z * a0x - a1x * a0z), a2z = -(a1x * a0y - a1y * a0x);
      const l2 = Math.hypot(a2x, a2y, a2z) + 1e-12; a2x /= l2; a2y /= l2; a2z /= l2;
      const s = (le + Math.abs(a2x * fx + a2y * fy + a2z * fz)) / 2;
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

/** Map a tracked FLAME pose (15) onto SMPL-X joints, on top of a body pose. */
export function applyFlamePose(pose55, flame15, scale = 1) {
  const map = [12, 15, 22, 23, 24];
  for (let m = 0; m < 5; m++) for (let c = 0; c < 3; c++) pose55[3 * map[m] + c] += scale * flame15[3 * m + c];
  return pose55;
}
