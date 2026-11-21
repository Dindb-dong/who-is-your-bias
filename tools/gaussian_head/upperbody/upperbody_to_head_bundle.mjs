// Re-bind an upper-body model's head + neck + collar gaussians to the FLAME head bundle the studio already renders.
//
//   node upperbody_to_head_bundle.mjs <upperbody_stem> <site_head_stem> <out_stem> [--keep 12,13,14] [--white 0.6]
//
// <upperbody_stem>: export_upperbody.py output (smplx-head-preview.v1). Its first faces are the FLAME head + teeth faces
//   in the same order as the head bundle, and its head vertices are the head bundle's FLAME rest vertices translated.
// <site_head_stem>: the gaussian-head.v1 bundle the studio uses (FLAME arrays, expression replay); copied unchanged
//   except the gaussians.
// Head gaussians keep their FLAME binding. Body gaussians are kept when their face is skinned mainly to --keep joints
// (SMPL-X neck 12, collars 13/14) and they are not near-white background haze (all DC channels > --white); they are
// re-expressed on the nearest FLAME face at rest, so the studio's existing head attachment, neck seam and expression
// replay drive them. Writes <out_stem>.json/.bin plus a recommended negative collar cut (collar_cut_m).
// Outputs are derived from FLAME/SMPL-X and research-licensed footage: never commit them.
import { readFileSync, writeFileSync } from 'node:fs';
import { parseBundle, FlameHead } from '../bundle/gaussian-head-core.js';
import { SmplxHead } from '../smplx/smplx-head-core.js';

const args = process.argv.slice(2);
const opt = (name, fallback) => { const i = args.indexOf(name); return i >= 0 ? args.splice(i, 2)[1] : fallback; };
const keep = new Set(opt('--keep', '12,13,14').split(',').map(Number));
const white = Number(opt('--white', '0.6'));
const [upperStem, siteStem, outStem] = args;
if (!outStem) throw new Error('usage: node upperbody_to_head_bundle.mjs <upperbody_stem> <site_head_stem> <out_stem> [--keep 12,13,14] [--white 0.6]');

const load = (stem) => {
  const header = JSON.parse(readFileSync(stem + '.json', 'utf8')), raw = readFileSync(stem + '.bin');
  const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  return { header, raw, B: parseBundle(header, buffer) };
};
const up = load(upperStem), site = load(siteStem);
if (up.header.schema !== 'smplx-head-preview.v1' || site.header.schema !== 'gaussian-head.v1') throw new Error('unexpected bundle schemas');

// ---- rest face frames of both meshes
const smplx = new SmplxHead(up.header, up.B);
smplx.forward(new Float32Array(up.header.n_joints * 3), new Float32Array(up.header.n_expr), [0, 0, 0]);
const upFrames = smplx.updateFaceFrames();
const flame = new FlameHead(site.header, site.B);
flame.forward(new Float32Array(site.header.n_expr), new Float32Array(15), new Float32Array(3));
const siteFrames = flame.updateFaceFrames();
const FH = site.header.n_faces, vmap = up.B.vmap;

// same FLAME topology, and head vertices differ from the site rest mesh by one translation
for (let f = 0; f < FH; f++) for (let c = 0; c < 3; c++) {
  if (up.B.faces_g[3 * f + c] !== vmap[site.B.faces[3 * f + c]]) throw new Error(`face ${f} differs between the bundles`);
}
const t = [0, 1, 2].map((c) => up.B.v_rest[3 * vmap[0] + c] - site.B.v_shaped[c]);
let drift = 0;
for (let i = 0; i < site.header.n_verts; i++) for (let c = 0; c < 3; c++) {
  drift = Math.max(drift, Math.abs(up.B.v_rest[3 * vmap[i] + c] - site.B.v_shaped[3 * i + c] - t[c]));
}
console.log(`head rest offset [${t.map((v) => v.toFixed(4))}] m, max deviation ${(drift * 1000).toFixed(3)} mm`);
if (drift > 0.002) throw new Error('head geometry differs from the site bundle by more than 2 mm');

// ---- quaternion helpers (w, x, y, z)
const qmul = (a, b) => [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
  a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
const conj = (q) => [q[0], -q[1], -q[2], -q[3]];
const norm = (q) => { const l = Math.hypot(...q) || 1; return q.map((v) => v / l); };
const qrot = (q, v) => qmul(qmul(q, [0, ...v]), conj(q)).slice(1);
const frame = (D, f) => ({ c: [D[8 * f], D[8 * f + 1], D[8 * f + 2]], q: norm([D[8 * f + 3], D[8 * f + 4], D[8 * f + 5], D[8 * f + 6]]), s: D[8 * f + 7] });

// ---- dominant skinning joint per upper-body vertex
const K = up.header.influences, dom = new Int32Array(up.header.n_verts);
for (let v = 0; v < up.header.n_verts; v++) {
  let best = -1, bw = -1;
  for (let k = 0; k < K; k++) if (up.B.w_val[v * K + k] > bw) { bw = up.B.w_val[v * K + k]; best = up.B.w_idx[v * K + k]; }
  dom[v] = best;
}
// candidate FLAME faces for re-binding: skin faces only (no teeth), by rest centroid
const skinFaces = 9976;
const out = { xyz: [], rot: [], scale: [], opacity: [], color: [], binding: [] };
let head = 0, body = 0, dropped = 0, maxDist = 0;
for (let n = 0; n < up.header.n_gauss; n++) {
  const f = up.B.g_binding[n];
  const local = [up.B.g_xyz[3 * n], up.B.g_xyz[3 * n + 1], up.B.g_xyz[3 * n + 2]];
  const rot = norm([up.B.g_rot[4 * n], up.B.g_rot[4 * n + 1], up.B.g_rot[4 * n + 2], up.B.g_rot[4 * n + 3]]);
  const scale = [up.B.g_scale[3 * n], up.B.g_scale[3 * n + 1], up.B.g_scale[3 * n + 2]];
  const color = [up.B.g_color[3 * n], up.B.g_color[3 * n + 1], up.B.g_color[3 * n + 2]];
  let bindTo = f, L = local, R = rot, S = scale;
  if (f >= FH) {
    const verts = [0, 1, 2].map((c) => up.B.faces_g[3 * f + c]);
    const isHead = (v) => false;                                   // body faces have no FLAME vertices by construction
    if (!verts.every((v) => isHead(v) || keep.has(dom[v]))) { dropped++; continue; }
    if (color.every((d) => 0.5 + 0.28209479 * d > white)) { dropped++; continue; }
    const F = frame(upFrames, f);
    const world = qrot(F.q, local.map((v) => v * F.s)).map((v, i) => v + F.c[i] - t[i]);   // into site FLAME space
    let best = -1, bestD = Infinity;
    for (let g = 0; g < skinFaces; g++) {
      const dx = siteFrames[8 * g] - world[0], dy = siteFrames[8 * g + 1] - world[1], dz = siteFrames[8 * g + 2] - world[2];
      const d = dx * dx + dy * dy + dz * dz; if (d < bestD) { bestD = d; best = g; }
    }
    const G = frame(siteFrames, best);
    bindTo = best;
    L = qrot(conj(G.q), world.map((v, i) => v - G.c[i])).map((v) => v / G.s);
    R = qmul(conj(G.q), qmul(F.q, rot));
    S = scale.map((v) => v + Math.log(F.s / G.s));
    maxDist = Math.max(maxDist, Math.sqrt(bestD));
    body++;
  } else head++;
  out.xyz.push(...L); out.rot.push(...R); out.scale.push(...S); out.opacity.push(up.B.g_opacity[n]); out.color.push(...color); out.binding.push(bindTo);
}
console.log(`kept ${head} head + ${body} neck/collar gaussians, dropped ${dropped}; farthest re-bound gaussian ${(maxDist * 100).toFixed(1)} cm from its FLAME face`);

// lowest kept centre relative to the FLAME rest mesh bottom -> recommended collar cut
let minY = Infinity, flameMinY = Infinity;
for (let i = 1; i < site.B.v_shaped.length; i += 3) flameMinY = Math.min(flameMinY, site.B.v_shaped[i]);
for (let i = 0; i < out.binding.length; i++) {
  const G = frame(siteFrames, out.binding[i]);
  minY = Math.min(minY, qrot(G.q, out.xyz.slice(3 * i, 3 * i + 3).map((v) => v * G.s))[1] + G.c[1]);
}
const collarCut = Math.max(-0.15, Math.floor((minY - flameMinY) * 100) / 100 - 0.01);
console.log(`lowest kept gaussian ${(minY - flameMinY).toFixed(3)} m relative to the FLAME rest bottom -> collar_cut_m ${collarCut}`);

// ---- write: unchanged site buffers byte-for-byte, new gaussian buffers as f32/u32
const N = out.binding.length, replaced = { g_xyz: [Float32Array, out.xyz, [N, 3]], g_rot: [Float32Array, out.rot, [N, 4]],
  g_scale: [Float32Array, out.scale, [N, 3]], g_opacity: [Float32Array, out.opacity, [N]], g_color: [Float32Array, out.color, [N, 3]],
  g_binding: [Uint32Array, out.binding, [N]] };
const parts = [], buffers = {};
let offset = 0;
for (const [name, b] of Object.entries(site.header.buffers)) {
  let bytes, dtype = b.dtype, shape = b.shape;
  if (replaced[name]) {
    const [Type, values, s] = replaced[name];
    bytes = Buffer.from(new Type(values).buffer); dtype = Type === Uint32Array ? 'u32' : 'f32'; shape = s;
  } else {
    const size = b.shape.reduce((a, c) => a * c, 1) * (b.dtype === 'f16' ? 2 : 4);
    bytes = site.raw.subarray(b.offset, b.offset + size);
  }
  buffers[name] = { dtype, shape, offset, bytes: bytes.length };
  parts.push(bytes);
  const pad = (4 - (bytes.length % 4)) % 4; if (pad) parts.push(Buffer.alloc(pad));
  offset += bytes.length + pad;
}
const header = { ...site.header, n_gauss: N, buffers, collar_cut_m: collarCut,
  upperbody: { source: upperStem.split(/[\\/]/).pop(), keep_joints: [...keep], white_threshold: white, head_gaussians: head, neck_collar_gaussians: body },
  source: `${site.header.source}; neck and collar gaussians from the jointly trained upper-body model` };
writeFileSync(outStem + '.json', JSON.stringify(header));
writeFileSync(outStem + '.bin', Buffer.concat(parts));
console.log('wrote', outStem + '.json/.bin', N, 'gaussians');
