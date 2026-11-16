// Node check: JS FLAME + face-frame math must match the numpy reference.
import { readFileSync } from 'node:fs';
import { parseBundle, FlameHead, gaussianCenters, frameParams } from './gaussian-head-core.js';

const stem = process.argv[2] || 'test_head';
const header = JSON.parse(readFileSync(stem + '.json', 'utf8'));
const bin = readFileSync(stem + '.bin');
const B = parseBundle(header, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
function npy(path) { // minimal float32 .npy reader
  const b = readFileSync(path); const hl = b.readUInt16LE(8); const off = 10 + hl;
  return new Float32Array(b.buffer.slice(b.byteOffset + off, b.byteOffset + b.byteLength));
}
const head = new FlameHead(header, B);
const p = frameParams(B, header, 37);
let t0 = performance.now();
const verts = head.forward(p.expr, p.pose, p.trans);
const fd = head.updateFaceFrames();
const cent = gaussianCenters(B, fd);
const ms = performance.now() - t0;
const maxErr = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const ev = maxErr(verts, npy(stem + '.ref_verts37.npy')), ec = maxErr(cent, npy(stem + '.ref_centers37.npy'));
// timing over 30 frames
t0 = performance.now();
for (let f = 0; f < 30; f++) { const q = frameParams(B, header, f); head.forward(q.expr, q.pose, q.trans); head.updateFaceFrames(); gaussianCenters(B, head.faceData, cent); }
const per = (performance.now() - t0) / 30;
console.log(JSON.stringify({ max_vertex_err_m: ev, max_center_err_m: ec, first_frame_ms: +ms.toFixed(1), per_frame_ms: +per.toFixed(2), gaussians: header.n_gauss }));
if (ev > 1e-4 || ec > 1e-4) { console.error('MISMATCH'); process.exit(1); }
