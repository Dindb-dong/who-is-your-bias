import { readFileSync } from 'node:fs';
import { parseBundle } from '../bundle/gaussian-head-core.js';
import { SmplxHead } from './smplx-head-core.js';
const header = JSON.parse(readFileSync('preview_cut0.05.json', 'utf8'));
const buf = readFileSync('preview_cut0.05.bin'); const B = parseBundle(header, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const h = new SmplxHead(header, B);
const cases = JSON.parse(readFileSync('ref_cases.json', 'utf8'));
let worst = 0;
for (const c of cases) {
  const t0 = performance.now(); const v = h.forward(Float32Array.from(c.pose), Float32Array.from(c.expr), Float32Array.from(c.trans)); const ms = performance.now() - t0;
  let e = 0; for (let i = 0; i < v.length; i++) e = Math.max(e, Math.abs(v[i] - c.verts[i]));
  worst = Math.max(worst, e); console.log(`case: max vertex err ${(e * 1e3).toFixed(4)} mm  forward ${ms.toFixed(1)} ms`);
}
const t0 = performance.now(); h.updateFaceFrames(); console.log(`face frames ${(performance.now() - t0).toFixed(1)} ms`);
console.log(worst < 1e-4 ? 'CORE TEST PASS' : 'CORE TEST FAIL');
