import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
const root = process.argv[2], outDir = process.argv[3], sets = JSON.parse(process.argv[4]);
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json' };
const server = createServer(async (req, res) => {
  try { const p = join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname)); const b = await readFile(p);
    res.writeHead(200, { 'content-type': types[extname(p)] || 'application/octet-stream' }); res.end(b); } catch { res.writeHead(404); res.end(); }
}).listen(0, '127.0.0.1');
await new Promise(r => server.once('listening', r));
const browser = await chromium.launch({ channel: 'msedge', headless: false, args: ['--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 1000, height: 760 }, deviceScaleFactor: 1 });
const errors = []; page.on('pageerror', e => errors.push(String(e))); page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
for (const [name, qs] of Object.entries(sets)) {
  await page.goto(`http://127.0.0.1:${server.address().port}/smplx/preview.html?${qs}`);
  await page.waitForFunction(() => window.__pv && window.__pv.ready && window.__pv.frames > 40, null, { timeout: 90000 }).catch(() => errors.push('timeout ' + name));
  const st = await page.evaluate(() => window.__pv);
  await page.screenshot({ path: join(outDir, name + '.png'), clip: { x: 0, y: 0, width: 720, height: 760 } });
  console.log(name, JSON.stringify({ fps: st.fps && +st.fps.toFixed(1), cpuMs: st.cpuMs && +st.cpuMs.toFixed(1), n: st.n_gauss }));
}
console.log('errors', JSON.stringify(errors.filter(e => !e.includes('favicon') && !e.includes('404')).slice(0, 6)));
await browser.close(); server.close();
