import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { extname, join } from 'node:path';
const root = process.argv[2], bundle = process.argv[3], cams = process.argv[4], outDir = process.argv[5];
const types = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json' };
const server = createServer(async (req, res) => {
  let p; try { p = join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname)); const b = await readFile(p);
    res.writeHead(200, { 'content-type': types[extname(p)] || 'application/octet-stream' }); res.end(b); } catch { res.writeHead(404); res.end(); }
}).listen(0, '127.0.0.1');
await new Promise(r => server.once('listening', r));
const browser = await chromium.launch({ channel: 'msedge', headless: false, args: ['--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 600, height: 600 }, deviceScaleFactor: 1 });
const errors = []; page.on('pageerror', e => errors.push(String(e))); page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
await page.goto(`http://127.0.0.1:${server.address().port}/cmp/compare.html?bundle=${bundle}&cams=${cams}`);
await page.waitForFunction(() => window.__out, null, { timeout: 120000 }).catch(() => {});
const res = await page.evaluate(() => window.__out || null);
await mkdir(outDir, { recursive: true });
if (res) for (const o of res.out) await writeFile(join(outDir, `${o.split}_${String(o.t).padStart(5, '0')}_webgl.png`), Buffer.from(o.png.split(',')[1], 'base64'));
console.log(JSON.stringify({ n: res && res.n, count: res && res.out.length, errors }));
await browser.close(); server.close();
