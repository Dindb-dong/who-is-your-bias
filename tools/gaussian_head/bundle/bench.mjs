// Open viewer.html in the local Edge (GPU), wait, report fps and save a screenshot.
import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';

const root = process.cwd(), bundle = process.argv[2] || 'test_head', shot = process.argv[3] || 'bench.png';
const types = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.bin': 'application/octet-stream' };
const server = createServer(async (req, res) => {
  let body, p;
  try { p = join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname)); body = await readFile(p); }
  catch { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'content-type': types[extname(p)] || 'application/octet-stream' }); res.end(body);
}).listen(0, '127.0.0.1');
await new Promise(r => server.once('listening', r));
const port = server.address().port;
const browser = await chromium.launch({ channel: 'msedge', headless: false, args: ['--window-size=900,900', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 860, height: 860 } });
const errors = []; page.on('pageerror', e => errors.push(String(e))); page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
await page.goto(`http://127.0.0.1:${port}/viewer.html?bundle=${bundle}&${process.argv[4] || ''}`);
await page.waitForFunction(() => window.__gh && window.__gh.frames > 120, null, { timeout: 60000 }).catch(() => {});
const stats = await page.evaluate(() => window.__gh || null);
const gpu = await page.evaluate(() => { const gl = document.createElement('canvas').getContext('webgl2'); const e = gl && gl.getExtension('WEBGL_debug_renderer_info'); return e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : 'unknown'; });
await page.screenshot({ path: shot });
console.log(JSON.stringify({ stats, gpu, errors }));
await browser.close(); server.close();
