import { chromium } from 'playwright';
const out = process.argv[2] || 'studio.png';
const browser = await chromium.launch({ channel: 'msedge', headless: false, args: ['--window-size=1440,900', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 1400, height: 860 } });
const errors = []; page.on('pageerror', e => errors.push(String(e))); page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
for (let i = 0; i < 40; i++) { try { await page.goto('http://127.0.0.1:5173/?head=gaussian' + (process.argv[3] || ''), { timeout: 5000 }); break; } catch { await new Promise(r => setTimeout(r, 1500)); } }
await page.waitForSelector('[data-testid="gaussian-head-stage"][data-state="ready"]', { timeout: 60000 }).catch(() => errors.push('stage not ready'));
await page.waitForTimeout(4000);
const state = await page.evaluate(() => document.querySelector('[data-testid="gaussian-head-stage"]')?.textContent);
await page.screenshot({ path: out });
console.log(JSON.stringify({ state, errors: errors.slice(0, 8) }));
await browser.close();
