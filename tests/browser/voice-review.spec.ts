import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { expect, test } from '@playwright/test';

const bundle = resolve('tools/voice_evaluation/results/qwen06-cpu-2026-11-11-listening');
const routeRoot = '/__voice-review__/listener/';
const sha256 = (value: Buffer) => createHash('sha256').update(value).digest('hex');

interface ListeningSample { id: string; path: string; sha256: string; language: string }
interface AnswerSample { condition: string; case_id: string; expected_text: string; synthesis_sha256: string }

/** Read the duration of the committed PCM WAV, independently of browser metadata. */
function wavDuration(bytes: Buffer): number {
  expect(bytes.toString('ascii', 0, 4)).toBe('RIFF');
  expect(bytes.toString('ascii', 8, 12)).toBe('WAVE');
  let byteRate = 0; let dataBytes = 0;
  for (let offset = 12; offset + 8 <= bytes.length;) {
    const kind = bytes.toString('ascii', offset, offset + 4);
    const size = bytes.readUInt32LE(offset + 4); const body = offset + 8;
    expect(body + size).toBeLessThanOrEqual(bytes.length);
    if (kind === 'fmt ') {
      expect(size).toBeGreaterThanOrEqual(16);
      expect(bytes.readUInt16LE(body)).toBe(1); // PCM, not a synthetic media fixture.
      byteRate = bytes.readUInt32LE(body + 8);
    }
    if (kind === 'data') dataBytes += size;
    offset = body + size + size % 2;
  }
  expect(byteRate).toBeGreaterThan(0); expect(dataBytes).toBeGreaterThan(0);
  return dataBytes / byteRate;
}

test('plays a committed generated WAV and exports a blinded review form without revealing its answer key', async ({ page }) => {
  // This is an automated transport/form check. Its ephemeral ratings and checkbox
  // values are synthetic test inputs, never human listening or similarity evidence.
  const [html, manifestBytes, templateBytes, keyBytes] = await Promise.all([
    readFile(resolve(bundle, 'listener/index.html')),
    readFile(resolve(bundle, 'listener/manifest.json')),
    readFile(resolve(bundle, 'listener/review-template.json')),
    readFile(resolve(bundle, 'private/answer-key.json')),
  ]);
  const manifest = JSON.parse(manifestBytes.toString()) as { reference: null; samples: ListeningSample[] };
  const template = JSON.parse(templateBytes.toString());
  const key = JSON.parse(keyBytes.toString()) as { samples: AnswerSample[] };
  expect(manifest.reference).toBeNull();
  expect(manifest.samples).toHaveLength(4);
  expect(template.listener_manifest_sha256).toBe(sha256(manifestBytes));

  const files = new Map<string, { body: Buffer; contentType: string }>([
    ['index.html', { body: html, contentType: 'text/html; charset=utf-8' }],
    ['manifest.json', { body: manifestBytes, contentType: 'application/json' }],
    ['review-template.json', { body: templateBytes, contentType: 'application/json' }],
  ]);
  const audioSamples = await Promise.all(manifest.samples.map(async (sample) => {
    expect(sample.path).toMatch(/^sample\d{3}\.wav$/);
    const bytes = await readFile(resolve(bundle, 'listener', sample.path));
    expect(sha256(bytes)).toBe(sample.sha256);
    files.set(sample.path, { body: bytes, contentType: 'audio/wav' });
    return { ...sample, duration: wavDuration(bytes) };
  }));
  const sample = audioSamples.reduce((shortest, next) => next.duration < shortest.duration ? next : shortest);
  expect(sample.duration).toBeGreaterThan(1);
  expect(sample.duration).toBeLessThan(20);

  const requestedFiles: string[] = []; const unexpectedRequests: string[] = []; const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.route('**/*', async (route) => {
    const url = new URL(route.request().url());
    const name = url.pathname.startsWith(routeRoot) ? url.pathname.slice(routeRoot.length) : '';
    const file = url.origin === 'http://127.0.0.1:8000' ? files.get(name) : undefined;
    if (file) {
      requestedFiles.push(name);
      await route.fulfill({ status: 200, body: file.body, contentType: file.contentType });
    } else if (url.origin === 'http://127.0.0.1:8000' && url.pathname === '/favicon.ico') {
      await route.fulfill({ status: 204 });
    } else {
      unexpectedRequests.push(url.href);
      await route.fulfill({ status: 404, body: 'Outside the listener test fixture' });
    }
  });
  await page.goto(`${routeRoot}index.html`);
  await expect(page.getByRole('heading', { level: 1, name: 'Blind voice listening review' })).toBeVisible();
  await expect(page.locator('audio')).toHaveCount(manifest.samples.length);
  await expect(page.locator('input[type=checkbox]:checked')).toHaveCount(0);
  expect(await page.locator('select').evaluateAll((nodes) => nodes.every((node) => (node as HTMLSelectElement).value === ''))).toBe(true);
  await expect(page.locator('[data-field=reference_similarity]')).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);

  // Check visible content and delivered HTML, including scripts/hidden attributes.
  // The private key is read by Node for assertions and is never served to the page.
  const visible = await page.locator('body').innerText(); const delivered = await page.content();
  expect(visible).not.toMatch(/\b(?:qwen|elon|musk)\b/i);
  expect(delivered).not.toMatch(/\b(?:expected_text|case_id|synthesis_sha256)\b/);
  for (const answer of key.samples) {
    for (const secret of [answer.condition, answer.case_id, answer.expected_text, answer.synthesis_sha256]) {
      expect(visible).not.toContain(secret);
      expect(delivered).not.toContain(secret);
    }
  }

  const section = page.locator(`section[data-id="${sample.id}"]`);
  const audio = section.locator('audio');
  await section.getByRole('heading').click(); // Trusted activation for native media policies.
  await audio.evaluate(async (element: HTMLAudioElement) => { element.muted = true; await element.play(); });
  await expect.poll(() => audio.evaluate((element: HTMLAudioElement) => element.currentTime), { timeout: 10_000 }).toBeGreaterThan(0.1);
  expect(await audio.evaluate((element: HTMLAudioElement) => element.duration)).toBeCloseTo(sample.duration, 3);
  expect(await audio.evaluate((element: HTMLAudioElement) => element.paused)).toBe(false);
  await audio.evaluate((element: HTMLAudioElement) => element.pause());
  expect(await audio.evaluate((element: HTMLAudioElement) => element.paused)).toBe(true);
  await audio.evaluate((element: HTMLAudioElement) => element.play());
  await expect.poll(() => audio.evaluate((element: HTMLAudioElement) => element.ended), { timeout: 10_000 }).toBe(true);
  expect(requestedFiles).toContain(sample.path);

  const downloads: string[] = [];
  page.on('download', (download) => downloads.push(download.suggestedFilename()));
  await page.getByRole('button', { name: 'Download review JSON' }).click();
  await expect(page.getByRole('status')).toContainText('Enter a reviewer identifier');
  expect(downloads).toEqual([]);

  await page.getByLabel('Reviewer initials or identifier').fill('  automated-browser-form-fixture  ');
  await page.getByLabel('I personally listened and entered these observations').check();
  await section.getByLabel('I listened to this complete clip').check();
  await section.getByLabel('I understand this language').selectOption('true');
  await section.getByLabel('What I heard').fill('  Synthetic transcript input  ');
  await section.getByLabel('Intelligibility (1–5)').selectOption('4');
  await section.getByLabel('Naturalness (1–5)').selectOption('3');
  await section.getByLabel('Unexpected repetition').selectOption('false');
  await section.getByLabel('Audible cutoff').selectOption('true');
  await section.getByLabel('Notes').fill('Synthetic UI input; no human listening or quality rating performed.');
  const downloaded = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download review JSON' }).click();
  const download = await downloaded;
  expect(download.suggestedFilename()).toBe('listening-review.json');
  const stream = await download.createReadStream(); expect(stream).not.toBeNull();
  const chunks: Buffer[] = [];
  for await (const chunk of stream!) chunks.push(Buffer.from(chunk));
  const exported = JSON.parse(Buffer.concat(chunks).toString());
  const createdAt = exported.created_at;
  expect(Number.isFinite(Date.parse(createdAt))).toBe(true);
  const expected = structuredClone(template);
  expected.reviewer = 'automated-browser-form-fixture'; expected.created_at = createdAt;
  expected.human_review_performed = true;
  Object.assign(expected.samples.find((row: { id: string }) => row.id === sample.id), {
    listened: true, understood_language: true, heard_text: 'Synthetic transcript input',
    intelligibility: 4, naturalness: 3, unexpected_repetition: false, audible_cutoff: true,
    notes: 'Synthetic UI input; no human listening or quality rating performed.',
  });
  // Exact equality also preserves every untouched sample as unrated and every
  // audio/manifest hash. No downloaded fixture is saved as an evaluation result.
  expect(exported).toEqual(expected);
  await expect(page.getByRole('status')).toContainText('Review downloaded');
  expect(downloads).toEqual(['listening-review.json']);
  expect(unexpectedRequests).toEqual([]);
  expect(errors).toEqual([]);
});
