import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { expect, test } from '@playwright/test';

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value !== null && typeof value === 'object') return `{${Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(',')}}`;
  return JSON.stringify(value);
}

test('auditions actual bundled geometry and local audio with hash-bound downloadable evidence', async ({ page }, info) => {
  const errors: string[] = []; const synthesisRequests: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('request', (request) => { if (request.url().includes('/api/speech')) synthesisRequests.push(request.url()); });
  // This case exercises the actual avatar and media bytes; unrelated conversation
  // bootstrap data is bounded so no inference provider or private corpus is needed.
  await page.route('**/api/persona', (route) => route.fulfill({ json: { id: 'elon-musk', name: 'Elon Musk',
    tagline: 'Research simulation', description: 'Browser audition fixture', example_questions: { en: [], ko: [] },
    stats: { sources: 0, evidence: 0, held_out_sources: 0 }, capabilities: { llm: false, speech: false }, notice: 'AI simulation' } }));
  await page.route('**/api/sources', (route) => route.fulfill({ json: { sources: [] } }));
  await page.route('**/api/motion/library', (route) => route.fulfill({ status: 503, json: { detail: 'No source motion is needed for this asset audition.' } }));
  await page.goto('/');
  const mobile = info.project.name === 'mobile';
  await page.getByRole('navigation', { name: mobile ? 'Mobile navigation' : 'Main navigation', exact: true })
    .getByRole('button', { name: mobile ? 'Lab' : 'Research lab', exact: true }).click();
  const stage = page.getByTestId('avatar-stage');
  await expect(stage).toHaveAttribute('data-rig-kind', 'glb', { timeout: 20_000 });
  await page.getByText('Rig inspection & calibration', { exact: true }).click();
  await page.getByText('Asset audition', { exact: true }).click();
  await expect(page.getByText(/52 face controls with measured vertex movement/)).toBeVisible();
  const assetDownload = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export asset inspection', exact: true }).click();
  const assetFile = await assetDownload; await assetFile.saveAs(info.outputPath('asset-inspection.json'));
  const asset = JSON.parse(await readFile(info.outputPath('asset-inspection.json'), 'utf8'));
  expect(asset.report.asset.sha256).toBe('aee550c9cb5b20c6afccfb038c42c2eb8d7aab2c8fb74ca977aefed56f633519');
  expect(asset.report.asset.meshes).toHaveLength(7);
  expect(asset.reportSha256).toBe(createHash('sha256').update(canonical(asset.report)).digest('hex'));
  await page.getByRole('button', { name: 'Load narration sample', exact: true }).click();
  await expect(page.getByText('Bundled generated narration', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Play & record audition', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Stop audition', exact: true })).toBeVisible();
  await expect.poll(async () => {
    const text = await page.getByText(/Articulation: measured audio energy ·/).textContent();
    return Number(text?.match(/· ([\d.]+) s/)?.[1] ?? 0);
  }).toBeGreaterThan(0.75);
  await page.getByRole('button', { name: 'Stop audition audio', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Record audition', exact: true })).toBeVisible();
  const auditionDownload = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export audition', exact: true }).click();
  const auditionFile = await auditionDownload; await auditionFile.saveAs(info.outputPath('audition.json'));
  const audition = JSON.parse(await readFile(info.outputPath('audition.json'), 'utf8'));
  expect(audition.reportSha256).toBe(createHash('sha256').update(canonical(audition.report)).digest('hex'));
  const narration = await readFile('apps/web/public/presentation/narration.wav');
  expect(audition.report.audioSha256).toBe(createHash('sha256').update(narration).digest('hex'));
  expect(audition.report.measurements.speakingFrames).toBeGreaterThan(0);
  expect(audition.report.measurements.renderedFrames).toBeGreaterThan(0);
  expect(audition.report.asset.review.likeness).toBe('ungraded');
  expect(audition.report.unmeasured).toContain('acoustic loudspeaker latency');
  expect(audition.report.stoppedReason).toBe('playback_inactive');
  expect(synthesisRequests).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole('region', { name: 'Motion & rig calibration', exact: true }).screenshot({ path: info.outputPath('avatar-audition.png') });
  expect(errors).toEqual([]);
});
