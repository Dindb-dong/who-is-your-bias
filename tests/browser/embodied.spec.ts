import { createHash } from 'node:crypto';
import { expect, test } from '@playwright/test';

function testAudio() {
  const rate = 16000; const frames = rate * 4; const output = Buffer.alloc(44 + frames * 2);
  output.write('RIFF', 0); output.writeUInt32LE(36 + frames * 2, 4); output.write('WAVEfmt ', 8);
  output.writeUInt32LE(16, 16); output.writeUInt16LE(1, 20); output.writeUInt16LE(1, 22);
  output.writeUInt32LE(rate, 24); output.writeUInt32LE(rate * 2, 28); output.writeUInt16LE(2, 32);
  output.writeUInt16LE(16, 34); output.write('data', 36); output.writeUInt32LE(frames * 2, 40);
  for (let i = 0; i < frames; i++) output.writeInt16LE(Math.round(Math.sin(i / rate * Math.PI * 440) * 1500), 44 + i * 2);
  return output;
}

test('plays a hash-bound speech package through the browser audio transport', async ({ page }) => {
  // This is a synthetic transport fixture. Real model HTTP runs are separate.
  const audio = testAudio(); const hash = createHash('sha256').update(audio).digest('hex');
  const audioUrl = `/api/speech/clips/${'f'.repeat(32)}`;
  await page.route('**/api/persona', async (route) => {
    const response = await route.fetch(); const data = await response.json();
    await route.fulfill({ response, json: { ...data, capabilities: { ...data.capabilities, speech: true } } });
  });
  await page.route('**/api/speech/prepare', (route) => route.fulfill({ json: {
    audio_url: audioUrl, audio_sha256: hash, timing_status: 'available',
    speech_timing: { schema_version: 'speech-timing.v1', audio_sha256: hash, audio_duration_ms: 4000,
      timing_basis: 'manual', label_set: 'rhubarb', provenance: { engine: 'synthetic-browser-fixture' },
      segments: [{ start_ms: 0, end_ms: 1000, label: 'B' }, { start_ms: 1000, end_ms: 3000, label: 'D' }, { start_ms: 3000, end_ms: 4000, label: 'X' }] },
  } }));
  await page.route(`**${audioUrl}`, (route) => route.fulfill({ body: audio, contentType: 'audio/wav' }));
  await page.goto('/');
  await expect(page.getByText('Studio connected', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Studio settings', exact: true }).filter({ visible: true }).click();
  await page.getByRole('button', { name: /Connected voice/ }).click();
  await page.getByRole('button', { name: 'Close settings' }).click();
  await page.getByRole('textbox', { name: 'Your message' }).fill('Why reuse rockets?');
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  await page.getByRole('button', { name: 'Play response' }).click();
  await expect(page.getByText('Provider voice · audio-timed mouth cues', { exact: true })).toBeVisible();
  await expect(page.getByText('Speaking', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Stop audio', exact: true }).click();
  await expect(page.getByText('Ready when you are', { exact: true })).toBeVisible();
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('requires an explicit geometry preview and exposes rig calibration', async ({ page }, info) => {
  const errors: string[] = []; page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/');
  const mobile = info.project.name === 'mobile';
  await page.getByRole('navigation', { name: mobile ? 'Mobile navigation' : 'Main navigation', exact: true })
    .getByRole('button', { name: mobile ? 'Lab' : 'Research lab', exact: true }).click();
  const clip = { schema_version: 'avatar-clip.v1', duration_ms: 1000, max_gap_ms: 1000,
    source: { source_id: 'synthetic-ui-fixture', source_timeline: 'test', source_start_seconds: 0, video_sha256: 'a'.repeat(64), observations_sha256: 'b'.repeat(64) },
    review: { status: 'geometry_preview', reviewer: 'automated-test', track_id: 'fixture', track_continuity_reviewed: false,
      source_clock_reviewed: false, speaker_association_reviewed: false, attribution_basis: 'Synthetic UI fixture; no person attribution.' },
    calibration: { neutral_frame_ms: 0, face_neutral: {}, coordinate_basis: 'right_handed_y_up', bone_rotation_space: 'local_delta' },
    frames: [{ time_ms: 0, face: { jawOpen: 0 } }, { time_ms: 1000, face: { jawOpen: .6 } }], limitations: ['Synthetic transport example; not a captured performance.'] };
  await page.getByLabel('Import motion clip', { exact: true }).setInputFiles({ name: 'fixture.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(clip)) });
  await expect(page.getByRole('button', { name: 'Replay source motion' })).toBeDisabled();
  await page.getByRole('checkbox', { name: 'Preview geometry without speaker attribution' }).check();
  await expect(page.getByRole('button', { name: 'Replay source motion' })).toBeEnabled();
  await page.getByRole('button', { name: 'Replay source motion' }).click();
  await expect(page.getByRole('button', { name: 'Replay source motion' })).toBeVisible();
  await page.getByText('Rig inspection & calibration', { exact: true }).click();
  await expect(page.getByRole('combobox', { name: 'Calibration control', exact: true })).toBeVisible();
  await page.getByRole('checkbox', { name: 'Manual pose preview' }).check();
  await page.getByRole('combobox', { name: 'Calibration control', exact: true }).selectOption('face:jawOpen');
  await page.getByRole('slider', { name: 'Calibration preview value' }).focus();
  await page.getByRole('slider', { name: 'Calibration preview value' }).press('ArrowRight');
  await expect(page.getByText('Preview value: 0.01', { exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: info.outputPath('motion-calibration.png'), fullPage: true });
  await page.getByRole('button', { name: 'Clear imported files' }).click();
  await expect(page.getByRole('button', { name: 'Replay source motion' })).toHaveCount(0);
  await page.getByRole('checkbox', { name: 'Manual pose preview' }).uncheck();
  await page.getByRole('button', { name: 'Load measured sample' }).click();
  await expect(page.getByText('ted-2013', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Replay source motion' })).toBeDisabled();
  await page.getByRole('checkbox', { name: 'Preview geometry without speaker attribution' }).check();
  await page.getByRole('button', { name: 'Replay source motion' }).click();
  await page.getByRole('button', { name: 'Pause replay' }).click();
  await page.getByRole('slider', { name: /Source position/ }).focus();
  await page.getByRole('slider', { name: /Source position/ }).press('PageUp');
  await expect(page.getByText('Face: geometry_preview · Speech: silent', { exact: true })).toBeVisible();
  await page.screenshot({ path: info.outputPath('measured-motion.png'), fullPage: true });
  await page.getByRole('button', { name: 'Load arm-motion sample' }).click();
  await expect(page.getByText('ted-2013-body-geometry-preview.json', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Replay source motion' })).toBeDisabled();
  await page.getByRole('checkbox', { name: 'Preview geometry without speaker attribution' }).check();
  await page.getByRole('slider', { name: /Source position/ }).focus();
  await page.getByRole('slider', { name: /Source position/ }).press('PageUp');
  await expect(page.getByText('Face: designed · Speech: silent', { exact: true })).toBeVisible();
  await page.screenshot({ path: info.outputPath('measured-arm-motion.png'), fullPage: true });
  expect(errors).toEqual([]);
});
