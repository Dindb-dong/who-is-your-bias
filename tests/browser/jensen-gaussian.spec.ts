import { expect, test, type Page } from '@playwright/test';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { build } from 'vite';
import { FINGER_JOINTS } from '../../apps/web/src/avatar/channels';
import { createEventParser, type Turn } from '../../apps/web/src/api';

/** Tests exercise production rendering with explicitly synthetic input. They
 * cannot establish Jensen likeness or trained expression/gesture quality. */
function syntheticBundle() {
  const arrays: Record<string, { shape: number[]; value: Float32Array | Uint32Array }> = {
    v_shaped: { shape: [3, 3], value: new Float32Array([-.15, -.1, 0, .15, -.1, 0, 0, .2, .02]) },
    expr_dirs: { shape: [1, 9], value: new Float32Array([0, .04, 0, 0, .04, 0, 0, .04, 0]) },
    pose_dirs: { shape: [36, 9], value: new Float32Array(36 * 9) },
    j_base: { shape: [5, 3], value: new Float32Array(15) },
    j_expr_dirs: { shape: [1, 15], value: new Float32Array(15) },
    lbs_weights: { shape: [3, 5], value: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]) },
    faces: { shape: [1, 3], value: new Uint32Array([0, 1, 2]) },
    g_binding: { shape: [1], value: new Uint32Array([0]) },
    g_xyz: { shape: [1, 3], value: new Float32Array(3) },
    g_rot: { shape: [1, 4], value: new Float32Array([1, 0, 0, 0]) },
    g_opacity: { shape: [1], value: new Float32Array([8]) },
    g_color: { shape: [1, 3], value: new Float32Array([1.7, -1.7, -1.7]) },
    g_scale: { shape: [1, 3], value: new Float32Array([-1, -1, -1]) },
    m_expr: { shape: [2, 1], value: new Float32Array([0, 1]) },
    m_pose: { shape: [2, 15], value: new Float32Array(30) },
    m_trans: { shape: [2, 3], value: new Float32Array(6) },
  };
  let offset = 0; const buffers: Record<string, unknown> = {}; const parts: Buffer[] = [];
  for (const [name, { shape, value }] of Object.entries(arrays)) {
    const bytes = Buffer.from(value.buffer, value.byteOffset, value.byteLength);
    buffers[name] = { dtype: value instanceof Uint32Array ? 'u32' : 'f32', shape, offset, bytes: bytes.length };
    parts.push(bytes); offset += bytes.length;
  }
  const binary = Buffer.concat(parts), hash = createHash('sha256').update(binary).digest('hex');
  return { binary, hash, header: { schema: 'gaussian-head.v1', n_verts: 3, n_faces: 1, n_expr: 1, n_gauss: 1,
    n_frames: 2, fps: 15, parents: [-1, 0, 1, 1, 1], buffers, camera: { position: [0, 0, 1], target: [0, 0, 0], fov_y_deg: 36 },
    source: 'Synthetic rendering fixture. No Jensen data or trained weights.' },
  driver: { schema: 'gaussian-expression-driver.v1', subject_id: 'jensen-huang', bundle_sha256: hash, n_expr: 1,
    input_names: ['browInnerUp'], input_bounds: [[0, 1]], weights: [[1]], bias: [0], output_bounds: [[0, 1]], neutral_expr: [0], jaw_open_radians: .3,
    training: { method: 'ridge_regression', dataset_sha256: 'a'.repeat(64), examples: 1, frame_indices: [0], validation_frame_indices: [1] } } };
}

async function routeBundle(page: Page, driver = true, failBinary = false) {
  const fixture = syntheticBundle();
  await page.route('**/models/gaussian-head/jensen_a_c1c3_240k_sh.json', (route) => route.fulfill({ json: fixture.header }));
  await page.route('**/models/gaussian-head/jensen_a_c1c3_240k_sh.bin', (route) => failBinary
    ? route.fulfill({ status: 404 }) : route.fulfill({ contentType: 'application/octet-stream', body: fixture.binary }));
  await page.route('**/models/gaussian-head/jensen_a_c1c3_240k_sh.expression-driver.json', (route) => driver
    ? route.fulfill({ json: fixture.driver }) : route.fulfill({ status: 404 }));
  return fixture;
}

function audioFixture() {
  const rate = 16000, count = rate * 8, audio = Buffer.alloc(44 + count * 2);
  audio.write('RIFF', 0); audio.writeUInt32LE(36 + count * 2, 4); audio.write('WAVEfmt ', 8);
  audio.writeUInt32LE(16, 16); audio.writeUInt16LE(1, 20); audio.writeUInt16LE(1, 22);
  audio.writeUInt32LE(rate, 24); audio.writeUInt32LE(rate * 2, 28); audio.writeUInt16LE(2, 32); audio.writeUInt16LE(16, 34);
  audio.write('data', 36); audio.writeUInt32LE(count * 2, 40);
  for (let i = 0; i < count; i++) audio.writeInt16LE(Math.round(Math.sin(i * 2 * Math.PI * 220 / rate) * 3000), 44 + i * 2);
  return audio;
}

function syntheticGestureBasis() {
  const group = (channels: string[]) => {
    const dimensions = 4 * channels.length * 3;
    return { channels, rank: 1,
      mean: Array.from({ length: dimensions }, (_, i) => i % 3 === 0 ? .35 : 0),
      basis: [Array.from({ length: dimensions }, (_, i) => i % 3 === 0 ? .1 : 0)],
      latent_examples: [[1], [-1]], latent_ids: ['synthetic:0', 'synthetic:5000'],
      training_source_events: ['synthetic-browser-fixture'], support_counts: Array(dimensions).fill(2),
      max_angle_rad: .65, iterations_completed: 1 };
  };
  return { schema_version: 'gesture-basis.v1', persona_id: 'jensen-huang', review_status: 'geometry_preview',
    coordinate_basis: 'right_handed_y_up', rotation_space: 'local_delta', flattening: 'time_channel_xyz',
    id: 'synthetic-browser-gesture-fixture', dataset_sha256: 'b'.repeat(64), sample_count: 4, duration_ms: 5000,
    groups: {
      left_arm: group(['leftUpperArm', 'leftForearm', 'leftHand']), right_arm: group(['rightUpperArm', 'rightForearm', 'rightHand']),
      left_fingers: group(FINGER_JOINTS.map((name) => `left${name}`).sort()),
      right_fingers: group(FINGER_JOINTS.map((name) => `right${name}`).sort()),
    }, limitations: ['Hand-authored synthetic browser fixture. Not trained weights or observations of Jensen.'] };
}

async function routeSpeech(page: Page) {
  const audio = audioFixture(), hash = createHash('sha256').update(audio).digest('hex');
  const clipUrl = `/api/speech/clips/${'e'.repeat(32)}`;
  await page.route('**/api/persona', async (route) => {
    const response = await route.fetch(), data = await response.json();
    await route.fulfill({ response, json: { ...data, capabilities: { ...data.capabilities, speech: true, live_speech: false },
      bindings: { ...data.bindings, voice: { status: 'matched', persona_id: 'jensen-huang', voice: 'reference' } } } });
  });
  const turn: Turn = { id: 'synthetic-gaussian-test', session_id: 'a'.repeat(32), text: 'Synthetic rendering and transport fixture.', language: 'en', mode: 'behavior', provider: 'evidence',
    citations: [], grounding: { status: 'unverified', retrieved_ids: [], invalid_citation_ids: [], reason: 'Synthetic test fixture.' },
    metrics: { retrieval_ms: 0, generation_ms: 0, total_ms: 0 },
    behavior: { duration_ms: 8000, timing_basis: 'estimated_text', motion: null, speech: { rate: 1, pitch: 1, pause_ms: 0 }, cues: [
      { channel: 'face', name: 'brow_raise', start_ms: 0, duration_ms: 7900, intensity: .9, provenance: 'designed' },
      { channel: 'gesture', name: 'open_palm', start_ms: 0, duration_ms: 7900, intensity: .9, provenance: 'designed' },
      { channel: 'gesture', name: 'emphasis', start_ms: 0, duration_ms: 7900, intensity: .5, provenance: 'designed' },
    ] } };
  const stream = [
    ['meta', { session_id: turn.session_id, turn_id: turn.id }], ['token', { text: turn.text }], ['result', { turn }], ['done', {}],
  ].map(([event, data]) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`).join('');
  // Fail a malformed synthetic response here instead of waiting for a Play
  // button that the production stream validator correctly refuses to expose.
  const parser = createEventParser(() => {}); parser.push(stream); parser.finish();
  await page.route('**/api/chat/stream', (route) => route.fulfill({ contentType: 'text/event-stream', body: stream }));
  await page.route('**/api/speech/prepare', (route) => route.fulfill({ json: { audio_url: clipUrl, audio_sha256: hash, timing_status: 'available',
    speech_timing: { schema_version: 'speech-timing.v1', audio_sha256: hash, audio_duration_ms: 8000, timing_basis: 'manual', label_set: 'rhubarb',
      provenance: { engine: 'synthetic-gaussian-browser-test' }, segments: [{ start_ms: 0, end_ms: 8000, label: 'D' }] } } }));
  await page.route(`**${clipUrl}`, (route) => route.fulfill({ contentType: 'audio/wav', body: audio }));
}

const errors = new WeakMap<Page, string[]>();
test.beforeEach(({ page }) => {
  const observed: string[] = []; errors.set(page, observed);
  page.on('pageerror', (error) => observed.push(error.message));
  page.on('console', (event) => { if (event.type() === 'error' && /WebGL|shader|THREE|texture/i.test(event.text())) observed.push(event.text()); });
});
test.afterEach(({ page }) => expect(errors.get(page)).toEqual([]));

test('draws real WebGL splats with expression displacement and correct opaque-hand depth', async ({ page }) => {
  const result = await build({ configFile: false, logLevel: 'silent', build: { write: false, minify: false,
    lib: { entry: resolve('tests/browser/fixtures/gaussian-composition.ts'), name: 'GaussianCompositionFixture', formats: ['iife'] } } });
  const outputs = (Array.isArray(result) ? result : [result]).flatMap((entry) => 'output' in entry ? entry.output : []);
  const script = outputs.find((entry) => entry.type === 'chunk'); if (!script || script.type !== 'chunk') throw new Error('Fixture bundling failed.');
  await page.route('**/__gaussian_fixture__.js', (route) => route.fulfill({ contentType: 'text/javascript', body: script.code }));
  await page.route('**/__gaussian_fixture__', (route) => route.fulfill({ contentType: 'text/html', body: '<!doctype html><script src="/__gaussian_fixture__.js"></script>' }));
  await page.goto('/__gaussian_fixture__');
  const pixels = await page.evaluate(() => (window as unknown as { GaussianCompositionFixture: { runCompositionFixture(): {
    face: number[]; occluded: number[]; behind: number[]; restY: number; movedY: number; canvasCount: number } } }).GaussianCompositionFixture.runCompositionFixture());
  expect(pixels.face[0]).toBeGreaterThan(220); expect(pixels.face[1]).toBeLessThan(10);
  expect(pixels.occluded[1]).toBeGreaterThan(240); expect(pixels.occluded[0]).toBeLessThan(10);
  expect(pixels.behind[0]).toBeGreaterThan(220); expect(pixels.behind[1]).toBeLessThan(15);
  expect(pixels.movedY - pixels.restY).toBeGreaterThan(10);
  expect(pixels.canvasCount).toBe(1); await expect(page.locator('canvas')).toHaveCount(0);
});

test('keeps Gaussian face, articulated body and speech on one playback path, then stops cleanly', async ({ page }, info) => {
  const fixture = await routeBundle(page); await routeSpeech(page); await page.goto('/?head=gaussian');
  const stage = page.getByTestId('avatar-stage'), canvas = stage.locator('canvas.avatar-canvas');
  await expect(stage).toHaveAttribute('data-gaussian-state', 'ready', { timeout: 25000 });
  await expect(stage).toHaveAttribute('data-framing', 'upper_body');
  await expect(canvas).toHaveAttribute('data-gaussian-sha256', fixture.hash);
  await expect(canvas).toHaveAttribute('data-gaussian-expression-source', 'calibrated');
  await expect(stage.locator('canvas')).toHaveCount(1);
  await page.getByRole('button', { name: 'Studio settings', exact: true }).filter({ visible: true }).click();
  await page.getByRole('button', { name: /Connected voice/ }).click(); await page.getByRole('button', { name: 'Close settings' }).click();
  await page.getByRole('textbox', { name: 'Your message' }).fill('Synthetic motion playback check.');
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  await page.getByRole('button', { name: 'Play response', exact: true }).click();
  await expect.poll(async () => Number(await canvas.getAttribute('data-gaussian-jaw'))).toBeGreaterThan(.05);
  await expect.poll(async () => Number(await canvas.getAttribute('data-gaussian-expression-energy'))).toBeGreaterThan(.3);
  await expect.poll(async () => Number(await canvas.getAttribute('data-rig-arm'))).toBeGreaterThan(.15);
  await expect.poll(async () => Number(await canvas.getAttribute('data-rig-hand'))).toBeGreaterThan(.1);
  await page.getByRole('button', { name: 'Stop audio', exact: true }).click();
  await expect(canvas).toHaveAttribute('data-speaking', 'false');
  await expect(canvas).toHaveAttribute('data-gaussian-jaw', '0.0000');
  await expect(canvas).toHaveAttribute('data-audio-clock-ms', '0.000');
  await expect.poll(async () => Number(await canvas.getAttribute('data-rig-arm'))).toBeLessThan(.02);
  await expect.poll(async () => Number(await canvas.getAttribute('data-gaussian-expression-energy'))).toBeLessThan(.02);
  await page.getByRole('button', { name: 'Studio settings', exact: true }).filter({ visible: true }).click();
  await page.getByRole('checkbox', { name: 'Reduce automatic motion' }).check(); await page.getByRole('button', { name: 'Close settings' }).click();
  await page.getByRole('button', { name: 'Play response', exact: true }).click();
  await expect.poll(async () => Number(await canvas.getAttribute('data-audio-clock-ms'))).toBeGreaterThan(800);
  expect(Number(await canvas.getAttribute('data-rig-arm'))).toBeLessThan(.15);
  expect(Number(await canvas.getAttribute('data-gaussian-jaw'))).toBeGreaterThan(.05);
  await page.getByRole('button', { name: 'Stop audio', exact: true }).click();
  const navigation = page.getByRole('navigation', { name: info.project.name === 'mobile' ? 'Mobile navigation' : 'Main navigation', exact: true });
  await navigation.getByRole('button', { name: /Sources/ }).click(); await expect(page.locator('canvas.avatar-canvas')).toHaveCount(0);
  await navigation.getByRole('button', { name: 'Studio', exact: true }).click();
  await expect(stage).toHaveAttribute('data-gaussian-state', 'ready', { timeout: 25000 });
  await expect(page.locator('canvas.avatar-canvas')).toHaveCount(1);
});

test('labels missing calibration and keeps the body visible when private face bytes are unavailable', async ({ page }) => {
  await routeBundle(page, false); await page.goto('/?head=gaussian');
  const stage = page.getByTestId('avatar-stage'), canvas = stage.locator('canvas.avatar-canvas');
  await expect(stage).toHaveAttribute('data-gaussian-state', 'ready', { timeout: 25000 });
  await expect(canvas).toHaveAttribute('data-gaussian-expression-source', 'static_reference');
  await expect(canvas).toHaveAttribute('data-gaussian-message', /No verified training split/);
  await page.route('**/models/gaussian-head/jensen_a_c1c3_240k_sh.bin', (route) => route.fulfill({ status: 404 }));
  await page.reload();
  await expect(stage).toHaveAttribute('data-gaussian-state', 'error', { timeout: 25000 });
  await expect(stage).toHaveAttribute('data-model-state', 'ready');
  await expect(stage).toHaveAttribute('data-rig-kind', 'glb');
  await expect(canvas).toBeVisible(); await expect(stage.locator('canvas')).toHaveCount(1);
});

test('activates the explicit learned-gesture preview and disables it on Stop, reduced motion and another persona', async ({ page }) => {
  await routeBundle(page); await routeSpeech(page);
  let modelRequests = 0;
  await page.route('**/models/gesture/jensen-gesture-basis-expanded.json', (route) => {
    modelRequests++; return route.fulfill({ json: syntheticGestureBasis() });
  });
  await page.goto('/?head=gaussian&gesture=learned&gestureBundle=jensen-gesture-basis-expanded');
  const stage = page.getByTestId('avatar-stage'), canvas = stage.locator('canvas.avatar-canvas');
  await expect(stage).toHaveAttribute('data-gaussian-state', 'ready', { timeout: 25000 });
  await expect.poll(() => modelRequests).toBeGreaterThan(0);
  const enableVoice = async () => {
    await page.getByRole('button', { name: 'Studio settings', exact: true }).filter({ visible: true }).click();
    await page.getByRole('button', { name: /Connected voice/ }).click(); await page.getByRole('button', { name: 'Close settings' }).click();
  };
  const askAndPlay = async () => {
    await page.getByRole('textbox', { name: 'Your message' }).fill('Synthetic learned-gesture activation check.');
    await page.getByRole('button', { name: 'Send message', exact: true }).click();
    await page.getByRole('button', { name: 'Play response', exact: true }).click();
  };
  await enableVoice(); await askAndPlay();
  await expect(stage).toHaveAttribute('data-bone-source', 'trained_geometry_preview');
  await expect.poll(async () => Number(await canvas.getAttribute('data-rig-arm'))).toBeGreaterThan(.2);
  await expect.poll(async () => Number(await canvas.getAttribute('data-rig-finger'))).toBeGreaterThan(.2);
  await expect.poll(async () => Number(await canvas.getAttribute('data-gaussian-jaw'))).toBeGreaterThan(.05);
  await page.getByRole('button', { name: 'Stop audio', exact: true }).click();
  await expect(stage).toHaveAttribute('data-bone-source', 'designed');
  await expect(canvas).toHaveAttribute('data-speaking', 'false');
  await expect.poll(async () => Number(await canvas.getAttribute('data-rig-finger'))).toBeLessThan(.02);
  await expect.poll(async () => Number(await canvas.getAttribute('data-rig-arm'))).toBeLessThan(.02);
  await page.getByRole('button', { name: 'Studio settings', exact: true }).filter({ visible: true }).click();
  await page.getByRole('checkbox', { name: 'Reduce automatic motion' }).check(); await page.getByRole('button', { name: 'Close settings' }).click();
  await page.getByRole('button', { name: 'Play response', exact: true }).click();
  await expect.poll(async () => Number(await canvas.getAttribute('data-audio-clock-ms'))).toBeGreaterThan(800);
  await expect(stage).toHaveAttribute('data-bone-source', 'designed');
  expect(Number(await canvas.getAttribute('data-rig-finger'))).toBeLessThan(.08);
  expect(Number(await canvas.getAttribute('data-gaussian-jaw'))).toBeGreaterThan(.05);
  await page.getByRole('button', { name: 'Stop audio', exact: true }).click();

  // Re-query a different service persona with the same opt-in URL. Neither
  // Jensen face bytes nor Jensen gesture weights may activate for that subject.
  const beforeOtherPersona = modelRequests;
  await page.route('**/api/persona', async (route) => {
    const response = await route.fetch(), profile = await response.json();
    await route.fulfill({ response, json: { ...profile, id: 'elon-musk', name: 'Other persona fixture',
      capabilities: { ...profile.capabilities, speech: true, live_speech: false },
      bindings: { ...profile.bindings, voice: { status: 'matched', persona_id: 'elon-musk', voice: 'reference' } } } });
  });
  await page.reload();
  await expect(stage).toHaveAttribute('data-model-state', 'ready', { timeout: 25000 });
  await expect(stage).toHaveAttribute('data-gaussian-state', 'disabled');
  await enableVoice(); await askAndPlay();
  await expect.poll(async () => Number(await canvas.getAttribute('data-audio-clock-ms'))).toBeGreaterThan(800);
  await expect(stage).toHaveAttribute('data-bone-source', 'designed');
  expect(modelRequests).toBe(beforeOtherPersona);
  await page.getByRole('button', { name: 'Stop audio', exact: true }).click();
});
