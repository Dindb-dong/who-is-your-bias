import { expect, test, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

const pageErrors = new WeakMap<Page, string[]>();
test.beforeEach(async ({ page }) => {
  const errors: string[] = []; pageErrors.set(page, errors);
  page.on('pageerror', (error) => errors.push(error.message));
});
test.afterEach(async ({ page }) => { expect(pageErrors.get(page)).toEqual([]); });

async function openStudio(page: Page) {
  await page.goto('/');
  await expect(page.getByText('Studio connected', { exact: true })).toBeVisible();
  await expect(page.getByText('JENSEN · STYLIZED PREVIEW', { exact: true })).toBeVisible();
  await expect(page.getByTestId('avatar-stage')).toHaveAttribute('data-rig-kind', 'procedural');
  await expect(page.getByTestId('attention-panel')).toHaveAttribute('data-input', 'off');
}

function canvas(page: Page) { return page.getByTestId('avatar-stage').locator('canvas.avatar-canvas'); }
async function angle(page: Page, name: string) { return Number(await canvas(page).getAttribute(`data-attention-${name}`)); }
function navigation(page: Page, mobile: boolean) {
  return page.getByRole('navigation', { name: mobile ? 'Mobile navigation' : 'Main navigation', exact: true });
}

type FaceBox = { x: number; y: number; width: number; height: number; score: number };
interface CaptureFixture {
  boxes: FaceBox[];
  streams: MediaStream[];
  frames: number;
  workers: number;
  terminated: number;
  stops: number;
  cameraCalls: number;
  screenCalls: number;
  screenHadActivation: boolean;
  cameraConstraints: MediaStreamConstraints | null;
  screenConstraints: DisplayMediaStreamOptions | null;
  grant: (() => void) | null;
}
type FixtureWindow = Window & { __attentionFixture: CaptureFixture };
type Trace = { at: number; phase: string; eyeYaw: number; headYaw: number };
type TraceWindow = Window & { __attentionTrace: Trace[]; __attentionObserver: MutationObserver };

/**
 * Synthetic detection fixture, not a detector-accuracy test. Only the device and
 * worker boundaries are replaced: video decoding, ImageBitmap creation, tracking,
 * target selection, animation and cleanup execute the production code.
 */
async function installCaptureFixture(page: Page, mode: 'granted' | 'delayed' | 'denied' = 'granted') {
  await page.addInitScript(({ mode }) => {
    const fixture: CaptureFixture = { boxes: [
      { x: 0.10, y: 0.20, width: 0.20, height: 0.35, score: 0.99 },
      { x: 0.66, y: 0.25, width: 0.19, height: 0.32, score: 0.98 },
    ], streams: [], frames: 0, workers: 0, terminated: 0, stops: 0, cameraCalls: 0, screenCalls: 0,
      screenHadActivation: false, cameraConstraints: null, screenConstraints: null, grant: null };
    (window as FixtureWindow).__attentionFixture = fixture;
    const createStream = () => {
      const source = document.createElement('canvas'); source.width = 640; source.height = 360;
      const ctx = source.getContext('2d')!;
      const draw = () => {
        ctx.fillStyle = '#20302b'; ctx.fillRect(0, 0, source.width, source.height);
        fixture.boxes.forEach((box, i) => {
          ctx.fillStyle = i === 0 ? '#99d0b2' : '#e0c39b';
          ctx.fillRect(box.x * source.width, box.y * source.height, box.width * source.width, box.height * source.height);
        });
        ctx.fillStyle = '#ffffff'; ctx.font = '14px sans-serif';
        ctx.fillText(`Synthetic capture fixture · ${Math.floor(performance.now())}`, 12, 340);
      };
      draw(); const stream = source.captureStream(12); const timer = setInterval(draw, 60);
      for (const track of stream.getTracks()) {
        const stop = track.stop.bind(track);
        track.stop = () => { fixture.stops++; clearInterval(timer); stop(); };
      }
      fixture.streams.push(stream); return stream;
    };
    const request = () => mode === 'denied'
      ? Promise.reject(new DOMException('Synthetic permission denial', 'NotAllowedError'))
      : mode === 'delayed' ? new Promise<MediaStream>((resolve) => { fixture.grant = () => { fixture.grant = null; resolve(createStream()); }; })
      : Promise.resolve(createStream());
    const devices = navigator.mediaDevices ?? {};
    Object.defineProperty(devices, 'getUserMedia', { configurable: true, value: (constraints: MediaStreamConstraints) => {
      fixture.cameraCalls++; fixture.cameraConstraints = constraints; return request();
    } });
    Object.defineProperty(devices, 'getDisplayMedia', { configurable: true, value: (constraints: DisplayMediaStreamOptions) => {
      fixture.screenCalls++; fixture.screenConstraints = constraints;
      fixture.screenHadActivation = navigator.userActivation.isActive; return request();
    } });
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: devices });
    class SyntheticFaceWorker {
      onmessage: ((event: MessageEvent) => void) | null = null;
      onerror: ((event: Event) => void) | null = null;
      onmessageerror: ((event: MessageEvent) => void) | null = null;
      private closed = false;
      constructor(url: string | URL) {
        if (!String(url).endsWith('/vision/attention.worker.js')) throw new Error(`Unexpected worker in capture fixture: ${url}`);
        fixture.workers++;
      }
      postMessage(message: { type: string; id?: number; bitmap?: ImageBitmap; observedAtMs?: number }) {
        if (this.closed) return;
        if (message.type === 'init') {
          queueMicrotask(() => { if (!this.closed) this.onmessage?.(new MessageEvent('message', { data: { type: 'ready' } })); });
        } else if (message.type === 'frame') {
          message.bitmap!.close(); fixture.frames++;
          const boxes = fixture.boxes.map((box) => ({ ...box }));
          queueMicrotask(() => { if (!this.closed) this.onmessage?.(new MessageEvent('message', {
            data: { type: 'result', id: message.id, boxes, observedAtMs: message.observedAtMs },
          })); });
        }
      }
      terminate() { if (!this.closed) { this.closed = true; fixture.terminated++; } }
    }
    Object.defineProperty(window, 'Worker', { configurable: true, value: SyntheticFaceWorker });
  }, { mode });
}

test('loads the actual MediaPipe worker and processes blank frames without capture permission', async ({ page }) => {
  // No worker or detector mocks in this test. Blank input establishes that the
  // pinned WASM/model loader and frame protocol execute, not face accuracy.
  const network: string[] = [];
  page.context().on('request', (request) => network.push(request.url()));
  await page.goto('/vision/assets.json');
  const result = await page.evaluate(async () => {
    const worker = new Worker('/vision/attention.worker.js', { name: 'actual-local-face-detector-smoke' });
    type Reply = { type: string; id?: number; observedAtMs?: number; boxes?: unknown[]; message?: string };
    const receive = (type: string, id: number | undefined, timeoutMs: number): Promise<Reply> => new Promise((resolve, reject) => {
      const finish = (error?: Error, reply?: Reply) => {
        clearTimeout(timer); worker.removeEventListener('message', message); worker.removeEventListener('error', failed);
        worker.removeEventListener('messageerror', unreadable);
        if (error) reject(error); else resolve(reply!);
      };
      const message = (event: MessageEvent<Reply>) => {
        if (event.data.type === 'error') finish(new Error(event.data.message ?? 'The actual detector worker failed.'));
        else if (event.data.type === type && event.data.id === id) finish(undefined, event.data);
      };
      const failed = (event: ErrorEvent) => finish(new Error(event.message || 'The actual detector worker could not load.'));
      const unreadable = () => finish(new Error('The actual detector worker returned an unreadable message.'));
      const timer = setTimeout(() => finish(new Error(`Actual detector ${type} timed out after ${timeoutMs} ms.`)), timeoutMs);
      worker.addEventListener('message', message); worker.addEventListener('error', failed); worker.addEventListener('messageerror', unreadable);
    });
    try {
      const ready = receive('ready', undefined, 25_000);
      worker.postMessage({ type: 'init' }); await ready;
      const source = new OffscreenCanvas(320, 240); const context = source.getContext('2d')!;
      context.fillStyle = '#000000'; context.fillRect(0, 0, source.width, source.height);
      const replies: Reply[] = []; let timestamp = 0;
      for (const id of [1, 2]) {
        const bitmap = await createImageBitmap(source); timestamp = Math.max(timestamp + 1, performance.now());
        const next = receive('result', id, 8_000);
        worker.postMessage({ type: 'frame', id, bitmap, observedAtMs: timestamp }, [bitmap]);
        const reply = await next;
        if (reply.observedAtMs !== timestamp) throw new Error('The real worker changed the capture timestamp.');
        replies.push(reply);
      }
      return replies;
    } finally { worker.terminate(); }
  });
  expect(result.map((reply) => reply.id)).toEqual([1, 2]);
  expect(result.every((reply) => reply.type === 'result' && Array.isArray(reply.boxes) && reply.boxes.length === 0)).toBe(true);
  const origin = new URL(page.url()).origin;
  expect(network.every((url) => new URL(url).origin === origin)).toBe(true);
  expect(network.some((url) => /vision_wasm_(?:nosimd_)?internal\.wasm/.test(url))).toBe(true);
  expect(network.some((url) => url.includes('blaze_face_short_range.tflite'))).toBe(true);
});

test('defaults to Jensen in the real API, source library and conversation without Musk data', async ({ page, request }, info) => {
  const profileResponse = await request.get('/api/persona'); expect(profileResponse.ok()).toBe(true);
  const profile = await profileResponse.json();
  expect(profile.id).toBe('jensen-huang'); expect(profile.name).toBe('Jensen Huang');
  expect(profile.bindings.persona_id).toBe('jensen-huang');
  expect(profile.bindings.voice.persona_id).not.toBe('elon-musk');
  const sourceResponse = await request.get('/api/sources'); expect(sourceResponse.ok()).toBe(true);
  const { sources } = await sourceResponse.json(); expect(sources.length).toBeGreaterThan(0);
  expect(sources.every((source: { persona_id: string }) => source.persona_id === 'jensen-huang')).toBe(true);
  const sourceIds = new Set(sources.map((source: { id: string }) => source.id));
  const response = await request.post('/api/chat', { data: {
    message: 'What did you mean by AI factories at COMPUTEX 2025?', mode: 'behavior', language: 'en', motion_mode: 'reviewed',
  } });
  expect(response.ok()).toBe(true); const answer = await response.json();
  expect(answer.citations.length).toBeGreaterThan(0);
  expect(answer.citations.every((citation: { source_id: string }) => sourceIds.has(citation.source_id))).toBe(true);
  expect(answer.behavior.motion).toBeNull();
  expect(answer.text).not.toMatch(/Elon Musk|SpaceX|Tesla/i);
  const motion = await (await request.get('/api/motion/library')).json(); expect(motion.available).toBe(false);
  await openStudio(page);
  await expect(page.locator('.stage-title')).toContainText('Jensen Huang');
  await expect(page.locator('body')).not.toContainText(/Elon Musk|SpaceX|Tesla/);
  await navigation(page, info.project.name === 'mobile').getByRole('button', { name: info.project.name === 'mobile' ? 'Sources' : /Source library/ }).click();
  await expect(page.locator('.source-card')).toHaveCount(sources.length);
  await expect(page.locator('body')).not.toContainText(/Elon Musk|SpaceX|Tesla/);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('runs actual idle avatar eyes-first motion, follows both directions and returns after Stop', async ({ page }, info) => {
  await openStudio(page);
  await canvas(page).evaluate((element) => {
    const state = window as TraceWindow; state.__attentionTrace = [];
    const record = () => {
      state.__attentionTrace.push({ at: performance.now(), phase: element.getAttribute('data-attention-phase') ?? '',
        eyeYaw: Number(element.getAttribute('data-attention-eye-yaw')), headYaw: Number(element.getAttribute('data-attention-head-yaw')) });
      if (state.__attentionTrace.length > 1000) state.__attentionTrace.shift();
    };
    state.__attentionObserver = new MutationObserver(record);
    state.__attentionObserver.observe(element, { attributes: true, attributeFilter: ['data-attention-phase', 'data-attention-eye-yaw', 'data-attention-head-yaw'] });
  });
  await page.getByRole('button', { name: 'Try pointer preview', exact: true }).click();
  await expect(page.getByTestId('attention-panel')).toHaveAttribute('data-input', 'demo');
  await expect.poll(() => angle(page, 'head-yaw'), { timeout: 4000 }).toBeGreaterThan(0.12);
  const trace = await page.evaluate(() => (window as TraceWindow).__attentionTrace);
  const eyes = trace.filter((sample) => sample.phase === 'eyes');
  const firstFollow = trace.find((sample) => sample.phase === 'following');
  expect(eyes.length).toBeGreaterThan(0); expect(firstFollow).toBeDefined();
  expect(Math.max(...eyes.map((sample) => sample.eyeYaw))).toBeGreaterThan(0.10);
  expect(Math.max(...eyes.map((sample) => Math.abs(sample.headYaw)))).toBeLessThan(0.03);
  // The artistic delay is 220 ms; allow one rendered frame of observation offset.
  expect(firstFollow!.at - eyes[0].at).toBeGreaterThanOrEqual(180);
  await expect(page.getByText('Ready when you are', { exact: true })).toBeVisible();
  await page.getByRole('slider', { name: 'Preview horizontal position', exact: true }).press('Home');
  await page.getByRole('slider', { name: 'Preview vertical position', exact: true }).press('End');
  await expect.poll(() => angle(page, 'head-yaw'), { timeout: 4000 }).toBeLessThan(-0.25);
  await expect.poll(() => angle(page, 'head-pitch'), { timeout: 4000 }).toBeGreaterThan(0.12);
  expect(await angle(page, 'eye-yaw')).toBeLessThan(0);
  expect(await angle(page, 'eye-pitch')).toBeGreaterThan(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: info.outputPath('jensen-attention-live-preview.png'), fullPage: true });
  await page.getByRole('button', { name: 'Stop attention tracking', exact: true }).click();
  await expect(page.getByTestId('attention-panel')).toHaveAttribute('data-input', 'off');
  await expect(canvas(page)).toHaveAttribute('data-attention-phase', 'idle', { timeout: 4000 });
  expect(await angle(page, 'eye-yaw')).toBe(0); expect(await angle(page, 'head-yaw')).toBe(0);
});

test('honors reduced motion while the keyboard-controlled preview remains usable', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await openStudio(page);
  await page.getByRole('button', { name: 'Try pointer preview', exact: true }).click();
  const pad = page.getByTestId('attention-demo-pad'); await pad.focus();
  await pad.press('Home'); await pad.press('Shift+ArrowLeft'); await pad.press('Shift+ArrowUp');
  await expect(canvas(page)).toHaveAttribute('data-attention-phase', 'reduced');
  await expect.poll(() => angle(page, 'eye-yaw'), { timeout: 3000 }).toBeLessThan(-0.04);
  expect(Math.abs(await angle(page, 'eye-yaw'))).toBeLessThanOrEqual(0.081);
  expect(Math.abs(await angle(page, 'eye-pitch'))).toBeLessThanOrEqual(0.081);
  expect(await angle(page, 'head-yaw')).toBe(0); expect(await angle(page, 'head-pitch')).toBe(0);
  await expect(page.getByText('Reduced motion is on. Head movement is limited.', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Stop attention tracking', exact: true }).click();
  await expect(canvas(page)).toHaveAttribute('data-attention-phase', 'idle', { timeout: 4000 });
});

test('navigation ends the pointer session and returning does not restart it', async ({ page }, info) => {
  await openStudio(page);
  await page.getByRole('button', { name: 'Try pointer preview', exact: true }).click();
  await expect(canvas(page)).toHaveAttribute('data-attention-phase', 'following');
  const nav = navigation(page, info.project.name === 'mobile');
  await nav.getByRole('button', { name: info.project.name === 'mobile' ? 'Sources' : /Source library/ }).click();
  await expect(page.getByTestId('attention-panel')).toHaveCount(0);
  await nav.getByRole('button', { name: info.project.name === 'mobile' ? 'Studio' : 'Conversation', exact: true }).click();
  await expect(page.getByTestId('attention-panel')).toHaveAttribute('data-input', 'off');
  await expect(canvas(page)).toHaveAttribute('data-attention-phase', 'idle');
  await expect(page.getByTestId('attention-demo-pad')).toHaveCount(0);
});

test('permission denial leaves capture off and allows a pointer preview retry', async ({ page }) => {
  await installCaptureFixture(page, 'denied'); await openStudio(page);
  expect(await page.evaluate(() => (window as FixtureWindow).__attentionFixture.cameraCalls)).toBe(0);
  await page.getByRole('button', { name: 'Camera', exact: true }).click();
  await expect(page.getByTestId('attention-panel')).toHaveAttribute('data-phase', 'error');
  await expect(page.getByTestId('attention-panel')).toHaveAttribute('data-input', 'off');
  await expect(page.getByRole('alert')).toHaveText('Permission was not granted. Choose Camera or Share screen to try again.');
  expect(await page.evaluate(() => (window as FixtureWindow).__attentionFixture.workers)).toBe(0);
  await page.getByRole('button', { name: 'Try pointer preview', exact: true }).click();
  await expect(canvas(page)).toHaveAttribute('data-attention-phase', 'following');
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('cancelling a pending permission request stops the subsequently granted stream', async ({ page }) => {
  await installCaptureFixture(page, 'delayed'); await openStudio(page);
  await page.getByRole('button', { name: 'Camera', exact: true }).click();
  await expect(page.getByTestId('attention-panel')).toHaveAttribute('data-phase', 'permission');
  await page.getByRole('button', { name: 'Stop attention tracking', exact: true }).click();
  await page.evaluate(() => { const fixture = (window as FixtureWindow).__attentionFixture; if (!fixture.grant) throw new Error('Permission fixture was not pending'); fixture.grant(); });
  await expect.poll(() => page.evaluate(() => (window as FixtureWindow).__attentionFixture.streams[0]?.getVideoTracks()[0]?.readyState)).toBe('ended');
  expect(await page.evaluate(() => (window as FixtureWindow).__attentionFixture.workers)).toBe(0);
  expect(await page.getByLabel('Camera preview', { exact: true }).evaluate((video) => (video as HTMLVideoElement).srcObject)).toBeNull();
  await expect(page.getByTestId('attention-panel')).toHaveAttribute('data-input', 'off');
  await expect(canvas(page)).toHaveAttribute('data-attention-phase', 'idle');
});

test('synthetic face detections preserve selection through reordering and require reselection after loss', async ({ page }, info) => {
  await installCaptureFixture(page); await openStudio(page);
  const outgoing: string[] = [];
  page.on('request', (request) => { if (request.method() !== 'GET') outgoing.push(request.url()); });
  await page.getByRole('button', { name: 'Camera', exact: true }).click();
  const panel = page.getByTestId('attention-panel');
  await expect(panel).toHaveAttribute('data-phase', 'running');
  await expect(page.getByRole('button', { name: 'Participant 1', exact: true })).toBeVisible();
  await expect(panel).toHaveAttribute('data-selected-participant', '');
  await expect(canvas(page)).toHaveAttribute('data-attention-phase', 'idle');
  await page.getByRole('button', { name: 'Participant 1', exact: true }).click();
  await expect(panel).toHaveAttribute('data-selected-participant', '1');
  // Source-left becomes display-right in the mirrored camera preview.
  await expect.poll(() => angle(page, 'head-yaw'), { timeout: 4000 }).toBeGreaterThan(0.18);
  const selectedTarget = await canvas(page).getAttribute('data-attention-target');
  const frames = await page.evaluate(() => {
    const fixture = (window as FixtureWindow).__attentionFixture;
    fixture.boxes = [fixture.boxes[1], { ...fixture.boxes[0], x: 0.12 }]; return fixture.frames;
  });
  await expect.poll(() => page.evaluate(() => (window as FixtureWindow).__attentionFixture.frames), { timeout: 3000 }).toBeGreaterThan(frames + 2);
  await expect(panel).toHaveAttribute('data-selected-participant', '1');
  await expect(canvas(page)).toHaveAttribute('data-attention-target', selectedTarget!);
  expect(await angle(page, 'head-yaw')).toBeGreaterThan(0.15);
  await page.evaluate(() => { (window as FixtureWindow).__attentionFixture.boxes = []; });
  await expect(panel).toHaveAttribute('data-selected-participant', '', { timeout: 2500 });
  await expect(page.getByText('Selection lost or ambiguous. Select a visible participant again.', { exact: true })).toBeVisible();
  await expect(canvas(page)).toHaveAttribute('data-attention-phase', 'idle', { timeout: 4000 });
  await page.evaluate(() => { (window as FixtureWindow).__attentionFixture.boxes = [{ x: 0.65, y: 0.2, width: 0.2, height: 0.35, score: 0.99 }]; });
  await expect(page.getByRole('button', { name: /^Participant \d+$/ })).toHaveCount(1);
  await expect(panel).toHaveAttribute('data-selected-participant', '');
  await expect(canvas(page)).toHaveAttribute('data-attention-phase', 'idle');
  expect(await page.evaluate(() => (window as FixtureWindow).__attentionFixture.cameraConstraints?.audio)).toBe(false);
  expect(outgoing).toEqual([]);
  await navigation(page, info.project.name === 'mobile').getByRole('button', { name: info.project.name === 'mobile' ? 'Sources' : /Source library/ }).click();
  await expect.poll(() => page.evaluate(() => (window as FixtureWindow).__attentionFixture.streams[0].getVideoTracks()[0].readyState)).toBe('ended');
  expect(await page.evaluate(() => (window as FixtureWindow).__attentionFixture.terminated)).toBe(1);
});

test('screen capture starts under user activation and external track end cleans up the worker', async ({ page }) => {
  await installCaptureFixture(page); await openStudio(page);
  await page.getByRole('button', { name: 'Share screen', exact: true }).click();
  await expect(page.getByTestId('attention-panel')).toHaveAttribute('data-phase', 'running');
  expect(await page.evaluate(() => (window as FixtureWindow).__attentionFixture.screenHadActivation)).toBe(true);
  expect(await page.evaluate(() => (window as FixtureWindow).__attentionFixture.screenConstraints?.audio)).toBe(false);
  await expect(page.getByRole('button', { name: 'Participant 1', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Participant 1', exact: true }).click();
  // Screen capture is not mirrored: the same source-left box turns the head left.
  await expect.poll(() => angle(page, 'head-yaw'), { timeout: 4000 }).toBeLessThan(-0.18);
  await page.evaluate(() => {
    const track = (window as FixtureWindow).__attentionFixture.streams[0].getVideoTracks()[0];
    track.stop(); track.dispatchEvent(new Event('ended'));
  });
  await expect(page.getByTestId('attention-panel')).toHaveAttribute('data-input', 'off');
  await expect(page.getByText('The video input ended. Choose an input to restart.', { exact: true })).toBeVisible();
  expect(await page.evaluate(() => (window as FixtureWindow).__attentionFixture.terminated)).toBe(1);
  await expect(canvas(page)).toHaveAttribute('data-attention-phase', 'idle', { timeout: 4000 });
});

test('Jensen fitted candidate renders its actual asset and resets preview motion cleanly', async ({ page }, info) => {
  const expectedAsset = JSON.parse(readFileSync('reports/jensen-avatar/runtime-asset-report.json', 'utf8'));
  const graphicsErrors: string[] = [];
  page.on('console', (message) => { if (message.type() === 'error' && /WebGL|shader|THREE|texture/i.test(message.text())) graphicsErrors.push(message.text()); });
  await openStudio(page);
  await page.getByRole('button', { name: 'Studio settings', exact: true }).filter({ visible: true }).click();
  await page.getByRole('button', { name: /^Jensen source-fitted candidate/ }).click();
  await page.getByRole('button', { name: 'Close settings' }).click();
  const stage = page.getByTestId('avatar-stage');
  await expect(stage).toHaveAttribute('data-model-state', 'ready', { timeout: 15000 });
  await expect(stage).toHaveAttribute('data-asset-sha256', expectedAsset.output_sha256);
  await expect(stage).toHaveAttribute('data-mapped-joints', '46');
  await expect(stage).toHaveAttribute('data-face-controls', '52');
  await expect(stage).toHaveAttribute('data-framing', 'upper_body');
  await expect(page.getByText('JENSEN · SOURCE-FITTED CANDIDATE', { exact: true })).toBeVisible();
  expect(await canvas(page).evaluate((element) => {
    const surface = element as HTMLCanvasElement; const gl = surface.getContext('webgl2');
    return Boolean(gl && !gl.isContextLost() && gl.drawingBufferWidth > 0 && gl.drawingBufferHeight > 0);
  })).toBe(true);
  await navigation(page, info.project.name === 'mobile').getByRole('button', { name: info.project.name === 'mobile' ? 'Lab' : 'Research lab', exact: true }).click();
  await expect(stage).toHaveAttribute('data-asset-sha256', expectedAsset.output_sha256, { timeout: 15000 });
  await page.getByRole('button', { name: 'Load Jensen face & hands', exact: true }).click();
  await expect(page.getByText('goldman-sachs-2024', { exact: true })).toBeVisible();
  const replay = page.getByRole('button', { name: 'Replay source motion', exact: true });
  await expect(replay).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Load narration sample', exact: true })).toHaveCount(0);
  await expect(page.getByRole('combobox', { name: 'Motion lab camera framing' })).toHaveValue('upper_body');
  await page.getByRole('checkbox', { name: 'Preview geometry without speaker attribution', exact: true }).check();
  await replay.click();
  await expect.poll(async () => Number(await page.getByRole('slider', { name: 'Source position' }).inputValue())).toBeGreaterThan(1800);
  await page.getByRole('button', { name: 'Pause replay', exact: true }).click();
  const evidence = info.outputPath(`jensen-fitted-hands-${info.project.name}.png`);
  await stage.locator('.avatar-visual').screenshot({ path: evidence });
  await info.attach('actual fitted avatar and hands', { path: evidence, contentType: 'image/png' });
  await page.getByRole('button', { name: 'Reset source replay', exact: true }).click();
  await expect(page.getByRole('slider', { name: 'Source position' })).toHaveValue('0');
  await page.waitForTimeout(180);
  await expect(page.getByRole('slider', { name: 'Source position' })).toHaveValue('0');
  await page.getByRole('checkbox', { name: 'Preview geometry without speaker attribution', exact: true }).uncheck();
  await expect(replay).toBeDisabled();
  await page.getByRole('button', { name: 'Clear imported files', exact: true }).click();
  await expect(page.getByText('goldman-sachs-2024', { exact: true })).toHaveCount(0);
  expect(graphicsErrors).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

type RecordedSample = { at: number; audioMs: number; mediaMs: number; speaking: boolean;
  jaw: number; blink: number; smile: number; arm: number; hand: number; finger: number };
type RecordedWindow = Window & { __recordedMedia: HTMLAudioElement[]; __recordedContexts: AudioContext[];
  __recordedTrace: RecordedSample[]; __recordedTraceFrame: number };

test('replays actual generated Jensen audio through the fitted avatar and cancels its clock and composed channels', async ({ page }, info) => {
  // The WAV is a saved neural output. The response envelope and gestures are
  // declared replay fixtures; no inference server or synthetic audio is used.
  const label = 'Recorded generated-audio replay — no live inference.';
  const audioPath = 'reports/voice/jensen-2026-11-13/listening/adapted-en_multisentence.wav';
  const audio = readFileSync(audioPath);
  const audioHash = createHash('sha256').update(audio).digest('hex');
  expect(audioHash).toBe('ca5664747699df7826d4c380c0d26f6b36ff34157df742276945ffbab62086f2');
  expect(audio.length).toBe(476204);
  const cases = readFileSync('tools/voice_evaluation/cases.json');
  expect(createHash('sha256').update(cases).digest('hex')).toBe('dc00ae852aeac5028914c68fbcbb019c00d85b46575912379976345752efe649');
  const spokenText = JSON.parse(cases.toString()).cases.find((item: { id: string }) => item.id === 'en_multisentence').text;
  const assetHash = 'dea0c921fb7d5668527bd8ee9082f382b14e8d300c43936842e15356dc7d90cd';
  const clipUrl = `/api/speech/clips/${'c'.repeat(32)}`;
  let preparedRequests = 0; let clipRequests = 0; let liveRequests = 0;
  const graphicsErrors: string[] = [];
  page.on('console', (message) => { if (message.type() === 'error' && /WebGL|shader|THREE|texture/i.test(message.text())) graphicsErrors.push(message.text()); });
  await page.addInitScript(() => {
    const state = window as RecordedWindow;
    state.__recordedMedia = []; state.__recordedContexts = []; state.__recordedTrace = []; state.__recordedTraceFrame = 0;
    // Observe real browser objects; do not replace decoding, clocks or analysis.
    Object.defineProperty(window, 'Audio', { configurable: true, value: new Proxy(window.Audio, {
      construct(target, args) { const media = Reflect.construct(target, args) as HTMLAudioElement; state.__recordedMedia.push(media); return media; },
    }) });
    Object.defineProperty(window, 'AudioContext', { configurable: true, value: new Proxy(window.AudioContext, {
      construct(target, args) { const context = Reflect.construct(target, args) as AudioContext; state.__recordedContexts.push(context); return context; },
    }) });
  });
  await page.route('**/api/persona', async (route) => {
    const response = await route.fetch(); const profile = await response.json();
    await route.fulfill({ response, json: { ...profile, capabilities: { ...profile.capabilities, speech: true, live_speech: false },
      bindings: { ...profile.bindings, voice: { status: 'matched', persona_id: 'jensen-huang', voice: 'reference' } } } });
  });
  const turn = { id: 'recorded-audio-replay', session_id: 'a'.repeat(32), text: spokenText, language: 'en', mode: 'behavior', provider: 'evidence',
    citations: [], grounding: { status: 'unverified', retrieved_ids: [], invalid_citation_ids: [], reason: label },
    metrics: { retrieval_ms: 0, generation_ms: 0, total_ms: 0 }, context: { condition: 'recorded_generated_audio_replay', live_inference: false },
    behavior: { duration_ms: 9920, timing_basis: 'estimated_text', motion: null, speech: { rate: 1, pitch: 1, pause_ms: 0 }, cues: [
      { channel: 'face', name: 'smile', start_ms: 0, duration_ms: 9600, intensity: .55, provenance: 'designed' },
      { channel: 'head', name: 'nod', start_ms: 500, duration_ms: 8500, intensity: .5, provenance: 'designed' },
      { channel: 'gesture', name: 'open_palm', start_ms: 400, duration_ms: 9000, intensity: .65, provenance: 'designed' },
      { channel: 'gesture', name: 'emphasis', start_ms: 400, duration_ms: 9000, intensity: .6, provenance: 'designed' },
    ] } };
  await page.route('**/api/chat/stream', (route) => route.fulfill({ contentType: 'text/event-stream', body: [
    ['meta', { session_id: turn.session_id, turn_id: turn.id }], ['token', { text: turn.text }], ['result', { turn }], ['done', {}],
  ].map(([event, data]) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`).join('') }));
  await page.route('**/api/speech/prepare', (route) => {
    preparedRequests++;
    expect(route.request().postDataJSON()).toEqual({ text: spokenText, language: 'en' });
    return route.fulfill({ json: { audio_url: clipUrl, audio_sha256: audioHash, timing_status: 'not_configured',
      speech_timing: null, timing_message: label+' Mouth motion uses actual waveform amplitude; no phoneme alignment.' } });
  });
  await page.route(`**${clipUrl}`, (route) => { clipRequests++; return route.fulfill({ body: audio, contentType: 'audio/wav' }); });
  for (const path of ['/api/speech', '/api/chat/live']) await page.route(`**${path}`, (route) => {
    liveRequests++; return route.fulfill({ status: 503, json: { detail: 'Live inference is excluded from this recorded playback check.' } });
  });

  await openStudio(page);
  await page.getByRole('button', { name: 'Studio settings', exact: true }).filter({ visible: true }).click();
  await page.getByRole('button', { name: /^Jensen source-fitted candidate/ }).click();
  await page.getByRole('button', { name: /Connected voice/ }).click();
  await page.getByRole('button', { name: 'Close settings' }).click();
  const stage = page.getByTestId('avatar-stage'); const surface = canvas(page);
  await expect(stage).toHaveAttribute('data-model-state', 'ready', { timeout: 20000 });
  await expect(stage).toHaveAttribute('data-asset-sha256', assetHash);
  await page.getByRole('textbox', { name: 'Your message' }).fill(label);
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  await expect(page.getByText(label, { exact: true })).toBeVisible();
  await surface.evaluate((element) => {
    const state = window as RecordedWindow;
    const frame = () => {
      const value = (name: string) => Number(element.getAttribute(`data-${name}`) ?? 0);
      const media = state.__recordedMedia.at(-1);
      state.__recordedTrace.push({ at: performance.now(), audioMs: value('audio-clock-ms'), mediaMs: (media?.currentTime ?? 0)*1000,
        speaking: element.getAttribute('data-speaking') === 'true', jaw: value('rig-jaw'), blink: value('rig-blink'),
        smile: value('rig-smile'), arm: value('rig-arm'), hand: value('rig-hand'), finger: value('rig-finger') });
      if (state.__recordedTrace.length > 1500) state.__recordedTrace.shift();
      state.__recordedTraceFrame = requestAnimationFrame(frame);
    };
    frame();
  });
  await page.getByRole('button', { name: 'Play response' }).click();
  await expect(surface).toHaveAttribute('data-speaking', 'true');
  await expect(surface).toHaveAttribute('data-audio-sha256', audioHash);
  await expect(surface).toHaveAttribute('data-speech-source', 'amplitude');
  await expect(page.getByText('Provider voice · approximate mouth movement', { exact: true })).toHaveAttribute('title', /Recorded generated-audio replay/);
  await expect.poll(async () => Number(await surface.getAttribute('data-audio-clock-ms')), { timeout: 12000 }).toBeGreaterThan(1800);
  expect(await page.evaluate(() => (window as RecordedWindow).__recordedMedia.at(-1)!.duration)).toBeCloseTo(9.92, 3);
  const screenshot = info.outputPath(`recorded-generated-audio-${info.project.name}.png`);
  await page.locator('.studio-grid').screenshot({ path: screenshot });
  await info.attach('recorded generated audio replay; no live inference', { path: screenshot, contentType: 'image/png' });
  await expect.poll(async () => Number(await surface.getAttribute('data-audio-clock-ms')), { timeout: 12000 }).toBeGreaterThan(6500);
  const beforeStop = await page.evaluate(() => {
    const state = window as RecordedWindow; const media = state.__recordedMedia.at(-1)!;
    return { trace: state.__recordedTrace, mediaMs: media.currentTime*1000, ended: media.ended, paused: media.paused,
      mediaCount: state.__recordedMedia.length, contextCount: state.__recordedContexts.length };
  });
  // Cancel before inspecting/exporting the collected trace; the real file lasts 9.92s.
  await page.getByRole('button', { name: 'Stop audio', exact: true }).click();
  expect(beforeStop.ended).toBe(false); expect(beforeStop.paused).toBe(false);
  expect(beforeStop.mediaMs).toBeLessThan(9920);
  expect(beforeStop.mediaCount).toBe(1); expect(beforeStop.contextCount).toBe(1);
  const trace = beforeStop.trace;
  const playing = trace.filter((sample) => sample.speaking && sample.audioMs > 100 && sample.audioMs < 9000);
  const maximum = (key: 'jaw' | 'blink' | 'smile' | 'arm' | 'hand' | 'finger') => Math.max(...playing.map((sample) => sample[key]));
  expect(playing.length).toBeGreaterThan(10);
  expect(maximum('jaw')).toBeGreaterThan(.02); expect(maximum('blink')).toBeGreaterThan(.03);
  expect(maximum('smile')).toBeGreaterThan(.1); expect(maximum('arm')).toBeGreaterThan(.1);
  expect(maximum('hand')).toBeGreaterThan(.1); expect(maximum('finger')).toBeGreaterThan(.005);
  const composed = playing.filter((sample) => sample.jaw > .02 && sample.smile > .1 && sample.arm > .1 && sample.hand > .1);
  expect(composed.length).toBeGreaterThan(0);
  const maximumClockDifference = Math.max(...playing.map((sample) => Math.abs(sample.audioMs-sample.mediaMs)));
  expect(maximumClockDifference).toBeLessThan(750); // Bounded rendered-frame lag on software WebGL, not acoustic latency.
  await expect(surface).toHaveAttribute('data-speaking', 'false');
  await expect(surface).toHaveAttribute('data-audio-clock-ms', '0.000');
  await expect(surface).toHaveAttribute('data-audio-sha256', '');
  await expect(surface).toHaveAttribute('data-speech-source', 'silent');
  for (const name of ['jaw', 'smile', 'arm', 'hand', 'finger']) await expect.poll(async () => Number(await surface.getAttribute(`data-rig-${name}`))).toBeLessThan(.001);
  await expect.poll(() => page.evaluate(() => (window as RecordedWindow).__recordedContexts.every((context) => context.state === 'closed'))).toBe(true);
  expect(await page.evaluate(() => (window as RecordedWindow).__recordedMedia.every((media) => media.paused && media.getAttribute('src') === null))).toBe(true);
  await page.waitForTimeout(250);
  await expect(surface).toHaveAttribute('data-audio-clock-ms', '0.000');
  await page.evaluate(() => cancelAnimationFrame((window as RecordedWindow).__recordedTraceFrame));
  expect(preparedRequests).toBe(1); expect(clipRequests).toBe(1); expect(liveRequests).toBe(0);
  expect(graphicsErrors).toEqual([]); await expect(page.getByRole('alert')).toHaveCount(0);
  await info.attach('actual generated audio playback measurements', { body: Buffer.from(JSON.stringify({
    schema_version: 'recorded-generated-audio-browser.v1', project: info.project.name,
    condition: label, audio_file: audioPath, audio_sha256: audioHash, audio_duration_seconds: 9.92, avatar_sha256: assetHash,
    articulation: 'actual waveform amplitude; no phoneme alignment', gestures: 'authored fixture cues; not a learned personal policy',
    live_inference: false, simultaneous_neural_providers: false, acoustic_latency_measured: false,
    cancelled_before_natural_end: !beforeStop.ended, media_ms_before_stop: beforeStop.mediaMs,
    real_media_instances: beforeStop.mediaCount, real_audio_contexts: beforeStop.contextCount,
    playing_frames: playing.length, composed_frames: composed.length, maximum_clock_difference_ms: maximumClockDifference,
    channel_maxima: Object.fromEntries((['jaw', 'blink', 'smile', 'arm', 'hand', 'finger'] as const).map((key) => [key, maximum(key)])),
    cancelled_to_zero_clock_and_neutral_channels: true, trace,
  }, null, 2)), contentType: 'application/json' });
});
