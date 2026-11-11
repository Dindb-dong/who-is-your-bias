import { createHash } from 'node:crypto';
import { createServer, type ServerResponse } from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import { expect, test as base, type Page, type TestInfo } from '@playwright/test';

// Synthetic SSE protocol and sine-wave PCM only. These tests exercise actual
// browser fetch cancellation, Web Audio and UI clocks, not neural TTS quality,
// likeness, acoustic output or target-machine model latency.
const FIXTURE = 'Synthetic live-segments-pcm.v1 transport; 16 kHz mono sine wave; no model inference or acoustic measurement.';
const RATE = 16000;
const PREFIX = 'Synthetic early sentence. ';
const SUFFIX = 'Synthetic final sentence.';

class Connection {
  closedBeforeEnd = false;
  ended = false;
  body = '';
  constructor(readonly response: ServerResponse) {
    response.on('close', () => { if (!this.ended) this.closedBeforeEnd = true; });
  }
  emit(event: string, data: unknown): boolean {
    if (this.response.destroyed || this.ended) return false;
    this.response.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
    return true;
  }
  end() { this.ended = true; this.response.end(); }
}

async function syntheticServer() {
  const connections: Connection[] = [];
  const server = createServer((request, response) => {
    response.setHeader('Access-Control-Allow-Origin', '*');
    response.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    response.setHeader('Access-Control-Allow-Headers', 'content-type, accept');
    response.setHeader('Access-Control-Allow-Private-Network', 'true');
    if (request.method === 'OPTIONS') { response.writeHead(204); response.end(); return; }
    if (request.method !== 'POST' || request.url !== '/synthetic-live') { response.writeHead(404); response.end(); return; }
    response.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' });
    response.flushHeaders();
    const connection = new Connection(response); connections.push(connection);
    request.setEncoding('utf8'); request.on('data', (chunk: string) => { connection.body += chunk; });
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => { server.removeListener('error', reject); resolve(); });
  });
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Synthetic SSE listener did not acquire a port.');
  return { url: `http://127.0.0.1:${address.port}/synthetic-live`, connections,
    async dispose() {
      for (const connection of connections) connection.response.destroy();
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    } };
}
type SyntheticServer = Awaited<ReturnType<typeof syntheticServer>>;
const test = base.extend<{ live: SyntheticServer }>({
  live: async ({}, use) => {
    const live = await syntheticServer();
    try { await use(live); } finally { await live.dispose(); }
  },
});

async function prepare(page: Page, live: SyntheticServer) {
  const errors: string[] = []; const synthesisRequests: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('request', (request) => { if (request.url().includes('/api/speech')) synthesisRequests.push(request.url()); });
  await page.route('**/api/persona', (route) => route.fulfill({ json: {
    id: 'synthetic-protocol-fixture', name: 'Synthetic protocol fixture', tagline: FIXTURE, description: FIXTURE,
    example_questions: { en: [], ko: [] }, stats: { sources: 0, evidence: 0, held_out_sources: 0 },
    capabilities: { llm: true, speech: true, live_speech: true, transcription: false }, notice: FIXTURE,
  } }));
  await page.route('**/api/sources', (route) => route.fulfill({ json: { sources: [] } }));
  await page.route('**/api/motion/library', (route) => route.fulfill({ json: {
    available: false, status: 'not_configured', reviewed_runtime_primitives: 0, preview_runtime_primitives: 0,
  } }));
  await page.addInitScript(({ url }) => {
    const fetcher = window.fetch.bind(window);
    window.fetch = (input, init) => typeof input === 'string' && new URL(input, location.href).pathname === '/api/chat/live'
      ? fetcher(url, init) : fetcher(input, init);
    // Observe real contexts; do not replace scheduling, analysers or sample time.
    const contexts: AudioContext[] = [];
    Object.assign(window, { __syntheticLiveAudioContexts: contexts });
    window.AudioContext = new Proxy(window.AudioContext, { construct(Target, args) {
      const context = Reflect.construct(Target, args) as AudioContext; contexts.push(context); return context;
    } });
  }, { url: live.url });
  await page.goto('/');
  await expect(page.getByText('Studio connected', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Studio settings', exact: true }).filter({ visible: true }).click();
  await page.getByRole('button', { name: /^Connected voice/ }).click();
  await page.getByRole('checkbox', { name: 'Read new responses aloud', exact: true }).check();
  await page.getByRole('button', { name: /^Stylized portrait/ }).click();
  await page.getByRole('button', { name: 'Close settings' }).click();
  await expect(page.getByTestId('avatar-stage')).toHaveAttribute('data-rig-kind', 'procedural');
  return { errors, synthesisRequests };
}

async function submit(page: Page, live: SyntheticServer) {
  const index = live.connections.length;
  await page.getByRole('textbox', { name: 'Your message', exact: true }).fill('Run the synthetic streaming protocol fixture.');
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  await expect.poll(() => live.connections.length).toBe(index + 1);
  return live.connections[index];
}
function begin(connection: Connection, turnId = 'synthetic-turn') {
  connection.emit('meta', { session_id: 'synthetic-session', turn_id: turnId, protocol: 'live-segments-pcm.v1' });
  connection.emit('token', { text: PREFIX });
}
function pcm(frames: number) {
  const bytes = Buffer.alloc(frames * 2);
  for (let frame = 0; frame < frames; frame++) bytes.writeInt16LE(Math.round(Math.sin(frame / RATE * 2 * Math.PI * 220) * 5000), frame * 2);
  return bytes;
}
function wavHash(bytes: Buffer) {
  const header = Buffer.alloc(44); header.write('RIFF'); header.writeUInt32LE(bytes.length + 36, 4); header.write('WAVEfmt ', 8);
  header.writeUInt32LE(16, 16); header.writeUInt16LE(1, 20); header.writeUInt16LE(1, 22); header.writeUInt32LE(RATE, 24);
  header.writeUInt32LE(RATE * 2, 28); header.writeUInt16LE(2, 32); header.writeUInt16LE(16, 34); header.write('data', 36); header.writeUInt32LE(bytes.length, 40);
  return createHash('sha256').update(header).update(bytes).digest('hex');
}
function segmentHeader(connection: Connection, sequence: number, offset: number, frames: number, turnId = 'synthetic-turn') {
  const bytes = pcm(frames); const identity = { turn_id: turnId, segment_id: `${turnId}:${sequence}`, sequence };
  connection.emit('segment', { ...identity, text: sequence ? SUFFIX : PREFIX, evidence_ids: [], state: 'committed' });
  connection.emit('audio_segment', { ...identity, audio_sha256: wavHash(bytes), sample_rate: RATE, channels: 1,
    sample_offset: offset, sample_count: frames, synthesis_ms: 0 });
  return { bytes, identity };
}
function segment(connection: Connection, sequence: number, offset: number, seconds: number, chunkStart: number) {
  const frames = RATE * seconds; const { bytes, identity } = segmentHeader(connection, sequence, offset, frames);
  let chunk = chunkStart;
  for (let frame = 0; frame < frames; frame += RATE) {
    const count = Math.min(RATE, frames - frame);
    connection.emit('audio', { ...identity, sequence: chunk++, segment_sequence: sequence, sample_rate: RATE, channels: 1,
      sample_offset: offset + frame, sample_count: count, encoding: 'pcm_s16le', pcm_base64: bytes.subarray(frame * 2, (frame + count) * 2).toString('base64') });
  }
  connection.emit('audio_segment_end', { ...identity, sample_end: offset + frames, state: 'delivered' });
  return chunk;
}
function finish(connection: Connection, chunks: number, frames: number) {
  connection.emit('result', { turn: { id: 'synthetic-turn', session_id: 'synthetic-session', text: PREFIX + SUFFIX,
    language: 'en', mode: 'behavior', provider: 'openai', citations: [],
    behavior: { duration_ms: frames / RATE * 1000, cues: [], speech: { rate: 1, pitch: 1, pause_ms: 0 } },
    grounding: { status: 'unverified', retrieved_ids: [], invalid_citation_ids: [] },
    metrics: { retrieval_ms: 0, generation_ms: 0, total_ms: 0 } } });
  connection.emit('audio_end', { turn_id: 'synthetic-turn', chunks, sample_count: frames, sample_rate: RATE,
    first_pcm_ready_ms: 0, total_ms: 0, state: 'delivered', spoken_complete: false, acoustic_latency_ms: null });
  connection.emit('done', {}); connection.end();
}
const timing = (page: Page) => page.getByRole('region', { name: 'Live response timing', exact: true });
const timingValue = (page: Page, label: string) => timing(page).locator('dl > div').filter({ has: page.getByText(label, { exact: true }) }).locator('dd');
const audioStates = (page: Page) => page.evaluate(() => ((window as unknown as { __syntheticLiveAudioContexts: AudioContext[] })
  .__syntheticLiveAudioContexts ?? []).map((context) => context.state));
type TimingExport = {
  status: string; first_audio_signal_ms: number | null; answer_complete_ms: number | null; stream_complete_ms: number | null; turn_complete_ms: number | null;
  streaming_pcm: { status: string; firstDigitalPcmMs: number | null; audioElapsedMs: number; elapsedMs: number; amplitude: number; queuedMs: number; interruptionMs: number | null; acousticLatencyMs: null };
};
async function exportTiming(page: Page, info: TestInfo, name: string): Promise<TimingExport> {
  const downloaded = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download this turn timing', exact: true }).click();
  const file = await downloaded; const raw = info.outputPath(`synthetic-${name}.raw.json`); await file.saveAs(raw);
  const report = JSON.parse(await readFile(raw, 'utf8')) as TimingExport;
  await writeFile(info.outputPath(`synthetic-${name}.json`), JSON.stringify({ fixture: FIXTURE, report }, null, 2));
  return report;
}

test('plays synthetic PCM before final text and seals the exact sample-clock completion', async ({ page, live }, info) => {
  const observed = await prepare(page, live); const connection = await submit(page, live); begin(connection);
  let chunks = segment(connection, 0, 0, 2, 0);
  await expect(page.getByText('Speaking', { exact: true })).toBeVisible();
  await expect(timingValue(page, 'First audio signal')).not.toHaveText('—');
  await expect(timingValue(page, 'Answer ready')).toHaveText('—');
  await expect(page.getByText(PREFIX + SUFFIX, { exact: true })).toHaveCount(0);
  const early = await exportTiming(page, info, 'live-early');
  expect(early.first_audio_signal_ms).not.toBeNull(); expect(early.answer_complete_ms).toBeNull();
  expect(early.turn_complete_ms).toBeNull(); expect(early.streaming_pcm.firstDigitalPcmMs).not.toBeNull();

  connection.emit('token', { text: SUFFIX }); chunks = segment(connection, 1, RATE * 2, 4, chunks);
  const networkDone = page.waitForEvent('requestfinished', { predicate: (request) => request.url() === live.url });
  finish(connection, chunks, RATE * 6); await networkDone;
  await expect(page.getByText(PREFIX + SUFFIX, { exact: true })).toBeVisible();
  const delivered = await exportTiming(page, info, 'live-delivered');
  expect(delivered.stream_complete_ms).not.toBeNull(); expect(delivered.turn_complete_ms).toBeNull();
  expect(delivered.streaming_pcm.audioElapsedMs).toBeLessThan(6000);
  await expect(timing(page).locator('.live-timing-heading > span')).toHaveText('completed', { timeout: 15_000 });
  const completed = await exportTiming(page, info, 'live-complete');
  expect(completed.status).toBe('completed'); expect(completed.streaming_pcm.status).toBe('completed');
  expect(completed.streaming_pcm.audioElapsedMs).toBe(6000); expect(completed.streaming_pcm.elapsedMs).toBe(6000);
  expect(completed.streaming_pcm.queuedMs).toBe(0); expect(completed.streaming_pcm.acousticLatencyMs).toBeNull();
  expect(completed.first_audio_signal_ms!).toBeLessThan(completed.answer_complete_ms!);
  expect(completed.stream_complete_ms!).toBeLessThan(completed.turn_complete_ms!);
  await expect.poll(() => audioStates(page)).toEqual(['closed']);
  expect(observed.synthesisRequests).toEqual([]); expect(observed.errors).toEqual([]);
  await timing(page).screenshot({ path: info.outputPath('synthetic-live-complete.png') });
});

test('cancels the live HTTP body and queued audio without claiming a completed turn', async ({ page, live }, info) => {
  const observed = await prepare(page, live); const connection = await submit(page, live); begin(connection);
  segment(connection, 0, 0, 6, 0);
  await expect(page.getByText('Speaking', { exact: true })).toBeVisible();
  await expect(timingValue(page, 'First audio signal')).not.toHaveText('—');
  await page.getByRole('button', { name: 'Stop response', exact: true }).click();
  await expect.poll(() => connection.closedBeforeEnd).toBe(true);
  await expect.poll(() => audioStates(page)).toEqual(['closed']);
  await expect(page.getByRole('textbox', { name: 'Your message', exact: true })).toBeEnabled();
  const report = await exportTiming(page, info, 'live-cancelled');
  expect(report.status).toBe('cancelled'); expect(report.answer_complete_ms).toBeNull();
  expect(report.stream_complete_ms).toBeNull(); expect(report.turn_complete_ms).toBeNull();
  expect(report.streaming_pcm).toMatchObject({ status: 'cancelled', queuedMs: 0, amplitude: 0, acousticLatencyMs: null });
  expect(report.streaming_pcm.interruptionMs).not.toBeNull();
  expect(connection.emit('token', { text: 'Synthetic stale text must not arrive.' })).toBe(false);
  await expect(page.getByText('Synthetic stale text must not arrive.', { exact: true })).toHaveCount(0);
  expect(observed.synthesisRequests).toEqual([]); expect(observed.errors).toEqual([]);
  await timing(page).screenshot({ path: info.outputPath('synthetic-live-cancelled.png') });
});

for (const defect of ['stale-turn', 'sample-boundary'] as const) {
  test(`fails closed on ${defect} PCM before scheduling audio`, async ({ page, live }, info) => {
    const observed = await prepare(page, live); const connection = await submit(page, live); begin(connection);
    const { bytes, identity } = segmentHeader(connection, 0, 0, RATE);
    connection.emit('audio', { ...identity, turn_id: defect === 'stale-turn' ? 'previous-turn' : identity.turn_id,
      segment_sequence: 0, sample_rate: RATE, channels: 1, sample_offset: defect === 'sample-boundary' ? 1 : 0,
      sample_count: RATE, encoding: 'pcm_s16le', pcm_base64: bytes.toString('base64') });
    await expect(page.getByRole('alert').filter({ hasText: 'Invalid live voice stream' }).first()).toBeVisible();
    await expect.poll(() => connection.closedBeforeEnd).toBe(true);
    await expect.poll(() => audioStates(page)).toEqual(['closed']);
    const report = await exportTiming(page, info, `live-${defect}`);
    expect(report.status).toBe('failed'); expect(report.turn_complete_ms).toBeNull();
    expect(report.first_audio_signal_ms).toBeNull(); expect(report.streaming_pcm.firstDigitalPcmMs).toBeNull();
    expect(report.streaming_pcm).toMatchObject({ status: 'failed', audioElapsedMs: 0, amplitude: 0, queuedMs: 0 });
    expect(observed.errors).toEqual([]); expect(observed.synthesisRequests).toEqual([]);
  });
}
