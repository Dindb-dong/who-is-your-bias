import { createHash } from 'node:crypto';
import { expect, test, type Page } from '@playwright/test';
import type { Turn } from '../../apps/web/src/api';

const QUESTION = 'Why do AI factories need fast networking?';
const CAPTION = 'Fast networking lets accelerators share work efficiently.';
const ANSWER = CAPTION + ' [source:fixture]';

function wavFixture() {
  const rate = 16_000;
  const audio = Buffer.alloc(44 + rate * 2);
  audio.write('RIFF', 0); audio.writeUInt32LE(audio.length - 8, 4); audio.write('WAVEfmt ', 8);
  audio.writeUInt32LE(16, 16); audio.writeUInt16LE(1, 20); audio.writeUInt16LE(1, 22);
  audio.writeUInt32LE(rate, 24); audio.writeUInt32LE(rate * 2, 28);
  audio.writeUInt16LE(2, 32); audio.writeUInt16LE(16, 34);
  audio.write('data', 36); audio.writeUInt32LE(rate * 2, 40);
  for (let index = 0; index < rate; ++index) audio.writeInt16LE(Math.round(5000 * Math.sin(index * Math.PI * 2 * 440 / rate)), 44 + index * 2);
  return audio;
}

async function fixture(page: Page, transcription: boolean) {
  const questions: string[] = [];
  const recordings: Buffer[] = [];
  const audio = wavFixture();
  const audioUrl = '/api/speech/clips/' + 'a'.repeat(32);
  await page.route('**/api/persona', (route) => route.fulfill({ json: {
    id: 'synthetic-live-question', name: 'Live question test', tagline: '', description: '', notice: '',
    example_questions: { en: [], ko: [] }, stats: { sources: 0, evidence: 0, held_out_sources: 0 },
    capabilities: { llm: true, speech: true, live_speech: false, transcription },
  } }));
  await page.route('**/api/sources', (route) => route.fulfill({ json: { sources: [] } }));
  await page.route('**/api/motion/library', (route) => route.fulfill({ json: {
    available: false, status: 'not_configured', reviewed_runtime_primitives: 0, preview_runtime_primitives: 0,
  } }));
  await page.route('**/api/transcribe?language=en', (route) => {
    expect(route.request().headers()['content-type']).toMatch(/^audio\/webm/);
    recordings.push(route.request().postDataBuffer()!);
    return route.fulfill({ json: { text: QUESTION } });
  });
  const turn: Turn = {
    id: 'live-question-turn', session_id: 'live-question', language: 'en', mode: 'behavior',
    text: ANSWER, provider: 'openai', citations: [],
    behavior: { duration_ms: 1000, cues: [], speech: { rate: 1, pitch: 1, pause_ms: 0 } },
    grounding: { status: 'unverified', retrieved_ids: [], invalid_citation_ids: [] },
    metrics: { retrieval_ms: 0, generation_ms: 0, total_ms: 0 },
  };
  await page.route('**/api/chat/stream', (route) => {
    questions.push(route.request().postDataJSON().message);
    const stream = [['meta', { session_id: turn.session_id }], ['token', { text: turn.text }], ['result', { turn }], ['done', {}]]
      .map(([event, data]) => 'event: ' + event + '\ndata: ' + JSON.stringify(data) + '\n\n').join('');
    return route.fulfill({ body: stream, contentType: 'text/event-stream' });
  });
  await page.route('**/api/speech/prepare', (route) => {
    expect(route.request().postDataJSON().text).toBe(CAPTION);
    return route.fulfill({ json: {
      audio_url: audioUrl, audio_sha256: createHash('sha256').update(audio).digest('hex'), timing_status: 'unavailable',
    } });
  });
  await page.route('**' + audioUrl, (route) => route.fulfill({ body: audio, contentType: 'audio/wav' }));
  return { questions, recordings };
}

test('presentation dictates one final question and automatically plays and captions the answer', async ({ page }) => {
  const { questions, recordings } = await fixture(page, false);
  await page.addInitScript(({ text }) => {
    class SpeechRecognitionFixture {
      onresult?: (event: { results: { isFinal: boolean; 0: { transcript: string } }[] }) => void;
      onend?: () => void;
      start() { this.onresult?.({ results: [{ isFinal: false, 0: { transcript: 'Partial question' } }] }); }
      stop() {
        this.onresult?.({ results: [{ isFinal: true, 0: { transcript: text } }] });
        this.onend?.();
      }
      abort() {}
    }
    Object.assign(window, { SpeechRecognition: SpeechRecognitionFixture });
  }, { text: QUESTION });
  await page.goto('/?present=1');
  await page.getByRole('button', { name: 'Use microphone', exact: true }).click();
  await expect(page.getByRole('textbox', { name: 'Your message' })).toHaveValue('Partial question');
  expect(questions).toEqual([]);
  await page.getByRole('button', { name: 'Finish and send question', exact: true }).click();
  await expect(page.getByLabel('Answer captions')).toHaveText(CAPTION);
  await expect(page.getByRole('log').getByText(QUESTION)).toBeVisible();
  await expect(page.getByLabel('Live response timing')).toContainText('Transcription');
  await expect(page.getByLabel('Live response timing')).toContainText('the end of microphone capture');
  await expect(page.locator('.live-timing-heading')).toContainText('completed');
  expect(questions).toEqual([QUESTION]);
  expect(recordings).toEqual([]);
});

test('presentation records to the configured transcription API before sending a question', async ({ page }) => {
  const { questions, recordings } = await fixture(page, true);
  await page.addInitScript(() => {
    Object.defineProperty(navigator.mediaDevices, 'getUserMedia', {
      value: async () => ({ getTracks: () => [{ stop: () => undefined }] }),
    });
    class RecorderFixture {
      static isTypeSupported(type: string) { return type.startsWith('audio/webm'); }
      readonly mimeType = 'audio/webm';
      state = 'inactive';
      ondataavailable?: (event: { data: Blob }) => void;
      onstop?: () => void;
      start() { this.state = 'recording'; }
      stop() {
        this.state = 'inactive';
        this.ondataavailable?.({ data: new Blob(['synthetic recording'], { type: this.mimeType }) });
        this.onstop?.();
      }
    }
    Object.assign(window, { MediaRecorder: RecorderFixture });
  });
  await page.goto('/?present=1');
  await page.getByRole('button', { name: 'Use microphone', exact: true }).click();
  await page.getByRole('button', { name: 'Finish and send question', exact: true }).click();
  await expect(page.getByLabel('Answer captions')).toHaveText(CAPTION);
  await expect(page.locator('.live-timing-heading')).toContainText('completed');
  expect(recordings.map((item) => item.toString())).toEqual(['synthetic recording']);
  expect(questions).toEqual([QUESTION]);
});

test('presentation review keeps the transcript editable until the user submits it', async ({ page }) => {
  const { questions } = await fixture(page, false);
  await page.addInitScript(({ question }) => {
    class RecognitionFixture {
      onstart?: () => void;
      onresult?: (event: unknown) => void;
      onend?: () => void;
      start() { this.onstart?.(); }
      stop() {
        this.onresult?.({ resultIndex: 0, results: [Object.assign([{ transcript: question }], { isFinal: true })] });
        this.onend?.();
      }
      abort() { this.onend?.(); }
    }
    Object.assign(window, { SpeechRecognition: RecognitionFixture });
  }, { question: QUESTION });
  await page.goto('/?present=1&reviewQuestion=1');
  await page.getByRole('button', { name: 'Use microphone', exact: true }).click();
  await page.getByRole('button', { name: 'Stop microphone', exact: true }).click();
  const composer = page.getByRole('textbox', { name: 'Your message' });
  await expect(composer).toHaveValue(QUESTION);
  expect(questions).toEqual([]);
  await composer.fill('Why does a fast interconnect help?');
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect(page.getByLabel('Answer captions')).toHaveText(CAPTION);
  expect(questions).toEqual(['Why does a fast interconnect help?']);
});
