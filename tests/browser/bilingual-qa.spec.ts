import { expect, test, type Page } from '@playwright/test';

// Synthetic text/translation and PCM: validates routing, timing and cancellation,
// not Korean ASR accuracy, translation quality, voice likeness or model latency.
const QUESTION = '이 프로젝트에서 음성은 어떻게 생성하나요?';
const ENGLISH = ['We use a configured speech model.', 'Voice quality still needs listening tests.'];
const KOREAN = '설정된 음성 모델을 사용합니다.';
const RATE = 16000;
const frame = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
function stream(translationAvailable: boolean) {
  const turnId = 'bilingual-test';
  let output = frame('meta', { turn_id: turnId, session_id: 'bilingual-session', protocol: 'live-segments-pcm.v1' });
  output += frame('token', { text: ENGLISH.join(' ') });
  for (let sequence = 0; sequence < 2; sequence++) {
    const id = { turn_id: turnId, segment_id: `${turnId}:${sequence}`, sequence };
    output += frame('segment', { ...id, text: ENGLISH[sequence], evidence_ids: [], state: 'committed' });
    const translated = translationAvailable && sequence === 0;
    output += frame('subtitle', { ...id, language: 'ko', text: translated ? KOREAN : '',
      state: translated ? 'ready' : 'unavailable', ...(translated ? {} : { reason: 'synthetic-timeout' }) });
    output += frame('audio_segment', { ...id, audio_sha256: 'a'.repeat(64), sample_rate: RATE, channels: 1,
      sample_offset: sequence * RATE * 2, sample_count: RATE * 2, synthesis_ms: 0 });
    for (let second = 0; second < 2; second++) {
      const bytes = Buffer.alloc(RATE * 2);
      for (let i = 0; i < RATE; i++) bytes.writeInt16LE(Math.round(Math.sin(i / RATE * 2 * Math.PI * 220) * 4000), i * 2);
      output += frame('audio', { ...id, sequence: sequence * 2 + second, segment_sequence: sequence,
        sample_rate: RATE, channels: 1, sample_offset: (sequence * 2 + second) * RATE, sample_count: RATE,
        encoding: 'pcm_s16le', pcm_base64: bytes.toString('base64') });
    }
    output += frame('audio_segment_end', { ...id, sample_end: (sequence + 1) * RATE * 2, state: 'delivered' });
  }
  output += frame('result', { turn: { id: turnId, session_id: 'bilingual-session', text: ENGLISH.join(' '),
    language: 'en', mode: 'behavior', provider: 'openai', citations: [],
    behavior: { duration_ms: 4000, cues: [], speech: { rate: 1, pitch: 1, pause_ms: 0 } },
    grounding: { status: 'unverified', retrieved_ids: [], invalid_citation_ids: [] },
    metrics: { retrieval_ms: 0, generation_ms: 0, total_ms: 0 } } });
  return output + frame('audio_end', { turn_id: turnId, chunks: 4, sample_count: RATE * 4, sample_rate: RATE,
    first_pcm_ready_ms: 0, total_ms: 0, state: 'delivered', spoken_complete: false, acoustic_latency_ms: null }) + frame('done', {});
}
async function prepare(page: Page, translationAvailable = true, url = '/?present=1&bilingual=1') {
  const requests: Record<string, unknown>[] = [];
  await page.route('**/api/persona', (route) => route.fulfill({ json: {
    id: 'jensen-huang', name: 'Synthetic bilingual fixture', tagline: '', description: '',
    example_questions: { en: [], ko: [] }, stats: { sources: 0, evidence: 0, held_out_sources: 0 },
    capabilities: { llm: true, speech: true, live_speech: true, transcription: false, translated_subtitles: translationAvailable, qa_knowledge: true, realtime_call: true }, notice: 'Synthetic protocol test.',
    bindings: { voice: { status: 'matched', persona_id: 'jensen-huang' } },
  } }));
  await page.route('**/api/sources', (route) => route.fulfill({ json: { sources: [] } }));
  await page.route('**/api/motion/library', (route) => route.fulfill({ json: { available: false, status: 'not_configured', reviewed_runtime_primitives: 0, preview_runtime_primitives: 0 } }));
  await page.route('**/api/chat/live', (route) => { requests.push(route.request().postDataJSON()); return route.fulfill({ contentType: 'text/event-stream', body: stream(translationAvailable) }); });
  await page.goto(url);
  await expect(page.getByText('Studio connected', { exact: true })).toBeVisible();
  return requests;
}
test('routes a Korean typed question to English speech with sentence-timed Korean captions and explicit fallback', async ({ page }) => {
  const requests = await prepare(page);
  await expect(page.getByLabel('Question language')).toHaveValue('ko');
  await page.getByLabel('Your message', { exact: true }).fill(QUESTION);
  await page.getByRole('button', { name: 'Send message', exact: true }).click();
  const caption = page.getByLabel('Korean answer captions');
  await expect(caption).toContainText(KOREAN);
  await expect(caption).toHaveAttribute('lang', 'ko');
  await expect(caption).not.toContainText(ENGLISH[1]);
  expect(requests).toHaveLength(1);
  expect(requests[0]).toMatchObject({ message: QUESTION, input_language: 'ko', language: 'en', subtitle_language: 'ko' });
  await expect(caption).toContainText(ENGLISH[1]);
  await expect(caption).toContainText('translation unavailable');
  await expect(caption).toHaveAttribute('lang', 'en');
  await page.getByRole('button', { name: 'Stop audio', exact: true }).click();
  await expect(caption).toHaveCount(0);
});
test('dictates Korean using ko-KR without switching the English answer language', async ({ page }) => {
  await page.addInitScript(({ text }) => {
    class SpeechRecognitionFixture {
      lang = '';
      onresult?: (event: { resultIndex: number; results: { isFinal: boolean; 0: { transcript: string } }[] }) => void;
      onend?: () => void;
      start() { Object.assign(window, { __capturedRecognitionLanguage: this.lang }); }
      stop() { this.onresult?.({ resultIndex: 0, results: [{ isFinal: true, 0: { transcript: text } }] }); this.onend?.(); }
      abort() {}
    }
    Object.assign(window, { SpeechRecognition: SpeechRecognitionFixture });
  }, { text: QUESTION });
  const requests = await prepare(page);
  await page.getByRole('button', { name: 'Use microphone', exact: true }).click();
  await page.getByRole('button', { name: 'Finish and send question', exact: true }).click();
  await expect(page.getByLabel('Korean answer captions')).toContainText(KOREAN);
  expect(await page.evaluate(() => (window as unknown as { __capturedRecognitionLanguage: string }).__capturedRecognitionLanguage)).toBe('ko-KR');
  expect(requests[0]).toMatchObject({ message: QUESTION, language: 'en', input_language: 'ko', subtitle_language: 'ko' });
});

test('keeps Jensen presentation answers and playback in English when Korean translation is unconfigured', async ({ page }) => {
  const requests = await prepare(page, false, '/?present=1');
  await expect(page.getByRole('button', { name: 'Start live call' })).toHaveCount(0);
  await page.getByLabel('Question language').selectOption('ko');
  await page.getByLabel('Your message', { exact: true }).fill(QUESTION);
  await page.getByRole('button', { name: 'Send message', exact: true }).click();

  await expect(page.getByLabel('Korean answer captions')).toContainText(ENGLISH[0]);
  await expect(page.getByLabel('Korean answer captions')).toHaveAttribute('lang', 'en');
  expect(requests).toHaveLength(1);
  expect(requests[0]).toMatchObject({
    message: QUESTION, input_language: 'ko', language: 'en', subtitle_language: 'ko', presentation_qa: true,
  });
});
