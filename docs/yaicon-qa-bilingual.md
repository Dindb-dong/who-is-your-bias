# Project/current-source Q&A and Korean captions

## Scope and audit

Task branch: `feat/yaicon-qa-context-bilingual-nov21`, based on main
`713232c74c890a61d7f6f6f4b67d5fed984b883c` (2026-11-21 audit).
PR28 was merged; PR27 (`ae724bc8c7e86e809f68b9cc5a0aabf1a5874644`) and
PR29 (`e210b3df1661baf299f5d58fc005d4a20325056c`) were open. No merge is part
of this change. Main and PR27 had Korean input foundations, but coupled
input/reply language and no translated, PCM-aligned Korean captions.

The supplied 28.57-second demonstration shows English questions, English
answers and English captions with a Qwen MLX label. That exact label is absent
from both audited Apps. A local/private runtime may differ from committed main;
the video is not proof of the selected checkpoint or private corpus.

## What is added

- Optional separate QA references in `data/qa`: ten audited project cards and
  two dated official announcements. They are retrieval context, **not training**,
  not Jensen's personal memories, and not persona-evaluation examples.
- Project and news route before the historical-interview speculative fallback.
  References have IDs, source links, revisions/publication dates, review dates
  and expiration. Unmatched/expired facts get a bounded uncertainty response.
  Trading instructions and personal allegations get explicit limitations.
- No automatic live web search. The news digest is limited first-party
  reporting, not exhaustive news or independent corroboration. Missing material
  does not establish that an event is false. Re-review before the demo.
- Independent input/answer/subtitle languages. Sentence translations run
  alongside audio synthesis; PCM sample offsets select the sentence actually
  playing. This is **sentence timing**, not word-level alignment or guaranteed
  simultaneous translation. Late/failed translations show labeled English.
- Cancel/new turn/replay clears captions. Text mode labels translations as an
  unsynchronized transcript. Replay does not promise translated captions.
- Only English source text goes to the existing English voice provider. Its
  Jensen reference/voice binding is unchanged; no new voice cloning here.

## Enable on the demo machine

Update **both** Python and web build. Retain the team's working LLM/TTS/ASR and
private assets; add to the existing launch environment:

```sh
export BIAS_QA_KNOWLEDGE_DIR=/absolute/path/to/who-is-your-bias/data/qa
export BIAS_TRANSLATION_BASE_URL=http://127.0.0.1:8120/v1
export BIAS_TRANSLATION_MODEL=your-non-thinking-translation-model
export BIAS_TRANSLATION_TIMEOUT=12
# BIAS_TRANSLATION_API_KEY is optional; keep it outside Git.
```

The translation address is an example, not an included running service. It uses
streaming Chat Completions like the answer provider. A separate fast endpoint
avoids contention when the local answer model serializes requests. Sharing one
busy model can increase delay despite parallel application scheduling. No
external paid provider is selected silently. Missing configuration advertises
`translated_subtitles: false` and the UI explains what is missing.

Run `npm ci` and `npm run build`, restart the existing API, then open its URL
with `?present=1&bilingual=1`. Select **English voice + Korean captions**. The
question selector controls Korean `ko-KR` recognition; requests use `language:
en`, `input_language: ko`, `subtitle_language: ko`. Configured server ASR or
browser ASR is used; browser support/microphone permission still matters. Typed
Korean works without ASR. Recording is transcribed after stopping, not continuous
speech-to-speech or interruption handling.

The workbench adapter forwards these options and preserves auxiliary QA
instructions. Its prepared runtime may point to another checkout; update and
rebuild that actual checkout too. This change targets `/api/chat/stream` and
`/api/chat/live`. PR27's Sophon websocket does **not** automatically inherit it.

## Initial checkpoint verification

Subtitle unit tests: 32 passed. Existing live/provider/language tests: 74 passed
(overlapping sets). Frontend typecheck, 282 unit tests and production build
passed. Cross-layer tests are being finalized; a test accessor typo was fixed.
Browser execution attempted two new bilingual and seven existing live tests,
but Chromium's process-singleton socket was denied before page initialization.
Browser assertions have **not** passed here. Actual-model checks are pending.
No production Mac ASR/TTS, voice-quality or audience-to-first-audio measurement
is claimed at this checkpoint.

```sh
python -m pytest -q tests/test_qa_knowledge.py tests/test_qa_delivery.py tests/test_subtitles.py tests/test_live_streaming.py tests/test_providers_and_api.py
npm run typecheck
npm run test:web
npm run build
npm run test:e2e -- tests/browser/bilingual-qa.spec.ts tests/browser/live-question.spec.ts tests/browser/live-streaming.spec.ts --project=desktop
```

## Persona result retained for the presentation

PR29's `reports/persona-delivery-2026-11-21/README.md` contains the separate
12-question Qwen3-4B pilot. It did **not** show a net gain in strict value
application. Graph/delivery options remain off by default and unmerged. Do not
present QA references or translated captions as a persona accuracy gain.
