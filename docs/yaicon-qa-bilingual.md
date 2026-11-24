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
- Known documentation gaps (model-selection rationale and MCCCD alignment)
  have exact cited replies. Fixed uncertainty/financial/allegation sentences
  have reviewed Korean translations, avoiding another model changing their
  meaning. General generated answers still need translation-quality review.

## Enable on the demo machine

Update **both** Python and web build. Retain the team's working LLM/TTS/ASR and
private assets; add to the existing launch environment:

```sh
export BIAS_QA_KNOWLEDGE_DIR=/absolute/path/to/who-is-your-bias/data/qa
export BIAS_TRANSLATION_PROTOCOL=google-web
```

The Google choices send only committed English answer sentences for translation
to Korean. The default response deadline is 2.5 seconds. Known reviewed phrases
use local translations first. When a subtitle is late or the service is
unavailable, the presenter can continue with English audio and English captions.

| `BIAS_TRANSLATION_PROTOCOL` | Setup |
| --- | --- |
| `google-web` | Orbit-compatible keyless web endpoint. Rehearse on the venue network because the web interface can change or rate-limit requests. |
| `google-cloud` | Official [Cloud Translation Basic v2](https://docs.cloud.google.com/translate/docs/reference/rest/v2/translate). Enable the API and billing and set `BIAS_TRANSLATION_API_KEY` in the server environment. |
| `openai-compatible` (default) | Retains `BIAS_TRANSLATION_BASE_URL`, `BIAS_TRANSLATION_MODEL` and the optional `BIAS_TRANSLATION_API_KEY`. A separate non-thinking Chat Completions backend keeps translation away from the answer model's queue. Its default deadline is 12 seconds. |

Keep provider keys outside Git. `BIAS_TRANSLATION_TIMEOUT` overrides the selected
provider's deadline. Before rehearsal, check `/api/persona` for
`capabilities.qa_knowledge` and `capabilities.translated_subtitles` and confirm
the selected backend with an actual audience question.

News review expires within 48 hours. The seeded review expires on November 23,
2026 at 23:40 KST; refresh it before using it as current context at a later demo.
Use `python tools/qa_news.py validate data/qa` to inspect freshness. The RSS
candidate command never approves material automatically. Read the source,
review the summary/dates, update the digest and restart the API after changes.
Update project cards whenever PR status or the deployed implementation changes;
their pinned old revision must not be described as the new deployment.

Run `npm ci` and `npm run build`, restart the API, then open its URL with
`?present=1&reviewQuestion=1`. This starts in English input. Add `&bilingual=1`
to start in Korean or `&bilingual=0` to disable Korean captions while keeping
English answers and voice. Select the audience's language before recording.
After stopping the mic, the operator can review and edit the transcript, then
submit it. Requests use `language: en` and pass the selected `input_language`
separately. Korean-caption requests also include `subtitle_language: ko`.
Configured server ASR or browser ASR still needs microphone permission. Typed
questions use the same answer path.

For a full-screen slide show, open **Avatar picture-in-picture** from the Studio
before switching to the slides. Presentation mode requests a 760 × 380 window
with a full-width caption strip above the avatar. The browser can clamp the
window dimensions and controls its placement. Move the entire window to the
audience display and confirm it stays above that display's slide show during
rehearsal. The caption strip and the avatar move together.

The workbench adapter forwards these options and preserves auxiliary QA
instructions. Its prepared runtime may point to another checkout; update and
rebuild that actual checkout too. This change targets `/api/chat/stream` and
`/api/chat/live`. PR27's Sophon websocket does **not** automatically inherit it.

## Verification and handoff

Implementation is saved in [PR30](https://github.com/Dindb-dong/who-is-your-bias/pull/30).
Checkpoint `840cf864b701a12379a80ad9607cce97f5da1b38` was read back before the
targeted rerun. No main/PR27/PR29 merge occurred.

159 focused Python tests, seven workbench tests, 282 web unit tests, typecheck,
production build, corpus validation and QA-reference schema/freshness checks
passed. Browser execution attempted two new bilingual and seven existing live
tests, but Chromium's process-singleton socket was denied before page
initialization; browser assertions remain unverified here. Another browser
regression test was added for immediate PCM finalization and was discovered,
not executed successfully.

The [actual-model report](../reports/qa-bilingual-2026-11-21/README.md) preserves
seven initial cases and five targeted follow-ups, including the errors that
motivated fixes. This does not establish general answer accuracy, translation
fluency, real Mac latency, Korean ASR accuracy or Jensen voice quality.
Pending acceptance: configure the translation provider, review fresh news,
and rehearse the exact demo checkout with its private voice/avatar assets.

PCM playback finalizes as soon as audio delivery ends. Subtitle work receives
only 250 ms extra grace after that event, then becomes explicitly unavailable.
This bounds completion delay but can drop a late subtitle while audio is still
queued in the browser. Keep this quality/latency tradeoff in the rehearsal.

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
