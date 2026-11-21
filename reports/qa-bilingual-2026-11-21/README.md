# QA and bilingual delivery pilot — 2026-11-21

This is a diagnostic content check, not a persona-fidelity experiment, blind
benchmark, audience study or production speech-latency result. Code and data
were saved on PR30 before executing actual inference. The original run used
checkpoint `3cbcd4635102e0e0ccc0f50b52798ff735031be4`.

## Runtime and retained outputs

- Qwen3-4B-Q4_K_M, non-thinking, llama-cpp-python 0.3.36, CPU, six threads,
  8192-token context. Reused the verified local runtime from the prior persona
  pilot; no new training or model download.
- Official model revision: `bc640142c66e1fdd12af0bd68f40445458f3869b`.
  GGUF SHA-256: `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`.
- `actual-model/manifest.json` pins source/data fingerprints and settings.
  Seven Korean typed prompts, English answers; four model-generated answers
  and three deterministic limitations. Nine sentence translations used the
  actual same Qwen model sequentially after the English answer.
- Each answer and translation is saved atomically. JSON contains generated
  outputs for inspection, including errors in wording. They are **not** facts
  to ingest into the runtime corpus.
- No ASR, actual TTS, Jensen-reference voice, webcam, Gaussian or EMAGE execution.
  Sequential CPU translation timing is not the live Mac/API pipeline.

## Findings and resulting fixes

| Case | Observed initial output | Action |
|---|---|---|
| Project pipeline | Described retrieval, generated answers and voice/avatar playback | Retain source-bound generation; not a blanket accuracy claim |
| Why Qwen? | Invented configuration flexibility as the team's selection reason, despite absent documentation | Exact cited limitation now states the reason/comparative superiority is not established |
| MCCCD alignment | Converted missing verification into a categorical claim that aligned learning did not occur | Exact cited response now distinguishes separate preparation from an unverified aligned dataset |
| Current executive announcement | Dated the November 12 company announcement and reported its named roles | Keep dates and company attribution; not private-event evidence |
| Buy/sell stock | Deterministic limitation, no LLM recommendation | Retain |
| Personal allegation | Deterministic English uncertainty; model translation changed rumor into false information | Exact reviewed Korean translation now preserves uncertainty without another model |
| Today's share price | Said current data is unavailable | Retain; no invented price |

The initial pipeline translation rendered *persona evidence* as `인사 증거`.
A technical-term glossary was added. Other initial translations were awkward
(for example, `시간 퍼리오드`); fixed policy messages now use reviewed authored
Korean phrases. General generated answers still depend on translation-model
quality and require rehearsal. Language/number shape checks alone do not
establish semantic fidelity.

Initial four generated answers took 18.9–40.7 seconds on this CPU. Nine model
translations took 6.6–15.3 seconds per sentence. These deliberately sequential
diagnostics do **not** establish live streaming latency. A separate fast
translation endpoint is recommended; same-model CPU timing here is unsuitable
as evidence that Korean captions will be available when speech starts.

## Follow-up checkpoint

Known documentation gaps use cited authored responses. Fixed uncertainty
sentences have exact authored Korean translations; unknown sentences still use
the configured model. Queue work is bounded, caption completion after audio
delivery is bounded, and wrong-language speech is rejected in bilingual mode.
This can show a labeled English fallback rather than delay audio for a subtitle.
The last 250 ms grace after PCM delivery may omit translations even while the
browser still has audio queued; it is an explicit latency/coverage tradeoff.

A targeted rerun of the project wording, fixed documentation limits and fixed
uncertainty translations is pending at this checkpoint. Preserve the original
outputs; do not overwrite them with improved examples or report an accuracy
percentage from these deliberately selected regression cases.
