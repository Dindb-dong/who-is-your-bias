# Optional conditional values and public communication style

This PR separates **what decision criterion to apply**, **how to respond to the
situation**, **how to phrase the answer**, and **how the audio sounds**. These are
different targets and require different evidence and measurements.

| Component | Implementation | Evidence and limit |
| --- | --- | --- |
| Conditional values | Compile one relevant v2 graph path into compact guidance, retaining claims, qualifications and exceptions | Existing 60-record source corpus and 12 strategies; lexical routing can miss paraphrases |
| Situational response | Select at most one reaction observation for feedback, uncertainty or disagreement | Public transcript observations; no claim about private feelings |
| Speaking style | Select at most one explanatory or invitation-only humor observation | Five observations from three events overall; no verbatim dialogue examples or trained style model |
| Voice identity/prosody | Keep the bound TTS reference and voice service separate | This PR does not train TTS or establish an acoustic improvement |

The compact format is an inference-time compilation of an editorial graph. It
does not train graph embeddings, infer a complete psychological profile or
reproduce ThinkPersona's training procedure. Conditions and exceptions are
checked by the answer model, not established by lexical matching. Source and
evidence hashes are checked before selection. The full graph remains available
for audit while the prompt contains a short selected portion.

Delivery observations never become factual citation authority. Their source
records are audited separately, and the prompt distinguishes communication
guidance from factual evidence. A historical joke does not justify inventing a
current scandal, feeling, endorsement or autobiographical story. Explicit humor
opt-outs and sensitive contexts override the humor invitation. These are
bounded lexical safeguards, not an exhaustive semantic safety classifier.

## Opt-in configuration

The shared team's default paths and voice/avatar services are unchanged. To
test the new layer with the source corpus bound to it, set these variables for
the Python API process, alongside the team's existing LLM configuration:

```bash
export BIAS_CORPUS_DIR=data/personas/jensen-huang-values-v2
export BIAS_PERSONA_SNAPSHOT=data/snapshots/jensen-values-2026-11-21.json
export BIAS_PERSONA_GRAPH=data/personas/jensen-huang-values-v2/persona_graph.v2.json
export BIAS_PERSONA_MEMORY_FORMAT=compact
export BIAS_PERSONA_DELIVERY=data/personas/jensen-huang-values-v2/persona_delivery.v1.json
export BIAS_LIVE_CONCISE=true
```

### Live-route boundary

This integration is in `bias.engine.Engine`, the Python RAG/text-generation
route. PR #27 at `ae724bc8c7e86e809f68b9cc5a0aabf1a5874644` has a separate
`bias.sophon` realtime route: it sends session instructions to Sophon's own
model and does not run the citation-checked text-conversation engine. Enabling
the variables above therefore does **not** automatically add per-question
persona selection to Sophon. PR #27's speech/gesture path is not modified here.

If that route becomes the final demo backend, integrate the selectors at a
verified per-turn text/transcript boundary, before answer generation, with the
same source/cutoff and guard checks. A static session prompt containing every
graph node is not equivalent to the conditional selection evaluated here.
Neither this document nor the CPU pilot establishes live Sophon integration.

`compact` requires the v2 conditional graph. The existing `flat` and `graph`
formats remain available. Unset `BIAS_PERSONA_DELIVERY` to remove the delivery
layer; unset the graph variable to remove value memory. Select the matching
corpus/snapshot when reverting to the team's prior data configuration.

The engine reserves factual evidence first, then fits complete value paths and
complete delivery observations into the declared context budget. It drops
whole units if they do not fit and records selected/included/dropped IDs in the
trace. It does not cut qualifications, create an extra LLM rewrite request or
silently convert dropped source IDs into allowed citations. `BIAS_LIVE_CONCISE`
applies only to the existing live segmented path; the frozen evaluation applies
the same production short-answer instruction to every condition explicitly.

## Evaluation

See [the execution record](persona-delivery-execution.md), the frozen
[registry](../reports/persona-delivery-2026-11-21/registry.json), and generated
artifacts in `reports/persona-delivery-2026-11-21/`. The three-condition comparison
holds factual evidence constant. Its matched-evidence comparator already
benefits from graph-selected source retrieval, so this does not measure the
total benefit versus unaugmented RAG. The judge rubric measures answer
relevance, fidelity, concrete application, situational response and wording.
It does not count matching a value keyword as success.

Each condition uses a pinned Qwen3-4B Q4_K_M CPU runtime, no thinking, 256 maximum
output tokens, deterministic decoding and no conversation history. Failure or
truncation stays in the denominator. Independent AI ratings are exploratory;
they are not human validation, proof of psychological similarity, or end-to-end
Mac/MLX voice latency measurements. Earlier negative v2 results remain intact.

## Voice follow-up without conflating it with persona text

Reuse the existing `tools.voice_evaluation` harness and verified Jensen-bound
reference configuration. With the same fixed English text in each condition,
compare at least two clean reference clips/transcriptions on the actual target
TTS runtime. Do not change text, model, reference and chunk size together.

- First measure the time to the first playable audio separately from complete
  synthesis, audio duration and real-time factor. The current batch synthesis
  harness records complete-request timing/RTF; it does not by itself establish
  end-to-end first-playback latency.
- Check intelligibility using an independent ASR pass and listening; ASR WER is
  not a speaker-similarity score.
- Use the existing blinded listening bundle with a reference recording to rate
  reference similarity and naturalness. For prosody, additionally listen for
  appropriate emphasis, pause placement and phrase rhythm; these are not
  inferred from the source transcript or from text-persona ratings.
- Keep the chosen reference identity across streamed chunks. Compare chunk
  boundary discontinuities under the same audio/text conditions before changing
  the live pipeline. Do not send unsupported emotion instructions to a TTS
  backend that does not expose that control.

Example command shape (requires an already configured compatible team voice
service; not executed as part of this text experiment):

```bash
python -m tools.voice_evaluation synthesize --base-url http://127.0.0.1:8092 --cases tools/voice_evaluation/cases.json --condition jensen-reference-a --output reports/local/jensen-reference-a
```

Historical generated voice samples in the repository are not evidence that this
new text layer improves Jensen's voice. No such improvement is claimed here.
