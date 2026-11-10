# Generated-answer evaluation

`tools.evaluation.run` executes the repository's actual `Engine` and `OpenAICompatibleLLM` adapter under three existing conditions. It records model output, citation formatting, retrieval gates, source boundaries, transport failures, measured latency, and provider-reported token usage. It leaves semantic support and persona resemblance for a separate reviewer.

| Condition | Mode | Actual application behavior |
| --- | --- | --- |
| A | `profile` | Public profile and editorial style guidance, without retrieved evidence. |
| B | `rag` | Profile plus reviewed train evidence selected by the normal retriever. |
| C | `behavior` | The RAG path with the existing situation-aware retrieval/prompt condition and designed animation plan. |

C does not train or measure Musk's gestures. All current corpus behavior observations remain null, and the generated animation cues are designed. The evaluator records the resulting `ChatTurn` so that these labels remain visible.

## Run it

First configure an actual model provider as described in [language-runtime.md](language-runtime.md). The runner uses the same `BIAS_LLM_BASE_URL`, `BIAS_LLM_MODEL`, `BIAS_LLM_API_KEY`, `BIAS_LLM_MAX_TOKENS`, and timeout settings as the application. Credentials are sent only in the provider request and are excluded from the report.

Check the declaration and corpus without inference:

```bash
uv run python -m tools.evaluation.run --validate-only
```

Run a small initial A/B/C comparison against the configured service:

```bash
BIAS_LLM_BASE_URL=http://127.0.0.1:8020/v1 \
BIAS_LLM_MODEL=qwen3-local \
BIAS_LLM_MAX_TOKENS=160 \
BIAS_REQUEST_TIMEOUT=150 \
uv run python -m tools.evaluation.run \
  --case g01 \
  --output reports/local/generation-g01.json
```

Run all declared cases, or select conditions explicitly:

```bash
uv run python -m tools.evaluation.run \
  --modes profile rag behavior \
  --output reports/local/generation-all.json

uv run python -m tools.evaluation.run \
  --case g02 --case g05 --modes rag behavior \
  --output reports/local/generation-ko-and-heldout.json
```

Eight bundled cases cover historical product strategy, critical feedback, uncertain goals, deadlines, a withheld source detail, an unsupported private experience, an injected endorsement instruction, and a Korean question about childhood influences. The declaration is [cases.json](../tools/evaluation/cases.json). It is separate from execution and from the older retrieval-only fixtures.

These questions are informed by the curated corpus and are explicitly development cases. They are not independently blinded test questions. The runner validates declarations before making provider requests and never silently rewrites a question to make its score improve. A separate [retrieval-only smoke evaluator](../src/bias/evaluate.py) remains available through `uv run bias evaluate`; its success is not a generated-answer result.

With no configured provider, execution is refused by default. `--allow-preview` explicitly enables the main application's existing evidence preview behavior, and each row records that no language model was requested. That option is useful for wiring checks; it is not model evaluation.

The runner processes requests sequentially so a single-worker local model does not receive an A/B/C burst. Every condition gets a new `Engine` and session. Condition order rotates across cases: A/B/C, then B/C/A, then C/A/B. `--repetitions` accepts one through five repetitions. Repeated fixed-seed runs are not automatically independent statistical samples. Server warm-up, package versions, hardware, and current load still affect latency.

The complete matrix has a 600-second deadline by default, configurable with `--max-total-seconds` up to 3,600. If the deadline interrupts a request, its partial text and an `evaluation_deadline` error are preserved. Remaining conditions are explicitly listed as skipped in `execution_scope`; they are not silently counted as passes or abstentions. A deadline-limited report is a partial evaluation, and the command exits nonzero.

## Data and source boundaries

Each case declares its question, language, category, expected action, reviewed retrieval targets, evaluator-only reference IDs, forbidden source IDs, and a manual review note. The schema rejects undeclared fields such as an accidentally supplied `golden_answer`. Only the case question, language, and mode become the application's `ChatRequest`.

Positive retrieval targets must be reviewed, source-grounded train records. Reference evidence must belong to a withheld source. Unknown identifiers and forbidden sources still marked train are errors. To change a source-disjoint experiment, create a separate corpus with reviewed event-family split assignments; do not rely on a case annotation to exclude an otherwise indexable train event.

The normal `Corpus` checks remain active: event families and canonical source URLs cannot cross splits. Immediately before the provider call, the observing adapter parses the actual research-context message, records its evidence identifiers, and refuses an unreviewed or withheld identifier. It delegates generation to the existing adapter; it does not synthesize a replacement response, append citations, alter the prompt, or insert reference answers.

The report records a canonical corpus hash, declaration hash, per-request prompt hash, and the evidence IDs actually sent. Hashes identify the input snapshot; they are not correctness scores. Withheld reference IDs and review notes appear only in evaluator output after the run. Application-level source isolation cannot prove that a pretrained public model never encountered the same public interview during its original training.

## Read the report correctly

The final `.json` report contains configuration, corpus quality information, condition summaries, and full per-run records. A companion `.jsonl` checkpoint is flushed after each completed run so cancellation or a later failure does not erase earlier measurements. Existing output/checkpoint paths are not overwritten; select a new report name. A Ctrl-C cancellation returns code 130 and reports the completed-run checkpoint. The currently interrupted generation is not represented as a completed response.

### Execution and errors

`execution.completed` means the actual engine emitted a final `ChatTurn`. `language_model_requested` records whether the engine selected the provider path; an authentication or connection error does not prove that a model executed. HTTP request counts/statuses, provider error codes, and ordered engine event names are recorded separately.

Partial generated text is retained when a provider times out, reaches its token limit, or fails midstream. Such a row has no completed turn. Provider error details use the main adapter's public messages rather than reflecting provider error bodies or credentials. Successful evidence-mode fallback is identifiable and is never counted as a language-model request.

The CLI returns nonzero for execution errors or a detected prompt split leak. Exit zero does not mean the model's claims, persona, or citations passed all quality checks; inspect the separate fields. The tool does not combine these different measurements into one success percentage.

### Citation format

The report checks exact `[source:RECORD_ID]` syntax against the retrieved IDs for that run. It records valid IDs, invented IDs, malformed markers, and a status:

- `valid`: at least one allowed marker and no invalid/malformed markers.
- `missing`: citations were required for a model response using retrieved evidence, but none appeared.
- `invalid`: a marker was malformed or named a record outside the retrieved set.
- `not_required`: no citation was required for this condition/path and none appeared.
- `not_evaluated`: the response did not complete. Any markers in partial text are retained for diagnosis and excluded from completed-answer citation counts.

A known identifier can still be attached to a false or unsupported sentence. `citation_format.semantic_entailment` therefore always starts as `ungraded`. The application's own `grounding.status="grounded"` likewise validates identifiers; its reason explicitly says factual entailment still requires review. Profile-only output cannot turn a source identifier into a valid citation because it was not supplied retrieved context.

An evidence preview may contain valid markers, but the summary's required citation-format checks cover completed generated responses that actually used retrieval. Provider failures with no output cannot become a model citation-compliance score. Markers in incomplete text remain available as a separate diagnostic.

### Retrieval and abstention

For an answerable case, `retrieval_check` asks whether the normal top-k result contains any declared acceptable evidence/source target. It does not grade the generated answer. In A, retrieval is `not_applicable`.

For an expected abstention, the retrieval check records whether the application returned no evidence. In the current engine, B/C bypass the language model when no acceptable evidence is retrieved. A correct evidence-mode abstention therefore tests the retrieval/application gate, not the model's own ability to decline an unsupported request. A may still generate a profile-only response to the same question, which requires review.

Injected instructions remain manually graded. Retrieving a correct Tesla record does not prove that the model refused an invented endorsement in the rest of its answer.

### Latency, tokens, and memory

`latency.wall_ms` measures the full per-case engine call. `first_visible_text_ms` measures the first engine token event; for evidence preview it is preview text, and for a model it is decoded text arriving through the actual stream. The turn also retains the application's retrieval, generation, and total timings. None is advertised as the first internal decoder-token time.

The observing HTTP transport captures usage only when the provider supplies nonnegative integer prompt/completion/total counts with consistent totals. Missing counts stay null. Invalid usage is ignored and labeled. Decoded character counts and visible chunk counts are separate fields and are never converted into guessed token totals. A normal environment-aware HTTP client carries the actual request; test-only injected transports are explicitly labeled `test_injected` in reports.

Condition summaries include observed median/minimum/maximum wall time and how many runs supplied token counts. A one-case run is a small execution sample, not a throughput benchmark or a latency distribution estimate. The language server's health response separately exposes process peak RSS on supported platforms. Collect it before stopping the model process; it does not establish simultaneous residency of the LLM, TTS, ASR, and vision models.

### Semantic review

Every run includes an ungraded review section with the case's review note and withheld reference IDs where applicable. A reviewer should examine the actual response alongside its cited train passages and separately assess:

1. Whether each factual statement is supported by its cited passage and keeps the original date/context.
2. Whether the answer addresses the question and preserves uncertainty.
3. Whether an unsupported request was appropriately declined without inventing private experience.
4. Whether Korean wording is accurate and natural when requested.
5. Whether an adversarial instruction was followed.
6. Whether any claimed persona resemblance has evidence beyond a plausible writing style.

The tool does not assign a synthetic persona score, voice score, gesture score, or automatic semantic pass. `persona_score` remains null and completed semantic review counts remain zero until a separate review process supplies real judgments. The current runner does not ingest or aggregate those judgments.

## Verification and observed limitations

Protocol tests exercise the real application engine/provider with explicitly fake HTTP responses. They verify A/B/C prompt differences, fresh-session isolation, withheld reference exclusion, errors and stream closure, valid/missing/fabricated citations, Unicode-split usage parsing, unknown-field rejection, and the separation of text chunks from token usage. They do not download or execute a model.

```bash
uv run pytest -q tests/language tests/evaluation
```

The first live HTTP integration run completed an ordinary model-generated main-app SSE answer and correctly labeled its missing citations. It also found that appending a long format instruction could weaken lexical retrieval enough to trigger the no-evidence gate; that attempted second model turn was not misreported as inference. The original failed smoke report is retained, and [language-runtime.md](language-runtime.md) records those measured limits.

## Measured eight-case run

On **2026-11-10 UTC**, the corrected offline runtime completed the declared eight-case A/B/C matrix using real `Qwen/Qwen3-0.6B` weights at revision `c1899de289a04d12100db370d81485cdf75e47ca`. The run used CPU bfloat16, four threads, a 160-token per-request cap, seed 42, and the normal non-thinking template. Optional-runtime telemetry was disabled before process startup and imports. No fake model or canned provider response participated.

The environment used Python 3.12.14, PyTorch 2.11.0+cpu, Transformers 4.57.3, huggingface-hub 0.36.2, accelerate 1.12.0, and safetensors 0.8.0. A direct JSON warm-up loaded the cached snapshot before the matrix; the matrix's timings therefore exclude cold model loading. The overall experiment stayed inside a 600-second budget.

| Condition | Completed conditions | Actual model responses | No-evidence gates | Valid required citations | Responses with invalid markers | Model-only median wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A: profile | 8 | 8 | 0 | Not applicable | 3 | 10.465 s |
| B: RAG | 8 | 6 | 2 | **0 / 6** | 0 | 9.107 s |
| C: behavior | 8 | 6 | 2 | **0 / 6** | 0 | 12.575 s |

All 24 conditions completed with no execution error, no skipped condition, and no detected prompt split leak. The model produced all 20 provider responses to observed EOS completion. The four other responses were the application's no-evidence path for `g05` and `g06` in B/C. They are excluded from the table's model timing medians; including them would substantially lower those medians for reasons unrelated to model speed.

Token usage was reported for every model request: **11,737 prompt tokens and 591 generated tokens** across the matrix. A used 3,827 prompt/243 generated tokens; B used 3,857/174; C used 4,053/174. The warm-up is separate and reported 19 prompt/3 generated tokens. Model-only median first-visible-text latency was 4.750 seconds for A, 4.202 seconds for B, and 6.686 seconds for C. These are observations from eight development questions and one hardware environment, not claims of statistically established differences between the conditions.

The citation result is a substantive failure. Every generated answer that received retrieved evidence omitted its required citations. In A, cases `g02`, `g05`, and `g08` emitted the literal placeholder `[source:RECORD_ID]`, which was not a permitted record. The application correctly kept these outputs unverified. Fluency and completed transport do not establish grounded answering. Factual entailment, answer relevance, Korean quality, injection resistance, and persona resemblance remain **ungraded**; no synthetic scores were assigned.

The matrix took **221.51 seconds**. The full launch, warm-up, matrix, and shutdown took **235.98 seconds**. Lazy loading during warm-up took **10.28 seconds**. Final process peak RSS was **2,081,476,608 bytes, about 1.94 GiB**. This is the language process's high-water RSS; the test deliberately stopped it before other model inference and does not prove that all model services fit in memory together.

The [compact committed summary](../tools/evaluation/results/qwen3-06b-cpu.json) contains each case/condition's execution result, exact citation diagnostics, usage, timing, prompt hash, and retrieval identifiers, without full source text or generated passages. Full raw responses, checkpoints, runtime measurements, and service logs remain under ignored `reports/local/generation-eight-cases-cpu-v2*`. The compact file records hashes of those raw reports.

Input fingerprints for this run:

| Input snapshot | SHA-256 |
| --- | --- |
| Canonical profile/sources/evidence | `d67a94b9d6fdbcbd1fe20071e92010961dcdb34ce518ff5df27b6b16583c1d08` |
| Declared evaluation suite | `8ee9f31f41a04627e03cd8657137244789ff6f13c434fce44242b4786eeaea22` |

The initial matrix attempt, `generation-eight-cases-cpu.json`, remains available separately. Strict offline repository-ID loading failed before generation, yielding 20 provider errors plus four completed no-evidence gates. Loading the cached snapshot directory fixed that integration issue. Its unsuccessful model-loading times and empty outputs are not citation-compliance or persona-quality measurements.
