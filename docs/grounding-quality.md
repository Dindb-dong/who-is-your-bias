# Grounded response quality and reproducible review

The dialogue pipeline distinguishes three different questions: did the model emit an exact citation, did it place citations after all sentence-like units, and do the cited sources actually support what it said? Only the first two are automatic checks. Every newly generated answer remains `grounding.status = unverified` and `semantic_status = unreviewed`, including an answer with mechanically complete citations. Curated source previews retain their separate `grounded` status because they display reviewed summaries without generating a new persona response.

## Why the prompts changed

The archived [Qwen3-0.6B measurement](../tools/evaluation/results/qwen3-06b-cpu.json) recorded 20 model responses across 24 conditions. All 12 retrieval-conditioned model responses omitted their required citations; three profile-only responses copied the literal `RECORD_ID` placeholder. That measurement remains unchanged.

The first compact candidate uses a shorter, condition-specific response contract:

- Profile-only condition A receives the same persona description and editorial presentation rules, and explicitly receives no evidence or citation examples.
- Retrieval conditions B and C receive each reviewed source summary, its kind, publication date, and its complete, usable citation string. There is no placeholder identifier to copy.
- Plans and predictions must retain their historical status. The prompt requests one to three short sentences and a supporting citation after each factual sentence.
- Source text and conversation history remain lower-priority, untrusted data. Retrieved source text never becomes a system instruction. Complete recent exchanges remain bounded to 12,000 characters.
- The generator makes one request. It does not add missing citations, rewrite an invalid citation, run a hidden repair pass, or replace model output with a canned source sentence.

The legacy builder in `src/bias/legacy_prompts.py` preserves the PR6 message bytes. An integration regression check reproduces all 20 archived prompt hashes when the archived corpus is present. For example, g01 condition B changes from 3,759 prompt characters to 3,203; the latter carries actual record citation strings and publication dates. Character count is a size observation, not a token-count or quality claim.

The measured compact v1 builder is frozen separately in `src/bias/compact_prompts.py`. The next candidate, `grounded_v2`, narrows factual composition to the supplied source records and separates it from a brief editorial presentation rule. It removes the long biography and personality-like style description from retrieval-conditioned context, explicitly prohibits adding unsupported motives or causal effects, and handles a mixed request by answering only its supported part. Profile-only A is deliberately byte-identical between compact v1 and grounded v2, while B/C retain the same retrieved record texts, dates, and citation strings. For g01 B, grounded v2 has 2,264 prompt characters. Use `--prompt-version grounded_v2` to evaluate it explicitly; naming a candidate does not establish its quality.

## Citation diagnostics

`src/bias/grounding.py` performs exact membership checks and a separate conservative coverage check. It accepts source markers before or after sentence-ending punctuation, attaches trailing markers to the preceding sentence, and handles common abbreviations, decimal numbers, numbered lists, and Korean sentence punctuation. A marker at the start or in the middle of a sentence does not automatically cover the rest of that sentence. One marker at the end of a multi-sentence paragraph does not cover all preceding sentences.

The live response adds the following fields without changing the existing status enum or streaming stages:

| Field | Meaning |
| --- | --- |
| `citation_status` | `complete`, `partial`, `missing`, `invalid`, or `not_required`; a formatting diagnostic |
| `citation_unit_count` | Number of nonempty sentence-like units found by the heuristic |
| `cited_unit_count` | Units with at least one trailing citation to a retrieved record |
| `malformed_citations` | Source-looking markers that do not match the exact grammar |
| `semantic_status` | `unreviewed` for generated output; `curated_source` for a source preview |

These are not factual claim counts. Headings, disclaimers, unusual abbreviations, or unconventional punctuation can produce extra units. A fluent false sentence with a valid source marker can receive `citation_status = complete`; it still remains unverified. Tests explicitly exercise that case. Raw model text is retained by the evaluator, while the existing speech/display cleanup removes exact citation tokens from the spoken text and presents permitted source links separately.

The evaluation report preserves `citation_format.status = valid` as a narrow syntax/membership metric for comparison with earlier reports. Its separate `citation_format.coverage` object and `complete_citation_coverage` summary measure sentence-marker placement. Neither metric is called entailment or answer accuracy.

## Controlled comparisons

Use identical declared cases, corpus, retrieval limit, language, model snapshot, sampling configuration, and output-token limit when changing only the prompt. Each condition still receives a fresh Engine and session; condition order rotates across cases. The transport audit records actual HTTP requests, prompt hashes, exact evidence-text hashes, provider token usage where available, and the response model. Held-out and unreviewed evidence is rejected before the provider request; evidence text must match the reviewed corpus exactly.

With a configured local language service:

```bash
python -m tools.evaluation.run \
  --prompt-version legacy \
  --max-total-seconds 1800 \
  --output reports/local/grounding-quality/legacy.json

python -m tools.evaluation.run \
  --prompt-version compact \
  --max-total-seconds 1800 \
  --output reports/local/grounding-quality/compact.json
```

Use different output paths for each model and repetition. Existing reports and their JSONL checkpoints are never overwritten. Preserve runtime health/configuration snapshots alongside them; prompt hashes do not bind the model weights, operating-system load, or sampling implementation. The no-evidence path bypasses the model and must be excluded from claims about model abstention quality and model generation latency.

Qwen3-1.7B was the first larger comparison model, using the same model family and Transformers runtime. Its [official model card](https://huggingface.co/Qwen/Qwen3-1.7B) documents non-thinking mode via `enable_thinking=False`. The [4B-Instruct-2507 card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) describes a separate, non-thinking-only model. It was subsequently measured with a different inference runtime and quantization. Neither model card establishes quality on this project's public-persona questions. A larger model is a candidate to measure, not a guarantee of faithful answers.

Pinned comparison revisions:

| Model | Revision |
| --- | --- |
| `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` |
| `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| `ggml-org/Qwen3-4B-Instruct-2507-Q8_0-GGUF` | `e6f794d44f9395d0184a966c27b5ae99ea356fcb` |

The [ggml-org Q8_0 conversion](https://huggingface.co/ggml-org/Qwen3-4B-Instruct-2507-Q8_0-GGUF) is published by the upstream runtime organization, not by Qwen. Its file is `qwen3-4b-instruct-2507-q8_0.gguf`, 4,280,403,520 bytes, SHA-256 `ae916ede1c010a26955ee8ae2e908bf8815a3f135ec860439ab924701c69d5f1`. The [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) documents CPU quantized inference through an OpenAI-compatible interface. This is a separate model/runtime/quantization experiment from the BF16 Transformers runs. Runtime memory, speed, and generation quality must be measured rather than inferred from file size.

The completed 4B comparisons used a **local Q4_K_M requantization of that Q8_0 file**, not the Q8_0 file itself and not an independently published native Q4 checkpoint. The local file is 2,497,279,040 bytes, SHA-256 `9b7d1268c8d2953d4d8ba24c8ba5ab211d18b8511b61d6ecb2c15d56e4fb875c`, served by `llama.cpp b12110`. The published runtime receipts retain both source and derived hashes, server arguments, and observed process-memory snapshots. They do not establish equivalence to another quantization of the original weights.

## Recovered measurements and default selection

The [published review artifact](../tools/evaluation/results/grounding-quality-review.json) preserves five complete evaluation reports' final responses and explicit review judgments: 120 case/condition executions, including 99 completed model responses, one truncated model response, and 20 deterministic application gates. The initial archived `qwen3-06b-cpu.json` report is a separate earlier run and is not pooled into these counts. The interrupted nine-row 4B candidate checkpoint is also excluded; its hash is retained in the final run's runtime receipt.

The table below includes only retrieval-conditioned **B and C model responses**. Each study attempted 12 such responses. Profile-only A has no prompt evidence by design; the four no-evidence application gates per study do not test model abstention and are excluded. “Supported answer” requires a relevant answer with at least one factual unit and **every** factual unit explicitly reviewed as supported by the supplied evidence. It does not require mechanically complete citations and is not an external-world accuracy score.

| Model and prompt | Completed B/C model responses | Complete sentence-marker coverage | Supported answers in the development review |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B BF16 / legacy | 12 / 12 | 0 / 12 | 4 / 12 |
| Qwen3-0.6B BF16 / compact v1 | 12 / 12 | 3 / 12 | 3 / 12 |
| Qwen3-1.7B BF16 / compact v1 | 12 / 12 | 1 / 12 | 0 / 12 |
| Qwen3-4B-Instruct-2507 Q4_K_M / compact v1 | 11 / 12 | 4 / 11 | 4 / 11 |
| Qwen3-4B-Instruct-2507 Q4_K_M / grounded v2 | 12 / 12 | 9 / 12 | 6 / 12 |

These are **nonblind, model-assisted development judgments by implementation agents**, with reviewer provenance and unit-specific rationales retained in the artifact. They are not independent human evaluation, held-out benchmark accuracy, or evidence of persona fidelity. The same eight questions informed prompt revisions, there is one repetition per condition, and sentence-like units can contain several claims. An unsupported detail causes a partially-supported judgment even when the main explanation is correct.

Two within-model prompt comparisons are available. On 0.6B, compact v1 improved citation placement but did not improve the reviewed supported-answer count. On 4B, one compact answer hit the output limit; among the **11 identical pairs completed in both runs**, complete coverage changed from 4 to 8 and supported answers from 4 to 5. The additional completed g01 RAG answer accounts for the candidate's sixth supported answer in the full table. This distinction avoids attributing a denominator change to semantic improvement.

The 0.6B/1.7B runs used BF16 Transformers inference; the 4B runs changed model variant, runtime, and precision to llama.cpp Q4_K_M. Their shared questions make them useful deployment comparisons, but they do **not isolate an effect of parameter count**. Runtime receipts record the captured model revisions and settings; the BF16 receipts do not include a full dependency lock or effective sampler snapshot. Host load and exact cross-runtime sampling behavior are not controlled by the common seed.

The candidate still has concrete failures:

- In g02 behavior, it attributes improved engineering accuracy and completeness to friends' criticism, although the supplied TED summary only calls the feedback unusually useful. Its exact citation passes the mechanical coverage check.
- In g07 RAG, it refuses the invented endorsement but also falsely denies that the supplied Tesla record contains a reason for starting with a sports car. The corresponding behavior answer states the financing reason and then contradicts it. Resisting the endorsement instruction does not establish that the supported part was answered correctly.
- In g01 behavior, the capital/scale explanation is supported by one retrieved record while the model attaches a different record's citation. Evidence support and citation support therefore receive different labels.

**Selection: compact v1 remains the application default; grounded v2 remains an explicit evaluation candidate.** The default change from the legacy prompt removes placeholder citations and improves the response contract, but these measurements do not justify claiming improved semantic accuracy. The candidate is not promoted merely because its citation count is higher. Every generated live answer continues to show unverified semantic status. Further prompt selection needs additional, separately declared questions and review of mixed supported/unsupported requests; no human-review or persona-fidelity result is implied by this implementation.

## Source-bound semantic review

The [ALCE research paper](https://arxiv.org/abs/2305.14627) evaluates fluency, correctness, and citation quality separately. The local review workflow follows that separation; it does not reproduce ALCE's benchmark or automatic entailment metrics.

Prepare a review packet from a complete raw report and its exact corpus snapshot:

```bash
python -m tools.evaluation.review prepare \
  --report reports/local/grounding-quality/compact.json \
  --output reports/local/grounding-quality/compact-review.json
```

This creates an immutable source packet and `compact-review.annotations.json`. The packet contains the exact model response, sentence-like units, actual prompt evidence with publication dates and locators, and any evaluator-only withheld references. It binds the report, corpus, suite, prompt, and response hashes. Full prompts, source texts, and raw outputs remain local research artifacts; a compact published summary cannot be substituted for the raw report.

All annotation fields start as `ungraded`. A reviewer must inspect the provided text and sources and supply their name, whether the work is human or model-assisted, whether it is independent or blinded, and a source-specific rationale for each graded unit. Positive support labels require evidence IDs actually provided to the model. A positive citation-support label additionally requires a record actually cited in that unit. Held-out references cannot be relabeled as prompt support.

Two unit judgments stay separate:

- `evidence_support` asks whether all material claims in the unit follow from the available prompt evidence. It may be supported even if the model forgot to cite it.
- `citation_support` asks whether the records the model actually cited support that unit. Valid spelling alone is insufficient. An unsupported extra numerical detail, a changed date, or a plan presented as an achieved result requires a weaker judgment and explanation.

Answer-level judgments cover relevance, language quality, abstention, injection resistance, and consistency with the editorial style rules. Style consistency does not measure identity, private thought, stable personality, or visual/vocal similarity. Persona fidelity remains ungraded.

After adding explicit annotations, validate and summarize them without modifying the generation report:

```bash
python -m tools.evaluation.review summarize \
  --packet reports/local/grounding-quality/compact-review.json \
  --annotations reports/local/grounding-quality/compact-review.annotations.json \
  --output reports/local/grounding-quality/compact-review-summary.json
```

The validator rejects mismatched packet hashes, missing or duplicate units, missing reviewer provenance, unsupported evidence references, positive citation judgments without actual cited records, and semantic labels on incomplete generations. Every unit remains visible in the denominator, including ungraded units. Model outputs and deterministic application gates have separate populations. A nonblind model-assisted review is labeled as such and cannot be presented as an independent human benchmark.

Keep durable review evidence alongside the code rather than retaining only aggregate counts and hashes of ephemeral local files:

```bash
python -m tools.evaluation.review publish \
  --review reports/local/grounding-quality/compact-review.json \
           reports/local/grounding-quality/compact-review.annotations.json \
  --output tools/evaluation/results/grounding-quality-review.json
```

Repeat `--review PACKET ANNOTATIONS` to include several versioned runs. Add one `--label` and `--runtime-metadata` per review, in matching order, to retain the actual model revision, quantization and runtime rather than relying on a request alias such as `qwen3-local`. The compact publication retains every exact final response, its hash, source IDs, unit labels and rationales, and reviewer/report/packet provenance. It includes failures and unsupported examples. Source summaries, original transcripts, and model reasoning traces are not duplicated. Per-condition response counts require all factual units to be explicitly supported; they remain exploratory reviewer judgments, with relevance and citation formatting separate. Profile-only output has no prompt evidence by design and should not be pooled with RAG conditions as if their source access were identical.

## Verification scope

Focused tests cover exact legacy prompt reproduction, all 100 requested prompt hashes in the five published studies, mode-specific context, complete-exchange history bounds, bilingual citation segmentation, malformed and fabricated markers, uncited sentence tails, a false claim with a valid citation, one-request generation, cancellation cleanup, held-out isolation, raw-output preservation, and review provenance. Protocol tests use explicit synthetic providers; frozen-hash checks use the published measurements and matching corpus snapshot. Actual local-model measurements and their review results remain separate from test pass counts.
