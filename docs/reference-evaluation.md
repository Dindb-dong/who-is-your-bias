# Actual-answer reference evaluation

This is the first executed original-Q&A pilot for R1–R4 of the [YAICON proposal](proposal-alignment.md). It preserves a negative coverage result rather than presenting application tests or lexical overlap as persona fidelity.

## Recorded result

[The frozen run](../tools/reference_evaluation/results/qa-v2-qwen4b-2026-11-11/run.json) contains **24 real HTTP/SSE executions: eight original held-out questions × three conditions, all complete**, without retries, substitutions or discarded failures. Total inference-loop wall time was 104.349 s on the execution host. Server startup was separate. The machine was a CPU execution environment, not the user's intended demonstration device.

| Condition | Requests complete | Questions with retrieved examples | Median request wall time | Mean unigram F1, diagnostic only |
| --- | ---: | ---: | ---: | ---: |
| A, profile only | 8/8 | 0/8 by design | 4.451 s | 0.1525 |
| B, topic Q&A | 8/8 | 0/8 | 4.441 s | 0.1571 |
| C, topic + situation Q&A | 8/8 | 0/8 | 4.018 s | 0.1482 |

**C's benefit is not demonstrated.** Every question's three complete provider payloads had the same hash. Retrieval made no treatment difference: both B and C abstained for all eight questions. Two questions produced byte-identical text across all conditions; six produced some different text despite identical temperature-zero requests. Such differences are runtime variation, not a retrieval effect. Latency is descriptive and affected by prompt-cache reuse and condition order; the table is not a speed benchmark.

The conservative lexical gate protects against generic words being treated as topical evidence, but gives inadequate coverage on this held-out selection. Expanding training from 22 to 80 exchanges before the run did not resolve that limitation. The working C development regression remains useful engineering evidence, but cannot replace held-out coverage or establish persona quality. Future semantic-retrieval work should use new versioned development data and a fresh independent test set. The exposed test questions must not be relabeled as untouched validation after tuning to them.

## Recorded model-assisted judgments

The [condition-blinded model-assisted review](../tools/reference_evaluation/results/qa-v2-qwen4b-2026-11-11/model-assisted-summary.json) graded all 24 candidates. Its reviewer disclosed prior corpus-author involvement and read only the public comparison material during grading. Event-balanced C−B was **0 on all three dimensions**: content/viewpoint, wording/structure and situation response. These are one assistant's descriptive judgments, not independent human ratings or evidence of causal equivalence. The absent retrieval intervention already prevents a C efficacy conclusion.

A separate [unblinded source-support pass](../tools/reference_evaluation/results/qa-v2-qwen4b-2026-11-11/source-support-summary.json) marked **18 responses unsupported by their supplied evidence** and six not assessable (social greetings or simulation-limit disclosures). Unsupported does not mean proven false: no example had been supplied, yet the 18 responses made person-attributed or factual claims. Stating that no interview example supports a claim does not itself support that claim. The frozen raw outputs remain unchanged. [Interpretation metadata](../tools/reference_evaluation/results/qa-v2-qwen4b-2026-11-11/interpretation.json) binds these findings to the run digest. **Human ratings: zero.**

## Data and controls

The full local v2 corpus contains 80 training exchanges, four development exchanges and eight test exchanges across five publisher events. Eighty-four are complete contiguous exchanges; eight are explicitly partial. Only the two training events enter retrieval. The test answers are accessible to the evaluator and reviewer, never the model prompt. Pretraining exposure to public interviews is unknown.

The [plan](../tools/reference_evaluation/results/qa-v2-qwen4b-2026-11-11/plan.json) freezes the complete reviewed test selection, original question/reference hashes, full corpus fingerprint, source manifest, prompt/taxonomy/retriever versions and seven implementation file hashes. Altered code or data fails plan validation. No held-out generated answer informed the preceding source-wide expansion. Publisher attribution and situation labels were checked by an AI agent; this is not a preregistered independent human study.

All conditions use the exact application `original-qa.v1` prompt builder and common profile/system instructions. A adds no examples. B and C differ only in topic versus situation-reranked selection. Whole exchanges that exceed the 9,000-character example budget are dropped and listed; no silent excerpting occurs. Each request has a fresh session, English output, at most 256 generated tokens, temperature 0, top-p 1 and seed 42. A/B/C order rotates between questions.

The dedicated runner deliberately calls the real language model even when B/C retrieves nothing. The ordinary studio can instead abstain through its deterministic evidence-preview gate. That product behavior is useful, but would confound a same-model experiment if substituted for actual B/C inference.

The pinned model is Qwen3-4B-Instruct-2507, locally Q4_K_M-quantized from the recorded Q8_0 source, served by llama.cpp b12110 on four CPU threads, a 4,096-token context and one inference slot. The [model identity](../tools/reference_evaluation/results/qa-v2-qwen4b-2026-11-11/model-identity.json) preserves source revision, quantized and source file hashes, runtime revision, quantization and cache policy. No fine-tuning was performed.

## Review and reconstruction

The exporter creates a condition-blinded comparison page, samples JSON and blank rating template. It removes condition labels, evidence IDs, scores and citation markers from the reviewer view. Original questions, actual answers, context kind and partial-reference notes remain visible. Candidate wording may still suggest a condition; blinding is not absolute. Private condition mappings and training-source support material stay separate.

Content/viewpoint, wording/sentence structure and situation response use separate ordinal rubrics. Factual support is a separate categorical judgment. Incomplete model outputs remain ungraded for fidelity. Unknown ratings stay null. Review import rejects mismatched report/bundle/key commitments, duplicate sample identities, invented reviewer identity, boolean/out-of-range ratings and ratings without an evidence note. Paired C–B differences are averaged within source event and then across events; missing pairs are explicit, with no significance claim.

A `model_assisted` review is never counted as a human review. Prior corpus-author involvement is disclosed in the actual review's method. Human scores remain ungraded until a real person submits them. ASR, video decode and software tests are not listening judgments.

The [reconstruction archive](../tools/reference_evaluation/results/qa-v2-qwen4b-2026-11-11/review-reconstruction.json) preserves sample order, private mapping, salt and content commitments without copying full source text into Git. Restore it against the pinned local corpus to recover the exact original blinded bundle hash. Do not give this private archive to blinded reviewers. Distribute only the generated `reviewer/` directory.

## Reproduction

First hydrate the pinned originals using [the original-Q&A instructions](original-qa.md). Reproduce the archived reviewer bundle without another model call:

```bash
python -m tools.reference_evaluation.review export \
  --corpus data/local/qna/elon-musk-v2 \
  --plan tools/reference_evaluation/results/qa-v2-qwen4b-2026-11-11/plan.json \
  --report tools/reference_evaluation/results/qa-v2-qwen4b-2026-11-11/run.json \
  --reconstruction tools/reference_evaluation/results/qa-v2-qwen4b-2026-11-11/review-reconstruction.json \
  --output reports/local/reference-review-restored
```

For a new run, start the declared OpenAI-compatible model server, declare its identity JSON, and freeze a new plan before executing. Existing outputs are never overwritten:

```bash
python -m tools.reference_evaluation freeze \
  --corpus data/local/qna/elon-musk-v2 --id my-new-study \
  --output reports/local/my-new-study/plan.json
BIAS_LLM_BASE_URL=http://127.0.0.1:8020/v1 \
BIAS_LLM_MODEL=qwen3-4b-instruct-q4 \
python -m tools.reference_evaluation run \
  --corpus data/local/qna/elon-musk-v2 \
  --plan reports/local/my-new-study/plan.json \
  --model-identity tools/reference_evaluation/results/qa-v2-qwen4b-2026-11-11/model-identity.json \
  --output reports/local/my-new-study/run.json
```

A new run on these exposed questions is a reproduction, not another independent holdout. The CLI preserves each raw response in a flushed JSONL checkpoint and records timeouts, incomplete finish reasons, invalid provider responses, total-deadline skips, usage and citation diagnostics. Citation formatting is not entailment; token overlap is not persona fidelity.
