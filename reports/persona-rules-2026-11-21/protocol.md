# Four-condition conditional-persona comparison

This is a prospective, author-designed application pilot. It is not a set of
held-out original Jensen Huang answers, a personality diagnosis, or a replication
of ThinkPersona's training procedure. No claim of improved accuracy is justified
until the actual outputs have been scored against the frozen rubric.

## Frozen inputs and estimands

The implementation checkpoint is `d65817048778ed87d48ad037dded05d6cba2b4b6`.
The corrected source-availability metadata and exact-token manifest were saved
in `c19b6e53b6302dd87ebc819b25c2757eb41e1eb6` before the first experiment request.
The manifest identity is
`87d81bbc50dcac93b05db0a14555dc0e953d723f64a84df3a9c695397e50ab6f`.
Its own byte hash is recorded separately by the Git checkpoint; the manifest
contains hashes of its model configuration, prompts, data, graph and runners.

| Condition | Generation context | What a comparison can establish |
| --- | --- | --- |
| `baseline` | Original persona card and original 32-record corpus, production lexical RAG | Current reference behavior on these questions |
| `expanded_rag` | Same card and lexical retriever, expanded 60-record corpus | Effect of expanding the retrievable data |
| `flat` | Expanded corpus plus selected source evidence and complete conditional rules in prose relationships | Combined effect of rule-based evidence selection and conditional memory |
| `graph` | Same evidence, claims, conditions and exceptions as `flat`, with typed relationships | Representation difference versus `flat`; token counts can differ |

The coverage dates in the profile may change to describe the expanded corpus.
All other profile fields and all original evidence/source records are checked
for equality. There is no fine-tuning, adaptive response length, shared history,
model-based question rewriting, or extra model call for graph retrieval.

**Observed before generation:** both the original and expanded lexical RAG
supplied zero evidence on all 16 questions at their unchanged production
thresholds. Twelve questions had weak matches and four encountered temporal
scope restrictions. Flat and graph memory each supplied evidence for eight of
the twelve advice questions and none of the four controls. The testability and
hardship pairs were not selected. These are retrieval diagnostics, not accuracy
results. They are retained without changing the questions, retrieval thresholds,
rule terms or data after seeing them.

Consequently, graph versus baseline combines improved access to source evidence
and conditional memory. It cannot isolate a graph benefit. `graph` versus `flat`
is the matched representation comparison. This run also does not directly compare
the previous six-principle graph against the new graph. The prior 83.3% result
used different questions and a different rubric, so subtracting it from this
run's result would be invalid.

## Questions and generation

The registry contains twelve English advice questions arranged in six pairs:
test availability, revisable conviction, experience-dependent supervision,
expertise and shared context, resource constraints, and interpreting hardship.
Four controls test current news, private allegations, unsupported project
measurements and fabricated quotations. The questions were authored before the
new rule data were inspected; judge-only source references were subsequently
bound to the actual verified records. Once model outputs have been inspected,
reuse of this set after tuning must be labeled development evaluation.

All 16 cases run in all four conditions, for **64 scheduled model requests**.
No case is discarded because memory is absent or an answer is poor. There are no
deterministic production guards in this frozen schedule. Condition positions are
balanced with Latin-square rotations and case order is seeded. Each request has
empty history and the local model is reset between requests.

- Model: official `Qwen/Qwen3-4B-GGUF`, Q4_K_M.
- Revision: `bc640142c66e1fdd12af0bd68f40445458f3869b`.
- Verified model SHA-256:
  `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`.
- Runtime: Python 3.12.14, llama-cpp-python 0.3.36, six CPU threads, no GPU.
- Exact model context window: 8,192 tokens. The original model chat template is
  used with thinking disabled.
- Decoding: temperature 0, top-p 1, presence penalty 0, seed 20261121,
  maximum 256 output tokens, one repeat.
- Exact input-token ranges: baseline 720–756; expanded RAG 722–758;
  flat 722–1,674; graph 722–1,765. All selected complete memories fit.

The run saves each attempt atomically. Batches of 16 are persisted before the
next batch, and resume skips every recorded attempt, including errors. A timeout,
empty output, truncation at the 256-token limit, or other invalid completion
remains a failed attempt; there are no implicit retries or rewritten answers.

## External, blinded assessment

After all scheduled attempts are recorded, the runner shuffles answers and
removes condition and attempt labels. Each answer receives the same union of
judge-only source references for its question. Evaluation rubrics and desired
answers never enter retrieval or generation prompts. Invalid completions remain
in the denominator with zero scores and are not silently omitted.

Advice dimensions are scored 0 (absent/contradictory), 1 (partial), or 2 (clear):

1. **Relevance:** directly addresses the question.
2. **Fidelity:** remains consistent with the available evidence and does not
   attribute unsupported beliefs, feelings, quotations or events to the person.
3. **Application:** applies the idea to a concrete action rather than mentioning
   a value by name.
4. **Context sensitivity:** uses the stated constraint and respects relevant
   qualifications or exceptions.

A conditional-advice pass requires relevance 2, fidelity 2, application at least
1, context sensitivity 2, and no unsupported autobiography. This is a rubric-based
judgment of plausible source-consistent advice, **not the probability that the
real person would give that answer**. The paired-context joint pass additionally
requires both variants to pass for the same rater. Controls use a separate pass
criterion and the same unsupported-autobiography restriction.

Raters must be identified as human or AI. AI assessments must not be described as
human validation. The summary retains per-answer judgments, dimension means,
failure counts and source-grounding limits. Confidence intervals are descriptive
paired bootstrap intervals over the six context pairs, with variants/repeats/
raters averaged within each pair. They are unstable with so few clusters.

The CPU request timings include model request processing but exclude speech
synthesis, network conditions on the team's machine, browser playback and avatar
rendering. They do not establish production or end-to-end latency.

## Reproduction

Use the pinned recovery instructions in
[`runtime-availability.json`](runtime-availability.json) and the earlier
[`runtime guide`](../persona-graph-2026-11-20/runtime/README.md). The same command
must launch the local server and client in an environment that isolates network
namespaces per command. Do not silently switch model precision or provider and
label it the same run.

From the repository root with the recovered runtime in `/tmp/persona-v2-runtime`:

```bash
PYTHONPATH=src:. /tmp/persona-v2-runtime/.venv/bin/python \
  /tmp/persona-v2-runtime/run_with_server.py -- \
  .venv/bin/python -m tools.persona_rules_eval run \
  --manifest reports/persona-rules-2026-11-21/manifest.json \
  --output reports/persona-rules-2026-11-21/results.json --limit 16
```

Repeat the identical command only after preserving the preceding batch. It
continues the same frozen schedule without repeating requests.
