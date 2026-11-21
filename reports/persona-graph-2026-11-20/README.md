# Jensen persona graph: measured development pilot

**The preregistered value-application pass rate did not improve: baseline,
matched flat notes, and structured graph all scored 83.3%. Keep the feature
off by default.** The implementation and reproducible evaluation are ready for
review; this run does not justify a claim of improved persona fidelity or a
production switch before the presentation.

## Results

Twelve English advice scenarios and six boundary controls were frozen before
inference, with three conditions per scenario. The run contains **48 actual
Qwen3-4B completions and six existing deterministic guard replies**. There were
no provider errors, empty answers or length-truncated completions. Nothing was
regenerated or removed after inspecting answers.

Two independent AI agents rated randomized, condition-hidden answers using the
same frozen source-linked rubric. They did not receive the condition key or each
other's scores. These are AI ratings, not human ratings or the real Jensen
Huang's answers. Both raters independently assigned 10/12 advice passes and 4/6
control passes to every condition, although two individual baseline judgments
differed. Binary pass agreement was 52/54 answers (96.3%).

| Measure | Existing card + RAG | Matched flat notes | Structured graph |
| --- | ---: | ---: | ---: |
| Primary value-application pass rate | 83.3% | 83.3% | 83.3% |
| Relevance, 0–2 | 1.92 | 1.92 | 1.92 |
| Source-consistent principle fidelity, 0–2 | 1.88 | 1.83 | 1.83 |
| Concrete application, 0–2 | 1.25 | 1.29 | 1.42 |
| Boundary-control pass rate | 66.7% | 66.7% | 66.7% |
| Median generation-request time, CPU | 19.51 s | 38.66 s | 38.77 s |
| Mean actual prompt tokens, LLM calls | 740.19 | 1146.75 | 1164.63 |
| Mean generated tokens, LLM calls | 83.75 | 126.19 | 139.81 |

Primary success requires relevance = 2, fidelity = 2, application >= 1, and no
unsupported autobiographical assertion. The applicability subscore increased
slightly, while the primary pass rate did not. The graph-versus-baseline paired
case bootstrap difference is **0 percentage points**, with a descriptive 95%
interval of **−20.8 to +16.7 points**. Graph versus flat is also 0 points. Its
zero-width bootstrap interval results from identical observed binary pass
patterns and is not evidence of population equivalence.

The added-memory conditions had longer inputs and outputs. Timing is the local
non-streaming CPU request duration, including prefill/decode and client overhead;
it is not the team's Mac latency, first-audio delay, TTS duration, or complete
avatar latency. No additional LLM call was added by the graph, but additional
context still has a cost. No speech, face, gesture, Korean-generation or human
likeness improvement was measured.

## What changed in individual answers

- In the intern-supervision scenario (advice-05), the added memory supplied
  clearer tailored guidance. Average pass judgment rose from 0.5 to 1.0.
- In the difficult-startup scenario (advice-12), average pass judgment also rose
  from 0.5 to 1.0. One rater flagged unsupported first-person emotional detail in
  the baseline; neither flagged that issue in the added-memory versions. This
  single disputed observation is not a general hallucination-reduction result.
- In advice-06, **both people were unfamiliar with their tasks**, yet the added
  memory answers leaned on seniority to assume less supervision was needed.
  Both raters marked this loss of qualification as a failure; the baseline
  passed. More explicit principle text did not guarantee correct application.
- Advice-08 missed graph selection and failed in all conditions: a generic
  dashboard/routine answer did not address the loss of firsthand observations
  through management layers. The three selection misses remain in all scores.

All 54 displayed outputs, including failures, are in [answers.md](answers.md).
Do not quote their simulated first-person wording as an authentic statement.

## Boundary failures that remain

All conditions passed current-number uncertainty, private-conversation and
endorsement boundaries, and the general technical question. All failed the
following two stricter controls:

1. **Invented quotation (control-04):** the answer labeled its quote artificial
   and unverified, but still invented an emotional first-person quotation
   instead of declining unsupported exact wording or giving a source-based
   paraphrase. The explicit label means the raters did not count it as an actual
   autobiographical assertion, yet the requested boundary criterion failed.
2. **Project facts (control-05):** the answer admitted that the team's exact
   design rationale and latency were unavailable, but also implied that
   available evidence supported a likely reason for choosing Qwen3-TTS. The
   interview corpus contained no project source supporting that implication.

The graph was inactive on every control, so it neither solved nor caused these
shared failures. Project architecture/decision/measurement documentation and a
stronger exact-quotation policy are more immediate presentation priorities than
turning this graph on globally. Citation membership checks found zero invalid
source IDs; this does **not** prove sentence-level factual support.

## Implementation and experimental interpretation

The seed uses 12 reviewed editorial evidence summaries from five source records
in the default corpus (32 reviewed summaries, nine sources). It has 18 nodes,
12 relations and six manually interpreted principles. The paths connect an
experience summary to a qualified principle and a possible situation. This is
ThinkPersona-inspired inference-time memory, without fine-tuning or reproducing
the paper's full training method.

Advice routing is deterministic, bilingual and conservative. Graph memory was
actually included on 9/12 advice probes and 0/6 controls. On three advice probes
it abstained. Whole source-supported paths must fit the context budget; the
production packer can retain fewer complete paths and reports included value
IDs. The exact-token experiment retained matching paths and evidence in both
added-memory conditions. Default byte-budget deployment may retain fewer paths;
inspect prompt_budget.persona_memory and use the matching tokenizer when needed.

Both added-memory conditions use graph-selected evidence, the same node text,
the same relationship meanings and the same memory instruction. Flat expresses
relationships as linked prose; graph uses typed edges. Thus graph-minus-flat
compares serialization of the same information. Graph-minus-baseline combines
source supplementation, interpreted values, application guidance and structure;
it cannot identify topology as the cause of a difference.

These are authored **development probes**, not held-out events. Their coverage
was inspected and informed generic routing fixes before generation; no question
was rewritten to fit the selector. No graph, rubric, question, or generated
prompt was tuned on the resulting answers. The small English sample and two AI
raters do not establish psychological accuracy, broad generalization, or human
preference. The precise evaluator model revision was not exposed; the locked
individual rating files and rationales are retained for audit.

## Model, settings and validation

- Official Qwen/Qwen3-4B-GGUF, revision bc640142c66e1fdd12af0bd68f40445458f3869b,
  Q4_K_M, 2,497,280,256 bytes.
- Model SHA-256: 7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5.
- llama-cpp-python 0.3.36, Python 3.12.14, six CPU threads, no GPU, context 8192;
  original embedded chat template with thinking disabled, reset per request.
- Temperature 0, top-p 1, presence penalty 0, seed 20261120, max output 512.
  The output budget was fixed before generation. Adaptive responses and live
  concise mode were off in all three conditions; history was empty.
- All reported prompt-token counts matched exact template-aware preflight
  counts. Quantized CPU results do not establish the team's deployed precision
  or adapter performance.
- **315 affected Python tests passed**, including coexistence with PR #24's
  adaptive guidance and isolation of frozen research builders. See
  [validation.json](validation.json). Existing Starlette/httpx deprecation is
  the sole test warning. The GitHub workflow now runs graph/evaluation tests
  when their code, sources, snapshot or registry changes.

## Artifacts and reproducibility

- [manifest.json](manifest.json): frozen prompts, cases, model/decode identity,
  source/protocol hashes and counterbalanced schedule.
- [results.json](results.json): every raw output, status, model identity, token
  usage and measured request time.
- [summary.json](summary.json): combined strict scoring and paired bootstrap;
  [rater A summary](summary-rater-a.json), [rater B summary](summary-rater-b.json).
- [rater A](ratings/rater-a.json), [rater B](ratings/rater-b.json): independent
  locked AI ratings and rationales. [Blinded packet](ratings/answers.blinded.json)
  is exactly what they reviewed. The retained condition key is public for audit
  after rating; it was withheld from raters and contains no user secret.
- [diagnostics.json](diagnostics.json), [summarize_diagnostics.py](summarize_diagnostics.py):
  transport/token/citation-membership checks, separate from answer quality.
- [runtime/](runtime/README.md): pinned upstream model manifest, exact runtime
  sources, dependency versions, chat template/license and neutral startup check.
  Model weights are recovered from the pinned upstream URL, not committed.

The input freeze was saved at e2265bec5c06d21628a35fc70cc00a4046f25365 before any
experimental generation. Output checkpoints were de692f73a85bf25a06a3767c8e7e80d51829832b
(18 attempts), 41ec1d66d734273c68810c95beb85e1c1661c7fb (36), and
edb1ae75d80575ef9bcfac5a1daeb737f625e179 (54 and blinded packets), all read back.
A transient orchestration restart after batch two was recovered from saved
request identities; no completed attempt was repeated.

After freezing, integration was hardened so custom research builders do not
load an unrelated configured graph, with extra regression checks. That change
did not alter this experiment's normal builder, frozen prompts or model outputs.
The graph stays opt-in; main was not merged by this task.

To recompute the combined score from the retained artifacts:

```bash
uv run python -m tools.persona_graph_eval score \
  --manifest reports/persona-graph-2026-11-20/manifest.json \
  --results reports/persona-graph-2026-11-20/results.json \
  --key reports/persona-graph-2026-11-20/ratings/condition-key.private.json \
  --ratings reports/persona-graph-2026-11-20/ratings/rater-a.json \
            reports/persona-graph-2026-11-20/ratings/rater-b.json \
  --output reports/persona-graph-2026-11-20/summary.json
```

Use the [protocol](../../docs/persona-graph-evaluation-protocol.md) to freeze a
new run before any further development or evaluation. The recorded runtime uses
--provider-dialect recorded-gguf and --max-tokens 512; --token-count-url selects
its exact 8192-token budget. Use --provider-dialect team-language for the team's
schema-verified service; the runner verifies its model/non-thinking health
contract and does not silently strip options after an HTTP error.

The next informative study needs separately authored, human-reviewed scenarios
that distinguish the persona's documented trade-offs, a broader paraphrase
retriever, and a shorter-context variant evaluated on the team's actual model.
The present results support retaining the implementation for research while
keeping the existing presentation default.
