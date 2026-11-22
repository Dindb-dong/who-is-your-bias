# Persona graph: frozen development-probe protocol

This protocol measures **source-consistent application of an interpreted value
to a new hypothetical situation**. It does not measure the real Jensen Huang's
private personality, emotions, agreement with an answer, or human likeness.
The source records are editorial summaries of public material. Their presence
does not establish a verbatim quotation or a newly acquired multimodal dataset.

## Conditions and estimands

All three conditions use the same production snapshot prompt builder, persona
card, query guards, temporal cutoff, retrieval backend, empty history, model
revision, decoding settings and output budget. Adaptive answer lengths and live
concise mode are disabled for this comparison. Speech and avatar rendering are
outside this experiment.

| Condition | Additional context | Interpretation |
| --- | --- | --- |
| Baseline | Existing card and ordinary RAG | Current lexical/snapshot reference |
| Flat | Graph-selected source anchors, interpreted nodes and relationships expressed as plain linked notes | Matched information without typed graph representation |
| Graph | The same selected source anchors, nodes and relationship meanings, expressed as typed graph edges | Structured graph representation |

Graph minus baseline estimates the **combined addition** of source selection,
interpreted value notes and structured relationships. It cannot attribute a
gain solely to graph topology. Graph minus flat compares representation with
the same semantic information, including relationship meanings. Prompt lengths
can still differ and are reported; this is not an equal-token padding study.
Flat and graph must supply identical factual evidence IDs and the same included
value paths under the common context budget. Production may retain fewer whole
paths when memory does not fit; both representations must retain the same ones
and record selected versus included IDs. Dropping memory or a path in only one
condition is an error, not an acceptable matched run. If a selector finds no
path, that coverage failure remains in the frozen run; it is not repaired after
examining outputs.

## Cases and freezing

`data/evaluation/jensen-persona-graph-v1.json` contains 12 English advice probes
(two per interpreted principle) and six controls. They are author-designed
synthetic **development probes** informed by known source summaries. They are
not withheld interview events, a representative distribution of audience
questions, a calibrated personality test, or evidence of generalization to
unseen real-world events. English matches the avatar's planned presentation
language; no Korean quality claim follows.

Pre-generation coverage inspection exposed overly broad time/approval filters
and narrow advice routing. Generic routing fixes were made during development
before the final freeze; the authored questions were not rewritten to match the
selector. This is an additional reason these probes are development evidence,
not a fresh held-out benchmark. After final freeze, no generated answer may be
used to change the graph, prompt, cases or scoring rubric within the same run.

The case file contains judge-only expected principles, source IDs and rubrics.
`prepare_case` accepts only the user's question and language, never these
labels. Retrieval and generation do not see expected answers or target-value
annotations. The reference rubric is available to raters because correctness
must be assessed against source-consistent interpretations rather than keyword
presence. Each rater also receives the union of legitimate retrieved sources
across conditions and the case's reference sources, without labels revealing
which condition retrieved which source. This avoids penalizing a supported
baseline fact merely because it was outside the suggested value's anchors.
A later genuinely held-out study needs separately authored scenarios
and independent human annotation before any further development.

Freeze the protocol, graph, corpus, snapshot, selected case IDs, model identity
and revision, production prompt builder and exact final prompts before any
answer generation. The manifest records their SHA-256 identities. A reduced
pilot is allowed only when its case IDs are selected **before** generation and
the reduced scope is reported. Never quietly remove an unsuccessful case.

Defaults are one run per case/condition, temperature 0, top-p 1, presence penalty
0, seed 20261120, a 256-token
answer budget, disabled thinking and 16384 conservative UTF-8 byte budget units.
Byte budget units are explicitly **not measured model tokens**. A model with a
smaller context needs a real tokenizer check of every frozen prompt and the
reserved answer budget before generation. `freeze_manifest(counter=...)`
accepts the production counter interface for exact template-aware token counts.
The CLI `--token-count-url http://127.0.0.1:8129/count_tokens` uses the local
inference server's exact embedded chat template and tokenizer, and adopts the
server's declared context window unless explicitly overridden with a smaller
one. The token counter is a counting request, not a generation request.
Endpoint-reported prompt/completion token usage is also saved. Record any
non-default settings in the manifest. A model alias without a pinned artifact
revision is a reproducibility limitation, not a pinned model.

Question/repeat order is seeded and shuffled. The six permutations of the
three conditions are cycled in shuffled order so a complete block of six cases
is counterbalanced. A partial block is only approximately counterbalanced.
Repeat averages are clustered within case; repeats do not multiply the number
of independent scenarios. Greedy repeats are usually redundant.

## Execution and failed answers

Use a real OpenAI-compatible endpoint or the Python callable-provider interface.
The tool has no mock or canned-answer fallback. It saves each attempted request
atomically and resumes only when all saved request/result identities match.
Requests are sequential, with no automatic retries. A failed request stays a
failure; to investigate a provider problem, freeze a separately named run and
retain the original run. `--limit` bounds a batch without altering the schedule.

The provider dialect is explicit and frozen. `recorded-gguf` (default) sends the
recorded local GGUF server's supported `chat_template_kwargs.enable_thinking=false`
and `presence_penalty=0` fields. `team-language` sends only fields accepted by
`services/language/schemas.py`; the versioned team service disables thinking in
`services/language/adapters.py` and has no presence-penalty generation option.
Before each request this dialect verifies the served model, declared revision
when available, and non-thinking template contract from `/health`, retaining
those identity fields. Unknown dialects or conflicting fields fail explicitly.
An API being generally OpenAI-compatible does not establish support for the
GGUF extension. Never silently strip fields or retry with a different dialect.

Save raw text, finish reason, provider model identity, system fingerprint when
available, token usage, elapsed generation-request time and any exception type.
Elapsed time includes local request overhead and is **not** end-to-end avatar
latency. Successful generation requires nonempty text, explicit `stop` finish,
and no raw-output truncation. `length`, missing finish, provider errors and empty
answers remain in the denominator as zero quality. Partial text is retained for
diagnosis but not mistaken for a complete answer. The maximum retained raw text
is 100000 characters; exceeding it marks the completion invalid and records the
original text hash. The ordinary 256-token budget should never reach this cap.

Existing public/private-evidence guards use the production deterministic
response without calling the model. These are recorded as `policy_guard`,
reported separately from valid model completions, and never treated as graph
generation improvements. Their control answers may still be rated for correct
behavior. A speculative no-evidence answer retains the application's explicit
simulation label; both raw and displayed versions are saved.

These ratings assess raw generated dialogue, with citation markers retained so
raters can inspect support. Production `Engine` removes exact citation markers
from its final display/speech text and exposes citation metadata separately.
The experiment also fixes greedy decoding and its declared output budget rather
than relying on a deployed provider's sampling defaults. Shared prompt/retrieval
logic does not make this an end-to-end application or spoken-avatar evaluation.

## Blind rating rubric

Generate the packet only after **all** frozen attempts have a recorded outcome.
It shuffles answers and replaces condition labels with opaque IDs. Give raters
`answers.blinded.json`, `ratings.template.json` and this protocol; do not give
them `condition-key.private.json`, manifests or other conditions' labels.
Semantic differences in an answer may reveal a condition, so blinding is
imperfect. Independent raters should not see each other's scores before filing
their own files. Declare `rater_kind` as `human` or `agent` and identify each
rater. AI review is never labeled human evaluation.

Rate each complete advice answer on these integer scales:

| Field | 0 | 1 | 2 |
| --- | --- | --- | --- |
| `relevance` | Ignores the practical question or chooses an unrelated principle | Partly addresses the problem, with a significant mismatch | Selects a relevant principle and addresses the stated dilemma |
| `fidelity` | Reverses/distorts the principle or introduces a categorical unsupported rule | Broadly compatible but loses an important qualification or is too vague to establish direction | Preserves the source-consistent direction and important qualifications |
| `application` | Only repeats an anecdote/value, abstains from advice, or gives no applicable step | Applies the principle to a concrete feature or next step in this scenario | Offers a specific, coherent action/decision criterion connected to the scenario |

Mark `unsupported_autobiography` true for invented personal experiences,
verbatim quotations, private emotions, endorsements or present-day personal
positions presented as actual. Conditional advice or an explicitly hypothetical
example is not automatically autobiographical. Source-grounded historical facts
with their qualifications are allowed; do not require a first-person story.

The **primary advice success** is relevance = 2, fidelity = 2, application >= 1,
and no unsupported autobiographical assertion. Mentioning an expected value,
retrieving its ID, or matching a keyword earns no automatic credit. A useful
different principle may receive full credit if source-consistent and applicable;
the suggested rubric is not a demand for a specific phrase. Raters explain their
decision in a short rationale. `control_pass` is null for advice answers.

Controls have no value-application score: `relevance`, `fidelity` and
`application` may remain null. Set `control_pass` to an explicit boolean using
the case criterion, and still rate unsupported autobiography. Control success
requires the criterion and no unsupported autobiography. The plain technical
question deliberately accepts a correct general explanation or an honest
coverage limitation; an interview corpus is not assumed to prohibit all novel
text generation.

## Aggregation and interpretation

The scorer rejects missing/duplicate/unknown ratings, altered answer hashes,
wrong manifests and incomplete runs. Failed generations are automatic zeros;
they do not disappear from denominators. The advice and control results are
reported separately, with total attempts, valid completions, deterministic
guards and failed/truncated completions per condition. Rater pass indicators
are averaged, not resolved by secretly selecting the favorable rater. An
unsupported-assertion rate also needs the failure count beside it, since a
crashed system cannot make such an assertion.

For graph minus baseline and graph minus flat, average repeats and raters within
each case, then bootstrap paired case differences (5000 resamples, seed 377).
Report the mean difference in percentage points and the descriptive percentile
95% interval with the number of independent cases. With 12 advice cases (or a
smaller pilot), intervals are unstable and often broad. There is no confirmatory
significance claim, no multiple-comparison testing claim, and no proven general
persona improvement. Report no gain, a regression or an inconclusive result as
such. Do not tune the graph or rubric on these outputs and reuse the same cases
as if they were a new held-out experiment.

## Commands

The endpoint must already serve the declared model. Choose an output directory
that will be checkpointed according to repository instructions. The graph path
below should be the actual committed seed path.

```bash
uv run python -m tools.persona_graph_eval freeze --graph data/personas/jensen-huang/persona_graph.v1.json --model MODEL_ID --revision PINNED_REVISION --output reports/persona-graph-run/manifest.json
uv run python -m tools.persona_graph_eval run --manifest reports/persona-graph-run/manifest.json --base-url http://127.0.0.1:8080/v1 --limit 6 --output reports/persona-graph-run/results.json
uv run python -m tools.persona_graph_eval blind --manifest reports/persona-graph-run/manifest.json --results reports/persona-graph-run/results.json --output-dir reports/persona-graph-run/ratings
uv run python -m tools.persona_graph_eval score --manifest reports/persona-graph-run/manifest.json --results reports/persona-graph-run/results.json --key reports/persona-graph-run/ratings/condition-key.private.json --ratings reports/persona-graph-run/ratings/rater-1.json --output reports/persona-graph-run/summary.json
```

Repeat `run` to resume subsequent bounded batches; the command skips already
recorded attempts, including failures. Do not share a condition key with raters
until independent rating is complete. Publicly saved raw prompts and model
answers are development artifacts, not new verified facts about Jensen Huang.
