# Source-bound persona graph for new advice

Task branch: `feat/jensen-persona-graph-nov20`, based on main
`472713e183e0dbdc8da8d35a22e6bd9e80be5f39` (PR #24).

This is a ThinkPersona-inspired **inference-time memory pilot**, not a
reproduction of that paper's fine-tuning or a measured psychological model of
Jensen Huang. Existing LLM, voice, avatar, and profile-only ablation remain in
place. No new LLM call is added to the response path.

## Behavior

The optional graph connects a reviewed source summary to an editorially
interpreted principle and a class of situations where it may be useful. Six
principles cover validation before commitment, system bottlenecks, contextual
delegation, firsthand observations, customer problems, and difficult work with
realism. The seed has 18 nodes, 12 edges, and 12 source-summary anchors.

English/Korean lexical rules select up to two relevant paths for advice
questions, requiring a specific topic term or multiple generic terms. Source passages join factual retrieval; interpreted values join a
separate `persona_memory` context. The prompt treats new applications as
hypothetical advice, not as actual memories, feelings, quotations, or current
positions. Selection rules are conservative and can miss paraphrases; they
are not a learned graph retriever.

The graph validates persona, reviewed training/factual eligibility, evidence
text hash, and source metadata hash at load and use. Snapshot eligibility is
applied before path selection. A personal deadline such as "now" can use dated
historical principles, while explicit dates/news/current-position requests
retain the conservative factual gate. This advice-only scope still excludes
post-cutoff evidence. Entire source-supported paths must fit the
prompt budget; the packer tries fewer complete ranked paths before omitting
memory. Included value IDs and the fallback reason are recorded. The
existing citation audit still cannot prove semantic faithfulness.

Graph memory does not answer factual questions about this project's
implementation, private claims, or recent news. Those require appropriate
evidence; the graph does not create it. Existing simulation labels and
uncertainty handling remain active.

## Opt in

From the repository root, add this to an existing working LLM configuration:

```bash
export BIAS_PERSONA_GRAPH=data/personas/jensen-huang/persona_graph.v1.json
export BIAS_PERSONA_MEMORY_FORMAT=graph
```

Use `BIAS_PERSONA_MEMORY_FORMAT=flat` for semantically matched linked prose.
Unset `BIAS_PERSONA_GRAPH` to restore the existing path. The feature is off by
default; it does not change provider or voice settings. `profile` mode and
custom/versioned prompt builders do not use graph memory.

Inspect `response.context.persona_memory` for selection and
`response.context.prompt_budget.persona_memory` for whether memory actually
fit. A selected path alone is not proof that it reached the model. For
larger prompts, configure the existing exact tokenizer support for the same
model; do not simply raise a byte-budget setting past a model's real context.

## Evaluation and task status

The planned comparison uses the same Qwen3-4B quantized model, decoding
settings, questions, and response budget:

1. Existing persona card and factual RAG.
2. Additional source evidence plus interpreted principles and relationship
   meanings in flat prose.
3. The same information represented as typed graph relationships.

Condition 3 versus 2 measures the representation difference; 3 versus 1 also
includes additional information and retrieval. Neither is a claim that a
graph by itself improves every answer.

The first implementation checkpoint is
[`f28f112`](https://github.com/Dindb-dong/who-is-your-bias/commit/f28f11212703f715900899e5f8fbe5f817be0192).
Subsequent pre-generation regression testing passed 310 tests; expanded routing,
bounded path fallback, and approval-queue disambiguation address issues found
before any model answers were generated. The same development probes informed
these fixes, so they are not a held-out generalization test.

See [the frozen evaluation protocol](persona-graph-evaluation-protocol.md) and
[the execution record](../reports/persona-graph-2026-11-20/README.md). The 54
scheduled attempts are now complete (48 LLM outputs and six deterministic
guards); independent condition-hidden quality ratings are in progress.

```bash
uv sync --locked --extra dev
uv run pytest tests/test_persona_graph_integration.py tests/test_context_budget_source_types.py tests/test_engine_and_sessions.py tests/test_answer_policy.py -q
```

Source paper: [ThinkPersona](https://aclanthology.org/2026.acl-long.449/).
