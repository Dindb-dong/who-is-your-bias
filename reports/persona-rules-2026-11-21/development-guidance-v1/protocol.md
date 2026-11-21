# Development diagnostic: concise use of conditional memory

This is a narrowly scoped development iteration after reviewing the original
64-request experiment. It is **not a held-out accuracy result**. The original
manifest, outputs and sealed ratings remain unchanged in the parent directory.
The response-guidance patch and original results were preserved in
`ed015c500c2a6f233c346dae2365c8ceb46ae8f5` before this diagnostic.

The original run showed that memory-rich answers could explain internal rule
identifiers or reach the frozen 256-token limit. The follow-up adds a generic
system-message instruction to answer the user directly, apply rules silently,
retain qualifications, and default to at most two short sentences/about 60
words when no stronger turn-specific instruction exists. There is no answer
rewriting, hard truncation, new source selection, or extra generation call.

## Fixed eight-case subset

Include **all eight advice questions that originally received graph memory**,
including original successes and failures. Do not select cases according to
their answer quality. This subset was determined by the original pre-generation
memory-inclusion trace. There are no new questions in this diagnostic.

Retain the exact original question, persona, source corpus, graph, snapshot,
selected rules, factual evidence, user/reference messages and empty history.
Retain the original model/revision, CPU runtime, temperature, seed, 8,192-token
context and 256-token output allowance. The default-language guidance is the
only permitted difference. The freezer checks full non-system message equality
and requires the system message to equal its original text plus the explicitly
versioned suffix. A context-budget change that drops memory fails the freeze.

The new manifest uses `conditional-values-development-guidance.v1` and only
`guided_graph`, with eight scheduled requests in their original relative order.
It records the original manifest/results hashes and each original prompt hash.
This cannot overwrite or resume into the original results file: the existing
runner rejects a mismatched manifest identity. Each attempt is saved atomically;
errors and length-limited answers remain in the denominator without retries.

## Outcomes and interpretation

Compare the same eight original graph attempts with the eight revised attempts:

- Normal completion and 256-token-limit counts.
- Whitespace word counts and model request latency.
- Exact, case-sensitive occurrences of the frozen graph's claim/rule IDs and
  `persona_memory`, `persona-memory`, `schema_version`, `rule_ids` in raw output.
  Count these markers on all eight attempts, including truncated answers.

These are execution diagnostics. Marker absence does not establish factuality,
correct value application, or absence of invented personal experiences. Latency
also depends on host load and excludes speech synthesis and avatar rendering.

After preserving the outputs and after the original ratings have been sealed,
the same four-dimension rubric may be applied independently to the valid new
answers. Invalid outputs count as primary failures and remain explicitly
unassessed for autobiographical claims. Any resulting score remains a **known
case development score**, not an independent generalization gain. Do not replace
the original four-condition result with this smaller subset.

Blinding removes condition/attempt labels. The condition key must be written
outside the Git checkout and archived privately; only the answer packet,
template and completed ratings may be added to the public report.

## Execution entry points

`python -m tools.persona_rules_diagnostic` provides `freeze`, `run`,
`diagnostics`, `blind` and `score`. Use the existing recovered model launcher
so server and client share the command's network namespace. Preserve this code
checkpoint before freezing, preserve the exact-token manifest before running,
and preserve the eight actual outputs before rating them. Test fixtures are not
model runs or quality results.
