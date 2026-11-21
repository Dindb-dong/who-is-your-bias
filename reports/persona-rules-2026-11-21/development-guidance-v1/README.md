# Guidance-only development follow-up

The response contract fixed the observed completion and internal-identifier
problems on these eight known questions. This is a **development result after
inspecting the first outputs**, not an unseen-question accuracy result or a
replacement for the [original four-condition comparison](../README.md).

## Same eight questions, unchanged evidence and rules

Every advice question that originally received graph memory was included,
whether its first answer succeeded or failed. The freezer verified byte-for-byte
equality of the user/reference messages, evidence, memory, question and history.
The model, decoding, 256-token allowance, source corpus, graph and snapshot were
unchanged. Only a generic system-message suffix changed. Exact input sizes were
1,554–1,893 tokens, within the same 8,192-token context window.

| Execution diagnostic | Original graph, same 8 cases | Revised response guidance |
| --- | ---: | ---: |
| Normal completions | 3/8 | 8/8 |
| Output-limit failures | 5/8 | 0/8 |
| Answers with exact internal identifiers | 4/8 | 0/8 |
| Median raw answer length | 206 words | 39 words |
| Median CPU model request time | 53.342 s | 41.504 s |

No retries, output rewriting or hard response cuts were used. A marker-free
answer can still narrate an abstract principle or cite an extrapolation badly.
Word counts are whitespace counts. Timing excludes TTS and rendering and can
vary with host load; it does not demonstrate real-time end-to-end operation.
The prompt's sentence/word targets remain advisory, not guaranteed constraints.

## Quality assessment

The same two AI raters apply the frozen four-dimension rubric to these new
answers after their original ratings have been sealed. This remains a known-case
development assessment. The original card/RAG reference passed 13/16 rater
judgments (81.25%) on this exact eight-case subset, and the initial graph passed
5/16 (31.25%); neither number should be confused with the full twelve-case
54.2% and 29.2% advice rates.

The revised guidance passed **6/16 rater judgments (37.5%)**, versus 5/16
(31.25%) for the initial graph and 13/16 (81.25%) for the original card/RAG on
this exact subset. Thus completion improved, but the revised graph still did
not outperform the simpler reference on source-consistent, context-sensitive
advice. The +6.25-point development change is only one additional passing rater
judgment and is not evidence of generalization.

Both AI raters passed the same three answers and failed the other five. No
unsupported autobiographical claim was flagged among these eight completed
answers; this does not certify factuality or citation accuracy. See the
[summary](summary.json), [rater A](ratings/rater-a.json) and
[rater B](ratings/rater-b.json).

The main residual failures were concrete: experienced-member advice did not
translate demonstrated expertise into reduced checking or review points;
novice advice did not give examples or a progression toward independence;
the missing-context answer emphasized project phase instead of having the lead
share the constraints; the cash-short answer did not protect payroll or specify
a cheaper/deferred learning path. One rater also downgraded its founding-story
citation as insufficient support for the delivery recommendation. Some outputs
still attached source markers to hypothetical advice despite the instruction
to distinguish source facts from new applications.

The result supports preserving this as an optional development candidate,
not promoting it as a value-fidelity improvement. Further work should test
situation-to-rule relevance and evidence-to-claim attribution on newly held-out
questions, with human source review, rather than tuning repeatedly on these
eight cases.

## Inspect and reproduce

- [Protocol](protocol.md), [frozen inputs](manifest.json),
  [eight actual outputs](results.json), and [execution diagnostics](diagnostics.json).
- The generic v2-only guidance is in
  [context_budget.py](../../../src/bias/context_budget.py).
- The integrity-checking runner is
  [persona_rules_diagnostic.py](../../../tools/persona_rules_diagnostic.py).

The exact-input manifest was saved at `b2bb53c`; actual outputs and blinded
packets were verified at `5a273ad`. Original 64-request artifacts remain intact.
The private mapping is outside Git and retained in a verified private artifact
(2,536 bytes; SHA-256
`db0ab2e4aa0e90203cc45c2783f3ceba6cc0cdafabe5a66e2f2b4ea03f1c1f8d`).

With the previously recovered runtime, reproduce the recorded requests with:

```bash
PYTHONPATH=src:. /tmp/persona-v2-runtime/.venv/bin/python \
  /tmp/persona-v2-runtime/run_with_server.py -- \
  .venv/bin/python -m tools.persona_rules_diagnostic run \
  --manifest reports/persona-rules-2026-11-21/development-guidance-v1/manifest.json \
  --output /tmp/persona-guidance-reproduction-results.json --limit 8
```

Use a distinct output path for a new reproduction; the preserved results path
resumes without rerunning recorded attempts. Runtime recovery is pinned in the
[parent runtime receipt](../runtime-availability.json). No training was run.
