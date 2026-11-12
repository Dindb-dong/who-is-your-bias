# Jensen bilingual retrieval development comparison

The inputs were frozen in `development-cases.json` before the candidate code
was implemented. Both policies index the same 410 factual records from the
saved original-dialogue-v4 corpus. The 24 questions contain 16 supported
technical/values questions (eight Korean and eight English) and eight controls.
The relevant-ID lists are non-exhaustive development labels, not entailment judgments.

| Policy | Relevant top 1 | Relevant top 4 | False-hit controls | Median search |
| --- | ---: | ---: | ---: | ---: |
| Archived lexical | 1/16 | 3/16 | 0/8 | 10.40 ms |
| Jensen concept lexical v1 | 9/16 | 11/16 | 0/8 | 14.02 ms |

The candidate improves these exposed development cases but **fails its frozen
80% recall gate and regresses the English TSMC case**. It remains opt-in and the
runtime default is unchanged. CUDA Korean, TSMC English, leadership English and
both top-five-email questions miss the recorded relevant IDs. These failures
are retained in `comparison-v1.json`; no score or reference set was edited to
make the result pass. Seven deterministic searches per question measured lookup
only, without language generation or target-device latency.

The opt-in setting is `BIAS_RETRIEVAL_BACKEND=jensen-lexical`. Its finite bilingual
concept dictionary preserves unknown content words, source-year filtering,
training-only eligibility and the shared unavailable-endorsement/private-data
guard. Jensen/Huang names alone do not count as subject evidence. It requires a
Jensen corpus and preserves its policy when temporal filtering rebuilds an index.
It is not an embedding model or a claim-support judge.

Checks at this checkpoint: 111 focused tests passed, including source isolation,
unsupported relationships, unknown-word coverage, subject binding and default
retrieval regression. A pinned multilingual embedding comparison is next; this
report does not inherit the old PR #17 model/API scores as a new measurement.

Reproduce the source-text-free report after restoring the v4 corpus:

```bash
PYTHONPATH=src uv run python -m tools.retrieval_development \
  --corpus /path/to/original-dialogue-v4/corpus \
  --protocol reports/jensen-retrieval-v2/development-cases.json \
  --output /path/to/new-comparison.json
```

No held-out No Priors reference was indexed or used for tuning. Citation support,
generated response quality and human persona fidelity remain separate evaluations.
