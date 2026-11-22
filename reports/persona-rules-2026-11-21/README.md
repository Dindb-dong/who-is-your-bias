# Conditional values: frozen four-condition pilot

**The initial new graph did not improve usable, source-consistent advice in
this pilot.** It produced more verbose answers, exposed internal rule labels,
and exceeded the common 256-token output limit in five of sixteen requests.
These failures are preserved. An output-guidance revision is a separate
development intervention, not a replacement for this experiment. The
[completed follow-up](development-guidance-v1/README.md) restored 8/8 completion
on the eight memory-included cases, but its 37.5% advice score remained below
the original card/RAG's 81.25% on that same subset. No value-fidelity gain has
been established.

## Actual model results

All 64 scheduled requests ran on the pinned Qwen3-4B Q4_K_M model, on CPU.
Twelve questions tested novel advice in six context-contrast pairs; four tested
unsupported news, private feelings, project measurements and invented quotes.
Two AI agents in fresh conversations independently rated 52 valid completions
without condition labels. Twelve length-limited completions remained failures.
There were no HTTP/runtime errors and no deterministic guard substitutions.

| Condition | Advice pass / 24 rater judgments | Advice pass rate | Both variants pass / 12 pair judgments | Completed / 16 requests | Request median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original card + RAG | 13/24 | 54.2% | 5/12 (41.7%) | 16/16 | 19.03 s |
| Expanded corpus + RAG | 15/24 | 62.5% | 5/12 (41.7%) | 16/16 | 19.57 s |
| Conditional flat memory | 2/24 | 8.3% | 0/12 (0%) | 9/16 | 30.09 s |
| Conditional graph memory | 7/24 | 29.2% | 1/12 (8.3%) | 11/16 | 30.82 s |

These are two judgments per question, not 24 independent questions. Advice
passes require relevance 2, source fidelity 2, concrete application at least 1,
context sensitivity 2, and no unsupported autobiography. See the
[frozen protocol](protocol.md) and [rating guide](rating-guide.md).
The two raters agreed on binary success for 45/52 valid outputs (86.5%).
This agreement does not establish psychological validity or human acceptance.

The graph-minus-original difference was -25.0 percentage points; its descriptive
six-pair bootstrap interval was [-75.0, +20.8] points. Graph-minus-flat was +20.8
points, interval [0.0, +45.8]. Six authored pairs are too few for a stable
generalization claim. Completed-advice-only pass rates were 54.2%, 62.5%, 20.0%
and 50.0%, respectively; reporting only those would hide output failures.

| Mean advice dimension (0–2; failed outputs = 0) | Original | Expanded RAG | Flat | Graph |
| --- | ---: | ---: | ---: | ---: |
| Relevance | 2.00 | 2.00 | 0.83 | 1.17 |
| Source fidelity | 1.83 | 1.83 | 0.67 | 1.00 |
| Concrete application | 1.38 | 1.58 | 0.58 | 0.92 |
| Context sensitivity | 1.71 | 1.79 | 0.67 | 1.04 |

## What the comparison does and does not show

The unchanged lexical RAG supplied no evidence for any of these 16 questions,
with either corpus. The baseline/expanded prompts still differ in snapshot
metadata, so their 8.3-point difference cannot be credited to newly retrieved
interview content. Conditional memory supplied evidence for eight of twelve
advice cases. The other four advice cases and all controls received no memory.

Graph versus original combines source selection and added conditional memory.
Only graph versus flat has matched evidence, claims, conditions and exceptions.
This experiment does not directly compare the previous six-principle graph
with v2, and its questions/rubric differ from the previous PR29 83.3% result.
There is no valid old-to-new percentage-point improvement to report.

## Observed failure modes

- All twelve truncations occurred in memory-augmented conditions. Seven flat
  and five graph outputs reached the common 256-token cap. Even text that
  appears nearly complete is counted as failed when the provider reports
  `finish_reason=length`; there are no selective retries.
- Four flat and four graph outputs contained at least one exact internal
  claim/rule ID or schema marker. This exploratory string diagnostic is not
  itself a quality score. See [execution diagnostics](execution-diagnostics.json).
- For experienced-team supervision, outputs spent much of the budget
  explaining principles instead of giving the requested practical balance.
  For a junior member missing project context, an adjacent project-phase rule
  distracted from sharing the missing constraints. Both illustrate limits of
  lexical selection and unbounded rule explanation.
- For refuted premises, the graph gave a source-consistent explanation of
  reassessing conviction. A successful example does not offset failed cases
  or establish superiority over the simpler reference answer.
- All conditions failed the request to invent and attribute an interview
  quote. A hypothetical header did not make a fabricated attribution valid.
  The model was evaluated without production guard substitutions; these are
  raw-model control results, not a full live-application audit.

Control pass rate was 5/8 judgments (62.5%) in every condition. The two raters
disagreed on whether the news refusal sufficiently challenged the event premise;
both rejected invented quotations. There was one unanimously flagged fabricated
attribution per condition among the valid rated outputs: 1/16 original, 1/16
expanded, 1/9 flat and 1/11 graph. The twelve truncated answers were unassessed
for this flag. The scorer's all-attempt `unsupported_autobiography_rate` must
not be read as evidence that unassessed outputs were free of invented claims.

## Artifacts, validation and reproducibility

- [Manifest](manifest.json): prompts, model/decoding, source bindings and schedule.
- [All recorded attempts](results.json): generated text, finish reason, usage,
  latency and integrity hashes. Generated quotes in failed controls are
  fabricated test outputs, not authentic Huang quotations.
- [Scored summary](summary.json), [rater A](ratings/rater-a.json),
  [rater B](ratings/rater-b.json), and [blinded packet](ratings/answers.blinded.json).
- [Validation receipt](validation.json): 322 relevant implementation tests;
  full Studio CI succeeded at implementation checkpoint `6b9287b`. A prior
  mobile gesture click timed out, then passed with unchanged executable code.
- [Method and twelve strategies](../../docs/persona-values-v2.md).

The private condition mapping is excluded from public Git and retained in a
verified private artifact (14,320 bytes; SHA-256
`cd516ac6eb4ea885e849cafc5c7e44c099e756f748509e3db580931f6b9521cf`).
The completed outputs were verified at `92dc45a`; their file SHA-256 is
`88af4d11d3c1ff7c7f1bed9d0cd33e22e5030769b314a6dca9d0f527f876cc46`.
The frozen manifest identity remains
`87d81bbc50dcac93b05db0a14555dc0e953d723f64a84df3a9c695397e50ab6f`.

CPU request times exclude TTS, avatar rendering, browser playback and team-Mac
conditions. No model weights were trained. The optional v2 feature remains off
by default; neither this small AI-rated pilot nor source counts justify live
promotion or a claim about Huang's real private thoughts.
