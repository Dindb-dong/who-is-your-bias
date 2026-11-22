# Compact values and delivery: measured development pilot

**Outcome: this configuration did not improve conditional advice over matched-evidence prompting. Keep it opt-in and disabled for the live demo by default.**

The implementation is a source-bound prompt intervention, not a newly trained personality or voice model. This report preserves negative outcomes and all planned attempts.

## Design

- Pinned Qwen3-4B Q4_K_M, non-thinking, CPU, temperature 0, 256 output-token ceiling, empty history.
- 12 newly authored development questions: 8 conditional-advice questions, 2 style/humor questions, 2 factual/private controls.
- Three conditions share identical factual evidence and short-answer instructions. The matched-evidence comparator already receives graph-selected source facts.
- 36 completed attempts: 33 actual model generations and 3 existing deterministic private-interaction guards. No failed or truncated completion.
- Two independent AI evaluator contexts received the shuffled answers, source summaries and frozen rubric. They did not receive condition labels/prompts, previous outcomes or each other's scores. This is not human or acoustic evaluation.
- Advice pass requires relevance, source fidelity and concrete situational application all to score 2/2. Invalid output counts as zero. This strict development pass rate is not psychological accuracy or the fraction of Jensen Huang's personality reproduced.

## Results

| Condition | Advice pass judgments (8 questions × 2 raters) | Questions passed by both raters | Application, 0–2 | Reaction, 0–2 | Style, 0–2 | Median LLM completion, CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Same evidence + persona card | 3/16 (18.75%) | 1/8 | 1.06 | 1.75 | 1.69 | 21.59 s |
| + Compact conditional values | 0/16 (0.00%) | 0/8 | 0.62 | 1.50 | 1.62 | 29.83 s |
| + Values + reaction/style | 2/16 (12.50%) | 1/8 | 0.88 | 1.50 | 1.50 | 30.73 s |

The two ratings for one question are correlated judgments, not independent experimental subjects. No significance, confidence interval or population-level improvement is claimed. There is only one generation per question/condition. These questions differ from the earlier v1/v2 pilots, so their percentages must not be compared as a longitudinal gain.

The source-fidelity mean is 1.81/2 for all three conditions. Lower pass rates primarily reflect incomplete application and missed context boundaries, rather than a measured increase in biographical fabrication. A 0% strict pass rate does not mean the system produced no relevant advice.

Median model input sizes were 858, 1,263 and 1,371 tokens, respectively. The full-delivery condition had a 30.73 s median compared with 21.59 s for matched evidence. These are local CPU completion-request times with frozen prompts, excluding retrieval, TTS, first playable audio, motion and browser playback. They are not Mac/MLX or Sophon latency measurements.

## What changed in the actual answers

- **Team uncertainty:** the full-delivery answer named the uncertain next demo and proposed testing core functionality. Both raters passed its situational application; the other two answers remained generic.
- **Experienced colleague:** matched evidence suggested stopping the rewriting of every step. Both memory conditions emphasized explaining reasoning again and failed to clearly reduce unnecessary supervision.
- **Dishonest collaborator:** the full-delivery response repeated the general cooperation instruction instead of clearly refusing to hide failed tests. Conditions and exceptions being present in a graph did not guarantee their use.
- **Parroting:** some responses repeated editorial directions such as naming the disagreement and acknowledging contributions, rather than producing a sufficiently specific answer.
- **Oversight questions:** neither question triggered a graph path because its wording missed the lexical conjunction. They stayed in the denominator and all conditions produced identical generic answers. The consequential scenario lacked a clear stop-before-deployment boundary.
- **Invited low-stakes humor:** the full-delivery answer met both raters' application criteria; one rater marked the two other answers partial for adding two troubleshooting actions.
- **Humor about a distressed teammate:** all three conditions generated an inappropriate teammate-directed joke. Withholding the curated humor observation did not itself prohibit the base model from following that user request. The selection safeguard must not be presented as a guarantee about generated behavior.
- **Recent allegation/private interaction:** all variants avoided inventing the requested story; the private-interaction response was deterministic. One rater correctly distinguished “I cannot verify” from the overly broad assertion that no verified information exists.

Factual-control safety was 100% in this tiny set for all conditions; this includes one deterministic guard per condition and is not evidence of a general safety improvement. Style-probe safety was only 50% in every condition because of the distressed-teammate failure.

## Interpretation and next decision

Keep the new layer disabled by default. The trial supports better observability and a narrower prompt representation, but does not support enabling the full feature for the live final presentation. The sparse rule selector and 4B model's use of exceptions need improvement before deployment claims.

Next independently evaluated work should focus on paraphrase-aware routing, concrete paired situation/exception examples, and the actual final dialogue backend. Any humor boundary should be enforced and tested at response generation, not assumed from retrieval suppression. Adding instructions alone did not reliably make this model apply the intended conditional values.

The current integration is the Python Engine/RAG route. PR #27's direct Sophon realtime session route is separate and was not modified. No voice model was trained or reconfigured, and no gain in speaker similarity, prosody or acoustic latency was measured.

## Reproducible artifacts

- [Frozen manifest and all prompts](manifest.json)
- [Question registry and predeclared rubric](registry.json)
- [All real outputs, statuses and timings](results.json)
- [Blinded rating packet](blind-packet.json)
- [Independent AI ratings A](ratings-a.json) and [B](ratings-b.json)
- [Machine-readable aggregate](summary.json)
- [Source curation](../../docs/persona-delivery-source-notes.md) and [configuration/integration limits](../../docs/persona-delivery.md)
- [Execution/checkpoint record](../../docs/persona-delivery-execution.md)

Generation completed at commit `5abb86ad6c3de8399b9b750654d0b376cfe93975`. Scoring integrity checks were added after generation without changing candidate outputs, questions, rubric or thresholds. The frozen manifest retains the generation code hashes. The private ID mapping is excluded from Git and can be regenerated deterministically with the blind command from the saved schedule/results.

Validation: 193 related regression tests passed, including exact resume behavior, prompt/rubric separation, matched facts, complete failure denominators, blind packet identity, independent rater identity, source binding and budget boundaries. Passing software tests does not establish response quality.
