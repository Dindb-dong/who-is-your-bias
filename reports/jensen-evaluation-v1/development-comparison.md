# Development comparison: execution and blinded agent assessment

**The completed29 adapter is not supported for default promotion by this development run.**
It completed two of twelve responses under the unchanged160-token budget, and
both completed answers used English first-person source wording despite the
applicable Korean instructions. This decision is bounded to this execution;
heldout results, actual neural session checks and human likeness remain separate.

The [machine-readable comparison](development-unblinded-comparison.json) binds
all cases and opaque sample IDs to their exact outputs. The [blinded findings](development-agent-review-blinded.md)
were saved and freshly restored at `24a7bad0f4897b338c6d180f764f41080debd481`
before the coordinator opened the condition key. The original findings were not
changed after that linkage. This is an agent semantic assessment; human ratings
remain null and no persona-fidelity score is supplied.

## Complete denominators

| Condition | Attempted | Normal completions | Failures | Language satisfied among completed | One sentence satisfied among applicable completed |
| --- | ---: | ---: | --- | ---: | ---: |
| Facts only | 12 | 6 | 6 length limits | 6/6 | 0/3 |
| Facts plus style examples | 12 | 5 | 1 length limit,1 timeout,5 HTTP429 rejections | 5/5 | 0/3 |
| Same style prompt plus learned adapter | 12 | 2 | 10 length limits | 0/2 | 0/2 |

All36 attempts,13 normal completions and23 failures are preserved. A normal
provider stop does not establish a correct answer. The one-sentence denominator
contains only completed responses governed by that Korean instruction; general
control prompts are not assigned it. Failed responses were not silently repaired,
retried or counted as completed semantic samples.

Style's five HTTP429 responses had no model field. The frozen runner gave them
an identity-validation label, while separate health identities remained equal.
The [runtime interpretation](development-style-runtime-errors.json) retains the
original transport errors and distinguishes them from actual model identity.
This transient admission/cancellation sequence prevents treating all completion
differences as an effect of wording examples alone.

## What can actually be paired

Only three cases completed in both facts and style: values, private information
and future information. Both values answers merely echoed the question; the
future answers were byte-identical and included unsupported negative claims.
The private-information answers withheld the requested private material while
asserting unsupported public unavailability.

Facts and adapter have one jointly completed case: follow-up. Facts gave a Korean
paraphrase with technical imprecision; adapter reproduced an81-word continuous
factual-source span in English first person without a citation. The latter
contains relevant mechanism information, but violates the requested language and
attribution. There are no jointly completed style/adapter cases. The small,
differing completed subsets do not support a general style or quality ranking.

The adapter's other completed case,TSMC, copied a17-word first-person trust
fragment and repeated emphasis while omitting the requested manufacturing,
delivery and service explanation. Its four unsupported-request control calls all
ended by length, leaving no completed adapter control response to grade.

## Other observed limits

Across all13 completed responses,eight had the restrictive Korean instruction;
all eight exceeded one sentence. Two values responses supplied no substantive
answer. Supported organization content still had sentence/citation-placement
problems. Technical and Sega responses had omissions or unsupported additions.
Five completed control answers acknowledged an evidence limit but also made
unsupported absence/current/future claims. Correct citation IDs did not by
themselves establish entailment or requested coverage.

All development reference answers are null. No similarity value or penalty was
invented for that absence. Source-copy observations are descriptive and do not
establish a causal memorization mechanism. Candidate wording examples were
available to the reviewer as a common comparison pool, without disclosing which
outputs actually received them.

The original complete outputs,failures,model identities,timings and private keys
are in the [verified collection](development-collection.json); exact semantic
review files are in the [review preservation receipt](development-agent-review-preservation.json).
The registered heldout evaluation continues without tuning on these outputs.
