# PR #19 completion report

The remaining registered execution and agent review are complete as of
2026-11-13 (Asia/Seoul). All **54 matched calls** and the **six separate RAG
calls** were attempted and preserved. The actual base and completed29-adapter
API probes each passed **46/46 functional checks**. The measured response
quality does **not** support promoting the learned adapter or changing the
default wording policy. Human persona, voice and avatar likeness remain
ungraded.

This closes execution of the registered experiment; it does not claim that all
quality or deployment acceptance gates passed. [PR #19](https://github.com/Dindb-dong/who-is-your-bias/pull/19)
remains the review and current CI entry point. No merge was performed as part
of this completion work.

## Matched response results

The [execution status](jensen-evaluation-v1/execution-status.json) links all six
condition results, exact frozen plans, runtime identities and verified output
archives. Normal completion means the provider ended normally; it is not a
factuality or persona score.

| Partition | Condition | Attempted | Normal completion | Failed |
| --- | --- | ---: | ---: | ---: |
| Development | Facts | 12 | 6 | 6 |
| Development | Facts plus style examples | 12 | 5 | 7 |
| Development | Same style prompt plus learned adapter | 12 | 2 | 10 |
| Heldout | Facts | 6 | 4 | 2 |
| Heldout | Facts plus style examples | 6 | 5 | 1 |
| Heldout | Same style prompt plus learned adapter | 6 | 0 | 6 |
| **Total** | | **54** | **22** | **32** |

The failures comprise **26 length limits, one timeout and five HTTP 429 busy
rejections**. The latter six arose in the development style condition. Its
[runtime note](jensen-evaluation-v1/development-style-runtime-errors.json)
distinguishes transport failures from the runner's missing-model-field labels;
the recorded health identities remained equal. No failed response was silently
repaired, retried or promoted to a semantic sample.

The resumed final adapter condition used the exact saved 29-update run and
unchanged CPU BF16/eight-thread contract, 160 output tokens, 4,096-token context,
greedy decoding and frozen messages. All six heldout adapter calls reached the
length limit. Their raw output and bounded partial checkpoints were preserved
before final collection. No heldout reference was supplied to generation.

### Independent review and decision

The [development comparison](jensen-evaluation-v1/development-comparison.md)
and [heldout comparison](jensen-evaluation-v1/heldout-review-v1/comparison.md)
retain their full attempted denominators. The heldout reviewer received only
opaque completed samples and their permitted source pools. Its exact findings
were saved, freshly restored and published at
`e3ed0796e0124a068f84856985c35839040b4e88` before the coordinator opened the
condition key or reference answers. Subsequent reference notes are descriptive
and do not alter the original source-support labels or human-null fields.

There are **no jointly completed style/adapter pairs** in either partition.
The two completed development adapter answers used English first-person source
wording despite the applicable Korean instructions. No heldout adapter answer
was complete enough to enter the semantic review.

Among the completed heldout answers, unsupported-claim flags occur in **2/4
facts** and **3/5 style** responses; applicable factual citation-presence
violations occur in **3/4 facts** and **4/5 style**. All nine completed answers
have no citation markers. Two are hypothetical normative advice, so their
citation absence is not itself treated as a factual defect. Missing requested
coverage remains separate from unsupported claims. These are agent judgments
on a small surviving subset, not aggregate human ratings or a causal ranking.

The saved adapter remains an optional experimental artifact. The additional
normal completion in the heldout style condition does not establish improved
support or justify a default prompt-policy change. A different future model or
budget would need a newly registered experiment; these heldout results were
not used to tune this run.

## Separate retrieval-to-generation diagnostic

The [six-call default lexical RAG diagnostic](jensen-evaluation-v1/heldout-rag-v1/README.md)
completed **4/6**, with two length failures. It used its actual frozen retrieved
passages, including two genuinely empty evidence pools, and has its own
[source-only agent review](jensen-evaluation-v1/heldout-rag-v1/review/support-findings.md).
It is outside the 54-call matched denominator.

The review distinguishes a supported deep-learning core from unsupported
additions and missing citations, identifies fallback-format violations in the
empty-evidence answers, and preserves an appropriate jacket-origin limitation.
Conditional speculation allowed by the actual fallback prompt is not
automatically relabeled as a false factual claim. Neither matched hand-selected
passages nor withheld references were used to repair retrieval failures.

## Actual neural API behavior

The [M7 comparison](jensen-text-live-v1/comparison.md) and the exact
[base](jensen-text-live-v1/base-result.json) /
[adapter](jensen-text-live-v1/adapter-result.json) receipts document actual
HTTP client → main API → local neural provider invocations.

| Functional observation | Base | Completed29 adapter |
| --- | ---: | ---: |
| Passed checks | 46/46 | 46/46 |
| Completed conversation turns | 4 | 4 |
| Active-token TCP disconnect | Passed | Passed |
| Provider idle after disconnect | 309.943 ms | 312.444 ms |
| Partial answer excluded from session memory | Passed | Passed |
| Same-session recovery, deletion and clean server teardown | Passed | Passed |

Each run completed five responses including recovery. First-token observations
were roughly **14.2–18.4 seconds** for base and **22.6–26.9 seconds** for adapter
on this CPU host. These are single sequential observations with different
outputs; they do not establish comparative model speed or real-time delivery.
Voice, transcription and selected motion were disabled.

All ten completed API responses emitted zero citation records. Four adapter
answers have the same response hash: the initial question, follow-up, explicit
correction and recovery all produced the same general CUDA definition. The
topic-switch answer merely restated the AI-factory term. The API's correct
session resolution and transport do not demonstrate meaningful contextual
answer quality. These limited output observations reinforce the decision to
keep the candidate optional; no human semantic or persona score was assigned.

### Corrected harness defect and retained first failure

The [initial base invocation](jensen-text-live-v1/base-first-preflight-failure.json)
loaded the correct model during warmup but rejected its identity before any
main-API case. The service includes `schema_version=language-model-identity.v1`;
the harness incorrectly compared that versioned object with the five-field
frozen payload alone.

The fix requires the exact supported schema envelope and every unchanged
identity field. Tests reject missing or changed schema versions, missing fields
and unexpected fields. The initial failure was preserved before the correction
and new invocations. [Corrected preparation](jensen-text-live-v1/schema-corrected-preparation.json)
binds the one changed harness file and the other 49 unchanged inputs. Models,
questions, snapshots and generation limits were unchanged.

A [separate two-request cancellation diagnostic](jensen-text-live-v1/cancel-recovery/README.md)
was registered as a fallback for a corrected original run that failed before
cancellation. Both original corrected runs reached and passed that stage, so
neither condition was eligible and the fallback was not executed.

## Preservation and validation

The [resumption record](jensen-recovery/pr19-resumption-v1/README.md) documents
the recovered original corpus, frozen experiments and completed adapter. Every
new reusable result was archived, uploaded, materialized into a distinct path,
and verified by archive byte count/SHA-256 plus every payload hash. The
[central manifest](../data/runs/jensen-2026-11-13/artifacts.json) retains the
actual object IDs and receipts. Source updates and public reports were saved in
small commits, fetched back and compared byte-for-byte against the remote head.

The final integrated targeted command passed **135 tests**:

```sh
.venv/bin/python -m pytest -q tests/adaptation_evaluation \
  tests/adaptation/test_supervise.py \
  tests/test_runtime_text_live.py tests/test_runtime_text_cancel_recovery.py
```

The [validation receipt](jensen-recovery/pr19-resumption-v1/final-validation.json)
binds this command and output. The Studio CI workflow now includes both neural
runtime regression modules in its Python gate, alongside TypeScript, web tests,
production build and desktop/mobile browser checks. CI is tracked separately
on PR #19 against its exact current head. The completion description must cite
the actual checked head; older run receipts remain historical evidence.

Finalization retained a [failed CI attempt](jensen-recovery/pr19-resumption-v1/ci-first-failure.json):
one desktop recorded-audio check observed a 754.924 ms rendered/media clock gap
against its unchanged 750 ms gate. Web/build, 134 CI Python tests, 30 historical
browser cases and 21 Jensen cases passed in that run. The
[raw-frame diagnostic](jensen-recovery/pr19-resumption-v1/browser-clock-isolation.json)
shows a previous-frame renderer sample across a 757.600 ms frame interval,
followed by a smaller gap. Concurrent desktop/mobile software rendering was
observed; CPU contention is an inference, not a measured CPU profile.

The Jensen suite now uses one worker to isolate those render loads. Every
assertion, all 22 Jensen cases, zero retries and the 750 ms threshold are
retained. Application code and frozen model inputs are unchanged. The failed
archive remains saved; a later CI result is a separate observation.

## Earlier deliverables retained

The [execution status](../docs/jensen-replication-status.md) retains the earlier
verified work: 554 original Q&A from nine events, 113 selected text examples,
29 saved text optimizer updates, 16 voice updates, eight generated EN/KO voice
samples, the optional 2.53 MB fitted avatar, measured motion clips, eleven
complete publisher video packages and the separate derived NTU address.
These were not retrained or reacquired to finish the remaining evaluation.

Both retrieval candidates remain optional after missing their registered
promotion gate. Human text/persona, voice and avatar likeness remain ungraded.
Full publisher recordings, reference transcripts and model weights remain in
their recorded artifacts; public reports expose provenance and measured
outcomes without duplicating those full private payloads.
