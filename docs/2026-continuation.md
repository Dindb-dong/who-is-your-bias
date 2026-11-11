# 2026 continuation: execution evidence and remaining acceptance work

Recorded on 2026-11-11, following the [persona roadmap](2026-persona-roadmap.md)
and the [PR #13 implementation](2026-implementation.md).

PR #13's final head, `13210fe0d26feeac3b8e3b5ed1a249a3b19e3f1f`, passed the
[final pull-request CI run](https://github.com/Dindb-dong/who-is-your-bias/actions/runs/43008779666),
including all 30 desktop/mobile browser scenarios. It was merged into `main`
at `cc23f47f5dfa25cced290e90a8f8c4ef9db22181` before this continuation.
Its historical experiments, failed pilot, prepared presentation, and source
recipes retain their original identities.

## What this continuation adds

| Work | Implemented and executed | Acceptance boundary |
| --- | --- | --- |
| Text adaptation | Completed CPU LoRA epoch; safe adapter loading; exact base, tokenizer, dataset, and output-file verification | A small development cohort; no validated improvement in likeness or answer quality |
| Controlled model comparison | Frozen facts and style context; actual base/style/adapter calls; raw failures; model/runtime identity checks; blinded review export | Four exposed development questions; human judgments remain ungraded |
| Additional original dialogue | Six official congressional hearing volumes; 41 retained exchanges; combined 15-event, 1,018-record corpus | Publisher labels and conservative extraction, not completed human review; still below the 20–30-event target |
| Dialogue | English/Korean date-only references, historical corrections, original-topic references, and preserved temporal context | Declared resolver rules; source-preview regression tests, not general coreference or model-quality validation |
| Evaluation isolation | Reserved-event acquisition, evidence/reference hashes, corpus overlap checks, and evaluation-only records | Inspected Lex pages yielded no acceptable new transcript cases; fresh evaluation count remains zero |
| Live neural transport | Real local LLM and Qwen3-TTS request; validated PCM hashes, order, totals, and provider identities | CPU observation; no browser playback, speaker, avatar, or intended-device latency acceptance |
| Avatar acquisition | Current official metadata and download-path checks for three free candidates | All three downloads require account authentication; no actual Musk package qualified |

## Original sources and reproducible data

Reacquiring the nine original dialogue events produced the same **977 selected
questions, answers, and completeness labels** as the earlier build. The YC2016
publisher document's canonical transcript hash changed. The old recipe correctly
rejected it; that check was retained. A
[new acquisition comparison](../reports/data-expansion/2026-reacquisition-v1/comparison.json),
[recipe](../reports/data-expansion/2026-reacquisition-v1/materialization-recipes.json),
and [actual-cutoff snapshot](../reports/data-expansion/2026-reacquisition-v1/persona-snapshot.json)
preserve the difference. Raw acquisition times and corpus digests can change
even when every selected question/answer remains identical.

The congressional extension separates live, explicitly attributed oral
discussion from prepared statements, witness biographies, procedural turns,
and later written answers. The additional build contains **39 complete
exchanges and two interruption excerpts** across six events. The combined
build contains **1,018 records across 15 families**, with **777 factual records,
961 style examples, zero accepted aligned-speech records, and zero accepted
visible-behavior records**. Exact normalized deduplication removed no records.
These are extraction counts, not human-approved quality scores.

Six natural questions were fixed before testing the added sources. The
[lexical diagnostic](../reports/persona-data/combined-2026-v2/api-availability.json)
found the intended passages in 2/6 cases. With the previously documented E5
and mMARCO revisions and unchanged 0.78 dense / −3 reranker gates, the
[actual hybrid API run](../reports/persona-data/combined-2026-v2/hybrid-api.v2.json)
completed all six requests and cited the intended passages in **5/6**. The
2004 prize case passed the dense gate but received a −8.3775101 reranker score,
so it abstained; lexical had found that passage. No threshold or question was
retuned after those results. This is a small training-event development
diagnostic, not fresh held-out retrieval accuracy. The
[source-extension guide](congressional-data.md) records the full comparison,
actual model/runtime identities, and the earlier inventory-check failure.

The comparison and training below deliberately retain the earlier nine-event
corpus. Adding hearing documents afterward does not silently change their
frozen evidence, data identities, or prompts. The later combined corpus has its
own recipe and snapshot at `2026-11-11T14:08:38.928202+00:00`, digest
`57cead316fd009be5d19e34c4a70a3df0290a67d4a90192b5d1e0c58c1bfc622`.

## Actual text training and serving

The [verified adapter runtime](text-adapter-runtime.md) fills the previous gap
between exporting data, training a LoRA adapter, and actually selecting that
adapter in the API's language service. Operator configuration selects one
completed adapter. User completion requests cannot select a filesystem path
or substitute a base checkpoint. Invalid or incomplete artifacts fail visibly.

The loader verifies the immutable base revision and actual checkpoint files,
all adapter/tokenizer files, training-only data provenance, and the tokenizer's
vocabulary, special tokens, template, and rendered English/Korean probes.
Training, serving, and exact prompt-budget tokenization use
`enable_thinking=False`. Health distinguishes configured, loaded, and verified
state and reports the observed base parameter count and execution stack.

### Final development epoch

| Setting or measured result | Value |
| --- | --- |
| Base | `Qwen/Qwen3-0.6B` |
| Immutable revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Selection | Seed 42; up to eight complete exchanges from each of nine training families |
| Exported / actually trained | 72 / 43 examples; 29 whole examples excluded by the 256-token limit |
| Epochs / optimizer steps | 1 / 11 |
| Adapter | Rank 8 LoRA, all linear layers; 5,046,272 trainable parameters |
| Optimization | Learning rate 0.0002; batch size 1; gradient accumulation 4 |
| Execution | CPU BF16; gradient checkpointing; two Torch threads |
| Actual trainer duration | 245.732 seconds |
| Receipt SHA-256 | `dd1e049aa38ea22e97188da4ee001b0ac0df1e493af22aef30177dc70cb06cf6` |

The selected examples were fixed before this run. Length filtering changes
their event balance; the [training summary](../reports/adaptation/2026-v1/training-summary.json)
reports the actual accepted count per family. All nine remain represented.
This is a complete epoch over those **43 examples**, not over the 595-example
larger export or the later combined corpus.

The [original completed receipt](../reports/adaptation/2026-v1/attempts/balanced8-one-epoch.json),
[process receipt](../reports/adaptation/2026-v1/balanced-process.json),
[matching training source archive](../reports/adaptation/2026-v1/training-implementation.tar.gz),
and [installed environment](../reports/adaptation/2026-v1/model-environment.txt)
record the executed computation. The separately delivered checkpoint package
contains the actual adapter and tokenizer, with its
[archive identity](../reports/adaptation/2026-v1/checkpoint-package.json).
Base weights and full original training wording are not bundled in that package.

Earlier attempts are preserved. The first float32 process exited with code 137;
the cause was not established as an operating-system OOM. Two slower full-cohort
CPU attempts were interrupted, with external exit code 130. Their literal last
receipts still say `running`; they are not complete or eligible for serving.
A separate four-step BF16 smoke completed, but its full exact trainer-source
snapshot was not preserved. Its role is an engineering smoke; the final
balanced epoch above is the reproducible training artifact.

## Controlled generation and negative evidence

The [new comparison protocol](adaptation-comparison.md) preserves the historical
A/B/C study and introduces separate `facts`, `style`, and `adapter` conditions.
Each condition receives the same selected factual passages. The latter two
receive identical complete messages; only the loaded adapter changes. Actual
base checkpoint, tokenizer, generation policy, and observed runtime identity
must match. The frozen implementation identity includes the language service,
adapter loader, evaluator, supervisor, and generation/observation code.

The four development questions cover childhood programming in Korean, first
principles in English, the published 2026 interview's hiring criteria, and a
current Mars-by-2030 request without current factual evidence. The three
answerable cases use explicitly fixed training-record IDs. This isolates
generation controls and does **not** measure retrieval effectiveness or fresh
held-out quality. The fourth case is a deliberate direct-model diagnostic:
the production API's earlier no-current-evidence gate remains active.

The initial four-step-adapter comparison attempted all 12 generations. Base
completed 4/4; style and adapter each completed 3/4, with one token-limit ending
each preserved as incomplete. The base and style conditions generated an
unsupported current-plan answer despite the empty current-evidence context.
This is a concrete reason to retain the API's evidence gate. Transport
completion does not establish correct factual grounding. See the
[preserved pilot comparison](../reports/adaptation/2026-v1/pilot-comparison/comparison.json)
and [exact pilot serving/evaluator source archive](../reports/adaptation/2026-v1/pilot-serving-implementation.json).

The final balanced-adapter comparison uses a new frozen identity,
`a10a6f8f764127e964fca7c2229577b49332c78449a2a85ece8999249c2a9f30`.
The [completed final report](../reports/adaptation/2026-v1/final-comparison/comparison.json)
records another 12 actual generations: base completed 4/4, style 3/4, and
adapter 3/4. Both incomplete outputs were the Korean question ending at the
192-token limit. **All nine evidence-bearing outputs lacked the required
citation markers.** The adapted Korean answer repeated a phrase until the
limit; its two English factual answers became short, nonspecific statements.
These observed failures do not support promoting this adapter to the default
runtime. The current-plan diagnostic produced a short denial in the adapted
condition, but that single output does not establish calibrated abstention.
The [development decision](../reports/adaptation/2026-v1/release-decision.json)
separates these checks and explicitly labeled AI inspection from still-null
human quality ratings. The adapter remains an opt-in engineering artifact.

Both studies retain raw generated
text, finish reasons, actual request hashes, failed attempts, and ungraded
human-review fields. No completion rate, training loss, surface word overlap,
or valid citation identifier is presented as measured persona improvement.
Timing is sequential and not counterbalanced; concurrent development inference
also makes these observations unsuitable for a model-speed ranking.

### Larger base control

The [additional larger-model comparison](../reports/adaptation/2026-v1/larger-comparison/README.md)
uses the same frozen style-condition messages and generation policy with
`Qwen/Qwen3-1.7B` at revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Its observed base parameter count is
1,720,574,976, compared with 596,049,920 in the smaller base, and the tokenizer
contract matches. CPU BF16 and two threads remain the same.

The completed attempt verified **3/4 generations**; the English first-principles
answer ended at the token limit. All three evidence-bearing outputs also lacked
required citations. Its current-plan answer was an unsupported personal denial,
so it is not evidence of calibrated abstention. This control does not establish
a successful quality fix or justify changing the default model.

An initial attempt hit a deliberately configured 4,000,000,000-byte server RSS
guard at 4,033,294,336 bytes. That incomplete attempt remains recorded as a
resource-budget interruption, not an OOM or model incompatibility. After the
hybrid process exited and available memory was recorded, one separate attempt
used a 5,000,000,000-byte guard and completed all four requests, peaking at
4,251,389,952 bytes. The
[four-condition report](../reports/adaptation/2026-v1/larger-comparison/comparison.json)
includes the larger control; the earlier three-condition report retains its
original identity and scope.

## Dialogue and fresh evaluation

The [dialogue suite](dialogue-evaluation.md) contains 29 scenarios and 100 steps.
It covers explicit earlier-topic references, short English/Korean follow-ups,
date corrections, before/through/exact-year intent, forecast targets, current
abstention, topic switches, source grounding, and context-budget pressure.
The fixes reserve original user references and temporal anchors even when
ordinary history is dropped. A temporal snapshot selects the admissible period
before lexical search, avoiding a second exact-year filter on a forecast target
or a historical upper bound. The archived lexical experiment remains unchanged.

The [held-out acquisition path](heldout-evaluation-data.md) verifies reserved
event families, original source spans, duplicate/overlap checks against the
exact runtime corpus, and incompatible training projections. The inspected
Lex episode pages did not provide usable speaker-labeled source transcripts.
They were rejected rather than turned into invented question/answer references.
**Accepted new held-out cases remain zero.** Development outputs are not renamed
to test evidence, and no human judgments have been filled automatically.

## Actual live neural transport

The [live benchmark](live-stream-benchmark.md) checks real HTTP delivery from
the main API, with actual local Qwen3-0.6B language inference and Qwen3-TTS
speech synthesis. Digital PCM hashes are now exposed per segment so the client
can verify the actual streamed bytes instead of comparing them with a WAV hash.
The recorder rejects wrong identities, missing/misordered segments, malformed
audio, byte/frame/sequence mismatches, terminal-total errors, and source previews
presented as real model generation. It records provider identity before and
after the request and supports explicit cancellation/release probes.

One bounded cold neural request completed with **first text at 7,093.753 ms,
completed text at 9,888.115 ms, first PCM at 52,715.145 ms, and verified stream
completion at 52,721.974 ms**. The first setup query had fallen back to a source
preview and was correctly rejected as a neural benchmark. Both attempts remain
recorded. An acquisition attempt also hit a temporary-filesystem limit and was
retried on the workspace volume, without changing the selected model revision.

The neural services each used two CPU threads. The controlled text comparison
overlapped on the same execution machine, so this is a real integration
observation under development load, not an isolated latency benchmark. It does
not meet a responsive voice-conversation target. First PCM followed the final
text result, so this request does not demonstrate overlapping text/audio delivery.
See the [actual neural report](../reports/live-stream/2026-v1/report.md) and its
[raw-receipt index](../reports/live-stream/2026-v1/receipt-manifest.json).
Browser playback completion,
audible speaker latency, perceptual similarity, avatar synchronization, and
intended-machine performance remain unmeasured in this request. The source
voice reference retains its candidate, human-unreviewed status; no new speech
training or accepted speech hours are claimed.

## Actual avatar acquisition boundary

The [acquisition follow-up](avatar-acquisition-followup.md) saved current
official metadata for all three shortlisted free candidates. All three official
download requests returned HTTP 401 requiring account authentication. No package
was acquired or qualified. The next concrete input is an original archive
downloaded through the authorized official account, together with its license
record, for the existing conversion, structural inspection, and same-audio
browser audition. The bundled generic proxy remains the runtime asset.

## Remaining roadmap acceptance

The project now has an executed text-adapter path, more original sources,
controlled generation tooling, broader dialogue regression coverage, and one
new real neural streaming observation. Completing the product target still
requires more independent events, fresh reviewed evaluation transcripts,
accepted aligned speech, speech adaptation and listening tests, a usable
Musk-specific visual asset, and intended-device audiovisual measurements.
Human assessment of grounding, language/voice likeness, facial identity, and
naturalness remains open. These are explicit remaining tasks, not inferred
successes from engineering checks.

## Engineering validation

The [local validation receipt](../reports/continuation/2026-v1/validation.json)
records **786 Python tests passed**, two optional presentation-test skips
(Torch and Pillow are absent from the default app environment), and one upstream
Starlette/httpx deprecation warning. TypeScript, **181 web tests**, the production
build, **four avatar-structure tests**, corpus validation, and the preserved
historical retrieval regression also passed. Actual neural training, generation,
and speech were separately executed in the documented optional environments.

The new branch's browser checks and exact final commit status are tracked in
the pull request's CI checks. A prior PR's successful run is not substituted for
validation of a later commit.

The first continuation CI run remained in FFmpeg package installation for several
minutes, whereas that step took 21 seconds in the prior successful PR #13 run.
The workflow now reuses FFmpeg when present and otherwise performs a bounded
package installation with explicit connection timeouts and a four-minute step
limit. FFmpeg was not preinstalled on the inspected earlier runner. The actual
browser suite and its job deadline were retained; setup failure remains an
explicit failed check.
