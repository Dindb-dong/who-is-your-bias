# Actual neural-text functional results

The corrected original M7 probe passed **46/46 checks for base and 46/46 for the
exact completed29 adapter**. Each condition ran once through the actual HTTP
client → main API → local neural language service. Each completed four session
turns, closed a fifth request at an actual nonempty token while the verified
model was busy, and completed one recovery response in the same session.

The [base result](base-result.json) and [adapter result](adapter-result.json)
contain the authoritative measurements and archive receipts. Both archives were
freshly restored and all **12 payload files per condition** verified. The
[machine-readable comparison](comparison.json) preserves exact timings,
response/trace hashes, loaded identities and stage outcomes without generated
text or private source passages.

## Functional behavior and memory

| Check | Base | Completed29 adapter |
| --- | --- | --- |
| Original question, follow-up, correction, topic switch | 4 completed, expected resolution methods | 4 completed, expected resolution methods |
| TCP disconnect after first active model token | Passed | Passed |
| Provider observed idle after disconnect | 309.943 ms | 312.444 ms |
| Interrupted answer omitted from session memory | Passed | Passed |
| Recovery completes as the fifth stored turn in the same session | Passed | Passed |
| Session deletion and provider idle after recovery | Passed | Passed |
| Main API and language child stopped without forced kill | Passed | Passed |
| Language-process peak RSS reported after recovery | 5.047 GiB | 5.850 GiB |

Both runs kept their loaded identity unchanged through recovery. The memory
values are the language process's recorded high-water RSS, including prior model
work, not total application memory. Voice, transcription and selected motion
were disabled in this probe.

## Five completed response timings

Values below are seconds from each actual main-API HTTP request. First-token
time means the first nonempty SSE token. Request times exclude warmup.

| Completed response | Base first token | Base HTTP completion | Adapter first token | Adapter HTTP completion |
| --- | ---: | ---: | ---: | ---: |
| Initial question | 18.408 | 25.129 | 24.588 | 28.190 |
| Follow-up | 15.459 | 22.138 | 24.294 | 28.215 |
| Explicit correction | 16.720 | 24.885 | 26.874 | 31.351 |
| Topic switch | 14.248 | 21.325 | 23.883 | 25.009 |
| Same-session recovery | 18.444 | 25.784 | 22.563 | 26.242 |

Whole-invocation elapsed time was **152.062 seconds** for base and **180.836
seconds** for adapter, including startup, warmup and lifecycle work. These are
single sequential observations with different generated outputs; they cannot
rank model speed or establish real-time performance.

## Exact identity and unchanged protocol

Both providers loaded `Qwen/Qwen3-1.7B` revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, with the same verified base checkpoint
manifest and tokenizer contract. The adapter additionally reported configured,
loaded and verified state for run
`e1102328f6cc85e9c096dc554e6255465d67870706df5d3a4f03bf091c6e3f42`,
29 optimizer updates and the frozen nine-event dataset.

The [BF16 protocol](protocol-bf16-v2.json) remained byte-identical at SHA-256
`6c699aaa81a5f90919be7c78579f400c58590db5924abd045e6c76bcf9cd331e`:
CPU BF16, eight threads, 4,096-token request context, 160 output tokens,
240-second request budget and 60-second cancellation-recovery budget. Observed
software was Torch `2.11.0+cpu` and Transformers `4.57.3`; the adapter also used
PEFT `0.18.1`. The model's reported native context capacity is separate from the
smaller enforced request budget.

The [first base attempt](base-first-preflight-failure.json) remains a separate
failed invocation with zero main-API cases. Its warmup loaded the correct model,
but the harness compared the service's versioned identity envelope with an
unversioned expected payload. The corrected harness requires the exact
`language-model-identity.v1` envelope and every frozen payload field. Its SHA-256
is `6421b108f7ec7d390507ba3015b7c55ed824c4c2e76e729fa5bc6f598c8ac014`.
The correction changed no question, model, dataset, snapshot or decoding limit.

## Interpretation

This establishes the bounded neural transport, session provenance, cancellation,
memory integrity, recovery and teardown behavior. All five completed responses
in each condition emitted **zero citation records**. Consequently, passing the
citation-binding checks does not establish citation coverage or factual
support. Four adapter completions—initial question, follow-up, correction and
recovery—also share the same response hash; this observation is retained without
assigning a semantic adequacy or persona score. The preserved generated text
shows the same general CUDA definition after the explicit correction; the
topic-switch answer merely restates the AI-factory term. Session-resolution
checks therefore do not establish that the response meaningfully uses context.

Semantic quality, persona fidelity and human likeness remain ungraded here.
These functional runs neither replace the separate M6 response evaluation nor
justify model promotion, voice/avatar integration or target-device acceptance.

Both conditions reached and passed cancellation in the original corrected
probe, so the [separate two-request diagnostic](cancel-recovery/README.md) is
**ineligible and was not executed**. The 38 identity/transport regression tests
are model-free fixtures, not additional neural trials.
