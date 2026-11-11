# Pinned larger-model development control

On 2026-11-11 the existing comparison's optional `larger` condition was executed
through an actual loopback HTTP language service using **Qwen/Qwen3-1.7B** at
revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. The official
[pinned repository](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e)
and [acquisition metadata](model-acquisition.json) identify the source; actual
downloaded inference files have SHA-256 receipts. Weights remain in the ignored
workspace cache and are not distributed with this report.

## Same frozen comparison

The plan remains
`a10a6f8f764127e964fca7c2229577b49332c78449a2a85ece8999249c2a9f30`.
The four questions, full style-condition messages, factual passages and
192-token output limit are unchanged. Requests use temperature 0, top-p 1 and
seed 42. No retrieval rerun, prompt editing or text-adapter training was added.
The larger model used CPU BF16, two Torch threads and no adapter.

The measured base parameter count is **1,720,574,976**, exceeding the baseline's
596,049,920. Its actual tokenizer contract matches the baseline. The existing
`compare_reports` validator accepted the earlier final facts/style/adapter
receipts plus the new larger receipt. See [combined comparison](comparison.json),
[frozen plan summary](plan-summary.json) and [larger raw run](attempt-02-5gb/larger/run.json).

| Condition | Verified completed / attempted | Required citations in factual-input outputs |
| --- | --- | --- |
| Existing 0.6B facts | 4 / 4 | Missing in 3 / 3 |
| Existing 0.6B with style context | 3 / 4 | Missing in 3 / 3 |
| Existing 0.6B with style context and trained adapter | 3 / 4 | Missing in 3 / 3 |
| New 1.7B with identical style context | 3 / 4 | Missing in 3 / 3 |

The new first-principles answer reached the 192-token limit and remains
incomplete. The Korean childhood-programming answer, 2026 hiring answer and
current-plan diagnostic reached normal stop with verified pre/post identities.
AI inspection of the current-plan text finds unsupported personal-plan claims;
it does not establish calibrated abstention. Human judgments remain ungraded.
The complete raw text, finish reasons and request hashes remain inspectable.

## Resource bound and preserved attempts

The first launcher failed an import-path setup check before acquisition or
inference. Its original source and receipt remain in
[setup-01-import-path](setup-01-import-path/execution.json).

The first actual model attempt successfully downloaded and loaded the pinned
checkpoint. During its first frozen question, sampled server RSS reached
**4,033,294,336 bytes**, above the configured 4,000,000,000-byte guard. The
supervisor sent SIGTERM. The provider returned text, but final health verification
failed after the server stopped, so the case correctly has **zero verified
completed generations** and three skipped cases. This was an explicit resource
guard, not an observed operating-system OOM or access failure. The literal
execution receipt's failure field is null because the condition recorder
returned normally with failure; the separate [derived outcome](attempt-01-4gb/outcome.json)
states this limitation and cites the unchanged memory log, source and receipts.

One new attempt was authorized with a **5,000,000,000-byte server-RSS guard**
after the other hybrid inference process ended. Available memory was checked
before launch; the exact observation is retained in the
[execution receipt](attempt-02-5gb/execution.json). It used the existing pinned
cache, completed all four attempts in **209.957 seconds**, and reached a sampled
server RSS peak of **4,251,389,952 bytes**. Warmup is excluded from case generation
timings. Both attempts and their exact supervisor sources are retained.

This is a bounded development comparison, not a held-out quality estimate or
an isolated speed benchmark. Conditions ran at different times; earlier
conditions had unrelated inference contention. Sampled server RSS is not total
host memory. No acoustic, voice or avatar test is part of this experiment, and
the application's default runtime remains unchanged.

The [machine-readable summary](summary.json) records the final decision and
scope. Original training wording, complete frozen prompts and the ungraded
blind-review package remain local; the public report contains generated outputs,
identifiers, hashes and metadata.
