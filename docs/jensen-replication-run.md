# Jensen persona, speech, and avatar execution

> Status as of 2026-11-13: the execution session was interrupted. This file
> preserves the original plan. Read the [recovery and remaining-work handoff](jensen-replication-handoff.md)
> for verified baseline evidence, reported intermediate progress, and resumption gates.

## Scope and baseline

This work continues merged PR #17 at `3038629f3c4aeefff568684a6a7ba10df2b61db8`.
That baseline has 32 editorial evidence records from nine publisher sources,
Korean retrieval/API experiments, a designed Jensen avatar, and isolated persona
bindings. It has no Jensen speech adaptation or acquired training videos.

The requested deliverable includes actual source acquisition, original-dialogue
persona adaptation, speech adaptation, collected video used in the pipeline,
a newly created Jensen avatar, and response experiments. The existing Musk
artifacts remain historical reference implementations.

## Execution plan

1. Acquire publisher transcripts and a bounded working set of source media.
   Record original URLs, event families, speaker labels, byte counts, hashes,
   timestamps, and the difference between transcript and media clocks.
2. Freeze event-level training and evaluation partitions before adaptation.
   Events already exposed in PR #17 remain development/training events.
   Original answers and full transcripts stay in ignored local data; public
   reports contain reproducible acquisition recipes and measured summaries.
3. Export complete Jensen dialogue and train a real text adapter with pinned
   model/tokenizer revisions. Record accepted examples, optimizer steps,
   artifact hashes, loss, exclusions, and execution settings.
4. Extract and acoustically align original Jensen speech, export a speaker-bound
   dataset, execute a bounded voice-weight adaptation experiment where the
   available CPU runtime supports it, and generate English/Korean samples.
   Voice conditioning and optimizer-based adaptation are recorded separately.
5. Create a new avatar from the collected reference material, connect its facial
   channels and existing eye/head attention behavior, and inspect the rendered
   result. Report the actual fitting method and identity/animation limitations.
6. Freeze matched response experiments with identical evidence across base,
   style-context, and adapted conditions. Record raw generated answers,
   completion/latency, citations, language quality, and held-out source isolation.
   Include unsupported/current/private-information controls.
7. Run focused regression tests and application checks, publish the resulting
   code, bounded generated assets, and a measured report in a pull request.

## Completion evidence

The final report will distinguish acquired bytes, prepared datasets, executed
optimizer updates, generated media, integrated runtime behavior, and unmeasured
quality. A completed training run does not by itself establish persona or voice
similarity. No human evaluation scores will be invented. A candidate adapter is
promoted only if its measured behavior supports using it.

## Environment

The interrupted session's plan recorded a Linux host with nine logical CPUs
and approximately 10 GB RAM, without CUDA or Apple MPS. These are historical
execution assumptions, not a guarantee about the next host. Recheck hardware,
disk, process lifetime, and persistent checkpoint storage before resuming.
Coordinate heavy training, speech, and rendering within the actual memory budget.

## Results

The audit found PR #18 still at its original plan-only commit, with no later
implementation connected to any audited remote branch. The interrupted dialogue
reported acquisition, preparation, training, and avatar work, but its supporting
artifacts were not recovered in the audit. Those claims remain pending recovery
and verification or a new reproducible run; they are not completed deliverables.

PR #18 now preserves [repository operating rules](../AGENTS.md) and the
[detailed handoff](jensen-replication-handoff.md). Its documentation-only merge
does not establish completion of text adaptation, speech adaptation, new avatar
generation, or response-quality experiments. Follow the handoff's M0-M7 gates
and save verified remote checkpoints throughout any later implementation.

