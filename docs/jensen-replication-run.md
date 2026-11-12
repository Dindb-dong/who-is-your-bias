# Jensen persona, speech, and avatar execution

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

The execution host is Linux with nine logical CPUs and approximately 10 GB RAM.
CUDA and Apple MPS are unavailable. Heavy training, speech, and rendering tasks
are coordinated to fit the memory budget; independent acquisition and code work
proceed in parallel.

## Results

Execution is in progress. Results and reproducible commands are added after the
corresponding runs finish.
