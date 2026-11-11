# Registered multi-turn runtime evaluation

The 2026 roadmap calls for at least twenty multi-turn scenarios. The new
[`dialogue-runtime-development.v1` registry](../tools/dialogue_evaluation/scenarios.v1.json)
contains **29 scenarios and 100 steps**, exercised through the actual chat and
session-context API. These are exposed, deterministic development and runtime
acceptance cases. They are not fresh independent test interviews, a semantic
grounding assessment, or evidence of convincing Musk style.

## Run the complete suite

```bash
uv sync --extra dev
uv run python -m tools.dialogue_evaluation \
  --output reports/local/dialogue-preview-new.json
```

The default ignores ambient provider settings and loads the bundled original-QA
preview and its exact registered snapshot. It makes no LLM or TTS calls. Reports
are created at a new path and never overwrite previous evidence. A failed step
produces a retained failure report and a nonzero process exit code.

To exercise the same conversation contracts with a materialized corpus, pinned
hybrid retrieval and optionally a real generation service, supply an explicit
environment file and opt into configured execution:

```bash
uv run --extra hybrid --env-file config/persona-2026.env.example \
  python -m tools.dialogue_evaluation --configured \
  --output reports/local/dialogue-configured-new.json
```

`--configured` can contact the LLM endpoint named in that environment. The runner
records configured models and counts turns that actually invoked generation.
It does not invoke the speech service. Preview-specific assertions that the
bundled archive lacks current evidence are disabled for a different corpus;
the temporal-state and user-reference assertions remain active. An explicit
corpus-bound snapshot is mandatory. Settings that retain too few history turns
can legitimately fail earlier-topic scenarios.

`--scenario en-year-change` selects an exact registered scenario for diagnosis.
A subset run records the smaller executed count; it must not be described as
the complete suite. Unknown scenario names, unknown assertions, duplicate
labels and references to nonexistent earlier steps are rejected.

## What the scenarios cover

| Area | Registered behavior |
| --- | --- |
| Chained references | English and Korean follow-ups retain the original user subject and requested audience |
| Elliptical questions | “Why?”, “Tell me more”, “왜?”, and explicit date-only follow-ups inherit user context |
| Temporal changes | Historical dates, current positions, forecasts, outcomes, before/through boundaries, and Korean date suffixes remain distinct |
| Corrections | A date-only correction retains the subject; a correction introducing a different subject resets it |
| Topic changes | Full new questions, explicit switches within the same named entity, and named earlier topics are handled separately |
| Session isolation | Two interleaved sessions retain their own subjects and temporal anchors |
| Bounded conversations | Long chains preserve the user anchor without retaining an unbounded dialogue history |
| Interruption | Cancelling a streaming request owner does not record a completed user turn or contaminate the next follow-up |
| Evidence boundary | User memories remain unverified context; citation IDs must come from retrieved evidence |

The interruption step drives the real ASGI streaming endpoint and cancels its
request owner during status delivery. It is a cancellation/state-ownership
check, not a measured network disconnect, mid-inference neural stop, browser
audio interruption or acoustic latency measurement. Existing live-stream and
browser tests cover those separate transport/playback contracts.

The report stores the registry digest, corpus/snapshot identity, implementation
file hashes, safe provider identities, exact synthetic user requests, resolved
queries, temporal traces, state provenance, citation IDs, response hashes and
individual assertions. Full source passages and generated answer text are not
copied into the report. In-process elapsed times exclude a browser, synthesis,
loudspeaker output and real network transport.

## Runtime fixes found while implementing the suite

The initial resolver recognized some explicit pronouns but dropped bare or
date-only follow-ups. It now recognizes a bounded set of English/Korean
ellipses. Shortness alone still does not establish a reference: “How are you?”
and unrelated complete questions start their own subjects.

Explicitly resuming a named earlier user topic can recover its anchor and
temporal policy. The lookup stays inside retained user records and never uses
unsupported generated assistant text as a fact. Date-only corrections retain
the subject, while rejected dates such as “2013, not 2017” no longer become a
two-period comparison. The full original request remains available for audit.

The resolved search string formerly added labels such as “Previous user
question” and “Follow-up question”. Those extra words could lower the existing
lexical match ratio below its relevance threshold. Search now uses only user
words; structured fields carry provenance. The historical lexical retriever
and its frozen experiments are unchanged.

When the snapshot has already selected an exact, before, through or forecast
period, the live lexical search no longer applies the archived retriever's
exact-year gate a second time. Otherwise “before 2020” could reject an eligible
2016 statement, and a plan for 2030 could require a source published in 2030.
The applied lexical query is included in the context trace. Temporal comparison
requests retain their existing year-matching behavior.

Finally, generated prompts explicitly carry the resolved temporal years and
period in addition to the mode. A large assistant-history pair can be dropped
from the context budget without losing a subsequent date correction or the
original user audience. API tests use an explicitly synthetic recording model
to inspect this prompt path; their result is not a neural inference claim.

## Validation

```bash
uv run pytest tests/dialogue_evaluation tests/test_temporal_dialogue.py \
  tests/test_engine_and_sessions.py tests/test_request_deadlines.py \
  tests/test_providers_and_api.py
```

The source-preview suite establishes runtime state and protocol behavior. Fresh
held-out source acquisition, actual adapter comparisons, human judgment of
factual entailment/style, and simultaneous LLM/TTS/avatar measurements remain
separate roadmap requirements.

The [recorded preview run](../reports/dialogue/2026-11-11-preview.v1.json) passed
all 29 scenarios and 100 steps, with zero LLM inference turns and no synthesis.
Its implementation digests identify the executed code independently of later
repository changes. The focused tests above passed 93 tests in that run.
