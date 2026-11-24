# Grounded first-person Jensen: execution and handoff

Repository: `Dindb-dong/who-is-your-bias`.
Task branch: `feat/jensen-grounded-persona-rag-nov24`.
Base and initially verified main: `0db299d8fa91d1909ac4e70e3ca3e22f6d0c9943`.
Scope: repair the production persona response path, broaden reviewed public
sources including social discovery, connect bilingual retrieval, and preserve
small remote checkpoints. A merge is not part of this authorization.

## Verified diagnosis

The production snapshot/context-budget path differs from the compact experiment
builder. It contains third-person guidance, including a Korean first-person
prohibition. Retrieval misses can enter the speculative Jensen path, allowing a
model to invent or deny biography. The separate current-source QA route can also
produce reporter-style language. More source files alone will not fix these
response contracts.

## Workstreams

- Public biography, university/leadership material and authenticated social
  discovery: `data/acquisition/jensen-public-2026-11-24/biography-*`.
- Long-form interview reasoning and technical constraints: the `judgment-*`
  shards in the same directory.
- Production first-person messages, evidence-miss behavior and news/project
  attribution: Python conversation runtime.
- Deterministic corpus build, bilingual retrieval, snapshot/configuration,
  regression evaluation and final integration: coordinating implementation.

## Evidence contract

Source and event IDs, publication/availability dates, speaker provenance and
partition labels remain explicit. New records use faithful bounded paraphrases
and source-linked decision annotations; short original excerpts are labeled
separately. They are not complete original question/answer pairs. A blocked or
unread Instagram link belongs in discovery, never active factual retrieval.
Personal Instagram ownership must not be inferred from a matching display name.
Newly authored scenario answers are not claimed to be authentic Jensen quotes.
Existing train events cannot become unseen evaluation events.

`tools/build_jensen_public_corpus.py` assembles the acquisition shards with the
existing 60-record values corpus and writes a hash-bound snapshot. It checks
source/event partition consistency and an aggregate short-excerpt budget.
This initial checkpoint does not claim that collection, runtime integration or
response-quality evaluation is complete.

## Status at initial checkpoint

Completed: main and instructions inspected; isolated task branch; dependency
installation; runtime diagnosis; deterministic build implementation.
In progress: source shards, persona contract, retrieval and regression cases.
Not performed: new training, audio/video acquisition, voice/gesture comparison,
private Mac runtime restart, human blind comparison or deployment.

Proposed build command (will be marked verified after execution):

```sh
uv run python tools/build_jensen_public_corpus.py
```

Remote checkpoint receipts, measured checks and exact activation commands will
be appended after those operations actually complete.
