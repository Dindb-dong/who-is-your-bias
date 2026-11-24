# Grounded first-person Jensen: execution and handoff

Repository: `Dindb-dong/who-is-your-bias`. Task branch:
`feat/jensen-grounded-persona-rag-nov24`. Base and verified unchanged main:
`0db299d8fa91d1909ac4e70e3ca3e22f6d0c9943`.
Last remote implementation checkpoint verified before this handoff:
`b22590f7ced4a6da41583da74df563eb44e9255d`.
The commit containing this handoff necessarily has a later SHA; inspect the
task branch or PR for that exact head. No merge is authorized by this task.

## Delivered behavior and its limits

The API production path now uses a grounded first-person Jensen contract and
the expanded bilingual public retriever. The local workbench prepares a merged
corpus that adds public evidence while retaining original interview Q&A, private
persona-card references, and the configured voice/avatar assets. These paths
are implemented and covered by automated regression tests. The user's running
Mac demo was not reachable from this workspace and was not restarted here.

The concrete reporter-style causes were a Korean prohibition on first-person
speech in the budgeted snapshot prompt, a production path that bypassed the
compact persona instructions, and third-person wording in current-source QA.
The new contract applies to the actual production prompt, including the
snapshot/context-budget path. Identity answers can say “I'm Jensen Huang in
this AI simulation”; the simulation label is preserved. Supported public
experience can use “I”, and a dated NVIDIA announcement can use “we announced”.
Project implementation remains attributed to the project team.

Retrieval failure for a personal-history question now produces a bounded
cannot-verify response instead of inviting invented memories or denial of real
experience. Narrow source-text checks also reject known tangential matches,
such as using an ordinary Denny's record to support a secret dismissal or an
unsupported meeting with Steve Jobs. Rejected citations are cleared before
speech. These are targeted guards, not general semantic entailment or a proof
that every generated answer is correct.

Hypothetical technology decisions use historical public criteria even when the
question includes “currently”. An investment decision addressed to Jensen no
longer becomes a securities question solely because it contains “invest”.
Actual stock/current-position questions retain their original restrictions.
The no-paying-customers case and invalid-performance-measurement case have
distinct retrieved decision annotations. The graph is not enabled by this
change, and frozen experiment prompt builders and baseline corpora remain
available for comparison.

## Evidence delivered

| Artifact or property | Verified result |
|---|---|
| Reviewed source catalog | 46 sources, 39 event families |
| New records | 101: 48 biography/social-context and 53 judgment/technical records |
| Built public corpus | 155 records: 54 retained plus 101 new |
| Short original excerpts | 26 new records contain explicitly labeled bounded excerpts |
| Snapshot general-query eligibility | 152 records; 3 undated-source records conservatively excluded |
| Instagram acquisition | Official-channel identity verified; profile and 3 posts remain discovery-only |
| Fresh independent comparison | NBA Technology Summit 2025 protocol prepared, not executed |
| Full new original Q&A dataset, model training, media synthesis | Not performed |

See the [complete source catalog and acquisition decisions](../data/acquisition/jensen-public-2026-11-24/README.md),
[corpus structure](../data/personas/jensen-huang-public-v3/README.md), and
[retrieval evaluation](../reports/jensen-public-rag-2026-11-24/README.md).
Sources cover interviews, early life and work, founding crises, engineering
tradeoffs, CUDA persistence, management and public disagreement, education,
philanthropy, commencement speeches, company history and product explanations.
Instagram access attempts returned rate limiting or unavailable-content errors;
no unread caption or video is treated as Jensen evidence. An official NVIDIA
channel is not asserted to be Jensen's personal account.

Runtime records are bounded editorial paraphrases, source-linked decision
annotations, bilingual search aliases and short original excerpts. They are
not complete original question/answer pairs or authentic newly authored Jensen
answers. The builder enforces an aggregate excerpt budget per source. Existing
original Q&A in a private workbench export is preserved by the merge.

Six inherited No Priors 2023 records are removed from the new index because the
older acquisition register reserved that event. The older values-v2 baseline
is unchanged. Prior development exposure means No Priors is not a fresh
holdout. The separate NBA event and its four authored reference/rubric cases
remain outside active retrieval and answer templates.

## Versions, hashes and durable source capture

- Knowledge version: `jensen-public-rag.v3`.
- Snapshot: `data/snapshots/jensen-public-2026-11-24.json`, ID
  `jensen-public-2026-11-24.v3`.
- Build/snapshot cutoff: `2026-11-24T11:23:47.521755+00:00`.
- Corpus SHA-256:
  `a3a8424bdd63aaec8a157656bdfd3f8fc9a8e0ff070eeeb8766acfab733cced5`.
- Source-shard hashes and per-source original-byte hashes:
  `data/personas/jensen-huang-public-v3/corpus-manifest.json`.
- Capture manifest and persistent receipt:
  `data/acquisition/jensen-public-2026-11-24/capture-manifest.json` and
  `capture-receipt.json`.

Eight original public documents, seven HTML and one PDF, were captured and
preserved separately from Git. The archive contains 848,160 bytes and has
SHA-256 `d8b88a2f9a9a4ac8fe43f2c9cc47839a02202487642986567f1dc3efbe65ae09`.
Its persistent file ID is
`libfile_fde3_f-ff5bb043f4088191ad97db06a7762428_a18ece08681081918d81df44f69c4b1f`.
Upload and an independent materialization both completed with matching bytes
and hash. The archive contains no weights, audio/video, or prepared complete
interview Q&A dataset.

Actual acquisition timestamps establish availability for those eight undated
documents; they do not establish original publication dates. The exact
Stanford 2003 time-management PDF URL returned HTTP 404. Its records
`JHJ-044`, `JHJ-045`, and `JHJ-046` remain in the reviewed corpus but are excluded
by this snapshot's conservative unknown-date bound. No alternate URL or
earlier availability was invented. Snapshot eligibility can differ with a
question's temporal constraints; 152 is the recorded general-query count.

## Retrieval and runtime integration

`jensen-public` is a dependency-free, source-aware BM25 retriever with explicit
English/Korean concepts, bounded aliases, decision metadata and event
diversity. Unknown query terms still count against confidence. It does not
download an embedding model or require a vector service. This is an initial
bilingual retrieval implementation, not a claim of unrestricted semantic
search.

`Settings.from_env()` defaults the production API to the new corpus, snapshot
and backend. Programmatic `Settings()` keeps the historical defaults for frozen
tests and experiments. An explicit `BIAS_CORPUS_DIR` preserves the existing
custom-corpus behavior: select the desired snapshot and backend explicitly.

The workbench defaults to `YAICON_PUBLIC_RAG=1` and materializes the combined
corpus during `prepare()`. Its merge preserves original IDs, profile, Q&A,
source content hashes and card links; duplicate incoming IDs retain the
original record. Shared URLs bind to existing sources. Reserved events,
connected event mirrors and split conflicts exclude the entire incoming
family. The merge report records exclusions and exact resulting hashes.

Missing public data, an old Python implementation, a changed merge report or
a different imported Python checkout stops startup explicitly. The launcher
records the Python checkout used during preparation and launches the API from
that checkout, independently of the selected web/avatar asset checkout.
It exposes backend, version, corpus counts and merge state at
`/api/local-runtime`. A controlled comparison can explicitly set
`BIAS_RETRIEVAL_BACKEND=lexical`. Explicit `hybrid` also forwards its embedding
configuration without changing the default backend, graph or voice settings.
Returning to the original private corpus requires `YAICON_PUBLIC_RAG=0` and a
fresh preparation.

## Checks actually performed

The [validation receipt](../reports/jensen-public-rag-2026-11-24/validation.json)
records commands and measured outcomes. The affected Python regression set
passed **533 tests**. The separate workbench merge/launcher set passed **21
tests**. The Python run emitted one dependency deprecation warning concerning
Starlette's `httpx` test-client integration; it did not fail. No frontend code
was changed, so no web rebuild or browser-likeness score is claimed.

The authored retrieval comparison uses 54 positive questions, half English and
half Korean, plus 10 abstention controls:

| Configuration | Expected source family in top 4 | Controls without hits |
|---|---:|---:|
| Original 32 records, legacy lexical | 5 / 54 | 10 / 10 |
| Values-v2 60 records, legacy lexical | 5 / 54 | 10 / 10 |
| Expanded 155 records, legacy lexical | 30 / 54 | 10 / 10 |
| Expanded 155 records, bilingual BM25 | 54 / 54 | 10 / 10 |

These questions were used during development, including five Korean coverage
fixes. The result is source-family retrieval on a seen regression set, not
answer correctness, blind persona likeness, or unseen-interview generalization.
It calls raw retrieval over 155 records; production routing, temporal selection
and model generation are separate. The runtime tests use controlled providers
and do not establish the real Qwen3 4B model's wording or judgment quality.

The corpus builder validates staged output before replacing a last-good bundle.
Rollback tests cover invalid source metadata, interruption after backup rename,
and an external-snapshot copy failure. A custom build output does not silently
overwrite the production snapshot.

## Reproduce and activate

From this branch's repository root, with its Python environment installed:

```bash
uv sync --locked --extra dev
uv run bias --corpus data/personas/jensen-huang-public-v3 validate
PYTHONPATH=src uv run python tools/evaluate_jensen_public_rag.py
```

The checked-in bundle is already built. To reproduce that exact timestamp:

```bash
uv run python tools/build_jensen_public_corpus.py --as-of 2026-11-24T11:23:47.521755+00:00
```

For an API launch using the existing configured providers, explicitly bind all
three public-RAG settings from the repository root:

```bash
export BIAS_CORPUS_DIR="$PWD/data/personas/jensen-huang-public-v3"
export BIAS_PERSONA_SNAPSHOT="$PWD/data/snapshots/jensen-public-2026-11-24.json"
export BIAS_RETRIEVAL_BACKEND=jensen-public
uv run bias serve --host 127.0.0.1 --port 8000
```

Provider credentials, voice/motion configuration and built web assets remain
installation prerequisites. The command is an activation procedure, not a
claim that the user's provider or Mac is running this revision. After startup,
`GET /api/health` must show `retrieval.backend` equal to `jensen-public` and the
intended snapshot. Health reports configuration, not provider answer quality.

For the existing Mac workbench, update the actual Python checkout and workbench
scripts to this branch, retain the configured `.env.live` and private assets,
stop the old demo, then run the existing entry point from the repository root:

```bash
export YAICON_PUBLIC_RAG=1
export BIAS_RETRIEVAL_BACKEND=jensen-public
zsh workbench/로컬페르소나시연.command
```

If the corpus is intentionally stored elsewhere, set
`YAICON_PUBLIC_CORPUS_DIR` to its absolute directory before preparation. The
default locates the public corpus in the imported Python checkout. The existing
launcher still requires its original local models, reviewed voice reference
and web/avatar assets. It refuses occupied ports; it does not silently replace
an already running demo.

After this Mac activation, inspect `http://127.0.0.1:8105/api/health` and
`http://127.0.0.1:8105/api/local-runtime`. Confirm `jensen-public`,
`public_rag.status=integrated`, the prepared merge hash and preserved
`corpus_counts.question_answers`. A merged private run's total record count
depends on its original corpus and reservations; do not require exactly 155.
This Mac activation and its live-provider checks remain unexecuted here.

## Remote checkpoints and remaining acceptance

Each completed checkpoint was pushed to the task branch and its affected files
re-read at the exact commit before continuing:

| Commit | Saved work |
|---|---|
| `08379467ce3fd3d09cbd6537e9979d3cda0bc240` | Builder and initial execution record |
| `7c3789e464dccd7c765003ae0df85ef86a9b1e07` | First-person runtime contract, retriever and regression tests |
| `3d43d5496566fa3a67cc708c2ef034986514b4fd` | Source-linked acquisition shards and social discovery |
| `89646e5c9111935a84b1c4acd5c33072fd28b718` | Hypothetical decision routing, biography guards and Korean concepts |
| `f25c49e5691cd8c4951ad741105107ece3f8cf1d` | Production/workbench integration, public bundle, capture receipt and builder safety |
| `b22590f7ced4a6da41583da74df563eb44e9255d` | Explicit hybrid embedding configuration in the local launcher |

On resume, read the current task-branch head, this handoff, the corpus manifest,
the capture receipt and the latest PR checks before repeating acquisition or
changing defaults. Do not merge solely because checkpoints exist.

Remaining acceptance is concrete: restart the accessible Mac demo on this
revision; run the real Qwen model on biography, judgment and three-turn followup
cases; execute the untouched NBA comparison with anonymous conditions; rate
facts, criteria and discourse separately; then compare speech using identical
answer text. No training, human blind scoring, acoustic comparison, gesture
timing assessment or real-avatar visual comparison was performed in this task.
