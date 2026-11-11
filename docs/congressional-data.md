# Congressional oral dialogue and the combined corpus

Six already registered GovInfo hearing volumes now have a narrowly reviewed
oral-discussion adapter. The new acquisition contains **41 adjacent question
and response exchanges from six event families**. A separate combined build
contains **1,018 records from 15 families**, including the existing 977-record,
nine-family corpus. The existing training and comparison inputs remain bound
to their earlier snapshots.

The executed records are the [Congress coverage report](../reports/persona-data/congress-2026-v1/coverage.json),
[combined coverage report](../reports/persona-data/combined-2026-v2/coverage.json),
[combined extraction recipe](../reports/persona-data/combined-2026-v2/materialization-recipes.json)
and [combined snapshot](../reports/persona-data/combined-2026-v2/persona-snapshot.json).
These shareable artifacts contain IDs, counts, provenance and hashes. Complete
publisher text, canonical dialogue and runtime passages stay in the local
materialization.

## Observed acquisition and extraction

All six registered publisher URLs were acquired successfully through the
existing acquisition command. The source and family assignments were already
`train`; prior summary exposure means these events are development/training
material, not fresh test references.

| Registered event | Hearing date | Retained exchanges | Publisher volume |
| --- | --- | ---: | --- |
| `house-space-cooperation-2004` | 2004-03-18 | 7 | [NASA–DOD cooperation](https://www.govinfo.gov/content/pkg/CHRG-108hhrg92514/html/CHRG-108hhrg92514.htm) |
| `senate-future-launch-2004` | 2004-05-05 | 1 | [Future launch vehicles](https://www.govinfo.gov/content/pkg/CHRG-108shrg20624/html/CHRG-108shrg20624.htm) |
| `house-commercial-space-2005` | 2005-04-20 | 5 | [Commercial space markets](https://www.govinfo.gov/content/pkg/CHRG-109hhrg20541/html/CHRG-109hhrg20541.htm) |
| `senate-advanced-energy-2007` | 2007-03-07 | 8 | [Advanced energy technologies](https://www.govinfo.gov/content/pkg/CHRG-110shrg36077/html/CHRG-110shrg36077.htm) |
| `house-commercial-crew-2011` | 2011-10-26 | 5 | [Commercial crew program](https://www.govinfo.gov/content/pkg/CHRG-112hhrg70800/html/CHRG-112hhrg70800.htm) |
| `senate-launch-2014` | 2014-03-05 | 15 | [National security launch programs](https://www.govinfo.gov/content/pkg/CHRG-113shrg49104594/html/CHRG-113shrg49104594.htm) |

The congressional build contains 39 complete printed exchanges and two
interruption excerpts. It has 32 factual-projection records and 39
text-style records. Short acknowledgments can be complete style examples
without meeting the factual projection's eight-word answer threshold.
Twenty-nine other adjacent Musk turns were excluded because their predecessor
was another witness, a procedural delegation, or lacked a direct question.
Those exclusions are retained in the report.

The combined build has 961 complete exchanges and 57 excerpts, with 777
factual and 961 style records. Its exact normalized question-plus-answer
deduplication removed **zero** records. This checks exact wording duplication;
it does not establish that semantically repeated claims are independent
evidence. The combined 15 events are still below the roadmap's 20–30-event
acquisition target.

Neither build has accepted speech or visible-behavior projections. Audio
duration, acoustic alignment, perceptual similarity, independent human
ratings and adaptation efficacy have not been established by this expansion.

## What the adapter accepts

`tools/persona_data/congress.py` identifies itself as
`govinfo-oral-discussion.v1`. It accepts only the six exact reviewed publisher
artifact SHA-256 values. Each hearing has an explicit oral-discussion start,
stop, participant-label list and permitted editorial headings. Changed
publisher bytes require a newly reviewed version; the parser cannot silently
reuse old evidence hashes.

The parser requires the full-name testimony heading before accepting the
short `Mr. Musk` label. Committee and other witness labels are preserved.
Only an immediately preceding committee utterance containing a question mark
can become a projected question. A name-only request such as `Mr. Musk?`
does not recover the earlier question addressed to a panel, so it is excluded.
The conservative rule also omits some valid declarative prompts and responses
separated from their original question by other witnesses. It does not invent
missing interviewer context to increase the record count.

The selected windows begin after the opening/prepared testimony and end
before written questions, appendices or the next panel. Prepared statements,
biographies and post-hearing written answers remain available only in the raw
publisher snapshot. They are never relabeled as live spoken answers. Runtime
source metadata preserves the original inventory's attribution separately
and describes the newly selected oral scope explicitly.

Each complete attributed turn retains all of its paragraphs. Whitespace is
collapsed; printed spelling and hard-wrap hyphens are retained. Audited
editorial headings and stage directions are kept as separate hash/line
metadata rather than spoken words. Unavailable written insertions break the
attribution boundary. An answer ending in an interruption dash is marked an
excerpt and excluded from style eligibility. No printed line number becomes
an audio timestamp.

The canonical transcript and shareable recipe bind the source-byte hash,
canonical turn hash, adapter version, adapter-code hash, exact preformatted
line window, excluded prefix/suffix hashes and skipped editorial-block hashes.
The existing parser version and frozen source/split registry are unchanged.
The nine current non-congressional canonical transcripts and all 977 existing
records were replayed without changes to their extracted content.

## Reproduce the separate acquisition

Run from the repository root and use new output paths:

```bash
uv run python -m tools.persona_data acquire \
  --source senate-launch-2014 \
  --source house-space-cooperation-2004 \
  --source senate-future-launch-2004 \
  --source house-commercial-space-2005 \
  --source senate-advanced-energy-2007 \
  --source house-commercial-crew-2011 \
  --output data/local/acquisition-congress-new

uv run python -m tools.persona_data build \
  --acquisition data/local/acquisition-congress-new \
  --recipe reports/persona-data/congress-2026-v1/materialization-recipes.json \
  --output data/local/persona-congress-new \
  --report reports/local/persona-congress-new
```

The pinned recipe verifies the selected transcript hashes and exchange
selection. An unavailable publisher or changed source fails explicitly; do
not edit a recorded digest to make a new document appear identical. Earlier
acquisition directories and build outputs are not overwritten.

## Build a usable combined corpus

The combination command copies verified acquired artifacts into a new root
and preserves their original acquisition records and timestamps. It records
the input acquisition-log hashes, rejects duplicate source IDs and reserved
test sources, and retains failed acquisition attempts as failures.

```bash
uv run python -m tools.persona_data combine-acquisitions \
  --acquisition data/local/persona-2026-acquisition \
  --acquisition data/local/acquisition-congress \
  --output data/local/persona-combined-acquisition-new

uv run python -m tools.persona_data build \
  --acquisition data/local/persona-combined-acquisition-new \
  --recipe reports/persona-data/combined-2026-v2/materialization-recipes.json \
  --output data/local/persona-combined-new \
  --report reports/local/persona-combined-new
```

The first input is the current original-data acquisition, whose nine usable
events were already materialized. Its additional TIME document still lacks
a reviewed original-dialogue parser and contributes no exchange. The
combination manifest therefore contains 16 acquired documents, while the
runtime corpus contains 15 accepted event families. The second input is the
separately reviewed congressional acquisition.

Register a snapshot using the actual time of the new build. This small script
sets an aware UTC cutoff and lets the snapshot preserve each source's stored
acquisition timestamp:

```bash
uv run python - <<'PY'
from datetime import date, datetime, timezone
from pathlib import Path
from bias.corpus import Corpus
from bias.snapshots import register_snapshot

root = Path("data/local/persona-combined-new")
now = datetime.now(timezone.utc)
snapshot = register_snapshot(
    Corpus.load(root / "corpus"),
    identifier="musk-combined-new",
    as_of=now,
    ingested_at=now,
    current_evidence_since=date(2026, 1, 1),
)
with (root / "snapshot.json").open("x", encoding="utf-8") as output:
    output.write(snapshot.model_dump_json(indent=2) + "\n")
PY
```

The executed combined snapshot's cutoff is
`2026-11-11T14:08:38.928202+00:00`. Its corpus digest is
`57cead316fd009be5d19e34c4a70a3df0290a67d4a90192b5d1e0c58c1bfc622`.
Use that snapshot only with those exact corpus bytes; a new acquisition or
changed metadata needs its own snapshot. A recent build cutoff does not make
a 2004 or 2014 hearing evidence of a current position.

Start the API on the new corpus by explicitly setting both paths:

```bash
export BIAS_CORPUS_DIR=data/local/persona-combined-2026-v2/corpus
export BIAS_PERSONA_SNAPSHOT=data/local/persona-combined-2026-v2/snapshot.json
uv run bias serve --host 127.0.0.1 --port 8000
```

Use the `persona-combined-new` paths after reproducing a new build. Provider,
retrieval and model settings otherwise follow the normal runtime setup.
The repository default has not been changed to this local materialization.

## Verify API availability and preserve negative diagnostics

```bash
uv run python -m tools.persona_data.congress_smoke \
  --corpus data/local/persona-combined-2026-v2/corpus \
  --snapshot data/local/persona-combined-2026-v2/snapshot.json \
  --output reports/local/congress-api-new.json
```

This runner constructs explicit lexical/source-preview settings, exercises
the actual FastAPI lifespan and `/api/chat` through in-process HTTP, and uses
no language or speech model. It checks that six developer-authored historical
queries retrieve and cite their intended event, then checks a Korean current
future-plan question and its follow-up for current-only abstention.

The [executed availability report](../reports/persona-data/combined-2026-v2/api-availability.json)
also preserves six earlier natural-paraphrase probes. Only **two of six**
retrieved their intended event; four abstained under the lexical topic gate.
The availability questions added explicit space/government topic anchors
after this observation. Their success shows that the new source passages are
usable through the API, not that paraphrase retrieval quality is solved.
The runner records those negative diagnostics separately and makes no
held-out or semantic-quality claim. A hybrid retrieval comparison requires
its own pinned model execution and report.

```bash
uv run pytest tests/persona_data -q
```

Synthetic parser tests cover exact snapshot and boundary rejection, explicit
speaker attribution, other-witness separation, full paragraph retention,
interruption excerpts, editorial material, written-appendix exclusion and
immutable acquisition combination. Actual source hashes and retained counts
come from the executed materialization reports, not those synthetic fixtures.
