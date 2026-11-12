# Package collected responses for blinded source review

The existing adaptation exporter preserves complete responses, supplied factual
passages and a separate condition key. It does not include candidate style
sources, and it places private reference answers in the primary sample records.
The additive `tools.adaptation_evaluation.blind_review` helper supplies the
common source pool required by the
[pre-output agent rubric](../reports/jensen-evaluation-v1/agent-review-rubric.md)
and separates reference comparison from initial support assessment.

This is an offline packaging step after the existing
[single-condition collector](adaptation-comparison.md). It does not call a model,
load neural weights, change any frozen plan/prompt, rerun failed generations,
or grade an answer. The original collector and pinned execution contract are
unchanged. Its synthetic tests establish packaging behavior, not Jensen quality.

## Inputs and command

Use the completed collection directory produced by
`python -m tools.adaptation_evaluation.local_condition collect`. All three
condition runs must have been collected with their validated persistence
receipts. Individual failures and unattempted cases are permitted and remain
in the denominator; a directory from a still-running condition is insufficient.

Run once per desired review package, repeating `--input PLAN COLLECTION` to
combine development and heldout outputs without mixing their source pools:

```bash
uv run python -m tools.adaptation_evaluation.blind_review \
  --input data/local/evaluation/development/plan.json data/local/evaluation/development-collected \
  --input data/local/evaluation/heldout/plan.json data/local/evaluation/heldout-collected \
  --output data/local/evaluation/agent-review-v1
```

These are illustrative paths. Pass the actual frozen plans and completed
collections. The output directory must be new; existing files are never
overwritten. `--rubric PATH` can explicitly select a review procedure; the
default is the registered M6 rubric, whose exact bytes and SHA-256 are retained.
Changing the rubric is a separately disclosed review version, not a silent
update to the registered assessment.

The helper checks the frozen plan seal and registered implementation hashes,
the collector/private comparison agreement, all three condition records,
the complete-answer key and exact output hashes, factual passages and reference
answers against the plan, and complete/failed/unattempted counts. Already graded
human fields are rejected rather than erased. The helper verifies consistency
of the saved collection; the coordinator's original receipt and fresh remote
readback verification remain necessary.

## Output contract

| File | Contents and audience |
| --- | --- |
| `reviewer/samples.json` | Complete unchanged answers, question, opaque sample ID, common source-pool pointer, seven null human rating fields and null reviewer fields. |
| `reviewer/source-pools.json` | One full factual-plus-candidate-style pool per frozen case, applicable system instruction, authored fixed history, and JSON pointers/source hashes. All anonymous outputs for that case use this same pool. |
| `reviewer/reference-answers.json` | Private reference text for a separate second-stage comparison. References are not supplied factual evidence. |
| `reviewer/rubric.md` | Exact selected review procedure. |
| `reviewer/README.md` | Support-first review sequence and sharing instructions. |
| `private/answer-key.json` | New sample-to-condition mapping, output hashes, plan identity and original sample IDs. Coordinator only. |
| `private/execution-diagnostics.json` | Full collected comparison, failures and unattempted cases, with separate development/heldout and per-condition denominators. Coordinator only. |
| `private/input-manifest.json` | Input locations and exact plan/collection file hashes. Coordinator only. |
| `manifest.json` | Packaging status, aggregate counts, tool/rubric hashes and hashes of every payload file; written last. |

Share only `reviewer/` with a blinded reviewer. The package uses a fresh
operating-system-random permutation, so the old exporter's fixed seed and order
do not identify the new sample mapping. Preserve the private key to reproduce
the realized mapping. The tool omits condition/model/runtime metadata from
reviewer JSON; it does not censor full model responses, which can themselves
contain identity or stylistic clues. Record any resulting unblinding honestly.

Style examples appear identically for every anonymous output of a case,
including outputs whose generation did not receive them. They help inspect
copying and are never factual support. Empty factual pools and missing style
contrasts remain empty. The case's reference answer is not inserted into any
generation input; packaging reads existing inputs without modifying them.

The presence of `manifest.json` marks a finished packaging operation. An I/O
failure can leave an incomplete directory without that manifest; preserve the
failure information and choose a new output directory when retrying. Keep the
entire package in the private artifact destination, save its manifest and
receipt, and freshly read it back before beginning expensive review or handing
it to another session. Original source wording is not a public Git artifact.

## Review and limitations

Follow the registered rubric and save categorical findings by opaque sample ID
before opening the condition key. Agent findings belong in a separate record;
the helper leaves human review ungraded and `efficacy_claim` false. Failed and
unattempted responses are not included as complete review samples, but their
counts and known failure reasons remain explicit in the private diagnostics.

This helper targets the matched three-condition collector. The separate
single-arm heldout RAG diagnostic is not a compatible input and must retain its
own actual supplied-evidence accounting. Do not manufacture missing comparison
conditions to put it into this package.

Focused, model-free verification:

```bash
uv run pytest tests/adaptation_evaluation/test_blind_review.py
```

The tests construct synthetic sources and outputs and exercise the existing
exporter before packaging: full Unicode/multiline answers, common pools and
evidence pointers, multiple plans with overlapping case IDs, reference/key
separation, null human ratings, tamper rejection, empty evidence/style pools,
length failures, unattempted cases and overwrite rejection. No live model,
publisher transcript or network response is used.
