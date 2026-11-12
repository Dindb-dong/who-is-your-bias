# Exact operational source copies

These are the actual run-specific scripts used or prepared for the recorded M6
campaign. Their bytes match the private operational/source artifacts; the
[source manifest](source-manifest.json) makes those identities reviewable in Git.
They are supplementary orchestration around the existing repository tools.
No frozen inference, question, prompt, decoding or comparison code is replaced.

| Script | Role | Execution boundary |
| --- | --- | --- |
| `launch_m6_condition.py` | One supervised matched condition | Prints by default; `--execute` launches it |
| `m6_condition_delivery.py` | Validate, package and finalize a complete condition | Does not run inference or upload; finalize requires genuine fresh-readback evidence |
| `m6_partial_delivery.py` | Preserve a finalized attempt prefix | Partial outputs are not collector inputs or generation-resume cursors |
| `collect_package_m6.py` | Collect three saved conditions and package opaque review | Supplying an exposure executes collection; the full archive contains private keys |
| `launch_m6_rag.py` | Separately registered six-case base-only retrieval diagnostic | Prints by default; `--execute` launches it |
| `m6_rag_delivery.py` | Preserve the separate RAG prefix, final run or failure bytes | No inference, transfer or central writes; verification requires an actual fresh archive and its external receipt |
| `m6_rag_reviewer_plan.json` | Prepare a separate support review from actual packed RAG evidence | Empty evidence stays empty; references remain isolated until categorical findings are saved |

The scripts retain the actual workspace paths in `ROOT`, `ART` and model-cache
settings. They are execution records, not environment-independent entrypoints.
After relocation, use the portable commands in the [condition execution guide](../condition-execution.md)
and [RAG guide](../heldout-rag-v1/README.md), or explicitly prepare a new
run-specific wrapper without misrepresenting its hash as the original.
The [v1](../operational-tools-checkpoint.json) and [v2](../operational-tools-v2-checkpoint.json)
receipts preserve original preparation/validation scopes. The collection script
is included in each complete collection/review archive.

Do not overwrite prior outputs. Upload and genuinely retrieve each produced
archive before registering a successful persistence receipt. Only a packet's
`agent-review/reviewer/` directory is shared with the blinded reviewer. The
coordinator retains private keys until categorical findings are saved.

The RAG delivery helper was compiled and reviewed before execution. Its partial,
final and failure branches are preservation paths, not evidence that a RAG model
run has completed. Actual outcomes are reported separately from the 54 matched
comparison calls.
