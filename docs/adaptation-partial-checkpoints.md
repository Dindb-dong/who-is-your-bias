# Preserve partial condition work without changing the frozen runner

`tools.adaptation_evaluation.partial_checkpoint` captures an already finalized
attempt prefix while a condition continues running. It reads the original
files, writes a new local snapshot outside the active condition directory, and
performs no inference, upload, download, Git write, remote verification or
generation resume.

Use this between complete-condition persistence boundaries to retain bounded
work if the process or conversation ends. The
[existing final collector](adaptation-comparison.md) and its full-condition
durability requirements remain unchanged. A partial snapshot cannot enter that
collector or the [blinded response package](adaptation-blind-review.md).

## Capture a cumulative prefix

The active directory must already contain `condition/run.json`,
`condition/attempts.jsonl`, `condition/raw-generations.jsonl`, `launcher.json`,
`warmup.json`, `observed-before.json` and the sealed supervisor observation.
There must be at least one finalized attempt.

```bash
uv run python -m tools.adaptation_evaluation.partial_checkpoint \
  --plan data/local/evaluation/development/plan.json \
  --contract reports/jensen-evaluation-v1/execution-contract.json \
  --condition-directory data/local/evaluation/development-condition-active \
  --attempts 4 \
  --include-server-log \
  --output data/local/evaluation/partial-checkpoints/development-attempts0004
```

These are illustrative paths. Supply the actual frozen plan, registered
contract and active condition directory. `--attempts N` selects exactly the
first N finalized attempts. Without it, the helper captures all attempts that
are finalized in its first stable report observation. A failed or length-cut
attempt is finalized and retained; it is not counted as a completed response.
The final row awaiting post-generation identity verification is excluded.

Capture a new cumulative snapshot every two to four finalized attempts using a
new output directory each time. The previous snapshot remains intact. A request
for more finalized attempts than are available fails without creating output.
The CLI prints only status, counts and the manifest seal, never response text.

## Consistency checks

Before writing, the helper checks the frozen plan and execution contract, the
run and supervisor seals, model identity, and launcher/warmup/observed-before
metadata. The selected rows must follow the exact frozen case order and
condition. The first N complete JSONL rows must match those report rows and
their original provider responses. Full answers, finish reasons, errors and
identity failures are retained.

The helper rereads the sealed report, finalized prefix, immutable metadata and
captured log prefixes. The report may advance to later cases while these reads
occur; previously finalized content must remain identical. A changed prefix or
incomplete/torn observation is retried at most three times by default, with a
50 ms pause. `--read-attempts` accepts one through ten bounded attempts. A
persistent mismatch fails before an output directory is created.

The original runner can rewrite `run.json` while this reader observes it.
Neither a transient parse error nor a locally saved line is accepted as a
finished case without the required seal and cross-file comparisons.

## Snapshot contents

| File | Meaning |
| --- | --- |
| `partial-run.json` | A separately sealed partial schema with only finalized captured rows, source invocation metadata, source report seals/hashes from both reads, counts and uncaptured case IDs. |
| `attempts.prefix.jsonl` | Exact original bytes of the first N finalized attempt rows. |
| `raw-generations.prefix.jsonl` | Exact original bytes of the corresponding first N original provider responses. |
| `metadata/launcher.json` | Captured immutable launcher declaration. |
| `metadata/warmup.json` | Captured excluded warmup record. |
| `metadata/observed-before.json` | Captured actual model/runtime identity before the condition. |
| `metadata/supervisor-observed.json` | A sealed supervisor observation, which may still say the server is running. It is not a final supervisor receipt. |
| `server.log.snapshot` | Optional exact first size-at-open bytes of the mutable server log; explicitly not the final log. |
| `manifest.json` | Written last, with a canonical seal and every payload's byte count/SHA-256, partial/non-collectible flags and false remote-persistence/readback flags. |

The full mutable source report is not copied: it may contain an unfinished
response beyond the captured prefix. Its verified seals and byte hashes remain
provenance observations in the partial record. The custom schema and filenames
deliberately do not resemble a complete `condition/run.json` plus final
`supervisor.json` directory.

All source files remain untouched. Existing destinations and destinations inside
the active condition directory are rejected. If writing fails after validation,
an incomplete new directory can remain without `manifest.json`; it is not a
finished snapshot. Inspect that failure and choose a new output path when
retrying.

## External preservation and later use

Archive the complete snapshot directory, upload it through the established
artifact route, freshly restore those actual remote bytes, and compare every
file against the preserved manifest. Save that separate transfer/readback
receipt and public source-free evidence. The helper's local success and hashes
do not establish that any remote operation occurred.

Keep raw responses and any included source/context wording private. A
condition-labelled partial snapshot should stay with the coordinator so the
later paired reviewer does not see the condition key prematurely. Use the final
complete collections to build the blinded review package.

These snapshots preserve work and failures; they are not a generation resume
cursor, do not authorize appending results to the frozen runner, and do not
replace the final full-condition save/readback. Any interrupted evaluation
continuation remains an explicit coordinator decision under the registered
protocol rather than a silent retry or selection of favourable answers.

## Focused verification

```bash
uv run pytest tests/adaptation_evaluation/test_partial_checkpoint.py -q
```

The fixtures are entirely synthetic. They check all three condition identities,
pending-row exclusion, exact full response/log preservation, length and identity
failures, bounded prefixes, concurrent append/report advancement, changing
prefix rejection, torn-read recovery, mutable log bounds, no overwrite, and
manifest-last behavior. They never read actual generated M6 responses or start
a neural process.
