# Repository instructions

## Scope and starting a task

- Follow the user's current task and permissions. Read-only requests do not authorize edits, training, uploads, commits, or merges. Perform authorized checkpoint saves without repeatedly asking for the same permission; do not infer permission to merge from permission to push.
- Before repository work, read this file and any applicable directory instructions or overrides from the target branch. In a web/GitHub-plugin session, fetch them explicitly; a repository or PR lookup does not establish that instruction files were read.
- Record the repository, target branch, base commit, and current remote head. Preserve unrelated work and inspect a moved head before retrying a write. Use a task branch and expected-head checks where supported; do not force-push or rewrite shared history.
- For interrupted work, first inspect the latest remote checkpoint, its actual artifacts, and the linked handoff. Reuse verified outputs before repeating expensive work. Conversation summaries and progress messages are leads, not proof of saved files.
- Keep this file focused on stable rules. Put changing task status and detailed execution plans in task documents. For Jensen work, start with [the execution record](docs/jensen-replication-run.md) and its linked handoff, then read only the relevant technical guides.

## Project map and invariants

- The active subject is Jensen Huang. Keep historical Elon Musk data, trained assets, and measurements explicitly separate; their results do not establish Jensen quality.
- `src/bias/`: Python API, retrieval, dialogue, timing, and runtime configuration.
- `apps/web/`: studio UI, avatar rendering, attention, and playback.
- `services/language/` and `services/voice/`: optional inference services.
- `tools/`: acquisition, preparation, adaptation, media, motion, avatar, and evaluation workflows.
- `data/`: versioned profiles, evidence, source metadata, recipes, and snapshots. `docs/` and `reports/`: methods, handoffs, and measured results.
- Preserve source attribution, event-level evaluation isolation, explicit persona/voice/motion binding, simulation labels, and the existing explicit participant-selection behavior.
- Distinguish editorial summaries, original dialogue, acquired media, prepared datasets, trained weights, generated samples, integrated behavior, and evaluated quality. Do not turn planned or reported work into a completion claim.

## Durable checkpoints are part of the work

Save a small, coherent checkpoint:

1. After a meaningful implementation or acquisition/preparation batch.
2. Before starting a long training, synthesis, rendering, or evaluation job.
3. After an expensive job produces a reusable output.
4. Before a planned handoff, context transition, or end-of-turn response.

Do not wait for all workstreams or the final test suite to finish before preserving work. An incomplete checkpoint may be marked WIP and list failing or pending checks; it is not a release or a merge approval.

### Code and documents

- Commit only files belonging to the task, in reviewable groups, and push to the authorized task branch. Include actual implementation/configuration changes; a plan-only commit does not preserve later implementation.
- With GitHub tools, finish the entire blob/tree/commit/branch-ref sequence, or use an atomic contents operation. Creating a blob or commit without updating a reachable branch is not a completed checkpoint.
- Re-read the remote branch and the affected paths at the saved commit. Verify the expected commit and file contents before reporting a successful save or starting the next expensive stage.
- If persistence fails, first resolve it or report the concrete blocker and last verified checkpoint. Do not continue accumulating expensive, unbacked work.

### Data, media, and model outputs

- Never commit secrets. Follow `.gitignore`; do not force-add ignored raw source recordings, full local transcripts, or large model checkpoints to Git.
- Before expensive acquisition or training, establish an accessible, authorized durable artifact destination and verify a small upload and readback. A temporary directory or a download recipe alone does not preserve acquired bytes.
- Keep a versioned artifact manifest with run ID, artifact role, persistent object/file ID or path, byte count, SHA-256, and upload/verification status. Include source/event IDs, acquisition URLs/dates, and local-to-source clock mapping where applicable. Do not commit credentials or expiring signed URLs.
- Retain code and preparation recipes in Git; retain the actual reusable datasets, permitted media, and weights in the durable destination. Report acquired, uploaded, verified, excluded, and unavailable material separately.
- Save partial results in bounded batches. Download completion, preprocessing completion, and remote persistence are separate states.

### Long-running jobs

- Check available hardware, memory, disk, dependencies, and process lifetime before choosing the execution plan. Do not assume a process survives a chat or container ending.
- Put periodic checkpointing in the job itself. For training, preserve weights, optimizer/scheduler state where applicable, progress counters, random state, pinned model/tokenizer revisions, dataset/split hashes, configuration, and the exact resume command.
- Write checkpoints atomically, retain the last known-good checkpoint, and persist them at a documented interval. Do not rely only on successful job completion or a final cleanup handler.
- Verify a bounded save/load/resume cycle before a long run. Report the actual last persisted step and any work since it; do not present a target step count as completed training.

## Handoff and evidence

Each active long task needs a durable status record linked from its entry document. Maintain:

- Task scope, branch, base revision, and the last verified remote checkpoint or external receipt.
- Completed, in-progress, blocked, and unverified items, with evidence for each.
- Artifact locations and hashes; code/model/data versions and execution settings.
- Checks actually run, outcomes, exclusions, known failures, and remaining acceptance gates.
- Exact next/resume commands when verified; clearly label proposed commands or missing implementation.
- Decisions needed to avoid repeating failed experiments or contaminating held-out evaluation.

Do not put a commit's own SHA inside content that determines that same SHA. Use an earlier verified checkpoint, or record the new SHA in a subsequent receipt or handoff update.

Treat an unavailable previous workspace as an access limitation, not proof that work never ran or that every artifact is permanently lost. Verify recovered bytes and provenance before restoring a success claim.

## Context and parallel work

- Keep detailed logs and large outputs in files. Read focused ranges and report concise findings; avoid repeatedly printing full files, registries, datasets, or generated payloads into the conversation.
- Delegate independent work when useful, with explicit file ownership and artifact contracts. The coordinating agent remains responsible for saving each completed subtask remotely; an agent message is not a checkpoint.
- Prefer bounded milestones and publish completed parts while other jobs continue. Never use conversation memory as the only record of the next step.

## Validation and delivery

Use the smallest meaningful checks for the changed paths. Do not launch model downloads, training, rendering, or full application suites solely for a documentation edit.

Existing entry points, subject to the checked-out revision and installed prerequisites:

- Setup: `uv sync --locked --extra dev`, `npm ci`.
- Corpus validation: `uv run bias validate`.
- Python: `uv run pytest`, narrowed to affected tests during development.
- Web: `npm run typecheck`, `npm run test:web`, `npm run build`.
- Browser behavior: `npm run test:e2e`; Jensen attention: `npm run test:attention`.
- Avatar tooling has separate dependencies and checks; consult its existing guide before invoking them.
- `make evaluate` explicitly targets the historical Musk corpus. It is not a Jensen acceptance result.
- Web build/dev provisioning may download a pinned face detector. Check network and asset prerequisites before interpreting failures.

For documentation-only changes, check consistency, relative links against the repository tree, formatting, and the exact remote diff. Observe required repository checks and merge rules without bypassing them.

Before delivery, verify the remote branch state and affected files. Before an authorized merge, confirm the current PR head, exact scope, and applicable checks; merge that verified head. After merging, re-read the PR and default-branch files. Report the actual commit/PR links and any remaining limitations. Checkpoint preservation and final acceptance are separate obligations.

