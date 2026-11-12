# Jensen text adapter: bounded execution and continuation

The immutable experiment uses 113 complete original exchanges across all nine
eligible training events. The earlier interrupted 105-answer selection was not
recovered. This version caps each event at 16 examples, requires at least 24
answer words and retains only whole exchanges within 768 tokens. Selection,
dataset and pinned tokenizer hashes are in `dataset-selection.json` and
`configuration.json`; no heldout answer enters this dataset.

## Verified checkpoint

The current saved counter and resume path are in [latest.json](latest.json).
Six optimizer updates have been preserved and freshly retrieved. The actual
second update ran in a new process restored from the remotely retrieved first
checkpoint, including adapter weights, AdamW state, random state and data cursor.
See `resume-step0002.json` for that real continuation gate and
`checkpoint-step0004.json` for the later saved checkpoint. The step-4 archive
was freshly retrieved, all internal file hashes matched, and the real loader
restored its AdamW, RNG and data cursor (epoch 0, next example 16).
The target is 29 updates for the complete one-epoch dataset, not a completed
step count. Training loss is not an answer-quality or likeness score.

The bounded invocation restored step 2, completed four additional updates and
paused at step 6. Its fresh archive and real optimizer/RNG/cursor load passed.
The actual runtime reports eight
Torch CPU threads. Earlier updates used four. The data order, seed, BF16
precision, gradient accumulation, optimizer and frozen 29-update budget are
unchanged. Small kernel measurements support trying eight threads but do not
establish an end-to-end speedup. The alternative cast-to-FP32 linear experiment
was rejected; the native implementation remains in use.

## Resume command

Restore the exact dataset and checkpoint archive using the stable artifact IDs
in `data/runs/jensen-2026-11-13/artifacts.json`, check the archive SHA-256, extract
it into a new directory, and validate its internal checkpoint manifest. `ART`
below denotes the restored artifact root; it is not a repository output path.

```sh
HF_HOME=/absolute/path/to/model-cache \
PYTHONPATH=src TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
.venv-inference/bin/python -m tools.adaptation train-text \
  --dataset "$ART/text-dataset-v1" \
  --output "$ART/text-adapter-v1" \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --device cpu --precision bfloat16 --rank 8 --epochs 1 \
  --max-length 768 --gradient-accumulation 4 --max-steps 29 \
  --checkpoint-every 2 --stop-after-steps 4 --gradient-checkpointing \
  --resume-from "$ART/readback/text-resume-step0002/step-00000002"
```

That exact bounded command completed. A coordinator-supervised continuation can
keep the model in memory while waiting at every second update for remote
verification. Resume from the **latest freshly restored** checkpoint, omit
`--stop-after-steps`, and add the following publisher argument (substitute the
actual absolute paths):

```sh
--checkpoint-publisher '["/absolute/path/to/.venv/bin/python","-m","tools.adaptation.checkpoint_gate","--handoff","/absolute/path/to/jensen-2026-11-13/text-checkpoint-gates","--timeout","1800"]'
```

The existing trainer calls the publisher synchronously after the checkpoint is
atomic. The gate creates `step-STEP-METADATA_HASH/request.json` and waits for
`verified.json` in that directory. It stops with an error after 30 minutes without
a valid receipt. The coordinator must archive the checkpoint, save it remotely,
freshly retrieve and hash-check every byte, run the actual checkpoint loader,
publish the corresponding public receipt and verify that Git checkpoint before
acknowledging the gate. The waiting process itself does not contact storage.

The acknowledgement is an atomic JSON object containing `status: "verified"`,
`verification_status: "fresh_materialization_and_checkpoint_load_verified"`,
the exact `optimizer_steps` and `checkpoint_metadata_sha256`, stable
`persistent_locator` (the saved artifact ID), `file_id`, `archive_sha256`,
`archive_bytes`, and the freshly verified `public_checkpoint_sha`. Missing or
mismatched identity never releases the next optimizer update. The gate also
applies to the final optimizer checkpoint before serving artifacts are exported.

The trainer writes an atomic local checkpoint every two updates and on pause.
After a bounded invocation exits, package the latest checkpoint and its paused
run receipt, upload them, retrieve a fresh copy, verify the complete archive and
internal hashes, and resume from that fresh directory. Record the actual local
and remote step counts separately. A later block changes only the resume path
and log filename, keeping the 29-step binding intact. On the final block the
trainer exports the adapter, tokenizer and completed serving receipt.

The serving path must load the saved completed adapter with its exact receipt
hash. The frozen matched generation experiment is a separate gate; no default
promotion follows merely from completing training.

## Recover an execution connection

At step 8 the external tool handles disappeared. The checkpoint archive was
already uploaded; fresh archive, payload and real state-loader verification
subsequently passed. The old process did not consume its valid acknowledgement,
and its mutable run receipt remained at step 7. The atomic checkpoint and its
verified remote receipt establish the actual saved step. The observation and
next bounded runtime choice are preserved in `session-recovery-step0008.json`.

`tools.adaptation.supervise` now keeps a shared nonblocking process lock and an
atomic file heartbeat independently of the interactive tool handle. It owns the
child process and records actual exit/cleanup and maximum waited-child RSS.
It also accepts an atomic `stop-NAME.json` containing the exact invocation
`name` and a nonempty `reason`, then stops only its owned child process group.
The supervisor's successful exit is not a completed-training claim: inspect the
trainer's separate run receipt. Existing state/log names cannot be overwritten.

Invoke it from the repository root with the inference virtualenv's actual path:

```sh
PYTHONPATH=src:. .venv/bin/python -m tools.adaptation.supervise \
  --state-dir "$ART/text-process" --name UNIQUE_INVOCATION_NAME \
  --log "$ART/UNIQUE_INVOCATION_NAME.log" \
  --inference-python /absolute/path/to/.venv-inference/bin/python \
  --interop-threads 1 --timeout 14400 -- train-text TRAINING_ARGUMENTS
```

Keep the frozen dataset/model/optimization arguments shown above, choose the
latest verified resume directory, and retain the synchronous checkpoint
publisher. The bounded recovery trial uses four intra-op threads and one
inter-op thread, with two additional updates before pausing. This is a recorded
native runtime choice; no new training algorithm or speedup result is claimed.
Eighteen gate/supervisor checks passed, including real child shutdown after a
file stop request and rejection of a simultaneous supervised invocation.
