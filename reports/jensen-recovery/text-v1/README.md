# Jensen text adapter: bounded execution and continuation

The immutable experiment uses 113 complete original exchanges across all nine
eligible training events. The earlier interrupted 105-answer selection was not
recovered. This version caps each event at 16 examples, requires at least 24
answer words and retains only whole exchanges within 768 tokens. Selection,
dataset and pinned tokenizer hashes are in `dataset-selection.json` and
`configuration.json`; no heldout answer enters this dataset.

## Verified checkpoint

Two optimizer updates have been preserved and freshly retrieved. The actual
second update ran in a new process restored from the remotely retrieved first
checkpoint, including adapter weights, AdamW state, random state and data cursor.
See `resume-step0002.json` for exact progress, hashes and artifact identity.
The target is 29 updates for the complete one-epoch dataset, not a completed
step count. Training loss is not an answer-quality or likeness score.

The next bounded invocation has started from the freshly retrieved step 2 and
will pause after four additional updates. The actual runtime reports eight
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
