# Deliver an atomic text checkpoint

`python -m tools.adaptation.checkpoint_delivery` provides three additive
coordinator commands: `package`, `verify`, and `acknowledge`. It leaves the
trainer, waiting gate and process supervisor unchanged. These commands handle
trusted training checkpoints, not user-uploaded inference packages.

The module performs **no upload, download or Git operation**. Stable object IDs,
an evidence reference and a commit SHA supplied on the command line are external
coordinator attestations. They are not independently verified remote facts.

## 1. Package the published checkpoint directory

Use the exact atomic `step-NNNNNNNN` directory named by the gate request. Choose
new archive and receipt names outside that directory. `ART` below is a restored
artifact root, not a repository output directory.

```sh
PYTHONPATH=src:. .venv/bin/python -m tools.adaptation.checkpoint_delivery package \
  --checkpoint "$ART/text-adapter-v1/checkpoints/step-00000008" \
  --archive "$ART/delivery/step0008-attempt2.tar.gz" \
  --receipt "$ART/delivery/step0008-attempt2.package.json"
```

Packaging validates the checkpoint schema, step, report, binding digest, every
registered payload hash, and the closed file/directory inventory. It writes
only regular files in the archive, then rereads its actual payloads before
publishing the archive and package receipt. Links, traversal, special files,
unregistered files/directories and replacements of existing outputs are refused.
No Torch import or state load occurs during packaging.

The immutable checkpoint already contains its report. The mutable trainer
`run.json` is deliberately not copied: it can lag behind an atomic checkpoint
while the process waits for acknowledgement. This package does not replace the
separate completed serving export, tokenizer and final run receipt.

Upload the archive through the authorized artifact route and retain the actual
stable object IDs and transfer receipt. Preserve the package receipt separately
so its expected hashes are available before readback. Freshly materialize the
same remote object to a different local file. A local copy alone does not prove
that upload or remote readback happened.

## 2. Verify the materialized bytes and actual state loader

Use an environment with Torch, without starting a language model:

```sh
PYTHONPATH=src:. .venv-inference/bin/python -m tools.adaptation.checkpoint_delivery verify \
  --archive "$ART/fresh-readback/step0008-attempt2.tar.gz" \
  --package-receipt "$ART/delivery/step0008-attempt2.package.json" \
  --output "$ART/fresh-readback/step0008-attempt2-restored" \
  --proof "$ART/delivery/step0008-attempt2.proof.json"
```

This compares the archive's size and SHA-256, accepts exactly the registered
regular-file members, and checks every restored payload. It calls the existing
`load_checkpoint(..., binding=...)` on the restored directory, loading the actual
saved optimizer, RNG and cursor on CPU. It requires populated optimizer state,
valid text progress/cursor, and usable Python and Torch CPU RNG state in isolated
RNG objects. It does not initialize a model, mutate the process's global RNG,
resume training or prove the adapter's response quality. Additional accelerator
RNG states are deserialized but not restored to a device.

Only successful validation emits the new proof. The command prints its SHA-256
for the acknowledgement step. The proof records `remote_storage_checked: false`,
`remote_materialization_origin_established: false`, and `git_commit_checked:
false`; a different local pathname does not establish those external facts.

`reported_elapsed_seconds` retains the trainer's reported cumulative elapsed
time. It includes checkpoint-publisher waits within a training invocation and
must not be relabeled as pure active compute time.

## 3. Publish and verify the public checkpoint, then acknowledge

The coordinator must preserve the proof and artifact receipt, publish the
corresponding source-text-free public evidence, freshly read the exact Git commit
and changed files, and compare their bytes. Retain references to both the actual
remote-materialization evidence and the actual Git-readback evidence. Do this
before acknowledging the waiting trainer.

Substitute the existing gate directory, actual proof SHA-256, stable artifact
IDs, already externally verified public commit SHA and evidence references:

```sh
PYTHONPATH=src:. .venv/bin/python -m tools.adaptation.checkpoint_delivery acknowledge \
  --proof "$ART/delivery/step0008-attempt2.proof.json" \
  --proof-sha256 FULL_PROOF_SHA256 \
  --request "$ART/text-checkpoint-gates/step-00000008-METADATA_SHA12/request.json" \
  --artifact-id jensen-text-checkpoint-step0008 \
  --file-id file_ACTUAL_OBJECT_ID \
  --library-file-id libfile_ACTUAL_OBJECT_ID \
  --externally-verified-public-commit FULL_VERIFIED_COMMIT_SHA \
  --materialization-record ACTUAL_TRANSFER_EVIDENCE_REFERENCE \
  --git-readback-record ACTUAL_GIT_READBACK_EVIDENCE_REFERENCE
```

Before writing, the command rechecks the proof's expected digest, required local
checks, materialized archive identity, restored payloads, original atomic
checkpoint and existing gate request. It then creates the gate's `verified.json`
atomically without replacing an earlier decision. The Python
`build_acknowledgement` API performs the same checks and returns the prospective
receipt without writing it.

The gate-compatible `verification_status` describes the combination of local
proof and the coordinator's named external checks. `external_verification`
explicitly records that this module did not check storage or Git. The proof is
an unsigned local record; it is not a remote attestation service or an authority
to bypass the external checks. An existing output or failed attempt requires a
new output name and inspection of the retained evidence, not an overwrite.

## Focused verification

```sh
.venv/bin/python -m pytest tests/adaptation/test_checkpoint_delivery.py -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-inference/bin/python -m pytest \
  tests/adaptation/test_checkpoint_delivery.py \
  tests/adaptation/test_checkpoint_delivery_state.py -q
```

The optional Torch tests use one scalar parameter and a real AdamW state. They
exercise the actual checkpoint loader and unchanged verification gate, with no
pretrained model, tokenizer, network request or training workload.
