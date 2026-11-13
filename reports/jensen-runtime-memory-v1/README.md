# FP32 loading memory diagnosis

This is a **read-only source/header diagnosis** of the pinned Qwen3-1.7B load.
The proposed native-load-then-upcast route is **UNEXECUTED**. Current frozen BF16
experiments continue with their existing code and contracts. No model was
initialized, no Torch import was needed, and no tensor payload was read or
compared for this diagnosis.

[diagnosis.json](diagnosis.json) preserves the installed source-file SHA-256
values, package versions, snapshot/header identities, exact byte accounting,
observed failure reference and candidate limitations. The inspected environment
has Transformers 4.57.3, PyTorch 2.11.0+cpu, Accelerate 1.12.0 and safetensors
0.8.0. Two official web lookups returned HTTP 503; the evidence is the exact
installed primary source and pinned local headers.

## Observed failure and identified loading path

The [preserved FP32 attempt](../jensen-evaluation-v1/runtime-amendment-bf16-v2.json)
ended with server return code `-9` during the separate M6 excluded warmup, before
any evaluation question or answer. Its supervisor recorded 91.894 seconds; the
process owner observed maximum waited-child RSS of 8,191,066,112 bytes. The
failed archive and all attempt files were freshly retrieved and verified. This
is consistent with memory pressure; no kernel OOM attribution was available.

The pinned snapshot is `Qwen/Qwen3-1.7B` revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Its configuration declares tied input
and output embeddings. The first safetensors shard contains 310 BF16 tensors;
the second contains only the declared tied `lm_head.weight`, shape
`[151936, 2048]`.

| Header-derived quantity | Elements | FP32 bytes |
| --- | ---: | ---: |
| Expected unique parameters after tying | 1,720,574,976 | 6,882,299,904 |
| Separately serialized output head | 311,164,928 | 1,244,659,712 |
| All serialized tensors before final tying | 2,031,739,904 | 8,126,959,616 |

Transformers 4.57.3 casts each serialized parameter and assigns it to its named
module, then repeats `tie_weights()` after all shards load. The extra FP32 head
allocation is therefore a concrete candidate for the peak. These counts do not
measure allocator overhead, source-page residency or the exact allocation that
triggered termination. Header inspection does not establish bitwise equality of
the two stored tensor payloads.

The service already passes `low_cpu_mem_usage=True`. This installed Transformers
version removes both that argument and `offload_state_dict` as unused kwargs.
Setting `device_map="cpu"` retains the per-name cast/assignment path and does not
remove the identified duplication.

## Unexecuted candidate: native load, tie, then upcast

Load the unchanged snapshot explicitly in its stored BF16 dtype, let its normal
weight tying finish, and convert the tied **base** to FP32 before loading PEFT.
Under the installed default conversion flags, PyTorch's `Module.float()` uses
`_apply` and retains ordinary compatible dense CPU `Parameter` objects through
in-place data assignment. The head should therefore share the already converted
embedding instead of allocating another FP32 head. This is an inference from
source, not an observed memory or sharing result.

Every stored tensor is BF16, so converting those original values to FP32 is
exact. Qwen3's default nonpersistent RoPE `inv_freq` buffer is explicitly
computed in FP32. The original snapshot files, index, tokenizer and their hashes
can remain unchanged. Loading mixed or FP32-source checkpoints as BF16 first
would require a different analysis because it could discard precision.

`from_pretrained` records the load dtype in `model.config.dtype`;
`Module.float()` does not update it. A prospective implementation must correct
only the **in-memory** configuration to FP32, retain the original cached files,
verify actual dtype/non-meta state/shared embedding-head identity, and record
its own implementation hash. PEFT loading should follow base conversion so it
receives the same FP32 base as the original serving path.

## Remaining limits

Unique FP32 parameters still require about 6.410 GiB. A batch-one FP32 KV cache
for 4,096 tokens adds 939,524,096 bytes (0.875 GiB), derived from 28 layers,
two key/value arrays, eight KV heads and 128 dimensions per head. Conversion
temporaries, mapped files, tokenizer/runtime allocations and attention workspace
also need headroom. A short warmup does not establish longest-prompt capacity.

Successful FP32 loading, whole-model numerical equivalence and FP32 throughput
relative to BF16 remain unmeasured. This report introduces no runtime change,
speedup claim or new model experiment. Finish the current frozen BF16 experiments
unchanged; M7 continues to wait for the verified completed 29-update adapter and
coordinator release of the model slot.
