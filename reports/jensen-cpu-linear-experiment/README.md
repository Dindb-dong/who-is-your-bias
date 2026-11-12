# Rejected CPU frozen-linear cast experiment

**Decision: keep the native BF16 training path.** A function-level prototype
cast frozen BF16 weights and inputs to FP32 for `F.linear`, then cast the result
back to BF16. Once nonreentrant activation checkpointing and input backward
were included, it provided no material speed gain at either tested thread count.
No production wrapper, training integration, checkpoint-binding change or full
model run was introduced. The M6 serving/protocol hashes remain unchanged.

## Measured timing

The input shape was `768 × 2048`, with a frozen `6144 × 2048` weight. Every
measurement included linear forward, GELU, input backward and, for the candidate,
all input/weight/output casts. Checkpointed cases used `use_reentrant=False`.
There was one excluded warmup per condition followed by three measured paired
iterations, alternating condition order. Torch was `2.11.0+cpu`.

| Threads | Checkpoint recomputation | Native BF16 median | FP32 cast median | Native time / cast time |
| --- | --- | ---: | ---: | ---: |
| 4 | No | 0.586 s | 0.499 s | 1.175 |
| 4 | Yes | 0.755 s | 0.788 s | 0.958 |
| 8 | No | 0.353 s | 0.541 s | 0.653 |
| 8 | Yes | 0.439 s | 0.461 s | 0.954 |

A ratio above one favors the candidate. The isolated uncheckpointed four-thread
improvement did not carry to the actual checkpointing contract. Host timings
were variable and other work could be concurrent; this small probe supports
stopping this candidate, not a precise whole-model slowdown estimate. The
[raw timings](kernel-results.json) retain every measured value.

## Numerical and memory diagnostics

The large-kernel maximum output difference was `0.015625`, with relative L2
difference about `3.45e-5`. Input-gradient maximum difference was
`0.000244140625`, with relative L2 difference about `1.74e-4`.

A separate real PEFT toy model used two LoRA layers, rank four, alpha eight,
dropout 0.05, FP32 trainable adapters, BF16 frozen base weights, nonreentrant
checkpointing and four AdamW updates at learning rate `0.0002`. Both variants
used identical seeds per update. All four losses and final trainable parameters
were equal in that toy run; the declared comparison tolerances were `rtol=0.02`,
`atol=2e-5` for parameters and `1e-4` absolute for loss. Input gradients were
finite, frozen parameters were unchanged, and state-dictionary keys were
unchanged. These toy equalities are not a full-model equivalence guarantee.

Saved-tensor hooks observed an additional **50,331,648-byte FP32 weight tensor**
retained for backward in the uncheckpointed cast path. The native path retained
a view of its existing BF16 parameter. With checkpointing, neither variant
retained an FP32 weight tensor between forward and backward in this probe;
the candidate still needed temporary casts during recomputation. These are
saved-tensor observations, not peak process RSS. GEMM workspace, allocator
behavior, complete transformer blocks and large output logits were not measured.
Blindly applying this approach to an uncheckpointed large language-model head
would require separate memory analysis.

This interpretation follows PyTorch's documented
[checkpoint recomputation contract](https://docs.pytorch.org/docs/2.11/checkpoint.html)
and [explicit autocast-disabled regions](https://docs.pytorch.org/docs/2.11/amp.html).
The prototype disables CPU autocast while executing FP32 linear arithmetic.
The [diagnostic record](diagnostics.json) contains the actual saved tensors,
losses, parameter differences and tolerances. The [decision record](decision.json)
also preserves hashes of the scripts actually executed.

## Reproduce the bounded probe

The optional inference environment supplies Torch and PEFT. From the repository
root, choose new output paths:

```sh
.venv-inference/bin/python reports/jensen-cpu-linear-experiment/probe.py \
  --output /tmp/jensen-linear-timing-new.json
.venv-inference/bin/python reports/jensen-cpu-linear-experiment/diagnostics.py \
  --output /tmp/jensen-linear-diagnostics-new.json
```

The committed scripts have the same numerical and timing operations as the
executed private scripts. Only their output destination was made a required
argument with an overwrite guard. They load no pretrained model. Their source
compilation and result JSON checks passed; the diagnostic assertions ran on
the actual optional Torch/PEFT environment. The candidate is rejected, so no
`tools/adaptation/cpu_linear.py` runtime utility or new training run is proposed.
