# Qwen3-4B LoRA on an Apple M4 Pro

## Korean citation regression

The production Q&A and snapshot prompts now explicitly reserve the literal
`[source:…]` marker while asking for Korean answer prose. The marker parser
still requires the original syntax and an allowed source ID. The archived
prompt builders retain their original behavior.

An actual studio SSE request about the Boring Company returned a localized
marker and no links before the change. The same request on the same pinned
llama.cpp Metal GGUF returned a valid `[source:qa-ted-2017-tunnels]` marker
and the linked evidence ID after the change. This checks citation formatting
and link delivery. Semantic support is evaluated separately.

## Training receipt

The full-precision Hub download stalled. Both arms of this pilot used the
same [pinned Q8_0 GGUF](https://huggingface.co/ggml-org/Qwen3-4B-Instruct-2507-Q8_0-GGUF/blob/e6f794d44f9395d0184a966c27b5ae99ea356fcb/qwen3-4b-instruct-2507-q8_0.gguf).
The runner dequantized 398 tensors to BF16 one at a time on the Mac, checked
their names and shapes, and restored the tied output embedding. Config,
tokenizer and generation metadata came from the
[pinned upstream Qwen revision](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554).
The resulting 4,022,468,096-parameter base reflects the Q8_0 source weights.

The Apple M4 Pro ran a rank-8 all-linear LoRA on `mps:0` with PyTorch 2.15.0.
One epoch processed 38 of 40 candidate examples in 10 optimizer steps, with
16,515,072 trainable parameters. Two whole exchanges exceeded 512 tokens.
Load, training, and serialization took 158.635 seconds. Gradient norms
were finite, frozen parameter versions stayed constant, and the saved
adapter was reloaded before comparison. [Training receipt](training.json)

The seed-42 dataset used eight complete training exchanges per event from
Dwarkesh 2026, Henry Ford 2008, Recode 2018, TED 2017 and WIRED 2012.
YC 2016 changed from the published transcript hash and was excluded.
The five retained events and their selections matched the committed
[materialization recipe](../../persona-data/combined-2026-v2/materialization-recipes.json).
The acquired originals, learned weights and raw outputs are local artifacts.

## Paired answer quality

Three development questions were evaluated in English and Korean. The supplied
evidence events were disjoint from the five adapter training events. The
same frozen prompts, one factual record per question, tokenizer, BF16 base,
greedy decoding, seed 42 and 192-token limit were used for both arms. The
baseline disabled PEFT adapters on the same frozen model. The second arm
activated the reloaded adapter. All 12 calls ended with EOS. The implementing
assistant scored answers against the fixed [rubric](rubric.json) with
condition names hidden, then joined the grades to the condition key.
Output length could still suggest a condition.

| Measure across six cases | Base | LoRA |
| --- | ---: | ---: |
| Answered all expected points | 6 | 3 |
| Fully supported, with all expected points | 1 | 2 |
| At least one sentence with a valid trailing citation | 5 | 2 |
| Complete sentence-level citation coverage | 1 | 2 |
| Korean answers with a valid trailing citation | 3 | 2 |
| Median generated tokens | 122.5 | 47.5 |

The adapter removed unsupported business claims from both Tesla answers and
kept the funded-development point. It omitted the distance caveat in English
Hyperloop and the reusable-launch rationale in both MIT answers. The base's
English MIT answer emitted a malformed marker with a space after the colon.
The adapter omitted the Tesla citations in both languages. The base handled
the Korean Hyperloop qualification more accurately.

[Per-case judgments and output hashes](comparison.json) record the observed
tradeoffs. Keep this adapter out of the default runtime. This small bilingual
development pilot leaves human quality and persona fidelity ungraded.

## Reproduce

Use the pinned GGUF and the Qwen metadata revision listed above. Download
only `config.json`, `generation_config.json`, `model.safetensors.index.json`,
`merges.txt`, `vocab.json`, and `tokenizer*.json` from the Qwen revision.
Build the five-source projection by filtering the committed recipe's
`sources` and `selections` to those source IDs and preserving their order
in the acquisition receipt. Use a new directory for each run.

```bash
uv run python -m tools.adaptation export-text \
  --projection data/local/mps-lora/corpus-build/projections/text_style.jsonl \
  --output data/local/mps-lora/training --seed 42 --max-per-event 8

report=reports/adaptation/m4pro-4b-q8-v1
uv run python -m tools.adaptation_evaluation freeze \
  --corpus data/personas/elon-musk --development-cases "$report/cases.json" \
  --selections "$report/selections.json" --id m4pro-4b-q8-lora-v1 \
  --base-model ggml-org/Qwen3-4B-Instruct-2507-Q8_0-GGUF \
  --base-revision e6f794d44f9395d0184a966c27b5ae99ea356fcb \
  --max-tokens 192 --context-window 8192 --output data/local/mps-lora/plan-gguf

HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONPATH=. \
  .venv-inference/bin/python -u "$report/run.py" \
  models/qwen3-4b-instruct-2507-q8_0.gguf "$QWEN_METADATA_SNAPSHOT" \
  data/local/mps-lora/training data/local/mps-lora/plan-gguf/plan.json \
  data/local/mps-lora/q8-run-replay
```

The optional inference environment uses `services/language/requirements-adapter.txt`
and `gguf==0.19.0`. The exact package versions are in the training receipt.
This experiment uses direct Transformers inference with a local Q8-derived
base. The existing verified serving path uses original Transformers weights
and its own receipt schema.
