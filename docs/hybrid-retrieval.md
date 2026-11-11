# Hybrid retrieval: actual execution, thresholds, and remaining errors

Recorded on **2026-11-11, Asia/Seoul**. This implements the retrieval portion of
[the 2026 roadmap](2026-persona-roadmap.md). The original BM25 retriever, its
version, and the failed frozen Q&A experiment remain unchanged. Hybrid retrieval
is an explicitly configured application capability, with separate development
results rather than a replacement for the historical experiment.

## Delivered behavior

`bias.hybrid_retrieval.HybridRetriever` indexes only reviewed, source-grounded
training records. The same corpus split guard runs before either lexical or
dense indexing. A caller's `eligible_ids` can remove records for temporal or
project reasons; it cannot add a held-out record to the index. Original-Q&A
projection flags are enforced independently: `factual` controls the factual
corpus, and `text_style` controls the complete original exchanges available for
style. Missing flags retain the historical corpus contract.

The new path combines three rankings: the original strict lexical baseline,
embedding search over answer subspans, and embedding search over original
questions with original preceding context. Editorial context summaries are not
silently inserted into the question view. Reciprocal rank fusion joins these
candidate lists by parent evidence ID.

Answer relevance is a separate gate. A high question similarity, high fused
rank, matching situation, or recent publication cannot by itself pass it. A
candidate must pass the strict lexical gate or the configured answer-similarity
threshold, and, when enabled, the independently configured cross-encoder gate.
Temporal eligibility runs before selection. Bounded temporal and situation
weights apply only to candidates that passed relevance. The application still
must check whether the supplied passage supports each generated claim; retrieval
scores are not entailment probabilities.

Long text is indexed in bounded, contiguous subspans, with exact character
offsets back to the unchanged parent exchange. The retriever returns that whole
parent, including qualifications. It never presents an index subspan as though
it were a complete original answer. Exact normalized answer duplicates are
removed and a default maximum of two factual candidates per recording event
limits domination by one long interview.

Style examples are opt-in through `include_style=True`. They are returned in
`result.style_hits`, separate from `result.hits`. A style-only example never
enters factual candidate rankings or the factual citation allowlist. The caller
can pass `eligible_style_ids` separately; supplying a factual eligibility gate
without a style gate restricts style to that factual set. Both paths retain
train/review restrictions. This opt-in is disabled in the controlled B/C
retrieval comparisons below, so style prompting is not an undeclared extra
intervention.

Every hybrid result has a structured trace containing the original and resolved
query, corpus digest, model/revision, candidate IDs, subspan offsets, lexical and
dense scores, fusion ranks, relevance decisions, temporal and situation weights,
filter reasons, and the final factual/style IDs. The trace omits complete source
passages and credentials. The transcript corpus remains a separate versioned
input.

## Real multilingual development comparison

The [registered case manifest](../tools/retrieval_evaluation/cases/multilingual-development.v1.json)
contains **100 answerable questions**: English/Korean paraphrase pairs for 50
reviewed summary records from 28 source events. It also contains **50
absent-evidence questions**, including nearby topics whose requested facts are
not present. An agent inspected the support records and authored the labels;
human review is ungraded. Translation pairs are correlated observations, not
independent events. These are development cases on an already exposed corpus,
not a newly held-out efficacy test.

All dense scores below came from actual local neural inference with
`intfloat/multilingual-e5-small`, pinned to
`614241f622f53c4eeff9890bdc4f31cfecc418b3`. The reranked run also executed
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, pinned to
`1427fd652930e4ba29e8149678df786c240d8825`. No fabricated vectors were used for
these measurements. Synthetic vectors appear only in isolated unit tests.

| Development configuration | Answerable questions with registered support in top 4 | English | Korean | Absent questions returning any candidate |
| --- | ---: | ---: | ---: | ---: |
| Preserved strict lexical baseline | 48/100 | 39/50 | 9/50 | 5/50 |
| First E5 run, cosine floor 0.84 | 58/100 | 47/50 | 11/50 | 5/50 |
| E5 floor 0.78, corrected Korean date/current guards | 94/100 | 48/50 | 46/50 | 9/50 |
| E5 floor 0.78 + cross-encoder raw-logit floor −4 | **92/100** | **46/50** | **46/50** | **2/50** |

The first two configurations exposed a real tradeoff: lowering the embedding
threshold recovered Korean support but also admitted more unsupported
candidates. The pair scorer improved that development tradeoff. These thresholds
were chosen on development data and must not be treated as universal or as
validated on unseen events. In particular, **do not apply the 0.78 recipe while
silently omitting its reranker**.

The final two false positives are the English/Korean versions of a question
about “Article 42” of an allegedly adopted Mars constitution. The archive has
no such article, but related source material still passes retrieval. This is a
concrete remaining false-premise/answerability failure. Eight registered
answerable questions also miss their specified supporting record. The score
improvement does not mean arbitrary unsupported questions are solved.

B and C have identical results on this summary set. No situation-retrieval
advantage, generated-answer quality, human likeness, or current-2026 accuracy
is established by this comparison.

The complete records retain every case, selected IDs, candidate scores and
filter reasons, runtime versions, model-file hashes, corpus and implementation
hashes, and failures:

- [First E5 run](../tools/retrieval_evaluation/results/e5-small-cpu-development-threshold084.v1.json.gz)
  and its [exact implementation archive](../tools/retrieval_evaluation/results/e5-small-threshold084-implementation.v1.tar.gz).
- [Lower-threshold run](../tools/retrieval_evaluation/results/e5-small-cpu-development-threshold078.v2.json.gz)
  and its [exact implementation archive](../tools/retrieval_evaluation/results/e5-small-threshold078-implementation.v2.tar.gz).
- [E5 plus actual cross-encoder run](../tools/retrieval_evaluation/results/e5-mmarco-cpu-development.v3.json.gz).

The `.json.gz` files are ordinary losslessly compressed UTF-8 JSON, with zero
gzip modification time for deterministic packaging. The earlier code snapshots
are retained because Korean query guards and inference caching changed during
development. They match the implementation digests in their corresponding
reports. Failed pilot artifacts elsewhere in the repository are untouched.

## Expanded original-Q&A check

The [separate original-Q&A case manifest](../tools/retrieval_evaluation/cases/original-qa-expanded-development.v2.json)
registers 36 bilingual paraphrases of 18 inspected original exchanges across the
nine newly acquired training events, plus 20 absent-evidence cases. Four
questions explicitly ask about positions expressed in the 2026 interview.
This checks retrieval on the actual expanded original text rather than the
historical summary projection. It remains a **held-in development** check.

The local corpus contains 977 original exchanges, of which **745 are eligible
for factual retrieval and 922 for text style**. The index embeds their union
once, then keeps factual and style membership separate before top-k selection.
The [data acquisition report](persona-data.md) records how these were obtained
and which projections are eligible. These counts do not represent 977
human-audited factual assertions or 977 audio-verified speech segments. Its new
situation annotations remain the explicit generic conversation label; a C
advantage cannot be manufactured from that label.

The first expanded attempt ran while source metadata was being corrected and
before the explicit factual/style projection flags were exported. Its
[diagnostic report](../tools/retrieval_evaluation/results/e5-mmarco-expanded-original-qa.v1.invalid-input-drift.json.gz)
is preserved with an [invalidity notice](../tools/retrieval_evaluation/results/e5-mmarco-expanded-original-qa.v1.invalid-input-drift.json).
It is excluded from completed comparison claims. The corrected case manifest
binds the stable corpus, validates each expected support record against factual
eligibility, and predeclares one stricter cross-encoder sensitivity check.

Both completed runs used dense cosine floor 0.78, the same pinned E5 and mMARCO
models above, the same corrected input hashes, and an unchanged implementation.
There were no provider failures or input changes during either run.

| Expanded development configuration | Registered support in top 4 | English | Korean | Absent questions returning any candidate | Explicitly dated 2026 support |
| --- | ---: | ---: | ---: | ---: | ---: |
| Preserved strict lexical baseline | 3/36 | 3/18 | 0/18 | 0/20 | 0/4 |
| E5 + cross-encoder floor −4 | 28/36 | 16/18 | 12/18 | **4/20** | **1/4** |
| Single stricter sensitivity, floor −3 | 27/36 | 16/18 | 11/18 | 0/20 | **1/4** |

The expanded corpus exposes a precision regression at the −4 floor. Related
passages are returned for questions whose requested information is absent,
including a password, a private money transfer, a full mathematical proof, and
a pet's vaccination history. These are false **retrieval candidates**, not
observed generated answers. The stricter −3 development setting removes those
four candidates while losing one additional Korean support hit. It is selected
only for the **optional expanded-corpus recipe**, based on these held-in cases.
Zero errors in 20 correlated development questions does not establish a zero
false-support rate on new questions or events. No additional threshold search
was performed after this single sensitivity run.

Only one of the four explicitly dated 2026 questions finds its registered
support under either hybrid threshold. The hiring-traits pair and the Korean
Starship-experience question still miss. Current-2026 answerability is therefore
not established. B and C again match exactly; the generic situation annotations
cannot substantiate a situation-retrieval advantage.

- [Corrected expanded baseline, floor −4](../tools/retrieval_evaluation/results/e5-mmarco-expanded-original-qa.v2.json.gz).
- [Single stricter sensitivity, floor −3](../tools/retrieval_evaluation/results/e5-mmarco-expanded-original-qa-stricter.v2.json.gz).
- [Exact evaluated source archive for both runs](../tools/retrieval_evaluation/results/e5-mmarco-expanded-implementation.v2.tar.gz).
- [Compact comparison and coverage summary](../tools/retrieval_evaluation/results/development-summary.v1.json).

The final public `warm_up()` startup method was added after these runs. The
archive preserves the evaluated code hashes; this method initializes the same
index and reranker without changing search decisions. Its startup behavior and
unchanged warmed/cold retrieval results are covered by focused contract tests.

## Configuration and execution

The application retains lexical retrieval unless hybrid retrieval is explicitly
selected. Local model packages are optional: the base Python install does not
pull Torch or download model weights. Install the `hybrid` extra for local
embeddings/reranking. For a CPU environment, the following dependency install
was exercised during validation:

```bash
uv venv .venv-retrieval
uv pip install --python .venv-retrieval/bin/python --torch-backend cpu \
  'sentence-transformers>=5,<6' 'httpx[socks]>=0.27,<1' pytest
```

The optional [expanded-corpus application recipe](../config/persona-2026.env.example)
uses explicit model revisions, the E5 query/passage prefixes, dense floor 0.78,
and cross-encoder floor **−3** selected in the held-in sensitivity check above.
The historical summary comparison below deliberately retains its measured
cross-encoder floor **−4**. The default application remains lexical unless the
hybrid backend is explicitly selected.

The real summary comparison can be repeated with:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src \
  .venv-retrieval/bin/python -m tools.retrieval_evaluation \
  --corpus data/personas/elon-musk \
  --cases tools/retrieval_evaluation/cases/multilingual-development.v1.json \
  --model intfloat/multilingual-e5-small \
  --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 \
  --dense-threshold 0.78 \
  --reranker-model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
  --reranker-revision 1427fd652930e4ba29e8149678df786c240d8825 \
  --reranker-min-score=-4 \
  --output reports/local/retrieval/repeated-development.json.gz
```

The runner refuses to overwrite an existing output. `--revision` and an enabled
reranker's revision require immutable commit IDs. It validates that registered
support belongs to the reviewed training corpus and that the reviewed support
text has not changed. Provider errors are recorded as failures, not credited as
successful abstentions. The report does not include full original passages.

After the pinned models have been downloaded, `HF_HUB_OFFLINE=1` makes cache-only
reproduction explicit. The runner sets Torch's CPU thread count to four by
default. For the expanded local corpus, use the original-Q&A case manifest and
its matching materialized corpus; do not substitute the bundled ten-excerpt
preview and retain the expanded-corpus label.

`OpenAIEmbeddingProvider` supports a separately configured compatible
`/embeddings` endpoint. It checks response indices, cardinality, dimensions,
finite values and nonzero norms, then normalizes vectors. Its tokenizer-less
input guard is a conservative UTF-8 byte budget, explicitly labelled as such.
The local Sentence Transformers adapter uses the actual model tokenizer. Both
reject oversized inputs rather than silently truncating them. Model-loading,
endpoint and reranker errors remain visible; there is no silent downgrade to
lexical retrieval. Exact duplicate query and pair-score computation uses bounded
in-memory caches; corpus embeddings are built once per retriever instance.

Call `HybridRetriever.warm_up()` before accepting interactive requests when
using a new process. It constructs the in-memory corpus index and initializes
the configured reranker; the API can await this through
`BIAS_RETRIEVAL_PREWARM`. The corrected expanded baseline's first hybrid query
took 117.2 seconds including model loading, cold indexing, and its first rerank.
That is a combined cold-query observation, not an isolated index or warm-query
latency measurement. A new process rebuilds the index; no persistent embedding
cache is implemented. Startup failures remain visible instead of reporting an
uninitialized hybrid capability as ready.

## Validation boundaries

Focused tests cover multilingual semantic candidates that have no lexical hit,
absent-evidence abstention, train/review/snapshot isolation, same-parent fusion,
duplicate removal, event caps, subspan continuity, separate style eligibility,
reranker rejection, Korean year suffixes, provider ordering and malformed
vectors, visible provider failure, immutable support labels, and preservation of
the historical lexical results.

The development runs do not establish target-machine conversational latency.
They run retrieval alone on CPU with pinned models, and cache/model-download
conditions differ across runs. The model/runtime identities and per-query times
are available for diagnosis, not as an isolated claim about a simultaneous
LLM/TTS/avatar workload. Fresh independent event families, human answerability
review, generated-claim entailment, and target-device interaction measurements
remain separate evaluation gates.

## Upstream references

- [E5 model card](https://huggingface.co/intfloat/multilingual-e5-small): input
  prefixes, 384-dimensional representation, model limitations and language
  coverage. Its benchmark is not a result on this project.
- [Sentence Transformers retrieve/rerank guide](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html).
- [Multilingual cross-encoder model card](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1):
  pair-ranking interface and multilingual MS MARCO training provenance. Korean
  outcomes are reported separately above rather than inferred from the model name.
