# Korean Jensen retrieval comparison

The default lexical and opt-in hybrid retrievers found and cited the same two of six expected passages. Neither completed a response within five seconds. This is a baseline on eight questions, with one request per question per backend.

| Metric | Lexical | Hybrid |
| --- | ---: | ---: |
| Expected passage returned in top four | 2/6 | 2/6 |
| Expected passage cited in final result | 2/6 | 2/6 |
| Unrelated passage on no-evidence controls | 0/2 | 0/2 |
| Median retrieval time | 1.4 ms | 77.6 ms |
| Median completion time | 11.97 s | 11.64 s |
| Completed within five seconds | 0/8 | 0/8 |

[Raw results](korean-retrieval-comparison.json) include the Korean prompts, expected passage IDs, selected and cited IDs, timings, and generated text. The harness called the existing FastAPI streaming route with real local Ollama generation (`qwen2.5:14b`, 160-token cap). Hybrid used `batiai/qwen3-embedding:0.6b`, a 0.84 dense gate, and no reranker. Each backend received a warmup, and each scored request had a fresh session. Completion time is wall time until receipt of the full SSE response, with no browser or speech playback. Generation dominated completion time. The small observed difference between backends does not establish a speed advantage.

A separate retrieval trace with the same hybrid settings put the correct answer passage first by cosine similarity for all six supported questions. The four misses had scores of 0.571 (rack design), 0.622 (TSMC), 0.449 (TPU), and 0.594 (employee updates), so the 0.84 gate rejected all four. The Mars control reached 0.437, close to the TPU score. The other control was excluded by the date policy. Calibrate the gate on a separate, broader Korean set before changing it; all six ranked first on this sample does not establish safe thresholds.

Both backends labeled the Mars answer as hypothetical with no citation and declined to supply the contents of the future private meeting. Some responses mixed Korean with Chinese or Vietnamese, and the no-evidence TPU answers included unsupported specifics despite the label. Citation hits measure retrieval and citation selection; they do not establish factual accuracy or persona fidelity.

## Korean threshold calibration

[Calibration and held-out questions](korean-calibration.json) were written before the sweep. Each positive has one reviewed target passage. A negative asks for information the corpus does not support, including some queries that share terms with the corpus. The selection rule maximized target recall at four without increasing the number of negatives that returned evidence compared to 0.84. It broke ties by top-one recall and then by the higher threshold.

| Split | Targets at 0.84 | Targets at 0.60 | Negatives with hits at 0.84 | Negatives with hits at 0.60 |
| --- | ---: | ---: | ---: | ---: |
| Calibration | 4/10 | 5/10 | 2/10 | 2/10 |
| Held-out | 2/6 | 2/6 | 0/6 | 0/6 |
| Original benchmark | 2/6 | 3/6 | 0/2 | 0/2 |

[Threshold sweep and held-out results](korean-threshold-results.json). The misses and the two calibration false positives justify keeping 0.84 as the general default. To reproduce the local experiment with the same embedding model and prefixes, opt in to `BIAS_RETRIEVAL_BACKEND=hybrid`, `BIAS_EMBEDDING_BACKEND=http`, `BIAS_EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1`, `BIAS_EMBEDDING_MODEL=batiai/qwen3-embedding:0.6b`, and `BIAS_DENSE_MIN_SIMILARITY=0.60`. Run `PYTHONPATH=src python reports/jensen-rag/evaluate_followup.py` to repeat the sweep.

## Smaller local models

The follow-up reused the original eight Korean prompts and the 160-token cap. Each backend had one warmup and eight fresh conversations through the existing FastAPI SSE route. Hybrid used 0.60. Latency is wall time until the whole response arrived. Cold model loads, browser rendering, and speech playback were outside the measurement.

Qwen2.5 3B ran through Ollama. Qwen3 4B used the repository's existing Q8_0 model and the [Metal llama.cpp recipe](../../docs/deployment.md#apple-silicon-text-inference) on the same M4 Pro. Runtime, quantization, and prompts can affect the comparison with the earlier 14B result. Each question was measured once per listed configuration. The 14B follow-up was recorded before the final Korean summary guard and its prompt.

| Model | Lexical median | Lexical within 5 s | Hybrid median | Hybrid within 5 s | Hybrid answers with target citation |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Qwen2.5 14B follow-up](qwen2.5-14b-korean-results.json) | 13.72 s | 2/8 | 9.02 s | 1/8 | 3/6 |
| [Qwen2.5 3B Instruct](qwen2.5-3b-instruct-korean-results.json) | 1.11 s | 7/8 | 2.22 s | 7/8 | 1/6 |
| [Qwen3 4B Instruct Q8_0](qwen3-4b-instruct-2507-korean-results.json) | 2.48 s | 7/8 | 3.48 s | 4/8 | 2/6 |

The three models retrieved the same three expected passages with hybrid. The citation counts exclude source-only responses produced by the output guard. The 3B model cited only the AI factory answer in hybrid, and exposed a prompt field name in that answer. It returned a source-only response for the founder question and missed the citation on TSMC. The 4B model answered and cited the TSMC and founder questions. It returned a source-only response for the AI factory question in hybrid. None of the three hybrid requests with retrieved target evidence finished within five seconds on the 4B model. One timed request per backend, the actual future quotation request, returned an application response without generation.

The Korean output guard checks a completed draft before sending it to the client. For supported answers, it checks for other scripts, direct quotations, and first-person speech. For missing evidence, it accepts one conditional criterion and supplies a generic conditional criterion if the draft fails. Both smaller models' final runs had no letters outside Hangul and ASCII in the displayed answers. The 4B model's six held-out controls did not return citations. On the school lunch hypothetical, it proposed examining which times have the most recurring delays. It declined to supply an actual quote from a future private board meeting. These checks do not establish factual entailment or the quality of all generated advice.

To repeat the 3B API run, use `PYTHONPATH=src python reports/jensen-rag/evaluate_followup.py --threshold 0.60 qwen2.5:3b-instruct`. For 4B, start the loopback llama.cpp server described above and pass its `--llm-base-url`, `--guard-checks`, and the configured model alias.
