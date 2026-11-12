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
