# Conditional persona values: execution record

Repository: `Dindb-dong/who-is-your-bias`.
Authorized target: PR #29, `feat/jensen-persona-graph-nov20`.
Starting remote head: `6631af9dd993e5134c642dfaf397e2156702898a`.
Implementation checkpoint: `d65817048778ed87d48ad037dded05d6cba2b4b6`.
Frozen inputs checkpoint: `c19b6e53b6302dd87ebc819b25c2757eb41e1eb6`.
Read [the method and twelve strategies](persona-values-v2.md) for the full design.
The existing default corpus, v1 graph, application defaults and other team work
are preserved. Main is not the write target.

## Implemented, pending empirical acceptance

- Backward-compatible Python graph v2 loader and deterministic situational
  retrieval. Values, beliefs, interaction style and conditional strategies are
  separate. Evidence can support, qualify or contradict interpretations.
- Conditions and exceptions remain in complete prompt units. Missing evidence,
  draft interpretations, mismatched personas/events, or hash drift exclude the
  affected memory. Multi-source interpretations require distinct events and
  experience groups. There is no additional runtime LLM call or training.
- Optional expanded corpus: original 32 records preserved, plus 28 bounded
  paraphrases; 60 records, 16 resources and 13 event families. The v2 graph has
  13 claims, 12 conditional strategies and 24 source anchors. Curation is by AI
  agents, not human certification or automatic extraction-model training.
- Four newly acquired events: Stanford GSB 2024, No Priors 2023, Milken 2026,
  Stanford ETL 2009. The last event is represented by four complete clip
  transcripts, not a full-event transcript. Acquired/Stripe were re-read.
- A four-condition evaluation runner preserves baseline, expanded RAG, flat
  rules and graph rules with matched model/decoding/output budgets. Novel
  authored scenario pairs are not held-out original Jensen answers.
- A hash-bound capture time can establish same-day availability when the true
  publication timestamp is unknown. Generic ingestion metadata cannot bypass
  the conservative publication cutoff. Historical captures do not become
  current stances. This resolves a seed integration failure without backdating
  a source or advancing the snapshot into the future.

## Durable source capture

Full source material is kept outside public Git. The private research archive
`jensen-persona-source-capture-2026-11-21.zip` was saved and independently
materialized with identical SHA-256:

`4b80acb5175ec9f1b6c8d388c369ab344f934bf2f3e52a2453c697a1ff2e3024`

Size: 954,869 bytes. It contains the seven new source resources and review
text, plus re-acquired Acquired/Stripe HTML and source receipts. Raw captures
and extracted review text retain their own hashes. The public acquisition
manifest records exact URLs, event identity and coverage. Captions/timestamps
have not been acoustically verified. No audio/video alignment is claimed.

## Acceptance gates

At this checkpoint the code and curated candidate exist. Model comparison,
blind source-aware ratings, failure analysis and promotion are still pending.
Do not infer improved value fidelity from retrieval hits or unit-test passes.
The optional feature remains off by default. This pilot does not establish
current news coverage, project knowledge or actual private feelings.

Final results and exact reproduction commands will be linked here after the
frozen experiment completes. The earlier PR29 result remains preserved in
`reports/persona-graph-2026-11-20/`; its 83.3% score is not directly comparable
with a different question set or rubric.
