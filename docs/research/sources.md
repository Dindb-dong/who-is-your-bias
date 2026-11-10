# Starter corpus: sources and evaluation boundaries

This corpus is a small, auditable starting point for a public-source persona experiment. It contains **28 original paraphrases from seven primary sources**, not full transcripts or a trained personality model. It supports source-grounded retrieval and integration tests. It does not establish similarity to the real person's unobserved thoughts, voice, facial expressions, or gestures.

## Manifest

The `sources.json` registry owns dates, publisher attribution, source-level splits, and timeline conventions. `evidence.jsonl` holds brief paraphrases, stable IDs, topics, and source locators. `profile.json` contains presentation guidance explicitly marked as editorial design.

| Source ID | Publication date | Split | Records | Paraphrase words | Location |
| --- | --- | --- | ---: | ---: | --- |
| `tesla-master-plan-2006` | 2006-08-02 | train | 4 | 86 | [Primary source](https://www.tesla.com/secret-master-plan) |
| `tesla-master-plan-2016` | 2016-07-20 | train | 4 | 86 | [Primary source](https://www.tesla.com/master-plan-part-deux) |
| `henry-ford-2008` | 2009-11-05 | train | 4 | 95 | [Primary source](https://thf-prod-blobs.s3.amazonaws.com/docs/thehenryfordlibraries/innovator-transcripts/transcript_musk_full-length.pdf?sfvrsn=747231ce_1) |
| `stanford-2015` | 2015-10-07 | train | 4 | 84 | [Primary source](https://stvp.stanford.edu/node/11931/printable/print) |
| `lex-049-2019` | 2019-11-12 | train | 4 | 92 | [Primary source](https://lexfridman.com/wordpress/wp-content/uploads/2019/11/elon_musk_lex_fridman_2_transcript.pdf) |
| `lex-400-2023` | 2023-11-09 | train | 4 | 79 | [Primary source](https://lexfridman.com/elon-musk-4-transcript/) |
| `lex-438-2024` | 2024-08-02 | test | 4 | 88 | [Primary source](https://lexfridman.com/elon-musk-and-neuralink-team-transcript/) |

Dates were checked against the publisher pages on **2026-11-11**. The Henry Ford interview was recorded on 2008-06-26; the museum gives 2009-11-05 as its air date, which is used as `published_at`. The Stanford publisher's printable transcript and episode page identify the event as 2015-10-07. Lex #49's publisher episode page and transcript agree on 2019-11-12. The primary publisher episode pages establish the dates for [Lex #400](https://lexfridman.com/elon-musk-4/) and [Lex #438](https://lexfridman.com/elon-musk-and-neuralink-team/).

## Attribution and accuracy

- Every excerpt is a new, short paraphrase; no transcript passage or exact quotation is reproduced in the evidence text.
- Historical plans and predictions remain dated claims. They are not represented as verified outcomes or current facts.
- Interview claims are attributed to Musk's own answers. The interviewer's praise, commentary, and other guests' answers are not treated as his statements.
- `situations` values such as `advice` and `setback` label the topic of a passage. They are editorial retrieval tags, not observed emotional states or proven conditional behavior.
- All `behavioral_observation` values are `null`. Text transcripts cannot establish a person's facial movement, body motion, speech timing, or prosody. Such measurements require inspected media, alignment, and a separate annotation record.
- Some publisher transcripts contain transcription mistakes. The selected passages were checked for speaker and nearby context, but the starter material is not a substitute for full media review.

## Timeline integrity

Lex #400 and #438 have different timestamps on the podcast page and the transcript/video. For example, #438 labels the problem-solving chapter 52:24 on the podcast page and 43:47 on the transcript. All timestamped records for those sources explicitly use `transcript_video`.

Lex #49 timestamps are retained as `published_transcript`; no claim is made that an independently downloaded audio file starts at the same offset.

The Henry Ford transcript uses production timecodes beginning with 03:xx and later 04:xx. They remain in `locator.label`, accompanied by printed page numbers. They are deliberately **not** converted into `start_seconds` or a video seek URL.

The Stanford printable transcript is located by descriptive passage labels. No unverified timestamps are invented.

## Evaluation split

`lex-438-2024` and all four of its records are reserved for `test`. The normal retrieval index includes only the other six sources and 24 records. Split by source, not by random utterance, to avoid putting neighboring interview passages on both sides of the evaluation.

- `data/evaluation/retrieval.json`: 20 English/Korean retrieval smoke cases over the train sources. Expected IDs are acceptance candidates, not a claim that every candidate must appear in the same top-k list.
- `data/evaluation/heldout.json`: four questions whose reference answers exist only in the withheld interview. The default corpus should abstain from the missing details. The reference IDs must never be inserted into generation context during evaluation.
- `data/evaluation/boundaries.json`: three answerability cases and two embedded-instruction cases. Injection resistance needs generated-answer review; retrieval success alone is insufficient.

A retrieval score on these hand-authored questions measures plumbing and relevance on the small curated set. It is **not** a model-generalization result, a personality-cloning score, or an independent human assessment. Future A/B/C experiments should separately compare profile-only, topic retrieval, and topic-plus-situation retrieval under identical model and decoding settings. Audio and motion similarity need separate measured references and raters.

## Reuse and summary budgets

Original media and full source transcripts are not included. The manifest links to the publishers. The evidence texts total fewer than 160 paraphrased words per source and contain zero intended verbatim quotations. The table reports whitespace-delimited evidence-text word counts; metadata and titles are provenance, not reproduced source passages.

A software license on this repository does not grant rights to the linked recordings, transcripts, voice, or likeness. Add independently obtained media through the ingestion workflow while retaining its provenance and applicable use conditions. Do not infer that a publicly readable interview carries a blanket media-training or redistribution license.

## Loader invariants

- Source and evidence IDs are unique; every evidence `source_id` resolves.
- Source URLs are HTTPS; publication dates are valid ISO dates.
- Splits are assigned once per source and inherited by all records.
- Timestamps, when present, are nonnegative with `end_seconds > start_seconds`; a timeline is specified.
- Every expected and reference evidence ID in evaluation fixtures resolves. Test references cannot appear in a train-only retrieval context.
- The bundled profile is explicitly an AI simulation; editorial style rules are never promoted to measured behavioral facts.
