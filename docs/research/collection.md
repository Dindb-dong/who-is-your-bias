# Collecting and auditing publisher material

The curated corpus and locally acquired originals serve different purposes. Reviewed evidence records are concise, attributed paraphrases used by retrieval. Publisher snapshots and complete transcripts preserve the material needed for subsequent segmentation, speaker review, and alignment; downloading them does not make them trusted runtime evidence.

## Inspect actual coverage

```bash
uv run python -m tools.research inventory --output reports/local/source-inventory.json
```

The inventory reports source events, publishers, publication dates, train/test counts, topics, and situations. It flags duplicate canonical URLs, identical evidence text, and source-event families spanning both training and evaluation splits. The same interview can appear as a video, podcast, transcript, and excerpt: those formats must retain one event identity. Tracking parameters are removed for duplicate detection, but document-identifying query parameters are preserved.

Date ranges describe the supplied collection. They do not establish that the persona knows current events. A source published later can discuss earlier events; a recording date must not be inferred from the upload date without supporting metadata.

## Acquire selected originals

```bash
uv run python -m tools.research collect \
  --source-id lex-049-2019 \
  --source-id lex-400-2023 \
  --destination data/raw/sources
```

Repeat `--source-id` to select known records explicitly. The collector uses all verified transcript URLs supplied for that source event, otherwise its component document URLs or original source URL. The three Stanford 2003 clip transcripts therefore produce three document outcomes under one event identity. The inventory also preserves supplied video, audio, and podcast-episode links without treating episode pages as audio files. Collection accepts bounded HTML, text, JSON, or PDF responses over HTTPS and records redirects. It does not infer a media download URL, crawl an entire site, authenticate to third-party services, or convert an article into a transcript.

Each source directory contains a content-addressed snapshot and manifest with the original and final URL, retrieval time, content type, byte length, SHA-256, and review state. Distinct URLs with identical content share the snapshot but retain separate acquisition manifests. The default output directory is ignored by Git. Existing symlink destinations are rejected. The network bounds apply to this local research CLI; it is not exposed as a public URL-fetching service.

The acquisition smoke run on 2026-11-10 retrieved the official Lex #49 transcript PDF: **144,047 bytes**, SHA-256 `999374c2f86694b31824dcc41bcc9f7b9c076a83a17327c6284af348416be6d2`. This proves the selected publisher document was acquired, not that speaker attribution, extraction completeness, or media alignment has been validated.

## Turn originals into reviewable examples

Use the [publisher transcript extractor](transcript-acquisition.md) for supported structured transcripts. Use `bias ingest` for SRT, VTT, or WhisperX-compatible JSON, and the [media pipeline](../media-pipeline.md) for local recordings. All automatic imports remain drafts until reviewed.

For each proposed example, retain the question and adjacent speaking turns, distinguish explicit speaker labels from inferred identity, record the exact media clock, and identify the situational evidence. A philosophical answer, a technical explanation, a joke, and a response to criticism should not be collapsed into one undifferentiated style label. Facial and body observations require the matching video and must be associated with the correct speaking turn separately.

Full original transcripts, recordings, downloaded model weights, and local analysis outputs are not committed. The repository contains source metadata, bounded editorial paraphrases, reproducible acquisition/extraction code, and measured coverage metadata. See [corpus status](corpus-status.md) for what is actually indexed and [source provenance](sources.md) for the initial source-clock notes.
