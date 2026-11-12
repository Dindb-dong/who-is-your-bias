# Jensen recovery: durable checkpoint and resume

This is an intermediate implementation checkpoint. The machine-readable
[source progress](source-progress.json) distinguishes located publisher players,
complete local media and complete durable videos. The
[artifact receipts](media-artifacts.json) contain exact saved object identities,
sizes and SHA-256 values. Do not infer persistence from a source URL or manifest.

## Actual reusable corpus

- `original-dialogue-v4`: 554 original exchanges from nine training events;
  551 complete exchanges and three excerpts. Public recipes are in
  [corpus-v4](corpus-v4/README.md); original wording remains in the saved archive.
- `heldout-original-v1`: 50 complete No Priors 2023 exchanges. The loader checks
  source/event isolation and normalized twelve-word spans against both v4 and
  the PR17 Jensen editorial corpus. This does not establish semantic independence
  or absence from a foundation model's pretraining. All training, retrieval,
  speech and avatar projections are disabled. Never put the reference answer
  into inference prompts. The [manifest](heldout-v1/manifest.json) and
  [selection recipe](heldout-v1/recipe.json) expose hashes and identifiers only.
- The WEF reserved event remains test-only. Its previously indexed transcript
  URL currently redirects to a landing page without transcript text. Cached
  search text has not been substituted for acquired publisher bytes.

## Recover acquired bytes

Use the current Library skill to materialize the **exact** `file_id`,
`library_file_id`, and returned filename in `media-artifacts.json`. Retain the
returned identity and transfer metadata. The heldout archive was freshly
materialized and hash checked, so it can already be restored.

Eight complete published videos have been acquired and validated: four keynote
replays and four interviews, totaling 10 h 35 min 29 s. Five have already passed
complete recovery from fresh remote copies. Every chunk for the remaining three
has uploaded successfully; their fresh readback and full reconstruction are the
remaining durability checks. Use the machine-readable count for newer updates.

Large media uses lossless contiguous raw byte chunks, wrapped in ordinary gzip
or single-member tar.gz containers. Initial unwrapped `.bin` transfers failed;
the first gzip chunk passed a 207,929,911-byte upload/readback and an independent
decoded SHA-256 comparison. Larger containers still transferred unreliably.
The final four videos use 24 MiB raw chunks: all 79 chunk, manifest and receipt
objects uploaded successfully on the first attempt. Retain the original
200 MiB profiles for Goldman, Stripe and GTC 2025, and the 80 MiB mixed gzip/tar
profile for WIRED. Do not substitute a partially saved earlier chunk profile.

For a fully saved video, materialize every listed gzip chunk into a new local
directory. Verify each **stored gzip** SHA-256 against its durable receipt,
retain the `.bin.gz` or single-member `.bin.tar.gz` names corresponding to the raw names in the reconstruction
manifest, and place that manifest at `reconstruction.json`. Then run:

```sh
PYTHONPATH=src:. python -m tools.persona_data.media_chunks verify RECOVERED_CHUNK_DIRECTORY --gzip --output RECOVERED_VIDEO.mp4
```

The command streams gzip decoding and checks part ordering, contiguous offsets, every raw chunk hash and
the reconstructed full-file hash before atomically naming the output. A failed
check leaves no completed output. Compare the result with the original video
SHA-256 in the source progress record. This is lossless recovery, not video
transcoding. A video is not durably complete while any part is unavailable.

## Resume source acquisition

The media event registration was frozen before each proposed download. Keep
`wef-davos-2026` and `no-priors-2023` excluded from every training modality.
The public source/player catalog is
[`media-event-registration.json`](../../data/acquisition/jensen-huang/media-event-registration.json).
Use the source URLs and player identifiers there to refresh expiring player
metadata. Public Kaltura metadata can be resolved with:

```sh
uv tool run --from yt-dlp yt-dlp --dump-single-json --no-playlist kaltura:2935771:PLAYER_ENTRY_ID > PRIVATE_PLAYER_METADATA.json
```

Select an actually available moderate-resolution public rendition. Metadata
format availability does not guarantee downloadable bytes. The NVIDIA Kaltura
recordings became downloadable when requests included the ordinary public
landing-page `Referer` and `User-Agent: Mozilla/5.0`; earlier headerless 404
failures remain in the record. Store these in `http_headers` and the resolved URL in
a private entry JSON with `source_id`, `event_family_id`, `split`,
`discovery_url`, and `url`; preserve publisher edit notes where present.

```sh
PYTHONPATH=src:. python -m tools.persona_data.media_acquire PRIVATE_ENTRY.json NEW_VIDEO.mp4
PYTHONPATH=src:. python -m tools.persona_data.media_chunks split NEW_VIDEO.mp4 NEW_CHUNK_DIRECTORY --chunk-bytes 25165824
```

The downloader retains incomplete bytes with `.part` and refuses to overwrite
existing completed artifacts or partial downloads. Inspect a partial result
before choosing a new output. A completion receipt requires HTTP byte-count
agreement, ffprobe streams/duration, and decoding the last four seconds. Local
elapsed time maps to the selected publisher rendition, not automatically to a
separate podcast or transcript clock. Preserve every completed batch promptly.

## Checks at this checkpoint

The Jensen adapters and heldout workflow passed 34 focused tests; media recovery
and complete HLS validation passed five focused tests, including corruption,
ordering and tar-member validation. The actual heldout loader
passed and the saved heldout archive passed fresh byte/hash readback. Retrieval
endorsement/private-interaction guards passed the earlier 104 affected tests.
The coordinating text workstream owns the opt-in bilingual retriever and its
frozen development comparison. Media acquisition does not establish retrieval,
voice, avatar or likeness quality.
