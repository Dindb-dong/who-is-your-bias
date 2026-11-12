# Registered regional replay acquisition

This additive lane completed the two already registered NVIDIA China CES 2026
and GTC 2026 public replays. Both event families were assigned to `train` before
media acquisition. The frozen text/voice datasets and evaluation inputs are not
modified by this acquisition.

## Two complete replays saved and restored

| Official replay | Registered fragments | Measured duration | Original retained bytes | Lossless chunks | Receipt |
| --- | ---: | ---: | ---: | ---: | --- |
| GTC 2026, China regional player | 834/834 | 8,335.646444 s (2 h 18 m 55.65 s) | 855,451,136 | 34 | [Complete GTC](gtc-complete.json) |
| CES 2026, China regional player | 550/550 | 5,499.813111 s (1 h 31 m 39.81 s) | 547,193,928 | 22 | [Complete CES](ces-complete.json) |

Together these are **two additional complete event recordings, 13,835.459555
seconds (3 h 50 m 35.46 s), and 1,402,645,064 bytes**. The 2 GiB final-media cap
was respected, leaving 744,838,584 bytes of that budget unused. Final observed
free disk was 10,659,983,360 bytes, above the 6 GiB combined working/training
reserve. Both contain H.264 video at 854×480 and 30 fps plus stereo AAC at
44.1 kHz.

All **56 source-media chunk objects plus four final manifest/metadata objects**
were saved and freshly materialized. Container hashes, decoded chunk hashes,
contiguous offsets and reconstructed whole-file hashes matched both retained
originals. Earlier prefix manifests are additional historical checkpoints;
they are not extra recordings.

The [aggregate artifact manifest](artifact-manifest.json) links complete object
IDs, byte counts, hashes and verification receipts. Full reconstructed hashes:

- GTC: `4070e63179f44e5de14d70bc615322a2821b2ed417610ffa0e92ccd371cd7b28`.
- CES: `e1ac9122cb2468d7759eeef5b2b2f67655eda6e6f72c3fe7b88389ccb5a05d45`.

Official discovery pages are [GTC](https://www.nvidia.cn/gtc/keynote/) and
[CES](https://www.nvidia.cn/events/ces/); the registered public player-state
endpoints are `https://live.nvidia.cn/20260317-gtc/state-u.json` and
`https://live.nvidia.cn/20260106-ces/state-u.json`. Saved manifests and exact
source-entry metadata are retained in each complete acquisition metadata bundle.

## Actual media and timeline validation

[gtc-complete.json](gtc-complete.json) records one complete additional publisher
video package: **8,335.646444 seconds (2 h 18 m 55.65 s), 855,451,136 bytes**.
All 834 registered fragments were acquired in order. The retained MPEG-TS file
contains H.264 video at 854×480 and 30 fps plus stereo AAC at 44.1 kHz.

The full decode-timestamp walk inspected 250,068 video packets and 358,987 audio
packets. Neither stream had a backward step over 1 ms; maximum gaps were 33.334 ms and
23.234 ms. First and last four-second decode checks passed. All 34 byte-chunk
objects, the final reconstruction manifest and the acquisition metadata archive
were saved and freshly materialized. Container hashes and decoded chunk hashes
matched; ordered reconstruction matched the retained original's full SHA-256:
`4070e63179f44e5de14d70bc615322a2821b2ed417610ffa0e92ccd371cd7b28`.

CES similarly passed its full 164,993-video/236,858-audio packet walk, with no
backward step over 1 ms and the same maximum inter-packet gaps. Its first and
last four-second decode checks passed. Both full packet scans produced empty
error logs. These checks cover all 1,010,906 packets across the two files.

After acquisition, [separate per-track coverage checks](stream-coverage-checks.json)
confirmed each audio and video track reaches the registered VOD boundaries
within the explicit 250 ms tolerance. These checks derive from saved actual
timestamps. [Collector hardening](collector-hardening.md) documents the added
future-run gate and early-error probe cleanup, with 21 focused tests passing.
Both acquisitions themselves used original published collector `9a6bada`; their
receipts retain its exact implementation hash. The later validation does not
change the acquisition implementation identity.

This is an exact concatenation of the public HLS fragment bytes, with no
transcoding or remux. The complete package includes the publisher's whole
timeline; it is not a measurement of isolated Jensen speech time. Audio
language, dubbing and burned-in subtitles remain unreviewed, and these new
recordings have not been admitted to frozen training/evaluation datasets.

The CES initial full-download attempt received HTTP 503 for the registered
manifest before any fragment download. One normal retry with unchanged URL and
registered headers succeeded. Both the failure and successful attempt logs are
preserved in the CES metadata archive. No alternate access path was used.

## Earlier durable media-prefix checkpoints

[ces-prefix001.json](ces-prefix001.json) first saved and freshly reconstructed
24 MiB (25,165,824 bytes); [ces-prefix009.json](ces-prefix009.json) extended that
to 216 MiB (226,492,416 bytes) in nine contiguous chunks. Their incomplete status
and zero complete-CES counts describe those earlier checkpoints.

[gtc-prefix007.json](gtc-prefix007.json) records 168 MiB (176,160,768 bytes)
from the beginning of the GTC replay, stored as seven lossless chunks plus a
reconstruction manifest. All eight objects were freshly materialized; container
hashes, decoded chunk hashes and the reconstructed ordered prefix hash matched.
This was an incomplete prefix, not a complete video. Whole-replay duration and
continuity validation had not yet run at this checkpoint.

[gtc-prefix019.json](gtc-prefix019.json) extends the saved, freshly reconstructed
prefix to 19 chunks: 456 MiB (478,150,656 bytes). Its full-prefix SHA-256 is
`f4577e83ea2d1ffb58fb27ba94d317eeae222d6a8c7c90157899e418947ca46a`.
Every added object was independently materialized before checking the complete
ordered prefix. This checkpoint still counts zero complete videos.

The CES first-fragment bounded retry subsequently returned HTTP 200 with
456,088 bytes, after the initial HTTP 503 responses. Its first-fragment probe
also identifies H.264 854×480 at 30 fps plus stereo AAC at 44.1 kHz. Neither short
sample establishes full acquisition or verified audio/subtitle language.

## Initial checkpoint

- Integration base: `2eb40726b8963d136edd0f25656a89b2dd77a67a`.
- Isolated branch: `feat/jensen-regional-replays-v1`.
- The official player states responded successfully at preflight and retained the
  registered SD rendition URLs. The saved finite manifests contain 550 and 834
  fragments, respectively (5,499.766667 and 8,335.6 seconds).
- Successful size samples projected approximately 474 MB for CES and 661 MB for
  GTC. These are estimates, not complete-file measurements or download bounds.
- The first GTC fragment has H.264 video at 854×480 and 30 fps, with stereo AAC
  audio at 44.1 kHz. It is only a short sample, not an acquired full video.
- The initial CES first-fragment HEAD and GET returned HTTP 503; other sampled
  fragment HEAD requests succeeded. A later bounded first-fragment retry returned 200.
- No complete replay is claimed by this initial checkpoint. Audio language,
  dubbing, burned-in subtitles and speech-training eligibility remain unverified.

See [preflight.json](preflight.json) for observed sizes and limits and
[storage-probe.json](storage-probe.json) for an actual small upload/fresh-readback
proof. Public reports omit expiring storage URLs and full source text.

## Bounded method

The additive `tools.persona_data.regional_hls` collector uses installed `httpx`,
`ffprobe` and `ffmpeg`; it installs no dependencies. It validates the frozen
manifest hash, count and duration; rejects encrypted, discontinuous or non-TS
playlists; downloads at most four fragments concurrently; and appends every
original fragment in playlist order to one MPEG-TS file. This is byte-exact
fragment concatenation, not original progressive MP4 or transcoded video.
Per-fragment hashes, sizes and contiguous byte offsets are journaled privately.
A failed or bounded-out run retains a `.part` prefix and never receives a
complete receipt. Existing outputs cannot be overwritten or silently resumed.

The combined final-media limit is 2 GiB. Each run uses its remaining share of
that limit and enforces a 6 GiB free-space reserve: 4 GiB working floor plus
2 GiB for continuing training/final export. Sources are acquired one at a time.
Completed media requires both streams, manifest-matched duration, a full decode
timestamp walk without backward steps over 1 ms or gaps over 250 ms, and first/last
four-second decode checks. This validates container/timeline continuity; it is
not a full audiovisual content review or a human language/identity assessment.

GTC invocation used for the complete receipt (replace local path placeholders):

```sh
PYTHONPATH=src:. /absolute/path/to/existing/.venv/bin/python -m tools.persona_data.regional_hls \
  /private/artifact/path/gtc-2026-cn.entry.json \
  /private/artifact/path/gtc-2026-cn.full.ts \
  --maximum-bytes 2147483648 --minimum-free-bytes 6442450944
```

Run from the checked-out repository. Entry JSON comes from the existing saved
publisher registration, including its ordinary public-player Referer. Never
work around an unavailable or access-restricted endpoint. A complete first
source must be saved and verified before the second source's budget is assigned.
The CES invocation used its own source/output paths and
`--maximum-bytes 1292032512`, the exact remaining budget after GTC.

After acquisition, preserve media in bounded lossless chunks with stable object
IDs. Fresh-materialize every saved object, check archive and decoded chunk
hashes, and hash the reconstructed ordered stream. A second complete duplicate
is unnecessary when the retained original and reconstructed stream match.
Acquired bytes, stored bytes and verified restored bytes remain separate states
until the final artifact manifest records all three.

## Restore from the saved objects

Use the exact `file_id`, `library_file_id` and filenames in a complete source
receipt to materialize its chunk archives and final reconstruction manifest into
a new directory. Check each materialized container's byte count and SHA-256
against the receipt before reconstruction. Preserve the downloaded manifest;
copy it to `reconstruction.json` in that directory for the existing verifier.

```sh
PYTHONPATH=src:. python -m tools.persona_data.media_chunks verify /new/gtc-recovery --gzip
```

This command checks each decoded chunk and hashes the reconstructed ordered
stream without creating another whole-file copy. To produce an independent
usable video file, add `--output /new/output/gtc-2026-cn.full.ts`; choose a new
path because the verifier rejects overwriting an existing output. Apply the
same recipe to CES with its own complete receipt and reconstruction manifest.
