# Registered regional replay acquisition

This additive lane attempts only the already registered NVIDIA China CES 2026
and GTC 2026 public replays. Both event families were assigned to `train` before
media acquisition. The frozen text/voice datasets and evaluation inputs are not
modified by this acquisition.

## Complete GTC replay saved and restored

[gtc-complete.json](gtc-complete.json) records one complete additional publisher
video package: **8,335.646444 seconds (2 h 18 m 55.65 s), 855,451,136 bytes**.
All 834 registered fragments were acquired in order. The retained MPEG-TS file
contains H.264 video at 854×480 and 30 fps plus stereo AAC at 44.1 kHz.

The full decode-timestamp walk inspected 250,068 video packets and 358,987 audio
packets. Neither stream had a backward step; maximum gaps were 33.334 ms and
23.234 ms. First and last four-second decode checks passed. All 34 byte-chunk
objects, the final reconstruction manifest and the acquisition metadata archive
were saved and freshly materialized. Container hashes and decoded chunk hashes
matched; ordered reconstruction matched the retained original's full SHA-256:
`4070e63179f44e5de14d70bc615322a2821b2ed417610ffa0e92ccd371cd7b28`.

This is an exact concatenation of the public HLS fragment bytes, with no
transcoding or remux. The complete package includes the publisher's whole
timeline; it is not a claim of 2.3 hours of isolated Jensen speech. Audio
language, dubbing and burned-in subtitles remain unreviewed, and these new
recordings have not been admitted to frozen training/evaluation datasets.

[artifact-manifest.json](artifact-manifest.json) tracks the current aggregate
state and links the exact object-level receipts. CES remains pending full
acquisition. Its first 24 MiB (25,165,824 bytes) are already saved and freshly
reconstructed in [ces-prefix001.json](ces-prefix001.json), with zero complete
CES videos counted. The initial full-download attempt received HTTP 503 for
the registered manifest before any fragment download; one normal retry with
unchanged URL and headers succeeded and acquisition is continuing.
Its remaining byte budget after GTC is 1,292,032,512 bytes, with
the 6 GiB free-space reserve still enforced.

## Earlier durable media-prefix checkpoints

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
- The official player states currently respond successfully and retain the
  registered SD rendition URLs. The saved finite manifests contain 550 and 834
  fragments, respectively (5,499.766667 and 8,335.6 seconds).
- Successful size samples project approximately 474 MB for CES and 661 MB for
  GTC. These are estimates, not complete-file measurements or download bounds.
- The first GTC fragment has H.264 video at 854×480 and 30 fps, with stereo AAC
  audio at 44.1 kHz. It is only a short sample, not an acquired full video.
- The initial CES first-fragment HEAD and GET returned HTTP 503; other sampled
  fragment HEAD requests succeeded. A bounded retry remains pending.
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
timestamp walk without backward steps or gaps over 250 ms, and first/last
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

After acquisition, preserve media in bounded lossless chunks with stable object
IDs. Fresh-materialize every saved object, check archive and decoded chunk
hashes, and hash the reconstructed ordered stream. A second complete duplicate
is unnecessary when the retained original and reconstructed stream match.
Acquired bytes, stored bytes and verified restored bytes remain separate states
until the final artifact manifest records all three.
