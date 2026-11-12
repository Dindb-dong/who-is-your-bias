# Registered regional replay acquisition

This additive lane attempts only the already registered NVIDIA China CES 2026
and GTC 2026 public replays. Both event families were assigned to `train` before
media acquisition. The frozen text/voice datasets and evaluation inputs are not
modified by this acquisition.

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

Proposed invocation, before an actual acquisition receipt exists:

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
