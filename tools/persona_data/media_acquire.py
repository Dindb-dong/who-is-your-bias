"""Acquire a complete public media rendition and retain a byte/decoder receipt.

Input manifests are private artifacts: resolved player URLs may expire. Public
recipes should identify the publisher landing and player entry instead. Partial
downloads retain a .part suffix and never receive a completion receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import httpx


def hls_manifest(raw: bytes) -> tuple[float, int]:
    text = raw.decode("utf-8")
    durations = [float(value) for value in re.findall(r"#EXTINF:([\d.]+)", text)]
    if not text.startswith("#EXTM3U") or "#EXT-X-ENDLIST" not in text or not durations:
        raise ValueError("Only complete finite publisher HLS VOD playlists are accepted")
    if any("METHOD=NONE" not in line for line in text.splitlines() if line.startswith("#EXT-X-KEY:")):
        raise ValueError("Encrypted HLS is not supported by this collector")
    return sum(durations), len(durations)


def acquire_hls(entry: dict, output: Path) -> dict:
    """Retain every VOD fragment; missing fragments abort instead of truncating."""
    staging = output.with_name(output.stem + ".download")
    if output.exists() or staging.exists() or output.with_suffix(".receipt.json").exists():
        raise ValueError("Choose a new output; inspect any retained HLS download directory first")
    if entry.get("split") not in {"train", "test", "validation"}:
        raise ValueError("HLS media requires a frozen event split")
    headers = entry.get("http_headers", {})
    response = httpx.get(entry["url"], headers=headers, follow_redirects=True, timeout=60)
    response.raise_for_status()
    expected_duration, fragments = hls_manifest(response.content)
    manifest_hash = hashlib.sha256(response.content).hexdigest()
    if (manifest_hash != entry.get("publisher_manifest_sha256")
            or fragments != entry.get("expected_fragments")):
        raise ValueError("Publisher HLS manifest changed since source registration")
    staging.mkdir(parents=True)
    (staging / "publisher.m3u8").write_bytes(response.content)
    command = ["uv", "tool", "run", "--from", "yt-dlp", "yt-dlp", "--no-playlist",
               "--abort-on-unavailable-fragments", "--fragment-retries", "3", "--retries", "3",
               "--concurrent-fragments", "4", "--remux-video", "mp4", "--write-info-json",
               "--newline", "--output", str(staging / "video.%(ext)s")]
    for name, value in headers.items():
        command.extend(["--add-headers", f"{name}:{value}"])
    command.append(entry["url"])
    with (staging / "download.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    media = staging / "video.mp4"
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(media)
    ]))
    duration = float(probe["format"]["duration"])
    if abs(duration - expected_duration) > 2:
        raise ValueError("Downloaded VOD duration differs from the complete publisher manifest")
    tail = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-ss", str(max(0, duration - 5)),
        "-i", str(media), "-t", "4", "-f", "null", "-"
    ], capture_output=True, text=True, check=True)
    digest = hashlib.sha256()
    with media.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    receipt = {
        "schema_version": "jensen-source-media.v1", **entry, "file": output.name,
        "bytes": media.stat().st_size, "sha256": digest.hexdigest(), "duration_seconds": duration,
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "source_clock": "publisher_VOD_manifest_elapsed", "local_source_clock_offset_seconds": 0,
        "transcript_clock_mapping_verified": False, "storage_transform": "lossless_fragment_concatenation_and_mp4_remux",
        "streams": [{key: stream.get(key) for key in (
            "codec_name", "codec_type", "width", "height", "r_frame_rate", "duration"
        )} for stream in probe["streams"]],
        "validation": {"publisher_manifest_sha256": manifest_hash, "expected_fragments": fragments,
                       "expected_duration_seconds": expected_duration, "missing_fragment_policy": "abort",
                       "ffprobe": "passed", "last_four_seconds_decode": "passed", "decoder_stderr": tail.stderr},
        "status": "acquired_complete_publisher_VOD_validated",
    }
    media.rename(output)
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def acquire(entry: dict, output: Path) -> dict:
    required = ("source_id", "event_family_id", "split", "discovery_url", "url")
    if any(not isinstance(entry.get(key), str) or not entry[key] for key in required):
        raise ValueError("Source, event, split, discovery and media URL are required")
    if entry["split"] not in {"train", "test", "validation"}:
        raise ValueError("Media requires a frozen event split")
    if output.exists() or output.with_suffix(".receipt.json").exists():
        raise ValueError("Choose a new output; completed artifacts are immutable")
    partial = output.with_name(output.name + ".part")
    if partial.exists():
        raise ValueError("A partial download exists; inspect it before a fresh acquisition")
    output.parent.mkdir(parents=True, exist_ok=True)
    digest, size = hashlib.sha256(), 0
    with httpx.stream("GET", entry["url"], headers=entry.get("http_headers", {}),
                      follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        expected = response.headers.get("content-length")
        with partial.open("xb") as stream:
            for block in response.iter_bytes(4 * 1024 * 1024):
                stream.write(block)
                digest.update(block)
                size += len(block)
        if expected is not None and size != int(expected):
            raise ValueError("Downloaded bytes do not match HTTP content length")
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(partial)
    ]))
    duration = float(probe["format"]["duration"])
    if duration <= 0 or not probe.get("streams"):
        raise ValueError("No decodable media duration/streams")
    tail = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-ss", str(max(0, duration - 5)),
        "-i", str(partial), "-t", "4", "-f", "null", "-"
    ], capture_output=True, text=True, check=True)
    receipt = {
        "schema_version": "jensen-source-media.v1", **entry,
        "role": entry.get("role", "full_source_video"), "file": output.name,
        "bytes": size, "sha256": digest.hexdigest(), "duration_seconds": duration,
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "source_clock": "publisher_rendition_elapsed", "local_source_clock_offset_seconds": 0,
        "transcript_clock_mapping_verified": False,
        "streams": [{key: stream.get(key) for key in (
            "codec_name", "codec_type", "width", "height", "r_frame_rate", "duration"
        )} for stream in probe["streams"]],
        "validation": {"http_content_length": expected, "ffprobe": "passed",
                       "last_four_seconds_decode": "passed", "decoder_stderr": tail.stderr},
        "status": "acquired_full_file_validated",
    }
    partial.rename(output)
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entry", type=Path, help="Private JSON with source/event/split/discovery/url")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    entry = json.loads(args.entry.read_text())
    value = (acquire_hls if entry.get("protocol") == "hls" else acquire)(entry, args.output)
    print(json.dumps({key: value[key] for key in (
        "source_id", "bytes", "sha256", "duration_seconds", "status"
    )}))


if __name__ == "__main__":
    main()
