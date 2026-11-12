"""Acquire a complete public media rendition and retain a byte/decoder receipt.

Input manifests are private artifacts: resolved player URLs may expire. Public
recipes should identify the publisher landing and player entry instead. Partial
downloads retain a .part suffix and never receive a completion receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import httpx


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
    with httpx.stream("GET", entry["url"], follow_redirects=True, timeout=120) as response:
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
    value = acquire(json.loads(args.entry.read_text()), args.output)
    print(json.dumps({key: value[key] for key in (
        "source_id", "bytes", "sha256", "duration_seconds", "status"
    )}))


if __name__ == "__main__":
    main()
