"""Bounded acquisition of a registered, plain MPEG-TS HLS VOD rendition.

This stores the publisher's fragment bytes in playlist order without transcoding
or remuxing. A complete receipt is emitted only after every fragment and the
full packet timeline validate. Private entries/receipts may contain source URLs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import urljoin, urlsplit

import httpx

from .media_acquire import hls_manifest


def parse_registered_playlist(raw: bytes, entry: dict) -> list[str]:
    duration, count = hls_manifest(raw)
    if hashlib.sha256(raw).hexdigest() != entry["publisher_manifest_sha256"]:
        raise ValueError("Registered publisher manifest hash differs")
    if count != entry["expected_fragments"] or abs(duration - entry["expected_duration_seconds"]) > .001:
        raise ValueError("Registered fragment count or duration differs")
    lines = raw.decode().splitlines()
    unsupported = ("#EXT-X-MAP", "#EXT-X-BYTERANGE", "#EXT-X-DISCONTINUITY", "#EXT-X-STREAM-INF")
    if any(line.startswith(unsupported) for line in lines):
        raise ValueError("Only continuous plain MPEG-TS VOD playlists are supported")
    urls = [urljoin(entry["url"], line) for line in lines if line and not line.startswith("#")]
    origin = urlsplit(entry["url"])
    if len(urls) != count or len(set(urls)) != count:
        raise ValueError("One unique URI is required for each fragment")
    for url in urls:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc != origin.netloc or not parsed.path.endswith(".ts"):
            raise ValueError("Fragment must use the registered HTTPS publisher origin and TS format")
    return urls


def packet_timeline(path: Path) -> dict:
    command = ["ffprobe", "-v", "error", "-show_packets", "-show_entries",
               "packet=stream_index,pts_time,dts_time,duration_time", "-of", "csv=p=0", str(path)]
    streams: dict[int, dict] = {}
    with path.with_suffix(path.suffix + ".packet-stderr.log").open("w") as err:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=err, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            fields = line.strip().split(",")
            if len(fields) < 4:
                continue
            index = int(fields[0])
            if fields[2] == "N/A":
                raise ValueError("Packet without decode timestamp")
            dts = float(fields[2])
            if not math.isfinite(dts):
                raise ValueError("Non-finite packet timestamp")
            state = streams.setdefault(index, {"packets": 0, "first_dts": dts, "last_dts": dts,
                                                "backward_steps": 0, "largest_gap_seconds": 0.0})
            if state["packets"]:
                delta = dts - state["last_dts"]
                state["backward_steps"] += delta < -.001
                state["largest_gap_seconds"] = max(state["largest_gap_seconds"], delta)
            state["last_dts"] = dts
            state["packets"] += 1
        if process.wait() != 0:
            raise ValueError("Full-file packet inspection failed")
    if len(streams) != 2 or any(s["backward_steps"] or s["largest_gap_seconds"] > .25 for s in streams.values()):
        raise ValueError("Unexpected stream count or packet-timeline discontinuity")
    return {"method": "full_file_ffprobe_decode_timestamp_walk", "maximum_allowed_gap_seconds": .25,
            "streams": streams, "status": "passed"}


def acquire(entry: dict, output: Path, *, maximum_bytes: int, minimum_free_bytes: int, workers: int = 4) -> dict:
    if entry.get("split") != "train" or entry.get("source_id") not in {"ces-2026-cn-video", "gtc-2026-cn-video"}:
        raise ValueError("This bounded collector is restricted to the two registered training sources")
    if maximum_bytes <= 0 or minimum_free_bytes < 0 or not 1 <= workers <= 4:
        raise ValueError("Invalid resource limits")
    partial = output.with_suffix(output.suffix + ".part")
    journal = output.with_suffix(".fragments.jsonl")
    receipt_path = output.with_suffix(".receipt.json")
    if any(p.exists() for p in (output, partial, journal, receipt_path)):
        raise ValueError("Choose a new output; existing source artifacts are immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    digest, size = hashlib.sha256(), 0
    with httpx.Client(headers=entry.get("http_headers", {}), follow_redirects=True,
                      timeout=httpx.Timeout(25), limits=httpx.Limits(max_connections=workers)) as client:
        response = client.get(entry["url"])
        response.raise_for_status()
        urls = parse_registered_playlist(response.content, entry)
        output.with_suffix(".publisher.m3u8").write_bytes(response.content)

        def fetch(item: tuple[int, str]) -> tuple[int, bytes, dict]:
            index, url = item
            for attempt in range(1, 4):
                try:
                    with client.stream("GET", url) as remote:
                        remote.raise_for_status()
                        blocks, fragment_size = [], 0
                        for block in remote.iter_bytes(1024 * 1024):
                            fragment_size += len(block)
                            if fragment_size > 16 * 1024**2:
                                raise ValueError("Fragment exceeds bounded-memory limit")
                            blocks.append(block)
                        data = b"".join(blocks)
                        expected = remote.headers.get("content-length")
                    if expected is not None and len(data) != int(expected):
                        raise ValueError("Fragment content-length differs")
                    if not data or len(data) % 188 or any(data[pos] != 0x47 for pos in range(0, len(data), 188)):
                        raise ValueError("Invalid MPEG-TS packet framing")
                    return index, data, {"index": index, "bytes": len(data),
                                         "sha256": hashlib.sha256(data).hexdigest(), "attempts": attempt}
                except (httpx.TransportError, httpx.HTTPStatusError) as error:
                    if attempt == 3 or (isinstance(error, httpx.HTTPStatusError) and error.response.status_code < 500):
                        raise
            raise AssertionError("Unreachable retry exit")

        with partial.open("xb") as sink, journal.open("x") as log, ThreadPoolExecutor(max_workers=workers) as pool:
            for first in range(0, len(urls), workers):
                for index, data, row in pool.map(fetch, list(enumerate(urls))[first:first + workers]):
                    if size + len(data) > maximum_bytes or shutil.disk_usage(output.parent).free - len(data) < minimum_free_bytes:
                        raise ValueError("Media cap or free-disk reserve reached; incomplete prefix retained")
                    row["offset_bytes"] = size
                    sink.write(data)
                    digest.update(data)
                    size += len(data)
                    log.write(json.dumps(row, sort_keys=True) + "\n")
                sink.flush(); os.fsync(sink.fileno())
                log.flush(); os.fsync(log.fileno())
                print(json.dumps({"fragments_saved": min(first + workers, len(urls)), "total_fragments": len(urls), "bytes": size}), flush=True)
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(partial)]))
    duration = float(probe["format"]["duration"])
    if abs(duration - entry["expected_duration_seconds"]) > 2:
        raise ValueError("Full transport-stream duration differs from registered VOD")
    if sorted(s["codec_type"] for s in probe["streams"]) != ["audio", "video"]:
        raise ValueError("Exactly one audio and one video stream required")
    timeline = packet_timeline(partial)
    for start in (0, max(0, duration - 5)):
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1", "-ss", str(start),
                        "-i", str(partial), "-t", "4", "-f", "null", "-"], check=True, capture_output=True)
    receipt = {"schema_version": "jensen-regional-hls-media.v1", **entry,
               "file": output.name, "bytes": size, "sha256": digest.hexdigest(), "started_at": started,
               "acquired_at": datetime.now(timezone.utc).isoformat(), "duration_seconds": duration,
               "storage_transform": "byte_exact_playlist_order_MPEG_TS_fragment_concatenation",
               "source_clock": "publisher_VOD_manifest_elapsed", "local_source_clock_offset_seconds": 0,
               "transcript_clock_mapping_verified": False, "language_review": "unverified",
               "speech_training_eligible": False,
               "streams": [{k: s.get(k) for k in ("codec_type", "codec_name", "width", "height", "r_frame_rate", "sample_rate", "channels")} for s in probe["streams"]],
               "validation": {"all_registered_fragments_saved": len(urls), "packet_framing": "passed",
                              "packet_timeline": timeline, "first_and_last_four_seconds_decode": "passed"},
               "status": "acquired_complete_publisher_VOD_locally_verified"}
    partial.rename(output)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entry", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--maximum-bytes", type=int, required=True)
    parser.add_argument("--minimum-free-bytes", type=int, default=6 * 1024**3)
    args = parser.parse_args()
    receipt = acquire(json.loads(args.entry.read_text()), args.output,
                      maximum_bytes=args.maximum_bytes, minimum_free_bytes=args.minimum_free_bytes)
    print(json.dumps({k: receipt[k] for k in ("source_id", "bytes", "sha256", "duration_seconds", "status")}))


if __name__ == "__main__":
    main()
