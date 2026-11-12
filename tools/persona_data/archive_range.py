"""Acquire an explicitly bounded public archive range without claiming a full ISO.

The entry must describe a reviewed source range. This collector preserves bytes;
it does not decide whether a range contains a complete speech. Its receipt keeps
that semantic review separate from transport, decoder, and timestamp checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx


def _digest(path: Path, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = path.stat().st_size if limit is None else limit
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError("Partial file is shorter than its checkpoint")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def download_range(entry: dict, output: Path, *, client: httpx.Client,
                   resume: bool = False, chunk_bytes: int = 16 * 1024 * 1024) -> dict:
    url = urlsplit(entry["source_url"])
    start, count, total = (entry[key] for key in
                           ("source_byte_start", "source_byte_count", "source_total_bytes"))
    if (url.scheme not in {"https", "http"} or not url.netloc or url.username
            or any(type(value) is not int for value in (start, count, total, chunk_bytes))
            or start < 0 or count <= 0 or count >= total or start + count > total or chunk_bytes <= 0):
        raise ValueError("Expected an in-bounds range of a public HTTP archive")
    if entry.get("split") != "train" or not entry.get("excluded_from_frozen_runs"):
        raise ValueError("This collector requires a separate train-only expansion")
    if output.exists():
        raise ValueError("Completed outputs are immutable")
    partial = output.with_name(output.name + ".part")
    state_path = output.with_name(output.name + ".state.json")
    binding = hashlib.sha256(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    state = {"entry_sha256": binding, "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest(),
             "etag": entry.get("expected_etag"), "last_modified": entry.get("expected_last_modified")}
    if resume:
        state = json.loads(state_path.read_text())
        if (state["entry_sha256"] != binding or not 0 <= state["bytes"] <= count
                or _digest(partial, state["bytes"]) != state["sha256"]):
            raise ValueError("Resume bytes or source binding differ from the checkpoint")
        # Discard only an interrupted, uncheckpointed append to this partial file.
        with partial.open("r+b") as stream:
            stream.truncate(state["bytes"])
    else:
        if partial.exists() or state_path.exists():
            raise ValueError("Inspect retained partial state and use explicit --resume")
        output.parent.mkdir(parents=True, exist_ok=True)
        partial.touch(exist_ok=False)
        _atomic_json(state_path, state)
    rolling = hashlib.sha256()
    with partial.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            rolling.update(block)
    while state["bytes"] < count:
        offset = start + state["bytes"]
        size = min(chunk_bytes, count - state["bytes"])
        headers = {"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity",
                   "Range": f"bytes={offset}-{offset + size - 1}"}
        if entry.get("publisher_referer"):
            headers["Referer"] = entry["publisher_referer"]
        if state["etag"] and not state["etag"].startswith("W/"):
            headers["If-Match"] = state["etag"]
        with client.stream("GET", entry["source_url"], headers=headers) as response:
            response.raise_for_status()
            expected = f"bytes {offset}-{offset + size - 1}/{total}"
            if response.status_code != 206 or response.headers.get("content-range") != expected:
                raise ValueError("Server did not honor the exact bounded range; body not consumed")
            if response.headers.get("content-length") not in {None, str(size)}:
                raise ValueError("Range Content-Length differs")
            for key, header in (("etag", "etag"), ("last_modified", "last-modified")):
                observed = response.headers.get(header)
                if state[key] is not None and observed != state[key]:
                    raise ValueError("Source archive identity changed between ranges")
                state[key] = observed
            written = 0
            with partial.open("ab") as stream:
                for block in response.iter_bytes(1024 * 1024):
                    written += len(block)
                    if written > size:
                        raise ValueError("Range body exceeds the registered size")
                    stream.write(block)
                    rolling.update(block)
                if written != size:
                    raise ValueError("Range body is truncated")
                stream.flush()
                os.fsync(stream.fileno())
        state["bytes"] += written
        state["sha256"] = rolling.hexdigest()
        _atomic_json(state_path, state)
    partial.rename(output)
    return {**state, "source_byte_start": start, "source_byte_count": count,
            "source_total_bytes": total, "full_source_archive_acquired": False}


def acquire(entry: dict, output: Path, *, resume: bool = False) -> dict:
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        transfer = download_range(entry, output, client=client, resume=resume)
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(output)
    ]))
    duration = float(probe["format"]["duration"])
    first_pts = float(probe["format"]["start_time"])
    if duration <= 0 or not any(s["codec_type"] == "video" for s in probe["streams"]):
        raise ValueError("Extracted range has no positive video duration")
    tail = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-ss", str(max(0, duration - 5)),
        "-i", str(output), "-t", "4", "-threads", "1", "-f", "null", "-"
    ], capture_output=True, text=True, check=True)
    receipt = {
        "schema_version": "public-archive-range-video.v1", **entry, **transfer,
        "file": output.name, "acquired_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration, "first_source_pts_seconds": first_pts,
        "source_clock": "original_DVD_MPEG_program_stream_PTS",
        "local_elapsed_to_source_pts_offset_seconds": first_pts,
        "youtube_or_other_player_clock_mapping_verified": False,
        "storage_transform": "exact_contiguous_archive_bytes; no re-encoding or timestamp rewriting",
        "status": "derived_archive_range_acquired_and_decoder_checked; semantic_completeness_requires_review",
        "streams": [{k: s.get(k) for k in ("codec_type", "codec_name", "width", "height",
                    "r_frame_rate", "start_time", "duration")} for s in probe["streams"]],
        "validation": {"exact_HTTP_ranges": "passed", "ffprobe": "passed",
                       "tail_decode": ("completed_with_decoder_warnings" if tail.stderr.strip()
                                       else "passed"),
                       "tail_decoder_exit_code": tail.returncode,
                       "tail_decoder_stderr": tail.stderr},
    }
    _atomic_json(output.with_name(output.name + ".receipt.json"), receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entry", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    value = acquire(json.loads(args.entry.read_text()), args.output, resume=args.resume)
    print(json.dumps({k: value[k] for k in ("bytes", "sha256", "duration_seconds", "status")}))


if __name__ == "__main__":
    main()
