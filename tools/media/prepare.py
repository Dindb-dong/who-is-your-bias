"""Clip local media into a reproducible review bundle."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .common import MediaError, finite_number, local_file, probe, run_media, sha256, source_identity, timestamp, write_json


def prepare_clip(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    source_id: str,
    source_url: str,
    source_timeline: str,
    start: float = 0,
    duration: float = 30,
    input_source_offset: float = 0,
    sample_rate: int = 16000,
    include_video: bool = True,
) -> dict[str, Any]:
    source_identity(source_id, source_url, source_timeline)
    start = finite_number(start, "start")
    duration = finite_number(duration, "duration", 0.1, 120)
    offset = finite_number(input_source_offset, "input_source_offset")
    if sample_rate not in {16000, 24000, 48000}:
        raise MediaError("sample_rate must be 16000, 24000, or 48000")
    media = local_file(input_path)
    if media.suffix.lower() not in {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".mp4", ".mov", ".mkv", ".webm"}:
        raise MediaError("Unsupported local audio/video extension; playlists and URLs are not accepted")
    metadata = probe(media)
    streams = metadata.get("streams", [])
    if not any(stream.get("codec_type") == "audio" for stream in streams):
        raise MediaError("The source has no audio stream")
    full_duration = finite_number(metadata.get("format", {}).get("duration"), "source duration", 0.001)
    if start >= full_duration or start + duration > full_duration + 0.05:
        raise MediaError("Requested clip lies outside the local media duration")
    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise MediaError("Choose a new output directory; existing review bundles are never overwritten")
    target.mkdir(parents=True, exist_ok=False)
    try:
        audio = target / "audio.wav"
        prefix = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-protocol_whitelist", "file,pipe", "-ss", str(start), "-i", str(media), "-t", str(duration)]
        run_media(prefix + ["-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", "-map_metadata", "-1", str(audio)])
        audio_metadata = probe(audio)
        actual_duration = finite_number(audio_metadata.get("format", {}).get("duration"), "prepared duration", 0.001, 120)
        artifacts = {"audio": {"path": "audio.wav", "sha256": sha256(audio), "bytes": audio.stat().st_size}}
        clip: dict[str, Any] = {"audio": "audio.wav", "duration_seconds": actual_duration, "sample_rate": sample_rate, "channels": 1}
        if include_video and any(stream.get("codec_type") == "video" for stream in streams):
            video = target / "video.mp4"
            run_media(prefix + ["-map", "0:v:0", "-an", "-vf", "scale=w='min(1280,iw)':h=-2,setsar=1,setpts=PTS-STARTPTS", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-map_metadata", "-1", "-movflags", "+faststart", str(video)])
            artifacts["video"] = {"path": "video.mp4", "sha256": sha256(video), "bytes": video.stat().st_size}
            clip["video"] = "video.mp4"
        manifest = {
            "schema_version": "media.v1", "created_at": timestamp(),
            "source_id": source_id, "source_url": source_url, "source_timeline": source_timeline,
            "source_start_seconds": offset + start,
            "source_end_seconds": offset + start + actual_duration,
            "clock_mapping": "source_seconds = clip_seconds + source_start_seconds",
            "input": {"name": media.name, "sha256": sha256(media), "bytes": media.stat().st_size, "duration_seconds": full_duration, "source_offset_seconds": offset},
            "preparation": {"start_seconds": start, "requested_duration_seconds": duration, "audio_codec": "pcm_s16le", "channel_operation": "mixdown_to_mono"},
            "clip": clip, "artifacts": artifacts,
            "review": {"status": "unreviewed", "speaker_identity": None, "source_clock_verified": False, "audio_single_speaker": None, "suitable_voice_reference": None, "notes": []},
        }
        write_json(target / "manifest.json", manifest)
        return manifest
    except Exception:
        # We created this directory exclusively in this call; no pre-existing work is removed.
        shutil.rmtree(target)
        raise
