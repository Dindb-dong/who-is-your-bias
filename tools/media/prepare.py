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
    crop: tuple[int, int, int, int] | None = None,
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
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    video_crop = None
    if crop is not None:
        if not include_video or video_stream is None:
            raise MediaError("A spatial crop requires a video stream and video output")
        if not isinstance(crop, (tuple, list)) or len(crop) != 4 or any(isinstance(value, bool) or not isinstance(value, int) for value in crop):
            raise MediaError("Crop must contain integer X, Y, WIDTH, HEIGHT values")
        x, y, width, height = crop
        input_width = int(finite_number(video_stream.get("width"), "video width", 1, 32768))
        input_height = int(finite_number(video_stream.get("height"), "video height", 1, 32768))
        if x < 0 or y < 0 or width < 32 or height < 32 or x + width > input_width or y + height > input_height or any(value % 2 for value in crop):
            raise MediaError("Crop must use even pixel coordinates/dimensions, be at least 32×32, and remain inside the source frame")
        rotations = [video_stream.get("tags", {}).get("rotate", 0), *[item.get("rotation", 0) for item in video_stream.get("side_data_list", [])]]
        if any(finite_number(value, "video rotation", -360, 360) % 360 for value in rotations):
            raise MediaError("Normalize rotated source video before selecting a fixed pixel crop")
        video_crop = {"x": x, "y": y, "width": width, "height": height, "input_width": input_width, "input_height": input_height,
                      "coordinate_basis": "input_pixels_before_scale", "mirrored": False}
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
        if include_video and video_stream is not None:
            video = target / "video.mp4"
            crop_filter = f"crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]}," if crop else ""
            run_media(prefix + ["-map", "0:v:0", "-an", "-vf", crop_filter + "scale=w='min(1280,iw)':h=-2,setsar=1,setpts=PTS-STARTPTS", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-map_metadata", "-1", "-movflags", "+faststart", str(video)])
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
        if video_crop is not None:
            manifest["preparation"]["video_crop"] = video_crop
        write_json(target / "manifest.json", manifest)
        return manifest
    except Exception:
        # We created this directory exclusively in this call; no pre-existing work is removed.
        shutil.rmtree(target)
        raise
