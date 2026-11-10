"""Optional ASR adapters that keep source clocks and unverified speaker identity."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

from .common import MediaError, finite_number, load_manifest, timestamp, verify_artifact, write_json


def normalize_segments(segments: list[dict[str, Any]], *, offset: float, duration: float) -> list[dict[str, Any]]:
    """Preserve absent word alignment rather than manufacturing a timestamp."""
    offset = finite_number(offset, "offset")
    duration = finite_number(duration, "duration", 0.001, 120)
    normalized = []
    for index, segment in enumerate(segments):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = finite_number(segment.get("start"), "segment start", 0, duration + 0.1)
        end = finite_number(segment.get("end"), "segment end", 0, duration + 0.1)
        if end <= start:
            raise MediaError("ASR returned a reversed or empty segment")
        words = []
        for word in segment.get("words") or []:
            item: dict[str, Any] = {"word": str(word.get("word", "")), "start": None, "end": None, "source_start": None, "source_end": None}
            if word.get("start") is not None and word.get("end") is not None:
                word_start = finite_number(word["start"], "word start", 0, duration + 0.1)
                word_end = finite_number(word["end"], "word end", 0, duration + 0.1)
                if word_end < word_start:
                    raise MediaError("ASR returned reversed word timestamps")
                item.update(start=word_start, end=word_end, source_start=offset + word_start, source_end=offset + word_end)
            probability = word.get("probability", word.get("score"))
            if probability is not None:
                item["confidence"] = finite_number(probability, "word confidence", 0, 1)
            words.append(item)
        normalized.append({"id": f"segment-{index:04d}", "start": start, "end": end, "source_start": offset + start, "source_end": offset + end, "text": text, "speaker": None, "review_status": "draft", "words": words})
    return normalized


def transcribe_clip(
    manifest_path: str | Path,
    *,
    engine: str = "faster-whisper",
    model_name: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    path, manifest = load_manifest(manifest_path)
    audio = verify_artifact(path, manifest, "audio")
    output = path.parent / "transcript.json"
    if output.exists() and not overwrite:
        raise MediaError("transcript.json already exists; use --overwrite explicitly")
    if device not in {"cpu", "cuda"}:
        raise MediaError("Supported ASR devices are cpu and cuda")
    if engine == "faster-whisper":
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise MediaError("Install tools/media/requirements-asr.txt in an isolated environment") from exc
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        segments, info = model.transcribe(str(audio), language=language, word_timestamps=True, vad_filter=True, beam_size=5, condition_on_previous_text=False)
        raw_segments = []
        for segment in segments:  # The upstream result is lazy; iteration performs transcription.
            raw_segments.append({"start": segment.start, "end": segment.end, "text": segment.text, "words": [{"word": word.word, "start": word.start, "end": word.end, "probability": word.probability} for word in (segment.words or [])]})
        detected_language = info.language
        alignment = "faster-whisper-word-timestamps"
    elif engine == "whisperx":
        try:
            import whisperx
        except ImportError as exc:
            raise MediaError("Install tools/media/requirements-whisperx.txt in an isolated environment") from exc
        if manifest["clip"].get("sample_rate") != 16000:
            raise MediaError("Prepare 16 kHz audio for the WhisperX adapter")
        audio_array = whisperx.load_audio(str(audio))
        model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)
        result = model.transcribe(audio_array, batch_size=4, language=language)
        detected_language = result["language"]
        align_model, metadata = whisperx.load_align_model(language_code=detected_language, device=device)
        aligned = whisperx.align(result["segments"], align_model, metadata, audio_array, device, return_char_alignments=False)
        raw_segments = aligned["segments"]
        alignment = "whisperx-forced-alignment"
    else:
        raise MediaError("Unknown ASR engine")
    try:
        version = importlib.metadata.version(engine)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    result = {
        "schema_version": "transcript.v1", "created_at": timestamp(),
        "source_id": manifest["source_id"], "source_url": manifest["source_url"],
        "source_timeline": manifest["source_timeline"], "source_start_seconds": manifest["source_start_seconds"],
        "audio_sha256": manifest["artifacts"]["audio"]["sha256"], "language": detected_language,
        "engine": {"name": engine, "version": version, "model": model_name, "device": device, "compute_type": compute_type, "alignment": alignment},
        "speaker_attribution": "unverified", "review_status": "draft",
        "segments": normalize_segments(raw_segments, offset=manifest["source_start_seconds"], duration=manifest["clip"]["duration_seconds"]),
    }
    write_json(output, result, overwrite=overwrite)
    return result
