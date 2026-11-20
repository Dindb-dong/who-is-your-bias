from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .common import PipelineError, interval

MEDIA_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aiff"}


def ffmpeg():
    exe = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        raise PipelineError("음성 변환 도구가 없습니다. requirements.txt를 설치하세요.") from None


def audio_info(path):
    with wave.open(str(path), "rb") as f:
        return {"sample_rate": f.getframerate(), "channels": f.getnchannels(),
                "sample_width": f.getsampwidth(), "duration": f.getnframes() / f.getframerate()}


def to_wav(media, output, sample_rate=24000):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.stem + ".partial.wav")
    try:
        proc = subprocess.run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                               "-i", str(Path(media).resolve()), "-vn", "-ac", "1", "-ar", str(sample_rate),
                               "-c:a", "pcm_s16le", str(tmp)], capture_output=True, text=True, timeout=3600)
        if proc.returncode:
            raise PipelineError("음성 변환 실패: " + proc.stderr[-1500:])
        info = audio_info(tmp)
        if info["duration"] <= 0:
            raise PipelineError("영상에 사용할 수 있는 음성 트랙이 없습니다.")
        os.replace(tmp, output)
        return info
    finally:
        tmp.unlink(missing_ok=True)


def cut_wav(source, output, start, end):
    info = audio_info(source)
    start, end = interval(start, end, info["duration"])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source), "rb") as src:
        src.setpos(round(start * info["sample_rate"]))
        frames = src.readframes(round((end - start) * info["sample_rate"]))
        with wave.open(str(output), "wb") as dst:
            dst.setparams(src.getparams())
            dst.writeframes(frames)
    return audio_info(output)


def youtube_video_id(url):
    host = (urlparse(url).hostname or "").lower()
    if urlparse(url).scheme != "https" or host not in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}:
        raise PipelineError("https 유튜브 영상 URL을 입력하세요.")
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    video_id = (parts[0] if host == "youtu.be" else
                parse_qs(parsed.query).get("v", [""])[0] if parsed.path == "/watch" else
                parts[1] if len(parts) == 2 and parts[0] in {"shorts", "live", "embed"} else "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise PipelineError("채널·재생목록 대신 영상 한 편의 URL을 입력하세요.")
    return video_id


def download_youtube(url, output_dir, language="en"):
    url = "https://www.youtube.com/watch?v=" + youtube_video_id(url)
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        raise PipelineError("유튜브 입력을 위해 requirements.txt를 설치하세요.") from None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # One combined video stream avoids a separate ffprobe/merge dependency.
    opts = {"noplaylist": True, "format": "best[height<=720]/best/bestaudio",
            "outtmpl": str(output_dir / "input.%(ext)s"), "writesubtitles": True,
            "writeautomaticsub": True, "subtitleslangs": [language], "subtitlesformat": "vtt",
            "socket_timeout": 30, "retries": 2, "ignoreerrors": False, "quiet": True}
    local_deno = Path(__file__).resolve().parents[1] / ".tools" / ("deno.exe" if os.name == "nt" else "deno")
    if local_deno.is_file():
        opts["js_runtimes"] = {"deno": {"path": str(local_deno)}}
    elif shutil.which("node"):
        opts["js_runtimes"] = {"node": {"path": shutil.which("node")}}
    try:
        with YoutubeDL(opts) as ydl:
            data = ydl.extract_info(url, download=True)
            if not data or data.get("_type") in {"playlist", "multi_video"}:
                raise PipelineError("재생목록 대신 영상 한 편의 URL을 입력하세요.")
        files = list(output_dir.iterdir())
        media = next((p for p in files if p.suffix.lower() in MEDIA_SUFFIXES), None)
        subtitle = next(iter(sorted(output_dir.glob("*.vtt"))), None)
        if media is None:
            raise PipelineError("다운로드된 미디어가 없습니다.")
        return media, subtitle, {"youtube_id": data.get("id"), "title": data.get("title"),
                                 "upload_date": data.get("upload_date"), "channel": data.get("channel")}
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError("유튜브 다운로드 실패. 로그인/JS 실행기/사이트 제한을 확인하거나 로컬 파일로 입력하세요. "
                            + str(exc)[:600]) from None


def transcribe(media, model="small", device="cpu", language="en", cache=None):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise PipelineError("자동 전사를 위해 requirements-asr.txt를 설치하세요.") from None
    if device not in {"cpu", "cuda"}:
        raise PipelineError("device는 cpu 또는 cuda입니다.")
    engine = WhisperModel(model, device=device, compute_type="int8" if device == "cpu" else "float16",
                          download_root=str(cache) if cache else None)
    segments, info = engine.transcribe(str(media), language=None if language == "auto" else language,
                                      vad_filter=True, word_timestamps=True, beam_size=5)
    rows = []
    for i, s in enumerate(segments, 1):
        rows.append({"segment_id": f"seg-{i:05d}", "start": s.start, "end": s.end, "text": s.text.strip(),
                     "words": [{"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
                               for w in (s.words or [])]})
    if not rows:
        raise PipelineError("자동 전사에서 발화를 찾지 못했습니다.")
    return rows, {"engine": "faster-whisper", "model": model, "device": device,
                  "language": info.language, "language_probability": info.language_probability}
