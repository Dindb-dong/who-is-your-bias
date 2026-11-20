"""Local speech recognition using the project's cached Whisper checkpoint."""
from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path


class LocalWhisper:
    model_name = "faster-whisper-small"

    def __init__(self, python: Path, checkpoint: Path):
        self.python = python
        self.checkpoint = checkpoint

    async def transcribe(self, data: bytes, mime: str, *, language: str | None = None) -> str:
        from bias.providers import ProviderError

        try:
            process = await asyncio.create_subprocess_exec(
                str(self.python), str(Path(__file__).resolve()), str(self.checkpoint), language or "auto",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as error:
            raise ProviderError("transcription_unavailable", "The local transcription service could not start.", 503) from error
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(data), timeout=25)
        except (TimeoutError, asyncio.CancelledError) as error:
            if process.returncode is None:
                process.kill()
                await process.wait()
            if isinstance(error, asyncio.CancelledError):
                raise
            raise ProviderError("transcription_timeout", "Local transcription took too long.", 504) from error
        if process.returncode:
            raise ProviderError("transcription_failed", "The local model could not transcribe the recording.")
        text = stdout.decode("utf-8").strip()
        if not text or len(text) > 2000:
            raise ProviderError("transcription_empty", "No usable question was found in the recording.", 422)
        return text


def find_local_whisper(project: Path) -> LocalWhisper | None:
    if not (project / ".venv/bin/python").is_file() or not any(project.glob(".venv/lib/python*/site-packages/faster_whisper")):
        return None
    cache = project / "data/local/models/models--Systran--faster-whisper-small/snapshots"
    for checkpoint in sorted(cache.glob("*")):
        if all((checkpoint / name).is_file() for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")):
            return LocalWhisper(project / ".venv/bin/python", checkpoint)
    return None


if __name__ == "__main__":
    from faster_whisper import WhisperModel

    audio = io.BytesIO(sys.stdin.buffer.read())
    model = WhisperModel(sys.argv[1], device="cpu", compute_type="int8", cpu_threads=4, local_files_only=True)
    language = None if sys.argv[2] == "auto" else sys.argv[2]
    keywords = "NVIDIA, Jensen Huang, CUDA, GeForce, GPU, GTC, NVLink" if language == "en" else None
    segments, _ = model.transcribe(audio, language=language, beam_size=3, hotwords=keywords, vad_filter=True)
    print(" ".join(segment.text.strip() for segment in segments))
