"""Lazy, single-generation Qwen3-TTS engine with one server-configured reference."""

from __future__ import annotations

import io
import os
import re
import threading
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


class VoiceUnavailable(RuntimeError):
    pass


class VoiceBusy(RuntimeError):
    pass


class VoiceOutputError(RuntimeError):
    pass


@dataclass(frozen=True)
class VoiceSettings:
    model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attention: str = "sdpa"
    reference_wav: Path | None = field(default=None, repr=False)
    reference_text: str | None = field(default=None, repr=False)
    x_vector_only_mode: bool = False
    api_key: str | None = field(default=None, repr=False)
    max_input_chars: int = 2000
    max_new_tokens: int = 1536
    max_audio_seconds: int = 120
    max_body_bytes: int = 32768

    def __post_init__(self) -> None:
        if not isinstance(self.x_vector_only_mode, bool):
            raise ValueError("x_vector_only_mode must be a boolean")
        if not re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", self.device):
            raise ValueError("Voice device must be cpu, mps, cuda, or cuda:N")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported voice dtype")
        if self.attention not in {"sdpa", "eager", "flash_attention_2"}:
            raise ValueError("Unsupported attention implementation")
        for name, low, high in (("max_input_chars", 1, 6000), ("max_new_tokens", 1, 4096), ("max_audio_seconds", 1, 180), ("max_body_bytes", 1024, 65536)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer between {low} and {high}")

    @classmethod
    def from_env(cls) -> "VoiceSettings":
        device = os.getenv("BIAS_VOICE_DEVICE", "cuda:0")
        reference = os.getenv("BIAS_VOICE_REFERENCE_WAV")
        x_vector_only = os.getenv("BIAS_VOICE_X_VECTOR_ONLY_MODE", "false").strip().lower()
        if x_vector_only not in {"true", "false", "1", "0"}:
            raise ValueError("BIAS_VOICE_X_VECTOR_ONLY_MODE must be true, false, 1, or 0")
        return cls(
            model_id=os.getenv("BIAS_VOICE_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
            device=device,
            dtype=os.getenv("BIAS_VOICE_DTYPE", "float32" if device == "cpu" else "bfloat16"),
            attention=os.getenv("BIAS_VOICE_ATTENTION", "sdpa"),
            reference_wav=Path(reference).expanduser().resolve() if reference else None,
            reference_text=os.getenv("BIAS_VOICE_REFERENCE_TEXT"),
            x_vector_only_mode=x_vector_only in {"true", "1"},
            api_key=os.getenv("BIAS_VOICE_API_KEY"),
            max_input_chars=int(os.getenv("BIAS_VOICE_MAX_INPUT_CHARS", "2000")),
            max_new_tokens=int(os.getenv("BIAS_VOICE_MAX_NEW_TOKENS", "1536")),
            max_audio_seconds=int(os.getenv("BIAS_VOICE_MAX_AUDIO_SECONDS", "120")),
            max_body_bytes=int(os.getenv("BIAS_VOICE_MAX_BODY_BYTES", "32768")),
        )


def read_reference(settings: VoiceSettings) -> tuple[Any, int]:
    path = settings.reference_wav
    if path is None:
        raise VoiceUnavailable("Configure a reference WAV before requesting speech")
    if not settings.x_vector_only_mode and (not settings.reference_text or not settings.reference_text.strip()):
        raise VoiceUnavailable("Configure a reference WAV and its accurate transcript before requesting speech")
    if not settings.x_vector_only_mode and not 1 <= len(settings.reference_text or "") <= 4000:
        raise VoiceUnavailable("The reference transcript must contain between 1 and 4000 characters")
    if not path.is_file() or path.stat().st_size > 6 * 1024**2:
        raise VoiceUnavailable("The configured reference WAV is missing or exceeds 6 MiB")
    try:
        import numpy as np
        with wave.open(str(path), "rb") as stream:
            rate = stream.getframerate()
            count = stream.getnframes()
            channels = stream.getnchannels()
            if stream.getcomptype() != "NONE" or stream.getsampwidth() != 2 or channels != 1 or not 16000 <= rate <= 48000:
                raise VoiceUnavailable("Reference audio must be mono PCM16 WAV at 16–48 kHz")
            if not 1 <= count / rate <= 30:
                raise VoiceUnavailable("Reference audio must last between 1 and 30 seconds")
            data = stream.readframes(count)
            if len(data) != count * 2:
                raise VoiceUnavailable("The reference WAV is truncated")
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        if not np.any(np.abs(samples) > 1e-4):
            raise VoiceUnavailable("The reference WAV contains no usable signal")
        return samples, rate
    except (wave.Error, EOFError, OSError) as exc:
        raise VoiceUnavailable("The configured reference WAV cannot be decoded") from exc
    except ImportError as exc:
        raise VoiceUnavailable("Install the optional voice service dependencies") from exc


def load_qwen_model(settings: VoiceSettings) -> Any:
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise VoiceUnavailable("Install services/voice/requirements.txt in a separate voice environment") from exc
    if settings.device.startswith("cuda") and not torch.cuda.is_available():
        raise VoiceUnavailable("CUDA is unavailable; configure a supported device before loading the voice model")
    return Qwen3TTSModel.from_pretrained(settings.model_id, device_map=settings.device, dtype=getattr(torch, settings.dtype), attn_implementation=settings.attention)


def encode_wav(samples: Any, sample_rate: int, max_seconds: int = 120) -> bytes:
    import numpy as np
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 96000:
        raise VoiceOutputError("The voice model returned an invalid sample rate")
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim != 1 or not 0 < data.size <= sample_rate * max_seconds or not np.all(np.isfinite(data)):
        raise VoiceOutputError("The voice model returned invalid or oversized mono audio")
    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm)
    return output.getvalue()


class VoiceEngine:
    def __init__(self, settings: VoiceSettings, model_loader: Callable[[VoiceSettings], Any] = load_qwen_model):
        self.settings = settings
        self._loader = model_loader
        self._model: Any = None
        self._prompt: Any = None
        self._generation_lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        reference_configured = bool(self.settings.reference_wav and (self.settings.x_vector_only_mode or (self.settings.reference_text and self.settings.reference_text.strip())))
        return {"status": "ready" if self._prompt is not None else "not_loaded", "reference_configured": reference_configured, "reference_mode": "speaker_embedding_only" if self.settings.x_vector_only_mode else "audio_and_transcript", "model_loaded": self._model is not None, "busy": self._generation_lock.locked(), "voice": "reference", "synthesis": "reference_conditioned", "streaming": False}

    def synthesize(self, text: str, language: str) -> bytes:
        if not text.strip() or len(text) > self.settings.max_input_chars or "\x00" in text:
            raise ValueError("Speech text is empty, contains NUL, or exceeds the configured character limit")
        if not self._generation_lock.acquire(blocking=False):
            raise VoiceBusy("The voice model is processing another request")
        # The synchronous lock is held even if an HTTP client disconnects while
        # asyncio.to_thread continues. Cancellation cannot admit overlapping GPU work.
        try:
            if self._prompt is None:
                reference = read_reference(self.settings)
                if self._model is None:
                    self._model = self._loader(self.settings)
                self._prompt = self._model.create_voice_clone_prompt(ref_audio=reference, ref_text=None if self.settings.x_vector_only_mode else self.settings.reference_text, x_vector_only_mode=self.settings.x_vector_only_mode)
            wavs, rate = self._model.generate_voice_clone(text=text, language=language, voice_clone_prompt=self._prompt, max_new_tokens=self.settings.max_new_tokens)
            if len(wavs) != 1:
                raise VoiceOutputError("The voice model did not return one waveform")
            return encode_wav(wavs[0], rate, self.settings.max_audio_seconds)
        finally:
            self._generation_lock.release()
