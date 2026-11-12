"""Lazy, single-generation Qwen3-TTS engine with one server-configured reference."""

from __future__ import annotations

import io
import math
import os
import re
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .provenance import ReferenceBindingError, verify_binding

# The official ONNX Runtime opt-out must precede initialization, including
# transitive imports by model libraries. Local inference needs no telemetry.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


class VoiceUnavailable(RuntimeError):
    pass


class VoiceBusy(RuntimeError):
    pass


class VoiceOutputError(RuntimeError):
    pass


@dataclass(frozen=True)
class VoiceSettings:
    model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    revision: str | None = None
    cache_dir: Path | None = field(default=None, repr=False)
    local_files_only: bool = True
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attention: str = "sdpa"
    reference_wav: Path | None = field(default=None, repr=False)
    reference_text: str | None = field(default=None, repr=False)
    persona_id: str | None = None
    speaker_label: str | None = None
    reference_manifest: Path | None = field(default=None, repr=False)
    allow_candidate_reference: bool = False
    x_vector_only_mode: bool = False
    api_key: str | None = field(default=None, repr=False)
    max_input_chars: int = 2000
    max_new_tokens: int = 1536
    max_audio_seconds: int = 120
    max_body_bytes: int = 32768
    threads: int = 4
    seed: int = 42
    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05

    def __post_init__(self) -> None:
        for name in ("x_vector_only_mode", "local_files_only", "do_sample", "allow_candidate_reference"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("Configure a voice model")
        if self.revision is not None and not re.fullmatch(r"[0-9a-f]{40,64}", self.revision):
            raise ValueError("Voice revision must be an immutable commit hash")
        if self.persona_id is not None and (not isinstance(self.persona_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,119}", self.persona_id)):
            raise ValueError("Voice persona ID must be a stable subject identifier")
        if self.speaker_label is not None and (not isinstance(self.speaker_label, str) or not self.speaker_label.strip() or len(self.speaker_label) > 200):
            raise ValueError("Voice speaker label must be the exact publisher label")
        if not re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", self.device):
            raise ValueError("Voice device must be cpu, mps, cuda, or cuda:N")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported voice dtype")
        if self.attention not in {"sdpa", "eager", "flash_attention_2"}:
            raise ValueError("Unsupported attention implementation")
        for name, low, high in (("max_input_chars", 1, 6000), ("max_new_tokens", 1, 4096), ("max_audio_seconds", 1, 180), ("max_body_bytes", 1024, 65536), ("threads", 1, 64), ("seed", 0, 2**32 - 1), ("top_k", 1, 100)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer between {low} and {high}")
        for name, low, high in (("temperature", 0.01, 2), ("top_p", 0.0001, 1), ("repetition_penalty", 1, 2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite and between {low} and {high}")

    @classmethod
    def from_env(cls) -> "VoiceSettings":
        device = os.getenv("BIAS_VOICE_DEVICE", "cuda:0")
        reference = os.getenv("BIAS_VOICE_REFERENCE_WAV")
        transcript = os.getenv("BIAS_VOICE_REFERENCE_TEXT") or None
        transcript_file = os.getenv("BIAS_VOICE_REFERENCE_TEXT_FILE") or None
        if transcript and transcript_file:
            raise ValueError("Configure only one reference transcript source")
        if transcript_file:
            path = Path(transcript_file).expanduser().resolve()
            if not path.is_file() or path.stat().st_size > 16000:
                raise ValueError("Reference transcript file is missing or exceeds 16 KB")
            transcript = path.read_text(encoding="utf-8").strip()
        if transcript is not None and (not 1 <= len(transcript.strip()) <= 4000 or "\x00" in transcript):
            raise ValueError("Reference transcript must contain 1–4000 characters and no NUL")
        x_vector_only = os.getenv("BIAS_VOICE_X_VECTOR_ONLY_MODE", "false").strip().lower()
        if x_vector_only not in {"true", "false", "1", "0"}:
            raise ValueError("BIAS_VOICE_X_VECTOR_ONLY_MODE must be true, false, 1, or 0")
        def boolean(name: str, default: str) -> bool:
            value = os.getenv("BIAS_VOICE_" + name, default).strip().lower()
            if value not in {"true", "false", "1", "0"}:
                raise ValueError(f"BIAS_VOICE_{name} must be true, false, 1, or 0")
            return value in {"true", "1"}
        cache = os.getenv("BIAS_VOICE_CACHE_DIR")
        manifest = os.getenv("BIAS_VOICE_REFERENCE_MANIFEST")
        return cls(
            model_id=os.getenv("BIAS_VOICE_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
            revision=os.getenv("BIAS_VOICE_REVISION") or None,
            cache_dir=Path(cache).expanduser() if cache else None,
            local_files_only=boolean("LOCAL_FILES_ONLY", "true"),
            device=device,
            dtype=os.getenv("BIAS_VOICE_DTYPE", "float32" if device == "cpu" else "bfloat16"),
            attention=os.getenv("BIAS_VOICE_ATTENTION", "sdpa"),
            reference_wav=Path(reference).expanduser().resolve() if reference else None,
            reference_text=transcript,
            persona_id=os.getenv("BIAS_VOICE_PERSONA_ID") or None,
            speaker_label=os.getenv("BIAS_VOICE_SPEAKER_LABEL") or None,
            reference_manifest=Path(manifest).expanduser().resolve() if manifest else None,
            allow_candidate_reference=boolean("ALLOW_CANDIDATE_REFERENCE", "false"),
            x_vector_only_mode=x_vector_only in {"true", "1"},
            api_key=os.getenv("BIAS_VOICE_API_KEY"),
            max_input_chars=int(os.getenv("BIAS_VOICE_MAX_INPUT_CHARS", "2000")),
            max_new_tokens=int(os.getenv("BIAS_VOICE_MAX_NEW_TOKENS", "1536")),
            max_audio_seconds=int(os.getenv("BIAS_VOICE_MAX_AUDIO_SECONDS", "120")),
            max_body_bytes=int(os.getenv("BIAS_VOICE_MAX_BODY_BYTES", "32768")),
            threads=int(os.getenv("BIAS_VOICE_THREADS", "4")),
            seed=int(os.getenv("BIAS_VOICE_SEED", "42")),
            do_sample=boolean("DO_SAMPLE", "true"),
            temperature=float(os.getenv("BIAS_VOICE_TEMPERATURE", "0.9")),
            top_k=int(os.getenv("BIAS_VOICE_TOP_K", "50")),
            top_p=float(os.getenv("BIAS_VOICE_TOP_P", "1.0")),
            repetition_penalty=float(os.getenv("BIAS_VOICE_REPETITION_PENALTY", "1.05")),
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


def model_location(settings: VoiceSettings) -> str:
    """Resolve cached weights before Qwen/Transformers can probe Hub metadata."""
    local = Path(settings.model_id).expanduser()
    if local.is_dir():
        return str(local.resolve())
    from huggingface_hub import snapshot_download
    return snapshot_download(settings.model_id, revision=settings.revision,
        cache_dir=str(settings.cache_dir) if settings.cache_dir else None,
        local_files_only=settings.local_files_only)


class SeededQwen:
    """Seed the production backend while keeping protocol tests dependency-free.

    The service admits one generation at a time. A seed helps repeat a run on
    the same stack; it does not promise bitwise identity across devices/releases.
    """
    def __init__(self, model: Any, torch: Any, seed: int):
        self._qwen, self.torch, self.seed = model, torch, seed

    def __getattr__(self, name: str) -> Any:
        # Preserve Qwen's public attributes, including .model for the existing
        # bounded smoke runner's parameter-count measurement.
        return getattr(self._qwen, name)

    def create_voice_clone_prompt(self, **kwargs: Any) -> Any:
        return self._qwen.create_voice_clone_prompt(**kwargs)

    def generate_voice_clone(self, **kwargs: Any) -> Any:
        self.torch.manual_seed(self.seed)
        with self.torch.inference_mode():
            return self._qwen.generate_voice_clone(**kwargs)


def load_qwen_model(settings: VoiceSettings) -> Any:
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise VoiceUnavailable("Install services/voice/requirements.txt in a separate voice environment") from exc
    if settings.device.startswith("cuda") and not torch.cuda.is_available():
        raise VoiceUnavailable("CUDA is unavailable; configure a supported device before loading the voice model")
    if settings.device == "mps" and not torch.backends.mps.is_available():
        raise VoiceUnavailable("MPS is unavailable; configure a supported device before loading the voice model")
    torch.set_num_threads(settings.threads)
    try:
        location = model_location(settings)
        model = Qwen3TTSModel.from_pretrained(location, device_map=settings.device,
            dtype=getattr(torch, settings.dtype), attn_implementation=settings.attention,
            local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        raise VoiceUnavailable("Voice model could not load; check its cache, revision, device, and dependencies") from exc
    return SeededQwen(model, torch, settings.seed)


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
        self._last_generation: dict[str, Any] | None = None
        self._reference_identity: dict[str, Any] | None = None

    def _verify_reference_binding(self) -> dict:
        try:
            return verify_binding(manifest_path=self.settings.reference_manifest,
                persona_id=self.settings.persona_id, speaker=self.settings.speaker_label,
                audio_path=self.settings.reference_wav, text=self.settings.reference_text,
                embedding_only=self.settings.x_vector_only_mode,
                allow_candidate=self.settings.allow_candidate_reference)
        except ReferenceBindingError as exc:
            raise VoiceUnavailable(str(exc)) from exc

    def status(self) -> dict[str, Any]:
        reference_configured = bool(self.settings.reference_wav and (self.settings.x_vector_only_mode or (self.settings.reference_text and self.settings.reference_text.strip())))
        return {"status": "ready" if self._prompt is not None else "not_loaded", "reference_configured": reference_configured, "reference_mode": "speaker_embedding_only" if self.settings.x_vector_only_mode else "audio_and_transcript", "model_loaded": self._model is not None, "busy": self._generation_lock.locked(), "voice": "reference", "synthesis": "reference_conditioned", "streaming": False,
            "persona_id": self.settings.persona_id, "reference_identity": self._reference_identity,
            "runtime": {"revision": self.settings.revision, "device": self.settings.device,
                "dtype": self.settings.dtype, "attention": self.settings.attention,
                "local_files_only": self.settings.local_files_only, "threads": self.settings.threads,
                "seed": self.settings.seed, "max_new_tokens": self.settings.max_new_tokens,
                "do_sample": self.settings.do_sample, "temperature": self.settings.temperature,
                "top_k": self.settings.top_k, "top_p": self.settings.top_p,
                "repetition_penalty": self.settings.repetition_penalty},
            "last_generation": self._last_generation}

    def synthesize(self, text: str, language: str) -> bytes:
        if not text.strip() or len(text) > self.settings.max_input_chars or "\x00" in text:
            raise ValueError("Speech text is empty, contains NUL, or exceeds the configured character limit")
        if not self._generation_lock.acquire(blocking=False):
            raise VoiceBusy("The voice model is processing another request")
        # The synchronous lock is held even if an HTTP client disconnects while
        # asyncio.to_thread continues. Cancellation cannot admit overlapping GPU work.
        try:
            started = time.monotonic()
            cold_start = self._prompt is None
            self._last_generation = None
            identity = self._verify_reference_binding()
            if self._reference_identity is not None and identity != self._reference_identity:
                raise VoiceUnavailable("Reference provenance changed after the voice prompt was loaded")
            self._reference_identity = identity
            if self._prompt is None:
                reference = read_reference(self.settings)
                if self._model is None:
                    self._model = self._loader(self.settings)
                self._prompt = self._model.create_voice_clone_prompt(ref_audio=reference, ref_text=None if self.settings.x_vector_only_mode else self.settings.reference_text, x_vector_only_mode=self.settings.x_vector_only_mode)
            wavs, rate = self._model.generate_voice_clone(text=text, language=language,
                voice_clone_prompt=self._prompt, max_new_tokens=self.settings.max_new_tokens,
                do_sample=self.settings.do_sample, temperature=self.settings.temperature,
                top_k=self.settings.top_k, top_p=self.settings.top_p,
                repetition_penalty=self.settings.repetition_penalty,
                subtalker_dosample=self.settings.do_sample,
                subtalker_temperature=self.settings.temperature,
                subtalker_top_k=self.settings.top_k, subtalker_top_p=self.settings.top_p,
                # This Qwen API still returns the complete waveform. Its false
                # mode simulates streaming text input; it is not audio streaming.
                non_streaming_mode=False)
            if len(wavs) != 1:
                raise VoiceOutputError("The voice model did not return one waveform")
            result = encode_wav(wavs[0], rate, self.settings.max_audio_seconds)
            duration = len(wavs[0]) / rate
            elapsed = time.monotonic() - started
            self._last_generation = {"language": language, "input_chars": len(text),
                "cold_start": cold_start, "elapsed_seconds": round(elapsed, 6),
                "audio_seconds": round(duration, 6), "real_time_factor": round(elapsed / duration, 6),
                "termination_reason": "not_exposed_by_backend"}
            return result
        finally:
            self._generation_lock.release()
