"""Lazy single-worker inference with bounded output and cooperative cancellation.

The admission lock belongs to the worker, not the HTTP coroutine. A disconnected
client cannot admit overlapping model work while a forward pass is still active.
"""

from __future__ import annotations

import logging
import math
import os
import queue
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from .schemas import CompletionRequest

logger = logging.getLogger(__name__)


def process_peak_rss_bytes() -> int | None:
    """Report the process high-water RSS only on platforms with known units."""
    if sys.platform not in {"linux", "darwin"}:
        return None
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


class LanguageError(RuntimeError):
    def __init__(self, code: str, message: str, status: int = 502):
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.status = status


@dataclass(frozen=True)
class LanguageSettings:
    model_id: str = "Qwen/Qwen3-0.6B"
    served_model: str = "qwen3-local"
    revision: str | None = None
    cache_dir: Path | None = field(default=None, repr=False)
    device: str = "cpu"
    dtype: str = "float32"
    local_files_only: bool = True
    api_key: str | None = field(default=None, repr=False)
    max_input_chars: int = 32_000
    max_input_tokens: int = 8192
    max_output_tokens: int = 1024
    max_output_chars: int = 12_000
    max_body_bytes: int = 131_072
    queue_size: int = 64
    generation_timeout: float = 120
    threads: int = 4
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.model_id or not self.model_id.strip() or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", self.served_model):
            raise ValueError("Configure a model and a simple public served_model alias")
        if not re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", self.device):
            raise ValueError("Language device must be cpu, mps, cuda, or cuda:N")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported language dtype")
        if not isinstance(self.local_files_only, bool):
            raise ValueError("local_files_only must be a boolean")
        for name, low, high in (("max_input_chars", 1, 64_000), ("max_input_tokens", 1, 32_768), ("max_output_tokens", 1, 4096), ("max_output_chars", 1, 64_000), ("max_body_bytes", 1024, 1_048_576), ("queue_size", 1, 256), ("threads", 1, 64), ("top_k", 1, 100), ("seed", 0, 2**32 - 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer between {low} and {high}")
        for name, low, high in (("generation_timeout", 0.01, 600), ("temperature", 0, 2), ("top_p", 0.0001, 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite and between {low} and {high}")

    @classmethod
    def from_env(cls) -> "LanguageSettings":
        prefix = "BIAS_LANGUAGE_"
        cache = os.getenv(prefix + "CACHE_DIR")
        local = os.getenv(prefix + "LOCAL_FILES_ONLY", "true").lower().strip()
        if local not in {"true", "false", "1", "0"}:
            raise ValueError("BIAS_LANGUAGE_LOCAL_FILES_ONLY must be true, false, 1, or 0")
        return cls(
            model_id=os.getenv(prefix + "MODEL", "Qwen/Qwen3-0.6B"),
            served_model=os.getenv(prefix + "SERVED_MODEL", "qwen3-local"),
            revision=os.getenv(prefix + "REVISION") or None,
            cache_dir=Path(cache).expanduser() if cache else None,
            device=os.getenv(prefix + "DEVICE", "cpu"),
            dtype=os.getenv(prefix + "DTYPE", "float32"),
            local_files_only=local in {"true", "1"},
            api_key=os.getenv(prefix + "API_KEY") or None,
            **{name: int(os.getenv(prefix + name.upper(), str(default))) for name, default in (("max_input_chars", 32_000), ("max_input_tokens", 8192), ("max_output_tokens", 1024), ("max_output_chars", 12_000), ("max_body_bytes", 131_072), ("queue_size", 64), ("threads", 4), ("top_k", 20), ("seed", 42))},
            **{name: float(os.getenv(prefix + name.upper(), str(default))) for name, default in (("generation_timeout", 120), ("temperature", 0.7), ("top_p", 0.8))},
        )


@dataclass(frozen=True)
class GenerationResult:
    prompt_tokens: int
    completion_tokens: int
    finish_reason: Literal["stop", "length"]

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (self.prompt_tokens, self.completion_tokens)) or self.finish_reason not in {"stop", "length"}:
            raise ValueError("Invalid generation usage or finish reason")

    def usage(self) -> dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens, "total_tokens": self.prompt_tokens + self.completion_tokens}


class GenerationJob:
    def __init__(self, settings: LanguageSettings):
        self.id = "chatcmpl-" + uuid.uuid4().hex
        self.created = int(time.time())
        self.started = time.monotonic()
        self.deadline = self.started + settings.generation_timeout
        self.settings = settings
        self.queue: queue.Queue[str] = queue.Queue(maxsize=settings.queue_size)
        self.finished = threading.Event()
        self.cancelled = threading.Event()
        self._cancel_lock = threading.Lock()
        self.cancel_reason: str | None = None
        self.stage = "loading"
        self.fragments: list[str] = []
        self.output_chars = 0
        self.first_text_ms: float | None = None
        self.result: GenerationResult | None = None
        self.error: LanguageError | None = None
        self.worker: threading.Thread | None = None

    def cancel(self, reason: str = "cancelled") -> None:
        with self._cancel_lock:
            if not self.cancelled.is_set():
                self.cancel_reason = reason
                self.cancelled.set()

    def should_stop(self) -> bool:
        if time.monotonic() >= self.deadline:
            self.cancel("timeout")
        return self.cancelled.is_set()

    def check(self) -> None:
        if self.should_stop():
            if self.cancel_reason == "timeout":
                raise LanguageError("generation_timeout", "Local language generation exceeded its deadline.", 504)
            raise LanguageError("generation_cancelled", "Local language generation was cancelled.", 499)

    def emit(self, text: str) -> None:
        self.check()
        if not text:
            return
        if not isinstance(text, str):
            raise LanguageError("invalid_output", "The model returned a non-text output.")
        if self.output_chars + len(text) > self.settings.max_output_chars:
            raise LanguageError("output_limit", "The model exceeded the configured text output limit.")
        while True:
            self.check()
            try:
                self.queue.put(text, timeout=0.05)
                break
            except queue.Full:
                continue
        self.output_chars += len(text)
        self.fragments.append(text)
        if self.first_text_ms is None:
            self.first_text_ms = (time.monotonic() - self.started) * 1000


class Backend(Protocol):
    def generate(self, request: CompletionRequest, job: GenerationJob) -> GenerationResult: ...


def model_location(settings: LanguageSettings) -> str:
    """Resolve a cached repo to a real directory before Transformers inspects it.

    Some optional Transformers metadata probes do not consistently propagate
    local_files_only. A local snapshot path avoids those Hub probes entirely.
    """
    local = Path(settings.model_id).expanduser()
    if local.is_dir():
        return str(local.resolve())
    if not settings.local_files_only:
        return settings.model_id
    from huggingface_hub import snapshot_download
    return snapshot_download(settings.model_id, revision=settings.revision, cache_dir=str(settings.cache_dir) if settings.cache_dir else None, local_files_only=True)


class TransformersBackend:
    def __init__(self, settings: LanguageSettings):
        # Optional dependency trees may import ONNX Runtime. Disable telemetry
        # before their first import, including its process-start event.
        os.environ["ORT_DISABLE_TELEMETRY"] = "1"
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise LanguageError("dependencies_missing", "Install the optional language service dependencies in its own environment.", 503) from exc
        if settings.device.startswith("cuda") and not torch.cuda.is_available():
            raise LanguageError("device_unavailable", "The configured CUDA device is unavailable.", 503)
        if settings.device == "mps" and not torch.backends.mps.is_available():
            raise LanguageError("device_unavailable", "The configured MPS device is unavailable.", 503)
        self.settings = settings
        self.torch = torch
        torch.set_num_threads(settings.threads)
        kwargs = {"revision": settings.revision, "cache_dir": str(settings.cache_dir) if settings.cache_dir else None, "local_files_only": settings.local_files_only, "trust_remote_code": False}
        try:
            location = model_location(settings)
            self.tokenizer = AutoTokenizer.from_pretrained(location, **kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(location, dtype=getattr(torch, settings.dtype), use_safetensors=True, attn_implementation="sdpa", low_cpu_mem_usage=True, **kwargs).to(settings.device).eval()
        except Exception as exc:
            logger.error("Language model loading failed (%s)", type(exc).__name__)
            raise LanguageError("model_unavailable", "The configured model could not be loaded; inspect its cache, revision, dependencies, and device.", 503) from exc
        if not self.tokenizer.chat_template:
            raise LanguageError("chat_template_missing", "The configured tokenizer does not supply a chat template.", 503)

    def generate(self, request: CompletionRequest, job: GenerationJob) -> GenerationResult:
        from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer

        job.check()
        job.stage = "tokenizing"
        try:
            prompt = self.tokenizer.apply_chat_template([message.model_dump() for message in request.messages], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        except Exception as exc:
            raise LanguageError("template_error", "The configured chat template could not encode these messages.", 422) from exc
        input_tokens = int(inputs["input_ids"].shape[-1])
        model_context = getattr(self.model.config, "max_position_embeddings", None)
        if input_tokens > self.settings.max_input_tokens or (isinstance(model_context, int) and input_tokens + request.max_tokens > model_context):
            raise LanguageError("context_limit", "The prompt and requested output exceed the configured context limit; no evidence was silently truncated.", 422)
        job.check()
        inputs = inputs.to(self.settings.device)
        self.torch.manual_seed(self.settings.seed if request.seed is None else request.seed)

        class OutputStreamer(TextStreamer):
            def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
                job.emit(text)

        class StopWhenCancelled(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                return job.should_stop()

        temperature = self.settings.temperature if request.temperature is None else request.temperature
        options: dict[str, Any] = {"max_new_tokens": request.max_tokens, "do_sample": temperature > 0, "streamer": OutputStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True), "stopping_criteria": StoppingCriteriaList([StopWhenCancelled()])}
        if temperature > 0:
            options.update(temperature=temperature, top_p=self.settings.top_p if request.top_p is None else request.top_p, top_k=self.settings.top_k)
        job.stage = "generating"
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **options)
        job.check()
        generated = output[0, input_tokens:]
        completion_tokens = int(generated.shape[-1])
        eos = self.model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        ended = bool(completion_tokens and int(generated[-1]) in eos_ids)
        if not ended and completion_tokens < request.max_tokens:
            raise LanguageError("unexpected_stop", "The model stopped without an observed EOS or reaching its token budget.")
        finish: Literal["stop", "length"] = "stop" if ended else "length"
        return GenerationResult(input_tokens, completion_tokens, finish)


def load_backend(settings: LanguageSettings) -> Backend:
    return TransformersBackend(settings)


class LanguageEngine:
    def __init__(self, settings: LanguageSettings, loader: Callable[[LanguageSettings], Backend] = load_backend):
        self.settings = settings
        self._loader = loader
        self._backend: Backend | None = None
        self._admission = threading.Lock()
        self._active: GenerationJob | None = None
        self._load_ms: float | None = None

    def status(self) -> dict[str, Any]:
        active = self._active
        return {"status": "ready" if self._backend is not None else "not_loaded", "model": self.settings.served_model, "model_loaded": self._backend is not None, "busy": self._admission.locked(), "stage": ("stopping" if active.cancelled.is_set() else active.stage) if active else None, "concurrency": 1, "cancellation": "cooperative_between_model_steps", "load_ms": self._load_ms, "process_peak_rss_bytes": process_peak_rss_bytes(), "limits": {"input_tokens": self.settings.max_input_tokens, "output_tokens": self.settings.max_output_tokens, "generation_seconds": self.settings.generation_timeout}}

    def start(self, request: CompletionRequest) -> GenerationJob:
        if request.model != self.settings.served_model:
            raise LanguageError("model_not_found", "The requested model is not served by this endpoint.", 404)
        if request.max_tokens > self.settings.max_output_tokens:
            raise LanguageError("output_budget", "max_tokens exceeds the server's configured output budget.", 422)
        if sum(len(message.content) for message in request.messages) > self.settings.max_input_chars:
            raise LanguageError("input_limit", "The messages exceed the configured character limit.", 422)
        if not self._admission.acquire(blocking=False):
            raise LanguageError("model_busy", "The local model is processing another request.", 429)
        job = GenerationJob(self.settings)
        self._active = job

        def run() -> None:
            try:
                if self._backend is None:
                    load_started = time.perf_counter()
                    self._backend = self._loader(self.settings)
                    self._load_ms = (time.perf_counter() - load_started) * 1000
                job.check()
                job.result = self._backend.generate(request, job)
                job.check()
                if not job.fragments:
                    raise LanguageError("empty_output", "The model completed without visible text.")
                job.stage = "complete"
            except LanguageError as exc:
                job.error = exc
            except Exception as exc:
                logger.error("Language generation failed (%s)", type(exc).__name__)
                job.error = LanguageError("generation_failed", "The local model failed during inference; inspect its runtime configuration.")
            finally:
                self._active = None
                self._admission.release()
                job.finished.set()

        job.worker = threading.Thread(target=run, name=job.id, daemon=True)
        try:
            job.worker.start()
        except Exception:
            self._active = None
            self._admission.release()
            raise
        return job

    def shutdown(self, wait_seconds: float = 2) -> None:
        active = self._active
        if active:
            active.cancel()
            active.finished.wait(wait_seconds)
