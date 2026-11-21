"""Optional speech-to-gesture generation (pretrained EMAGE) with an explicit fallback state.

EMAGE runs in its own Python environment as a resident worker process
(tools/gaussian_head/gesture/emage_worker.py), so torch and the research-licensed
EMAGE code never become API dependencies. Without configuration the API behaves
exactly as before and reports ``not_configured``.

Environment:
    BIAS_EMAGE_PYTHON   Python executable of the EMAGE environment
    BIAS_EMAGE_REPO     EMAGE (character-policy) checkout containing models/emage_audio.py
    BIAS_EMAGE_DEVICE   cpu | cuda (default: worker picks cuda when available)
    BIAS_EMAGE_TIMEOUT  seconds per clip (default 60)
    BIAS_EMAGE_WORKER   override for the worker script path

The motion is generated from audio alone. It follows speech rhythm; it is not a
measurement of the persona's own gestures.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from .lipsync import wav_duration_ms

POSE_WIDTH = 165  # 55 SMPL-X joints x axis-angle
FPS = 30
DEFAULT_WORKER = Path(__file__).resolve().parents[2] / "tools" / "gaussian_head" / "gesture" / "emage_worker.py"


@dataclass(frozen=True)
class GestureResult:
    status: str
    motion: bytes | None = None
    frames: int = 0
    message: str | None = None
    compute_ms: float | None = None


class GestureEngine:
    def __init__(self, python: str | None = None, repo: str | None = None, *, worker: str | os.PathLike | None = None,
                 device: str = "", timeout: float = 60, startup_timeout: float = 300):
        if not 1 <= timeout <= 600 or not 1 <= startup_timeout <= 1800 or device not in {"", "cpu", "cuda"}:
            raise ValueError("Invalid gesture resource bounds")
        self.python = python
        self.repo = repo
        self.worker = Path(worker) if worker else DEFAULT_WORKER
        self.device = device
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._start_failed: str | None = None

    @classmethod
    def from_env(cls) -> "GestureEngine":
        return cls(os.getenv("BIAS_EMAGE_PYTHON") or None, os.getenv("BIAS_EMAGE_REPO") or None,
                   worker=os.getenv("BIAS_EMAGE_WORKER") or None, device=os.getenv("BIAS_EMAGE_DEVICE", ""),
                   timeout=float(os.getenv("BIAS_EMAGE_TIMEOUT", "60")))

    @property
    def configured(self) -> bool:
        return bool(self.python and self.repo)

    @property
    def ready(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def _read(self, timeout: float) -> dict:
        assert self._process is not None and self._process.stdout is not None
        while True:
            line = await asyncio.wait_for(self._process.stdout.readline(), timeout=timeout)
            if not line:
                raise EOFError("The gesture worker exited")
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue  # stray library output; the protocol lines are JSON objects
            if isinstance(message, dict):
                return message

    async def _start(self) -> None:
        if self.ready:
            return
        await self._stop()
        arguments = [self.python, "-u", str(self.worker), self.repo]
        if self.device:
            arguments += ["--device", self.device]
        self._process = await asyncio.create_subprocess_exec(*arguments, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, limit=1_000_000)
        try:
            message = await self._read(self.startup_timeout)
        except BaseException:
            await self._stop()
            raise
        if message.get("ready") is not True or message.get("fps") != FPS:
            await self._stop()
            raise RuntimeError("The gesture worker did not report readiness")

    async def _stop(self) -> None:
        process, self._process = self._process, None
        if process is None or process.returncode is not None:
            return
        try:
            process.kill()
        except ProcessLookupError:
            pass
        await process.wait()

    async def start(self) -> bool:
        """Load the model ahead of the first request; failures leave a fallback state."""
        if not self.configured:
            return False
        async with self._lock:
            try:
                await self._start()
                self._start_failed = None
                return True
            except asyncio.CancelledError:
                raise
            except (OSError, EOFError, TimeoutError, RuntimeError) as exc:
                self._start_failed = type(exc).__name__
                return False

    async def close(self) -> None:
        async with self._lock:
            await self._stop()

    async def generate(self, data: bytes) -> GestureResult:
        if not self.configured:
            return GestureResult("not_configured", message="Speech-driven gestures are not configured; the avatar uses designed movement.")
        duration = wav_duration_ms(data) if len(data) <= 20_000_000 else None
        if duration is None:
            return GestureResult("unsupported_audio", message="Speech-driven gestures require a bounded PCM16 WAV; the audio remains playable.")
        async with self._lock:
            try:
                await self._start()
            except asyncio.CancelledError:
                raise
            except (OSError, EOFError, TimeoutError, RuntimeError):
                return GestureResult("unavailable", message="The gesture model could not be started; the avatar uses designed movement.")
            with tempfile.TemporaryDirectory(prefix="bias-gesture-") as directory:
                root = Path(directory)
                audio_path, out_path = root / "speech.wav", root / "poses.f32"
                audio_path.write_bytes(data)
                request_id = uuid.uuid4().hex
                try:
                    assert self._process is not None and self._process.stdin is not None
                    self._process.stdin.write((json.dumps({"id": request_id, "wav": str(audio_path), "out": str(out_path)}) + "\n").encode())
                    await self._process.stdin.drain()
                    while True:
                        message = await self._read(self.timeout)
                        if message.get("id") == request_id:
                            break
                except asyncio.CancelledError:
                    # The worker would still answer this request; restart rather than desynchronise.
                    await self._stop()
                    raise
                except TimeoutError:
                    await self._stop()
                    return GestureResult("timeout", message="Gesture generation timed out; the avatar uses designed movement.")
                except (OSError, EOFError, ConnectionError):
                    await self._stop()
                    return GestureResult("failed", message="The gesture worker stopped; the avatar uses designed movement.")
                frames = message.get("frames")
                if message.get("ok") is not True or message.get("fps") != FPS or isinstance(frames, bool) or not isinstance(frames, int):
                    return GestureResult("failed", message="Gesture generation failed; the avatar uses designed movement.")
                expected = duration / 1000 * FPS
                if not 1 <= frames <= FPS * 180 or abs(frames - expected) > max(6, expected * 0.05) or not out_path.is_file():
                    return GestureResult("failed", message="Generated gestures do not match the audio duration.")
                motion = out_path.read_bytes()
                if len(motion) != frames * POSE_WIDTH * 4:
                    return GestureResult("failed", message="Generated gestures have an unexpected size.")
                compute = message.get("compute_ms")
                return GestureResult("available", motion, frames, compute_ms=float(compute) if isinstance(compute, (int, float)) else None)
