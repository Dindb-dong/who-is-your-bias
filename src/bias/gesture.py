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
    BIAS_EMAGE_WINDOW   streaming window in frames, 8..64 (default: the model's 64)

Prepared speech sends one finished WAV (``generate``). Live voice opens a
``GestureStream`` and pushes each sentence's PCM as it is synthesized; the
worker runs EMAGE's own 64-frame windows as soon as their audio exists.

The motion is generated from audio alone. It follows speech rhythm; it is not a
measurement of the persona's own gestures.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from .lipsync import wav_duration_ms

POSE_WIDTH = 165  # 55 SMPL-X joints x axis-angle
FPS = 30
ENGINE_LABEL = "emage-audio (pretrained on BEAT2)"
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
                 device: str = "", timeout: float = 60, startup_timeout: float = 300, window: int = 0):
        if not 1 <= timeout <= 600 or not 1 <= startup_timeout <= 1800 or device not in {"", "cpu", "cuda"} \
                or not (window == 0 or 8 <= window <= 64):
            raise ValueError("Invalid gesture resource bounds")
        self.python = python
        self.repo = repo
        self.worker = Path(worker) if worker else DEFAULT_WORKER
        self.device = device
        self.window = window
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._start_failed: str | None = None
        self.stream_window: int | None = None

    @classmethod
    def from_env(cls) -> "GestureEngine":
        return cls(os.getenv("BIAS_EMAGE_PYTHON") or None, os.getenv("BIAS_EMAGE_REPO") or None,
                   worker=os.getenv("BIAS_EMAGE_WORKER") or None, device=os.getenv("BIAS_EMAGE_DEVICE", ""),
                   timeout=float(os.getenv("BIAS_EMAGE_TIMEOUT", "60")), window=int(os.getenv("BIAS_EMAGE_WINDOW") or 0))

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
        if self.window:
            arguments += ["--window", str(self.window)]
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
        window = message.get("window")
        self.stream_window = window if isinstance(window, int) and not isinstance(window, bool) and 8 <= window <= 64 else None

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

    async def _call(self, payload: dict, timeout: float) -> dict:
        """One protocol request (stream ops); raises GestureStreamError with a fallback status."""
        if not self.configured:
            raise GestureStreamError("not_configured", "Speech-driven gestures are not configured; the avatar uses designed movement.")
        async with self._lock:
            try:
                await self._start()
            except asyncio.CancelledError:
                raise
            except (OSError, EOFError, TimeoutError, RuntimeError) as exc:
                raise GestureStreamError("unavailable", "The gesture model could not be started; the avatar uses designed movement.") from exc
            request_id = uuid.uuid4().hex
            try:
                assert self._process is not None and self._process.stdin is not None
                self._process.stdin.write((json.dumps({**payload, "id": request_id}) + "\n").encode())
                await self._process.stdin.drain()
                while True:
                    message = await self._read(timeout)
                    if message.get("id") == request_id:
                        break
            except asyncio.CancelledError:
                await self._stop()
                raise
            except TimeoutError as exc:
                await self._stop()
                raise GestureStreamError("timeout", "Gesture generation fell behind; the avatar uses designed movement.") from exc
            except (OSError, EOFError, ConnectionError) as exc:
                await self._stop()
                raise GestureStreamError("failed", "The gesture worker stopped; the avatar uses designed movement.") from exc
        if message.get("ok") is not True:
            raise GestureStreamError("failed", "Gesture generation failed; the avatar uses designed movement.")
        return message

    def stream(self) -> "GestureStream":
        return GestureStream(self)


class GestureStreamError(Exception):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class GestureStream:
    """Incremental motion for one live turn. Frames are on the turn's audio clock: frame f starts at f/30 s
    of the concatenated PCM. Each push returns the newly finished frames, contiguous from frame 0."""

    MAX_FRAMES = FPS * 600

    def __init__(self, engine: GestureEngine):
        self.engine = engine
        self.session = uuid.uuid4().hex
        self.next_frame = 0
        self.audio_seconds = 0.0
        self.window: int | None = None
        self.opened = False
        self.finished = False

    async def open(self) -> None:
        reply = await self.engine._call({"op": "open", "session": self.session}, self.engine.startup_timeout)
        window = reply.get("window")
        if not isinstance(window, int) or isinstance(window, bool) or not 8 <= window <= 64:
            raise GestureStreamError("failed", "The gesture worker does not support streaming.")
        self.window = window
        self.opened = True

    async def push(self, pcm: bytes, rate: int, channels: int, *, final: bool = False) -> list[tuple[int, bytes]]:
        if not self.opened or self.finished:
            raise GestureStreamError("failed", "The gesture stream is not open.")
        if channels not in {1, 2} or not 8000 <= rate <= 48000 or len(pcm) % (2 * channels) or len(pcm) > rate * channels * 2 * 60:
            raise GestureStreamError("unsupported_audio", "Speech-driven gestures need 16-bit PCM sentences up to 60 s.")
        self.audio_seconds += len(pcm) / (2 * channels * rate)
        reply = await self.engine._call({"op": "push", "session": self.session, "pcm": base64.b64encode(pcm).decode("ascii"),
                                         "rate": rate, "channels": channels, "final": final}, self.engine.timeout)
        chunks = reply.get("chunks")
        if not isinstance(chunks, list):
            raise GestureStreamError("failed", "The gesture worker sent an invalid stream reply.")
        result = []
        for chunk in chunks:
            start, frames = chunk.get("start") if isinstance(chunk, dict) else None, chunk.get("frames") if isinstance(chunk, dict) else None
            if any(not isinstance(v, int) or isinstance(v, bool) for v in (start, frames)) or start != self.next_frame \
                    or not 1 <= frames <= 64 or start + frames > min(self.MAX_FRAMES, self.audio_seconds * FPS + 2):
                raise GestureStreamError("failed", "Generated gestures do not follow the audio.")
            try:
                data = base64.b64decode(chunk.get("poses", ""), validate=True)
            except (ValueError, TypeError) as exc:
                raise GestureStreamError("failed", "Generated gestures have an unexpected size.") from exc
            if len(data) != frames * POSE_WIDTH * 4:
                raise GestureStreamError("failed", "Generated gestures have an unexpected size.")
            result.append((start, data))
            self.next_frame += frames
        self.finished = final
        return result

    async def close(self) -> None:
        if self.opened and not self.finished and self.engine.ready:
            try:
                await self.engine._call({"op": "close", "session": self.session}, 10)
            except GestureStreamError:
                pass
        self.finished = True


CANCEL = "cancel"


async def stream_gestures(engine: GestureEngine, audio: asyncio.Queue, emit: Callable[[str, dict], Awaitable[object]],
                          turn_id: Callable[[], str | None]) -> None:
    """Feed one turn's PCM to EMAGE and emit gesture_state / gesture / gesture_end events.

    Queue items: ``(pcm, rate, channels)`` in playback order, ``None`` when the audio is complete, or ``CANCEL``
    (barge-in: stop without finishing). Frames are on the turn's PCM clock (frame f = f/30 s of the concatenated
    audio). Failures only stop the gestures; the caller's speech continues with designed movement.
    """
    stream: GestureStream | None = engine.stream()
    status, message = "streaming", None
    try:
        await stream.open()
    except GestureStreamError as exc:
        stream, status, message = None, exc.status, exc.message

    def state(status: str, message: str | None, window: int | None) -> dict:
        return {"turn_id": turn_id(), "status": status, "message": message, "fps": FPS, "width": POSE_WIDTH,
                "format": "smplx-axis-angle-f32le", "window_frames": window, "engine": ENGINE_LABEL}

    announced, sequence, frames, audio_format = False, 0, 0, None
    while True:
        item = await audio.get()
        if item == CANCEL:
            if stream is not None:
                await stream.close()
            if announced:
                await emit("gesture_end", {"turn_id": turn_id(), "chunks": sequence, "frames": frames, "status": "stopped"})
            return
        if not announced:   # after the first audio, so the caller has already sent turn ownership
            announced = True
            await emit("gesture_state", state(status, message, stream.window if stream else None))
        final = item is None
        if stream is not None and (item or audio_format):
            pcm, rate, channels = item if item else (b"", *audio_format)
            audio_format = rate, channels
            # One-second pieces: each EMAGE window is sent as soon as its audio exists.
            piece = rate * channels * 2
            parts = [pcm[i:i + piece] for i in range(0, len(pcm), piece)] or [b""]
            for index, part in enumerate(parts):
                try:
                    chunks = await stream.push(part, rate, channels, final=final and index == len(parts) - 1)
                except GestureStreamError as exc:
                    stream = None
                    await emit("gesture_state", state(exc.status, exc.message, None))
                    break
                for start, data in chunks:
                    count = len(data) // (POSE_WIDTH * 4)
                    await emit("gesture", {"turn_id": turn_id(), "sequence": sequence, "start_frame": start, "frame_count": count,
                                           "encoding": "f32le", "poses_base64": base64.b64encode(data).decode("ascii")})
                    sequence += 1
                    frames = start + count
        if final:
            break
    if announced:
        await emit("gesture_end", {"turn_id": turn_id(), "chunks": sequence, "frames": frames,
                                   "status": "complete" if stream is not None else "stopped"})
