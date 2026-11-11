"""Measure a real HTTP live LLM/TTS turn; never infer browser/acoustic latency."""
from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import codecs
import hashlib
import ipaddress
import json
import math
import os
import platform
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from bias.grounding import validate_citations
from bias.models import ChatRequest


PROTOCOL = "live-segments-pcm.v1"
SCHEMA = "live-http-benchmark.v1"


class ProtocolError(ValueError):
    """A bounded public explanation; never includes raw provider content."""


def _integer(value, *, minimum=0, maximum=1_000_000_000) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ProtocolError("Invalid integer in live protocol")
    return value


def _number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ProtocolError("Invalid timing number in live protocol")
    return float(value)


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: dict) -> str:
    return _hash(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode())


async def sse_frames(chunks):
    """Bound incremental UTF-8/SSE parsing without buffering an entire reply."""
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    buffer, event, lines, frame_size = "", None, [], 0
    async for chunk in chunks:
        try:
            buffer += decoder.decode(chunk)
        except UnicodeDecodeError as exc:
            raise ProtocolError("SSE is not valid UTF-8") from exc
        if len(buffer) > 1_048_576:
            raise ProtocolError("SSE frame exceeds 1 MiB")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.removesuffix("\r")
            frame_size += len(line)
            if frame_size > 1_048_576:
                raise ProtocolError("SSE frame exceeds 1 MiB")
            if not line:
                if lines:
                    try:
                        value = json.loads("\n".join(lines))
                    except ValueError as exc:
                        raise ProtocolError("SSE payload is not valid JSON") from exc
                    if not isinstance(value, dict) or event is None:
                        raise ProtocolError("Live SSE requires a named JSON-object event")
                    yield event, value
                event, lines, frame_size = None, [], 0
            elif line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                lines.append(line[5:].removeprefix(" "))
            elif line.startswith(":") or line.startswith(("id:", "retry:")):
                pass
            else:
                raise ProtocolError("Unknown live SSE field")
    try:
        buffer += decoder.decode(b"", final=True)
    except UnicodeDecodeError as exc:
        raise ProtocolError("Truncated SSE UTF-8 sequence") from exc
    if buffer or lines or event:
        raise ProtocolError("Truncated live SSE frame")


class LiveProtocol:
    """Check a stream prefix and require full integrity before completion."""

    def __init__(self):
        self.session_id = self.turn_id = None
        self.raw_text = ""
        self.segments, self.audio_segments = [], []
        self.active = None
        self.rate = self.channels = None
        self.chunks = self.frames = self.pcm_bytes = 0
        self.pcm_digest = hashlib.sha256()
        self.result = self.audio_end = None
        self.done = False
        self.events = Counter()
        self.timings = {key: None for key in ("first_text_received_ms", "first_pcm_received_ms", "result_received_ms", "audio_end_received_ms", "done_received_ms")}
        self.last_at = 0.0

    def _turn(self, data):
        if data.get("turn_id") != self.turn_id:
            raise ProtocolError("Live event has a stale or missing turn identity")

    def _segment(self, data, *, key="sequence"):
        self._turn(data)
        if self.active is None or data.get("segment_id") != self.active["segment_id"] or _integer(data.get(key)) != self.active["sequence"]:
            raise ProtocolError("Live audio does not match the active segment")

    def accept(self, event: str, data: dict, at_ms: float) -> None:
        at_ms = _number(at_ms)
        if at_ms < self.last_at:
            raise ProtocolError("Client observation clock moved backwards")
        self.last_at = at_ms
        self.events[event] += 1
        if sum(self.events.values()) > 50_000:
            raise ProtocolError("Live event count exceeds the benchmark bound")
        if self.done:
            raise ProtocolError("Received a live event after done")
        if event == "meta":
            if self.turn_id is not None or data.get("protocol") != PROTOCOL:
                raise ProtocolError("Duplicate or incompatible live meta event")
            for key in ("turn_id", "session_id"):
                value = data.get(key)
                if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", value):
                    raise ProtocolError("Live session/turn identity is invalid")
            self.turn_id, self.session_id = data["turn_id"], data["session_id"]
            return
        if self.turn_id is None:
            raise ProtocolError("Live stream must start with meta")
        if event in {"error", "turn_state"}:
            raise ProtocolError("Server reported a failed live turn")
        if event == "status":
            if self.result is not None or not isinstance(data.get("stage"), str):
                raise ProtocolError("Invalid live generation status")
        elif event == "token":
            text = data.get("text")
            if self.result is not None or not isinstance(text, str) or len(self.raw_text) + len(text) > 100_000:
                raise ProtocolError("Invalid, late or oversized live text token")
            self.raw_text += text
            if text.strip() and self.timings["first_text_received_ms"] is None:
                self.timings["first_text_received_ms"] = at_ms
        elif event == "segment":
            self._turn(data)
            sequence = _integer(data.get("sequence"))
            start, end = _integer(data.get("char_start")), _integer(data.get("char_end"))
            previous_end = self.segments[-1]["char_end"] if self.segments else 0
            if (self.result is not None or sequence != len(self.segments) or data.get("segment_id") != f"{self.turn_id}:{sequence}"
                    or not previous_end <= start < end <= len(self.raw_text) or data.get("raw_text") != self.raw_text[start:end]
                    or data.get("state") != "committed" or not isinstance(data.get("text"), str) or not data["text"].strip()
                    or validate_citations(data["raw_text"], set())[0] != data["text"]):
                raise ProtocolError("Committed segment differs from received text or order")
            if start > previous_end and validate_citations(self.raw_text[previous_end:start], set())[0]:
                raise ProtocolError("Committed segments skipped meaningful text")
            self.segments.append(data)
        elif event == "audio_segment":
            self._turn(data)
            sequence = _integer(data.get("sequence"))
            rate, channels = _integer(data.get("sample_rate"), minimum=8000, maximum=48000), _integer(data.get("channels"), minimum=1, maximum=2)
            count = _integer(data.get("sample_count"), minimum=1, maximum=rate * 180)
            if (self.active is not None or self.audio_end is not None or sequence != len(self.audio_segments) or sequence >= len(self.segments)
                    or data.get("segment_id") != self.segments[sequence]["segment_id"] or _integer(data.get("sample_offset")) != self.frames
                    or (self.rate is not None and (self.rate, self.channels) != (rate, channels))):
                raise ProtocolError("Audio segment order, format or sample extent changed")
            if any(not isinstance(data.get(key), str) or not re.fullmatch("[a-f0-9]{64}", data[key]) for key in ("audio_sha256", "pcm_sha256")):
                raise ProtocolError("Audio segment requires original WAV and verifiable PCM digests")
            self.rate, self.channels = rate, channels
            self.active = {"sequence": sequence, "segment_id": data["segment_id"], "sample_offset": self.frames,
                           "sample_count": count, "audio_sha256": data["audio_sha256"], "pcm_sha256": data["pcm_sha256"],
                           "received_frames": 0, "digest": hashlib.sha256(), "synthesis_ms": _number(data.get("synthesis_ms"))}
        elif event == "audio":
            self._segment(data, key="segment_sequence")
            count = _integer(data.get("sample_count"), minimum=1, maximum=16384)
            if (data.get("encoding") != "pcm_s16le" or _integer(data.get("sequence")) != self.chunks or _integer(data.get("sample_offset")) != self.frames
                    or _integer(data.get("sample_rate")) != self.rate or _integer(data.get("channels")) != self.channels
                    or self.active["received_frames"] + count > self.active["sample_count"]):
                raise ProtocolError("Live PCM sequence, sample offset or format is invalid")
            try:
                encoded = data.get("pcm_base64")
                if not isinstance(encoded, str) or len(encoded) > 90_000:
                    raise ValueError("bounded encoding required")
                pcm = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ProtocolError("Invalid or oversized PCM base64") from exc
            if len(pcm) != count * self.channels * 2 or self.pcm_bytes + len(pcm) > 512 * 1024 * 1024:
                raise ProtocolError("Live PCM byte count differs from its sample extent")
            self.active["digest"].update(pcm)
            self.active["received_frames"] += count
            self.pcm_digest.update(pcm)
            self.pcm_bytes += len(pcm)
            self.frames += count
            self.chunks += 1
            if self.timings["first_pcm_received_ms"] is None:
                self.timings["first_pcm_received_ms"] = at_ms
        elif event == "audio_segment_end":
            self._segment(data)
            active = self.active
            if (data.get("state") != "delivered" or _integer(data.get("sample_end")) != self.frames
                    or active["received_frames"] != active["sample_count"]
                    or active["digest"].hexdigest() != active["pcm_sha256"]):
                raise ProtocolError("Audio segment is incomplete or its PCM digest changed")
            self.audio_segments.append({key: value for key, value in active.items() if key != "digest"})
            self.active = None
        elif event == "result":
            turn = data.get("turn")
            if (self.result is not None or not isinstance(turn, dict) or turn.get("id") != self.turn_id or turn.get("session_id") != self.session_id
                    or turn.get("text") != validate_citations(self.raw_text, set())[0]):
                raise ProtocolError("Live text result differs from its session, turn or received tokens")
            if turn.get("provider") != "openai":
                raise ProtocolError("Source-preview response did not execute language-model inference")
            if not isinstance(turn.get("metrics"), dict):
                raise ProtocolError("Live result lacks measured language metrics")
            for key in ("retrieval_ms", "generation_ms", "total_ms"):
                _number(turn["metrics"].get(key))
            self.result = turn
            self.timings["result_received_ms"] = at_ms
        elif event == "audio_end":
            self._turn(data)
            if (self.audio_end is not None or self.active is not None or not self.audio_segments or len(self.audio_segments) != len(self.segments)
                    or self.result is None or _integer(data.get("chunks")) != self.chunks or _integer(data.get("sample_count")) != self.frames
                    or _integer(data.get("sample_rate")) != self.rate or data.get("spoken_complete") is not False
                    or data.get("state") != "delivered" or data.get("acoustic_latency_ms") is not None):
                raise ProtocolError("Final audio totals/state differ from validated delivery")
            if self.segments[-1]["char_end"] < len(self.raw_text) and validate_citations(self.raw_text[self.segments[-1]["char_end"]:], set())[0]:
                raise ProtocolError("Final text contains an unspoken committed tail")
            _number(data.get("total_ms"))
            _number(data.get("first_pcm_ready_ms"))
            self.audio_end = data
            self.timings["audio_end_received_ms"] = at_ms
        elif event == "done":
            if self.audio_end is None or self.result is None or data:
                raise ProtocolError("Live done arrived before complete verified text/audio")
            self.done = True
            self.timings["done_received_ms"] = at_ms
        else:
            raise ProtocolError("Unknown event for the frozen live protocol")

    def summary(self) -> dict:
        first_pcm, result_ms = self.timings["first_pcm_received_ms"], self.timings["result_received_ms"]
        return {"session_id": self.session_id, "turn_id": self.turn_id, "protocol": PROTOCOL,
                "completed": self.done, "events": dict(self.events), "timings": dict(self.timings),
                "llm_result_confirmed": self.result is not None, "generated_text_sha256": _hash(self.raw_text.encode()),
                "pcm_sha256": self.pcm_digest.hexdigest(), "received_frames": self.frames, "received_chunks": self.chunks,
                "received_pcm_bytes": self.pcm_bytes, "sample_rate": self.rate, "channels": self.channels,
                "delivered_audio_duration_ms": self.frames / self.rate * 1000 if self.rate else None,
                "complete_verified_segments": len(self.audio_segments), "segments": self.audio_segments,
                "active_segment_incomplete": self.active is not None,
                "first_pcm_preceded_text_result": first_pcm < result_ms if first_pcm is not None and result_ms is not None else None,
                "server_metrics": {"language": {key: self.result["metrics"][key] for key in ("retrieval_ms", "generation_ms", "total_ms")} if self.result else None,
                                   "first_pcm_ready_ms": (self.audio_end or {}).get("first_pcm_ready_ms"),
                                   "total_ms": (self.audio_end or {}).get("total_ms")},
                "original_wav_hash_verification": "not_possible_without_original_wav_headers; actual PCM digests verified per complete segment",
                "browser_first_visible_text_ms": None, "browser_playback_first_pcm_ms": None,
                "speaker_first_audible_ms": None, "playback_completion_ms": None, "spoken_complete": None}


def _url(value: str) -> str:
    parts = urlsplit(value)
    try:
        local = parts.hostname == "localhost" or ipaddress.ip_address(parts.hostname or "").is_loopback
    except ValueError:
        local = False
    if (not parts.hostname or parts.username or parts.password or parts.query or parts.fragment
            or parts.scheme not in {"http", "https"} or (parts.scheme == "http" and not local)):
        raise ValueError("Benchmark endpoints require credential-free HTTPS or loopback HTTP URLs without query strings")
    return value.rstrip("/")


def _local(url: str) -> bool:
    host = urlsplit(url).hostname
    try:
        return host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _headers(env_name: str | None) -> dict:
    if env_name is None:
        return {}
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name):
        raise ValueError("Credential option must name an environment variable")
    value = os.getenv(env_name)
    if not value:
        raise ValueError("The explicitly named credential environment variable is empty")
    return {"Authorization": f"Bearer {value}"}


def _health_view(value: dict, kind: str) -> dict:
    # Never retain unknown response fields: proxy/service configurations may
    # contain secrets. Selected nested identities are reduced separately.
    common = {key: value.get(key) for key in ("status", "model_loaded", "busy")}
    if kind == "language":
        common.update({key: value.get(key) for key in ("model", "load_ms", "stage")})
        common["identity"] = {key: (value.get("identity") or {}).get(key) for key in
            ("schema_version", "base_model", "base_revision", "base_checkpoint_manifest_sha256", "tokenizer_contract_sha256")}
        common["identity"]["chat_template_options"] = {
            "enable_thinking": ((value.get("identity") or {}).get("chat_template_options") or {}).get("enable_thinking")}
        common["adapter"] = {key: (value.get("adapter") or {}).get(key) for key in
            ("configured", "loaded", "verified", "schema_version", "run_sha256", "artifact_manifest_sha256",
             "base_model", "base_revision", "base_checkpoint_manifest_sha256", "dataset_sha256", "tokenizer_contract_sha256", "optimizer_steps")}
        runtime = value.get("runtime") or {}
        common["runtime"] = {key: runtime.get(key) for key in
            ("backend", "requested_device", "requested_dtype", "device", "dtype", "context_window", "base_parameters", "threads", "attention")}
        common["runtime"]["versions"] = {key: (runtime.get("versions") or {}).get(key) for key in ("torch", "transformers", "peft")}
        common["runtime"]["generation_defaults"] = {key: (runtime.get("generation_defaults") or {}).get(key) for key in
            ("seed", "temperature", "top_p", "top_k")}
        common["limits"] = {key: (value.get("limits") or {}).get(key) for key in ("input_tokens", "output_tokens", "generation_seconds")}
    elif kind == "voice":
        common.update({key: value.get(key) for key in ("reference_configured", "reference_mode", "voice", "synthesis", "streaming")})
        common["runtime"] = {key: (value.get("runtime") or {}).get(key) for key in
            ("revision", "device", "dtype", "attention", "local_files_only", "threads", "seed", "max_new_tokens", "do_sample", "temperature", "top_k", "top_p", "repetition_penalty")}
        common["last_generation"] = {key: (value.get("last_generation") or {}).get(key) for key in
            ("language", "input_chars", "cold_start", "elapsed_seconds", "audio_seconds", "real_time_factor", "termination_reason")}
    else:
        common = {key: value.get(key) for key in ("status", "version", "provider", "live_protocol", "configuration_only")}
        common["providers"] = {name: {key: (value.get("providers", {}).get(name) or {}).get(key) for key in ("configured", "model")}
                               for name in ("llm", "speech")}
        common["retrieval"] = {key: (value.get("retrieval") or {}).get(key) for key in
            ("backend", "embedding_model", "embedding_revision", "style_examples", "prewarmed", "startup_warmup_ms")}
        common["persona_snapshot"] = {key: (value.get("persona_snapshot") or {}).get(key) for key in ("id", "as_of", "corpus_sha256")}
    return common


async def _health(url: str, kind: str, headers: dict) -> dict:
    async with httpx.AsyncClient(timeout=15, trust_env=not _local(url), follow_redirects=False, headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()
        if len(response.content) > 1_048_576:
            raise ValueError("Provider health response exceeds 1 MiB")
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError("Provider health must be a JSON object")
        return _health_view(value, kind)


def _configured(app: dict, language: dict, voice: dict) -> None:
    providers = app.get("providers", {})
    if (app.get("live_protocol") != PROTOCOL or app.get("provider") != "openai"
            or any(providers.get(name, {}).get("configured") is not True or not providers[name].get("model") for name in ("llm", "speech"))):
        raise ValueError("Real live benchmark requires configured language and speech providers")
    if language.get("model") != providers["llm"]["model"]:
        raise ValueError("Observed language service does not match the application's model alias")
    if voice.get("reference_configured") is not True or voice.get("synthesis") != "reference_conditioned" or providers["speech"]["model"] != "qwen3-tts":
        raise ValueError("This benchmark requires the configured local reference-voice service")


def _health_continuity(before: dict, after: dict) -> list[str]:
    """Reject changed observed service identities, not legitimate lazy loading.

    These are service-reported identities, not independent process attestation.
    A cold service may discover previously unknown artifact/runtime fields and
    finish loading its configured adapter. Already observed values stay fixed.
    """
    changed = []

    def compare(old, new, path: str, *, cold: bool) -> None:
        if isinstance(old, dict):
            for key, value in old.items():
                compare(value, new.get(key) if isinstance(new, dict) else None, path + "." + key, cold=cold)
        elif old != new and not (cold and old is None) and not (
            cold and path in {"language.adapter.loaded", "language.adapter.verified"} and old is False and new is True
        ):
            changed.append(path)

    fields = {
        "language": ("model", "identity", "adapter", "runtime", "limits"),
        "voice": ("reference_configured", "reference_mode", "voice", "synthesis", "streaming", "runtime"),
        "api": ("provider", "live_protocol", "providers", "persona_snapshot"),
    }
    for kind, keys in fields.items():
        old, new = before.get(kind, {}), after.get(kind, {})
        cold = kind != "api" and old.get("model_loaded") is False
        for key in keys:
            compare(old.get(key), new.get(key), kind + "." + key, cold=cold)
        if kind != "api" and old.get("model_loaded") is True and new.get("model_loaded") is not True:
            changed.append(kind + ".model_loaded")
    # Retrieval warmup timing can change on the first request, but the evidence
    # and encoder selection used by the same run must remain stable.
    for key in ("backend", "embedding_model", "embedding_revision", "style_examples"):
        compare(before.get("api", {}).get("retrieval", {}).get(key), after.get("api", {}).get("retrieval", {}).get(key),
                "api.retrieval." + key, cold=False)
    return changed


async def _release_session(base: str, session: str | None, headers: dict, timeout: float) -> dict:
    if session is None:
        return {"observed": False, "elapsed_ms": None, "reason": "no_session_identity"}
    started = time.perf_counter()
    try:
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(timeout=min(timeout, 5), trust_env=not _local(base), headers=headers) as client:
                while True:
                    response = await client.delete(base + "/api/sessions/" + session)
                    if response.status_code == 200 and response.json().get("deleted") is True:
                        return {"observed": True, "elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "basis": "benchmark-owned_session_deletion_accepted"}
                    if response.status_code != 409:
                        break
                    await asyncio.sleep(0.1)
    except (httpx.HTTPError, TimeoutError, ValueError):
        pass
    return {"observed": False, "elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "reason": "session_cleanup_not_confirmed"}


async def run_benchmark(*, base_url: str, language_health_url: str, voice_health_url: str, message: str, language: str = "en",
                        mode: str = "behavior", timeout: float = 600, cleanup_timeout: float = 10,
                        cancel_after_first_pcm: bool = False, api_key_env: str | None = None,
                        language_key_env: str | None = None, voice_key_env: str | None = None) -> dict:
    base_url, language_health_url, voice_health_url = map(_url, (base_url, language_health_url, voice_health_url))
    if not 1 <= timeout <= 1800 or not 0 < cleanup_timeout <= 60:
        raise ValueError("Benchmark deadlines must be bounded")
    request = ChatRequest(message=message, language=language, mode=mode).model_dump(mode="json")
    headers, language_headers, voice_headers = _headers(api_key_env), _headers(language_key_env), _headers(voice_key_env)
    clock = time.get_clock_info("perf_counter")
    report = {"schema_version": SCHEMA, "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
              "transport": "real_http_sse", "request": request, "cancel_after_first_pcm": cancel_after_first_pcm,
              "endpoints": {"api": base_url, "language_health": language_health_url, "voice_health": voice_health_url},
              "clock": {"source": "time.perf_counter", "monotonic": clock.monotonic, "adjustable": clock.adjustable, "resolution_seconds": clock.resolution},
              "benchmark_host": {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(), "logical_cpus": os.cpu_count(),
                                 "scope": "HTTP client host; does not establish remote model hardware"},
              "implementation_sha256": {"tools/live_benchmark.py": _hash(Path(__file__).read_bytes())},
              "timeout_seconds": timeout, "health_before": {}, "health_after": {}, "events": [],
              "limitations": ["HTTP receipt times are digital transport measurements, not first visible text, playback or acoustic output.",
                              "Application-level committed-sentence WAV synthesis is not native neural audio-token streaming.",
                              "Configured provider aliases and observed health are not independent proof of all remote checkpoint bytes.",
                              "A cancelled prefix cannot establish complete segment integrity, text generation or spoken completion.",
                              "Session release does not prove that an external synchronous voice worker has stopped; inspect post-run busy state."]}
    state, started, cancelled = LiveProtocol(), None, False
    try:
        for key, url, auth in (("api", base_url + "/api/health", headers), ("language", language_health_url, language_headers), ("voice", voice_health_url, voice_headers)):
            report["health_before"][key] = await _health(url, key, auth)
        _configured(report["health_before"]["api"], report["health_before"]["language"], report["health_before"]["voice"])
        report["warm_state"] = {key: report["health_before"][key].get("model_loaded") for key in ("language", "voice")}
        started = time.perf_counter()
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=min(15, timeout)), trust_env=not _local(base_url), follow_redirects=False, headers=headers) as client:
                async with client.stream("POST", base_url + "/api/chat/live", json=request) as response:
                    report["http_status"] = response.status_code
                    response.raise_for_status()
                    report["headers_received_ms"] = round((time.perf_counter() - started) * 1000, 3)
                    if response.headers.get("content-type", "").split(";", 1)[0].lower() != "text/event-stream":
                        raise ProtocolError("Live response is not an SSE stream")
                    async for event, data in sse_frames(response.aiter_bytes()):
                        at_ms = round((time.perf_counter() - started) * 1000, 3)
                        state.accept(event, data, at_ms)
                        report["events"].append({"event": event, "received_at_ms": at_ms, "payload_sha256": _digest(data)})
                        if cancel_after_first_pcm and event == "audio":
                            cancelled = True
                            report["cancellation_requested_ms"] = at_ms
                            break
        if not cancelled and not state.done:
            raise ProtocolError("HTTP body ended before a verified done event")
        report["status"] = "cancelled" if cancelled else "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = {"type": type(exc).__name__, "reason": str(exc) if isinstance(exc, ProtocolError) else "Benchmark configuration, provider health or HTTP execution failed; no completed inference is claimed."}
    finally:
        report["request_closed_ms"] = round((time.perf_counter() - started) * 1000, 3) if started is not None else None
        report["stream"] = state.summary()
        report["session_release"] = await _release_session(base_url, state.session_id, headers, cleanup_timeout)
        for key, url, auth in (("api", base_url + "/api/health", headers), ("language", language_health_url, language_headers), ("voice", voice_health_url, voice_headers)):
            try:
                report["health_after"][key] = await _health(url, key, auth)
            except Exception as exc:
                report["health_after"][key] = {"status": "unavailable", "failure_type": type(exc).__name__}
        loaded = all(report["health_after"].get(key, {}).get("model_loaded") is True for key in ("language", "voice"))
        report["model_services_loaded_after"] = loaded
        changed = _health_continuity(report["health_before"], report["health_after"])
        report["provider_identity_continuity"] = {"verified": bool(report["health_before"]) and not changed,
            "changed_fields": changed, "scope": "service_reported_identity_and_runtime"}
        report["measured_inference_complete"] = report["status"] == "completed" and state.result is not None and loaded and not changed
        if report["status"] == "completed" and not loaded:
            report["status"] = "failed"
            report["failure"] = {"type": "ProviderEvidenceMissing", "reason": "A complete protocol stream lacked post-run loaded model evidence."}
        elif report["status"] == "completed" and changed:
            report["status"] = "failed"
            report["failure"] = {"type": "ProviderIdentityChanged", "reason": "Provider identity or runtime changed during the measured request."}
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-url", "language-health-url", "voice-health-url", "message"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--language", choices=("en", "ko"), default="en")
    parser.add_argument("--mode", choices=("profile", "rag", "behavior"), default="behavior")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--cleanup-timeout", type=float, default=10)
    parser.add_argument("--cancel-after-first-pcm", action="store_true")
    parser.add_argument("--api-key-env")
    parser.add_argument("--language-key-env")
    parser.add_argument("--voice-key-env")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output
    if output.exists() or any(path.is_symlink() for path in (output, *output.parents)):
        parser.error("Choose a new report path without symlink ancestors")
    values = vars(args).copy()
    del values["output"]
    try:
        report = asyncio.run(run_benchmark(**values))
    except (ValueError, OSError) as exc:
        report = {"schema_version": SCHEMA, "status": "failed", "measured_inference_complete": False,
                  "failure": {"type": type(exc).__name__, "reason": "Invalid benchmark setup; no inference was measured."}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "measured_inference_complete": report["measured_inference_complete"], "report": str(output)}))
    return 0 if report["status"] in {"completed", "cancelled"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
