"""Synthetic protocol fixtures only; no model-latency or audio-quality claim."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import wave
from copy import deepcopy

import pytest

from tools.live_benchmark import LiveProtocol, ProtocolError, _configured, _health_continuity, _health_view, _url, run_benchmark, sse_frames


def wav(pcm):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(pcm)
    return output.getvalue()


def fixture():
    prefix, suffix = "First synthetic sentence. ", "Second synthetic sentence."
    identity = {"turn_id": "fixture-turn"}
    segments = [{**identity, "segment_id": f"fixture-turn:{i}", "sequence": i, "text": text.strip(), "raw_text": text,
                 "char_start": offset, "char_end": offset + len(text), "state": "committed"}
                for i, text, offset in ((0, prefix, 0), (1, suffix, len(prefix)))]
    pcm = [b"\x01\x00" * 4, b"\x02\x00" * 4]
    headers = [{**identity, "segment_id": f"fixture-turn:{i}", "sequence": i, "audio_sha256": hashlib.sha256(wav(data)).hexdigest(),
                "pcm_sha256": hashlib.sha256(data).hexdigest(), "sample_rate": 8000, "channels": 1,
                "sample_offset": 4 * i, "sample_count": 4, "synthesis_ms": 10.0}
               for i, data in enumerate(pcm)]
    def chunk(segment, sequence, offset, data):
        return {**identity, "segment_id": f"fixture-turn:{segment}", "segment_sequence": segment, "sequence": sequence,
                "sample_rate": 8000, "channels": 1, "sample_offset": offset, "sample_count": len(data) // 2,
                "encoding": "pcm_s16le", "pcm_base64": base64.b64encode(data).decode()}
    return [
        ("meta", {"turn_id": "fixture-turn", "session_id": "fixture-session", "protocol": "live-segments-pcm.v1"}),
        ("token", {"text": prefix}), ("segment", segments[0]), ("audio_segment", headers[0]),
        ("audio", chunk(0, 0, 0, pcm[0][:4])),
        ("token", {"text": suffix}), ("segment", segments[1]),
        ("audio", chunk(0, 1, 2, pcm[0][4:])),
        ("audio_segment_end", {**identity, "segment_id": "fixture-turn:0", "sequence": 0, "sample_end": 4, "state": "delivered"}),
        ("result", {"turn": {"id": "fixture-turn", "session_id": "fixture-session", "text": prefix + suffix, "provider": "openai",
                              "metrics": {"retrieval_ms": 1, "generation_ms": 20, "total_ms": 21}}}),
        ("audio_segment", headers[1]), ("audio", chunk(1, 2, 4, pcm[1])),
        ("audio_segment_end", {**identity, "segment_id": "fixture-turn:1", "sequence": 1, "sample_end": 8, "state": "delivered"}),
        ("audio_end", {**identity, "chunks": 3, "sample_count": 8, "sample_rate": 8000, "first_pcm_ready_ms": 10,
                       "total_ms": 30, "state": "delivered", "spoken_complete": False, "acoustic_latency_ms": None}),
        ("done", {}),
    ]


def consume(events):
    state = LiveProtocol()
    for i, (event, data) in enumerate(events):
        state.accept(event, data, i * 10)
    return state


def test_real_format_protocol_fixture_validates_pcm_before_completion():
    state = consume(fixture())
    report = state.summary()
    assert state.done and report["completed"]
    assert report["complete_verified_segments"] == 2
    assert report["received_frames"] == 8 and report["received_chunks"] == 3
    assert report["delivered_audio_duration_ms"] == 1.0
    assert report["first_pcm_preceded_text_result"] is True
    assert report["browser_playback_first_pcm_ms"] is None
    assert report["speaker_first_audible_ms"] is None and report["spoken_complete"] is None
    assert "First synthetic" not in json.dumps(report)


@pytest.mark.parametrize("index,key,value", [
    (0, "protocol", "unknown"),
    (2, "char_end", 500),
    (2, "raw_text", "Changed after streaming."),
    (2, "sequence", 1),
    (3, "sample_offset", 1),
    (3, "pcm_sha256", None),
    (4, "turn_id", "stale-turn"),
    (4, "segment_id", "fixture-turn:1"),
    (4, "sequence", 1),
    (4, "sequence", False),
    (4, "sample_offset", 1),
    (4, "sample_count", 3),
    (4, "sample_rate", 44100),
    (4, "channels", 2),
    (4, "pcm_base64", "%%%"),
    (4, "pcm_base64", base64.b64encode(b"\0" * 4).decode()),
    (8, "sample_end", 5),
    (10, "sample_rate", 16000),
    (13, "chunks", 8),
    (13, "sample_count", 9),
    (13, "spoken_complete", True),
    (13, "acoustic_latency_ms", 200),
])
def test_corrupt_or_stale_live_protocol_never_becomes_success(index, key, value):
    events = fixture()
    events[index][1][key] = value
    with pytest.raises(ProtocolError):
        consume(events)


@pytest.mark.parametrize("field,value", [("session_id", "wrong-session"), ("id", "wrong-turn"), ("text", "Replaced reply."), ("provider", "evidence")])
def test_result_identity_content_and_actual_generation_are_required(field, value):
    events = fixture()
    events[9][1]["turn"][field] = value
    with pytest.raises(ProtocolError):
        consume(events)


def test_done_requires_result_audio_end_and_all_committed_audio():
    events = fixture()
    for cut in (2, 5, 9, 12, 13):
        with pytest.raises(ProtocolError):
            consume(events[:cut] + [("done", {})])


def test_valid_cancelled_prefix_is_neither_complete_nor_fully_hash_verified():
    state = consume(fixture()[:5])
    report = state.summary()
    assert not state.done
    assert report["first_pcm_preceded_text_result"] is None
    assert report["llm_result_confirmed"] is False
    assert report["active_segment_incomplete"] is True
    assert report["complete_verified_segments"] == 0
    assert report["timings"]["done_received_ms"] is None


def test_error_and_post_done_events_fail_closed():
    for events in (fixture()[:4] + [("error", {"code": "provider_error"})], fixture() + [("token", {"text": "late"})]):
        with pytest.raises(ProtocolError):
            consume(events)


def test_late_result_token_or_repeated_audio_segment_is_invalid():
    events = fixture()
    for extra in (("token", {"text": "late"}), deepcopy(events[3])):
        with pytest.raises(ProtocolError):
            consume(events[:10] + [extra] + events[10:])


def test_actual_live_generator_emits_digest_of_decoded_pcm(corpus):
    from bias.config import ProviderSettings, Settings
    from bias.engine import Engine
    from bias.live import live_events
    from bias.models import ChatRequest
    class LLM:
        async def stream(self, messages):
            yield "Reusable rockets reduce launch costs. [source:rocket-reuse]"
    pcm = b"\x11\x00" * 1000
    class Voice:
        settings = ProviderSettings("http://fixture.invalid/v1", "fixture")
        async def synthesize(self, text, **kwargs):
            return wav(pcm), "audio/wav"
    async def run():
        state = LiveProtocol()
        engine = Engine(Settings(llm=ProviderSettings("http://fixture.invalid/v1", "fixture")), corpus, llm=LLM())
        i = 0
        async for event, data in live_events(engine, Voice(), ChatRequest(message="Why reuse rockets?")):
            state.accept(event, data, i)
            i += 1
        assert state.done
        assert state.audio_segments[0]["pcm_sha256"] == hashlib.sha256(pcm).hexdigest()
        assert state.audio_segments[0]["audio_sha256"] == hashlib.sha256(wav(pcm)).hexdigest()
    asyncio.run(run())


async def byte_chunks(value):
    for byte in value:
        yield bytes([byte])


def test_sse_parser_handles_split_utf8_comments_crlf_and_multiline_json():
    payload = ': comment\r\nevent: token\r\ndata: {\r\ndata: "text":"안녕"}\r\n\r\n'.encode()
    async def run():
        return [event async for event in sse_frames(byte_chunks(payload))]
    assert asyncio.run(run()) == [("token", {"text": "안녕"})]


@pytest.mark.parametrize("payload", [b'event: done\ndata: {}\n', b'event: done\ndata: [1]\n\n', b'event: done\ndata: nope\n\n', b'event: token\ndata: {"text":"\xff"}\n\n'])
def test_sse_parser_rejects_truncation_and_invalid_payloads(payload):
    async def run():
        return [event async for event in sse_frames(byte_chunks(payload))]
    with pytest.raises(ProtocolError):
        asyncio.run(run())


def test_unbounded_sse_line_is_rejected_before_json_decoding():
    async def chunks():
        yield b"data: " + b"x" * 1_048_576
    async def run():
        return [event async for event in sse_frames(chunks())]
    with pytest.raises(ProtocolError, match="1 MiB"):
        asyncio.run(run())


def health_fixture():
    app = {"provider": "openai", "live_protocol": "live-segments-pcm.v1", "providers": {
        "llm": {"configured": True, "model": "local-model"}, "speech": {"configured": True, "model": "qwen3-tts"}}}
    return app, {"model": "local-model", "model_loaded": False}, {"reference_configured": True, "synthesis": "reference_conditioned"}


def test_configured_services_required_and_base_only_preview_rejected():
    app, llm, voice = health_fixture()
    _configured(app, llm, voice)
    for name in ("llm", "speech"):
        modified = deepcopy(app)
        modified["providers"][name]["configured"] = False
        with pytest.raises(ValueError):
            _configured(modified, llm, voice)
    with pytest.raises(ValueError, match="model alias"):
        _configured(app, {"model": "other"}, voice)


def test_health_export_uses_allowlist_instead_of_serializing_credentials():
    value = {"model": "local-model", "status": "ready", "api_key": "private-key", "endpoint": "https://private:secret@example.org",
             "identity": {"base_model": "Qwen/Model", "base_revision": "a" * 40, "authorization": "secret-token"},
             "adapter": {"configured": True, "loaded": True, "verified": True, "local_path": "/private/secret"},
             "runtime": {"dtype": "bfloat16", "device": "cpu", "backend": "transformers", "threads": 2,
                         "context_window": 40960, "base_parameters": 600000000, "attention": "sdpa", "token": "secret",
                         "versions": {"torch": "2.11.0+cpu", "unknown": "private-value"},
                         "generation_defaults": {"seed": 42, "temperature": 0.7, "private_field": "private"}}}
    safe = json.dumps(_health_view(value, "language"))
    assert "private" not in safe and "secret" not in safe
    assert "Qwen/Model" in safe
    runtime = json.loads(safe)["runtime"]
    assert runtime["dtype"] == "bfloat16" and runtime["device"] == "cpu"
    assert runtime["versions"]["torch"] == "2.11.0+cpu"
    assert runtime["generation_defaults"]["seed"] == 42 and runtime["base_parameters"] == 600000000


def observed_health():
    app, language, voice = health_fixture()
    language.update(model_loaded=True, identity={"base_model": "Qwen/Model", "base_revision": "a" * 40,
        "base_checkpoint_manifest_sha256": "b" * 64, "tokenizer_contract_sha256": "c" * 64,
        "chat_template_options": {"enable_thinking": False}},
        adapter={"configured": True, "loaded": True, "verified": True, "run_sha256": "d" * 64,
                 "artifact_manifest_sha256": "e" * 64},
        runtime={"backend": "transformers", "requested_dtype": "bfloat16", "device": "cpu", "dtype": "bfloat16",
                 "threads": 2, "context_window": 40960, "base_parameters": 600000000,
                 "versions": {"torch": "2.11.0+cpu", "transformers": "4.57.3"},
                 "generation_defaults": {"seed": 42}})
    voice.update(model_loaded=True, runtime={"revision": "f" * 40, "device": "cpu", "dtype": "bfloat16", "threads": 2})
    return {kind: _health_view(value, kind) for kind, value in (("api", app), ("language", language), ("voice", voice))}


@pytest.mark.parametrize("path", [
    "language.model", "language.identity.base_model", "language.identity.base_revision",
    "language.identity.base_checkpoint_manifest_sha256", "language.identity.tokenizer_contract_sha256",
    "language.identity.chat_template_options.enable_thinking", "language.adapter.run_sha256",
    "language.adapter.artifact_manifest_sha256", "language.adapter.loaded", "language.adapter.verified",
    "language.runtime.dtype", "language.runtime.device", "language.runtime.threads",
    "language.runtime.base_parameters", "language.runtime.context_window", "language.runtime.versions.torch",
    "language.runtime.generation_defaults.seed", "voice.runtime.revision", "voice.runtime.dtype",
    "api.providers.llm.model", "api.providers.speech.model",
])
def test_observed_model_or_runtime_replacement_cannot_pass_as_same_run(path):
    before = observed_health()
    after = deepcopy(before)
    target = after
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = "changed"
    assert path in _health_continuity(before, after)


def test_cold_identity_discovery_and_adapter_load_preserve_known_configuration():
    after = observed_health()
    before = deepcopy(after)
    before["language"]["model_loaded"] = False
    for key in ("base_checkpoint_manifest_sha256", "tokenizer_contract_sha256"):
        before["language"]["identity"][key] = None
    for key in ("device", "dtype", "context_window", "base_parameters"):
        before["language"]["runtime"][key] = None
    before["language"]["runtime"]["versions"] = {key: None for key in after["language"]["runtime"]["versions"]}
    before["language"]["adapter"].update(loaded=False, verified=False, run_sha256=None, artifact_manifest_sha256=None)
    assert _health_continuity(before, after) == []
    before["language"]["model_loaded"] = True
    assert "language.runtime.dtype" in _health_continuity(before, after)
    before["language"]["model_loaded"] = False
    after["language"]["identity"]["base_revision"] = "different-revision"
    assert "language.identity.base_revision" in _health_continuity(before, after)


def test_unchanged_warm_runtime_ignores_dynamic_work_and_latency_state():
    before = observed_health()
    after = deepcopy(before)
    after["language"].update(load_ms=1234, stage=None, busy=False)
    after["voice"]["last_generation"] = {"elapsed_seconds": 12, "audio_seconds": 3}
    after["api"]["retrieval"]["startup_warmup_ms"] = 100
    assert _health_continuity(before, after) == []


def test_complete_stream_is_failed_when_postrun_loaded_service_identity_changes(monkeypatch):
    import httpx
    health = observed_health()
    wire = b"".join(("event: " + event + "\ndata: " + json.dumps(data) + "\n\n").encode() for event, data in fixture())
    visits = {}
    async def handler(request):
        path = request.url.path
        if path == "/api/chat/live":
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=wire)
        if request.method == "DELETE":
            return httpx.Response(204)
        kind = "api" if path == "/api/health" else path.strip("/")
        visits[kind] = visits.get(kind, 0) + 1
        value = deepcopy(health[kind])
        if kind == "language" and visits[kind] > 1:
            value["identity"]["tokenizer_contract_sha256"] = "0" * 64
        return httpx.Response(200, json=value)
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    report = asyncio.run(run_benchmark(base_url="http://127.0.0.1:8000", language_health_url="http://127.0.0.1:8000/language",
        voice_health_url="http://127.0.0.1:8000/voice", message="Synthetic orchestration fixture only.", cleanup_timeout=0.1))
    assert report["status"] == "failed" and not report["measured_inference_complete"]
    assert report["model_services_loaded_after"] is True and report["stream"]["completed"] is True
    assert report["failure"]["type"] == "ProviderIdentityChanged"


@pytest.mark.parametrize("alteration", ["missing_weights", "extra_file", "changed_weights"])
def test_local_launcher_requires_complete_embedded_tokenizer_identity(tmp_path, alteration):
    from tools.live_benchmark_local import verify_embedded_tokenizer
    directory = tmp_path / "speech_tokenizer"
    directory.mkdir()
    files = {"config.json": b"{}", "model.safetensors": b"synthetic identity fixture only"}
    identity = {"files": [{"path": name, "sha256": hashlib.sha256(data).hexdigest()} for name, data in files.items()]}
    for name, data in files.items():
        (directory / name).write_bytes(data)
    verify_embedded_tokenizer(tmp_path, identity)
    if alteration == "missing_weights":
        (directory / "model.safetensors").unlink()
    elif alteration == "changed_weights":
        (directory / "model.safetensors").write_bytes(b"changed")
    else:
        (directory / "extra.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="Embedded speech tokenizer"):
        verify_embedded_tokenizer(tmp_path, identity)


@pytest.mark.parametrize("url", ["http://example.org/health", "https://user:password@example.org", "https://example.org?token=secret", "https://example.org#secret"])
def test_report_urls_never_embed_credentials(url):
    with pytest.raises(ValueError):
        _url(url)
    assert _url("http://127.0.0.1:8000/") == "http://127.0.0.1:8000"
