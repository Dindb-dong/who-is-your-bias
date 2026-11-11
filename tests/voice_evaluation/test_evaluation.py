"""Protocol/metric tests use synthetic signals; they do not measure voice quality."""

from __future__ import annotations

import asyncio
import io
import json
import math
import struct
import subprocess
import sys
import types
import wave
from pathlib import Path

import httpx
import pytest

from tools.voice_evaluation.metrics import EvaluationError, edit_counts, lexical_metrics, wav_metrics
from tools.voice_evaluation.asr import transcribe_run
from tools.voice_evaluation.review import create_bundle, validate_review
from tools.voice_evaluation.run import checkpoint, local_base_url, read_json, run_synthesis


@pytest.fixture
def signal() -> bytes:
    data = io.BytesIO()
    with wave.open(data, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b''.join(struct.pack('<h', int(5000 * math.sin(index * 0.1))) for index in range(1600)))
    return data.getvalue()


@pytest.fixture
def declared(tmp_path: Path) -> Path:
    path = tmp_path / "cases.json"
    checkpoint(path, {"schema_version": "voice_cases.v1", "cases": [
        {"id": "en", "text": "A short test.", "language": "English", "asr_language": "en"},
        {"id": "ko", "text": "짧은 테스트입니다.", "language": "Korean", "asr_language": "ko"}]})
    return path


def test_levenshtein_counts_include_insertions_and_omissions():
    result = edit_counts(["a", "b", "c"], ["a", "wrong", "c", "extra"])
    assert result == {"reference_units": 3, "hypothesis_units": 4, "substitutions": 1,
        "deletions": 0, "insertions": 1, "distance": 2, "error_rate": 0.666667}
    assert edit_counts(["a", "b", "c"], ["a"])["deletions"] == 2
    assert edit_counts(["a"], ["a", "b", "c"])["error_rate"] == 2.0
    with pytest.raises(EvaluationError):
        edit_counts([], ["extra"])
    with pytest.raises(EvaluationError, match="budget"):
        edit_counts(["a"] * 3000, ["a"] * 3000)


def test_korean_spacing_and_unicode_normalization_are_not_confused_with_char_errors():
    result = lexical_metrics("가정을 적고 비교합니다.", "가정을적고 비교합니다", "ko")
    assert result["primary_metric"] == "cer"
    assert result["cer"]["distance"] == 0
    assert result["wer"]["distance"] > 0
    assert result["screens"]["missing_expected_suffix"] is False


def test_lexical_screens_detect_obvious_repetition_and_missing_suffix_without_quality_claims():
    result = lexical_metrics("We check the design and finish the experiment.", "We check the design we check the design we check the design", "en")
    assert result["screens"]["excess_repeated_ngrams"] > 0
    assert result["screens"]["missing_expected_suffix"] is True
    assert "do not prove" in result["limitations"]
    assert lexical_metrics("Trial, trial, trial.", "Trial trial trial.", "en")["screens"]["excess_repeated_ngrams"] == 0


def test_wav_stats_validate_real_frame_bytes_and_never_establish_quality(signal: bytes):
    result = wav_metrics(signal)
    assert result["duration_seconds"] == 0.1
    assert result["signal_present"] is True
    assert result["clipped_sample_fraction"] == 0
    with pytest.raises(EvaluationError, match="truncated"):
        wav_metrics(signal[:-10])
    with pytest.raises(EvaluationError):
        wav_metrics(b"bad wave" * 20)


@pytest.mark.parametrize("url", ["https://example.com", "http://127.0.0.1:8092/private", "http://user:pass@localhost", "http://localhost/?token=secret", "http://localhost:bad", "file:///tmp", "http://[invalid"])
def test_runner_only_accepts_credential_free_loopback_origins(url):
    with pytest.raises(EvaluationError):
        local_base_url(url)


def make_run(tmp_path, declared, signal):
    calls = []
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"streaming": False, "runtime": {"seed": 42}})
        calls.append(json.loads(request.content))
        return httpx.Response(200, content=signal, headers={"content-type": "audio/wav"})
    report = asyncio.run(run_synthesis(base_url="http://127.0.0.1:8092", output=tmp_path / "run",
        cases_path=declared, condition="protocol-test", api_key="private-test-secret", transport=httpx.MockTransport(handler)))
    assert [call["language"] for call in calls] == ["English", "Korean"]
    assert report["status"] == "completed"
    assert "private-test-secret" not in json.dumps(report)
    assert all(row["termination_reason"] == "not_exposed_by_backend" for row in report["cases"])
    assert all(row["audio"]["sha256"] for row in report["cases"])
    return tmp_path / "run" / "synthesis.json"


def test_sequential_runner_records_hashes_and_refuses_overwrite(tmp_path, declared, signal):
    file = make_run(tmp_path, declared, signal)
    report = read_json(file)
    assert report["human_listening"] == "not_performed"
    with pytest.raises(EvaluationError, match="not overwritten"):
        asyncio.run(run_synthesis(base_url="http://localhost", output=file.parent, cases_path=declared, condition="protocol-test"))


def test_timeout_preserves_checkpoint_and_does_not_admit_later_work(tmp_path, declared):
    requests = []
    async def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"streaming": False})
        requests.append(request)
        await asyncio.sleep(0.2)
        return httpx.Response(500)
    report = asyncio.run(run_synthesis(base_url="http://localhost", output=tmp_path / "run", cases_path=declared,
        condition="timeout-test", per_case_timeout=0.02, total_timeout=1, transport=httpx.MockTransport(handler)))
    assert len(requests) == 1
    assert [row["status"] for row in report["cases"]] == ["timed_out", "not_run_after_failure"]
    assert read_json(tmp_path / "run" / "synthesis.json")["status"] == "incomplete"


def test_invalid_audio_is_not_saved_as_a_success(tmp_path, declared):
    def handler(request):
        return httpx.Response(200, json={}) if request.method == "GET" else httpx.Response(200, content=b"broken", headers={"content-type": "audio/wav"})
    report = asyncio.run(run_synthesis(base_url="http://localhost", output=tmp_path / "run", cases_path=declared,
        condition="invalid-wave", transport=httpx.MockTransport(handler)))
    assert report["cases"][0]["status"] == "failed"
    assert not list((tmp_path / "run").glob("*.wav"))


def test_listening_bundle_starts_unrated_hides_conditions_and_checks_tampering(tmp_path, declared, signal):
    run = make_run(tmp_path, declared, signal)
    output = tmp_path / "listening"
    result = create_bundle([run], output=output, seed=9)
    assert result["status"] == "unrated"
    manifest = output / "listener" / "manifest.json"
    template = output / "listener" / "review-template.json"
    assert "protocol-test" not in (output / "listener" / "index.html").read_text()
    assert "protocol-test" not in manifest.read_text()
    assert "protocol-test" in (output / "private" / "answer-key.json").read_text()
    key = read_json(output / "private" / "answer-key.json")
    assert {sample["expected_text"] for sample in key["samples"]} == {"A short test.", "짧은 테스트입니다."}
    assert all(sample["expected_text"] not in (output / "listener" / "index.html").read_text() for sample in key["samples"])
    with pytest.raises(EvaluationError, match="personally|personal listening"):
        validate_review(manifest, template)
    review = read_json(template)
    review.update(human_review_performed=True, reviewer="Synthetic test fixture", created_at="2026-11-11T00:00:00Z")
    review["samples"][0].update(listened=True, understood_language=True, intelligibility=4, naturalness=3)
    submitted = tmp_path / "submitted.json"
    checkpoint(submitted, review)
    summary = validate_review(manifest, submitted)
    assert summary["listened_samples"] == 1
    assert summary["ratings"]["reference_similarity"] == {"count": 0, "mean": None}
    review["samples"][0]["reference_similarity"] = 5
    checkpoint(submitted, review)
    with pytest.raises(EvaluationError, match="comparison recording"):
        validate_review(manifest, submitted)
    review["samples"][0]["reference_similarity"] = None
    checkpoint(submitted, review)
    (output / "listener" / "sample001.wav").write_bytes(b"replaced")
    with pytest.raises(EvaluationError, match="hash changed"):
        validate_review(manifest, submitted)


def test_bundle_rejects_audio_changed_after_synthesis(tmp_path, declared, signal):
    run = make_run(tmp_path, declared, signal)
    (run.parent / "en.wav").write_bytes(signal + b"extra")
    with pytest.raises(EvaluationError, match="mismatched hash"):
        create_bundle([run], output=tmp_path / "listening")


def test_asr_receives_only_audio_and_language_without_target_text_hints(tmp_path, declared, signal, monkeypatch):
    run = make_run(tmp_path, declared, signal)
    model_directory = tmp_path / "recognizer"
    model_directory.mkdir()
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (model_directory / name).write_bytes(b"protocol fake, not model weights")
    calls = []
    class FakeRecognizer:
        def __init__(self, location, **kwargs):
            assert kwargs["local_files_only"] is True
            assert kwargs["compute_type"] == "int8"

        def transcribe(self, samples, **kwargs):
            assert samples.ndim == 1
            calls.append(kwargs)
            text = "A short test." if kwargs["language"] == "en" else "짧은 테스트입니다."
            segment = types.SimpleNamespace(start=0, end=0.1, text=text, words=[])
            return iter([segment]), types.SimpleNamespace(language=kwargs["language"])
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeRecognizer))
    monkeypatch.setattr("tools.voice_evaluation.asr.importlib.metadata.version", lambda _: "protocol-test")
    monkeypatch.setattr("tools.voice_evaluation.asr.subprocess.run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, stdout=b"\x01\x00" * 1600, stderr=b""))
    result = transcribe_run(run, model_directory=model_directory, model_repo="test/fake", revision="a" * 40,
        output=tmp_path / "asr.json")
    assert result["status"] == "completed"
    assert [call["language"] for call in calls] == ["en", "ko"]
    assert all(call["initial_prompt"] is call["prefix"] is call["hotwords"] is None for call in calls)
    assert all(call["condition_on_previous_text"] is False for call in calls)
    assert all("text" not in call for call in calls)
    assert result["cases"][1]["metrics"]["primary_metric"] == "cer"
    assert result["human_listening"] == "not_performed"


def test_listening_similarity_is_bound_to_the_exact_comparison_file(tmp_path, declared, signal):
    run = make_run(tmp_path, declared, signal)
    reference = tmp_path / "reference.wav"
    reference.write_bytes(signal)
    bundle = tmp_path / "listening"
    create_bundle([run], output=bundle, reference=reference)
    review = read_json(bundle / "listener" / "review-template.json")
    review.update(human_review_performed=True, reference_listened=True, reviewer="Synthetic fixture", created_at="2026-11-11T00:00:00Z")
    review["samples"][0].update(listened=True, reference_similarity=3)
    submitted = tmp_path / "submitted.json"
    checkpoint(submitted, review)
    assert validate_review(bundle / "listener" / "manifest.json", submitted)["ratings"]["reference_similarity"]["mean"] == 3
    (bundle / "listener" / "comparison.wav").write_bytes(signal + b"changed")
    with pytest.raises(EvaluationError, match="Comparison recording"):
        validate_review(bundle / "listener" / "manifest.json", submitted)
