"""Synthetic text and tone fixtures; no real voice assets or model downloads."""

from __future__ import annotations

import array
import builtins
import json
import math
import os
import shutil
import wave
from pathlib import Path

import pytest

from tools.media.common import MediaError
from tools.media.prepare import prepare_clip
from tools.references.__main__ import main
from tools.references.asr import transcribe_reference_context
from tools.references.workflow import CHECKS, align_transcripts, prepare_reference, review_reference, verify_reference

WORDS = "alpha bravo cedar delta ember forest garden harbor island jasmine kettle lemon".split()


@pytest.fixture
def context(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe are required for the local WAV integration fixture")
    original = tmp_path / "synthetic-tone.wav"
    rate = 16000
    samples = array.array("h", (int(2500 * math.sin(2 * math.pi * 220 * i / rate)) for i in range(rate * 12)))
    with wave.open(str(original), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(samples.tobytes())
    media = prepare_clip(original, tmp_path / "context", source_id="synthetic", source_url="https://publisher.example/interview",
        source_timeline="publisher_native_audio", duration=12, include_video=False)
    publisher = {"schema_version": "publisher_transcript.v1", "source_id": "synthetic",
        "source_url": "https://publisher.example/interview", "source_timeline": "publisher_caption_clock", "segments": []}
    for index, words in enumerate((WORDS[:6], WORDS[6:])):
        basis = "explicit_publisher_label" if index == 0 else "same_paragraph_label_continuation"
        publisher["segments"].append({"id": f"cue-{index + 1}", "start": index * 2.1 + 0.2,
            "speaker": "Speaker One", "speaker_basis": basis, "text": " ".join(words),
            "speaker_spans": [{"kind": "caption_text", "text": " ".join(words), "speaker": "Speaker One"}]})
    asr = {"schema_version": "transcript.v1", "source_id": "synthetic", "source_timeline": "publisher_native_audio",
        "source_start_seconds": 0, "audio_sha256": media["artifacts"]["audio"]["sha256"],
        "engine": {"name": "synthetic_fixture_no_model_run"}, "segments": [{"text": " ".join(WORDS), "words": [
            {"word": word, "start": 1 + index * 0.35, "end": 1.3 + index * 0.35, "confidence": 0.9}
            for index, word in enumerate(WORDS)
        ]}]}
    publisher_file, asr_file = tmp_path / "publisher.json", tmp_path / "asr.json"
    publisher_file.write_text(json.dumps(publisher))
    asr_file.write_text(json.dumps(asr))
    return {"original": original, "media_manifest": tmp_path / "context/manifest.json", "publisher_file": publisher_file,
        "asr_file": asr_file, "publisher": publisher, "asr": asr, "output": tmp_path / "reference"}


def make_reference(context):
    return prepare_reference(context["media_manifest"], context["publisher_file"], context["asr_file"],
        context["original"], context["output"], speaker="Speaker One", cue_ids=["cue-1", "cue-2"], start_seconds=0.5, duration_seconds=8)


def assessment(*, listening=False, status="verified"):
    return {"reviewer": "synthetic_test_reviewer", "listening_review_completed": listening,
        "checks": {name: {"status": status, "grounds": "Synthetic fixture explicitly exercises this review transition."} for name in CHECKS},
        "notes": "This is a synthetic test, not review of a real speaker."}


def test_actual_word_matching_retains_different_source_clocks(context):
    report = align_transcripts(context["publisher"], context["asr"])
    assert report["matched_tokens"] == report["asr_tokens"] == 12
    assert report["token_coverage"] == 1
    assert report["media_timeline"] != report["publisher_timeline"]
    assert report["offset_summary"]["median_seconds"] == pytest.approx(0.8)
    assert report["review_status"] == "candidate"
    assert "no global clock offset" in report["scope"]


def test_same_subject_metadata_cannot_replace_source_identity(context):
    context["asr"]["source_id"] = "different-interview"
    with pytest.raises(MediaError, match="source IDs differ"):
        align_transcripts(context["publisher"], context["asr"])


def test_untimed_publisher_text_never_invents_subtitle_clock_or_offset(context):
    context["publisher"].update(timing_kind="untimed_text", source_timeline=None)
    for cue in context["publisher"]["segments"]:
        cue["start"] = None
    context["publisher_file"].write_text(json.dumps(context["publisher"]))
    report = align_transcripts(context["publisher"], context["asr"])
    assert report["matched_tokens"] == 12
    assert report["offset_summary"] is None
    assert report["anchors"] == []
    assert report["publisher_timeline"] is None
    assert all(match["publisher_cue_start_seconds"] is None for match in report["matches"])
    assert make_reference(context)["automatic_checks"]["exact_normalized_text_agreement"] is True


def test_missing_publisher_times_need_explicit_untimed_declaration(context):
    context["publisher"]["segments"][0]["start"] = None
    with pytest.raises(MediaError, match="publisher cue onset"):
        align_transcripts(context["publisher"], context["asr"])
    context["publisher"]["timing_kind"] = "untimed_text"
    with pytest.raises(MediaError, match="Untimed publisher text"):
        align_transcripts(context["publisher"], context["asr"])


def test_real_ffmpeg_candidate_keeps_full_provenance_and_never_auto_approves(context):
    manifest = make_reference(context)
    assert manifest["review"]["status"] == "candidate"
    assert manifest["automatic_checks"]["exact_normalized_text_agreement"] is True
    assert manifest["publisher_attribution"]["biometric_identity_inference"] is False
    assert manifest["clip"]["sample_rate"] == 24000
    assert manifest["clip"]["duration_seconds"] == pytest.approx(8, abs=0.001)
    assert (context["output"] / "reference.txt").read_text().strip().split() == WORDS
    assert verify_reference(context["output"] / "reference.json")["review_ledger"]["entries"] == 1
    assert (context["output"] / "publisher-context.json").is_file()
    assert (context["output"] / "asr-context.json").is_file()


@pytest.mark.parametrize("speaker,basis", [(None, "unknown"), (None, "mixed_speakers"), ("Speaker Two", "explicit_publisher_label")])
def test_unknown_or_mixed_publisher_cues_cannot_establish_target_speaker(context, speaker, basis):
    context["publisher"]["segments"][1].update(speaker=speaker, speaker_basis=basis)
    context["publisher_file"].write_text(json.dumps(context["publisher"]))
    with pytest.raises(MediaError, match="Every selected cue"):
        make_reference(context)
    assert not context["output"].exists()


def test_asr_audio_and_original_media_hashes_are_required(context):
    context["asr"]["audio_sha256"] = "0" * 64
    context["asr_file"].write_text(json.dumps(context["asr"]))
    with pytest.raises(MediaError, match="verified context audio"):
        make_reference(context)


def test_cutting_through_a_word_is_rejected(context):
    with pytest.raises(MediaError, match="cut an ASR word"):
        prepare_reference(context["media_manifest"], context["publisher_file"], context["asr_file"], context["original"],
            context["output"], speaker="Speaker One", cue_ids=["cue-1", "cue-2"], start_seconds=1.1, duration_seconds=8)


def test_adjacent_word_is_not_included_by_floating_point_roundoff(context):
    words = context["asr"]["segments"][0]["words"]
    for index, word in enumerate(words):
        word["start"] = 1.22 + index * 0.7
        word["end"] = word["start"] + 0.6
    words[-1]["end"] = 10.16
    words.append({"word": "unrelated", "start": 10.16, "end": 10.88, "confidence": 0.9})
    context["asr_file"].write_text(json.dumps(context["asr"]))
    result = prepare_reference(context["media_manifest"], context["publisher_file"], context["asr_file"], context["original"],
        context["output"], speaker="Speaker One", cue_ids=["cue-1", "cue-2"], start_seconds=1.22, duration_seconds=8.94)
    assert result["automatic_checks"]["exact_normalized_text_agreement"] is True


def test_reference_duration_is_bounded(context):
    with pytest.raises(MediaError, match="reference duration"):
        prepare_reference(context["media_manifest"], context["publisher_file"], context["asr_file"], context["original"],
            context["output"], speaker="Speaker One", cue_ids=["cue-1", "cue-2"], start_seconds=0.5, duration_seconds=7)


@pytest.mark.parametrize("name", ["reference.txt", "audio.wav", "publisher-context.json", "asr-context.json"])
def test_changed_reference_artifacts_invalidate_review(context, name):
    make_reference(context)
    file = context["output"] / name
    file.write_bytes(file.read_bytes() + b"changed")
    with pytest.raises(MediaError, match="artifact changed"):
        verify_reference(context["output"] / "reference.json")


def test_review_requires_explicit_checks_and_listening_does_not_happen_implicitly(context):
    make_reference(context)
    path = context["output"] / "reference.json"
    incomplete = assessment()
    del incomplete["checks"]["audio_quality"]
    with pytest.raises(MediaError, match="Supply each"):
        review_reference(path, incomplete)
    first = review_reference(path, assessment(listening=False))
    assert first["review"]["status"] == "candidate"
    assert first["review_ledger"]["entries"] == 2
    second = review_reference(path, assessment(listening=True))
    assert second["review"]["status"] == "reviewed"
    assert second["review_ledger"]["entries"] == 3
    assert verify_reference(path)["review"]["status"] == "reviewed"


def test_any_rejected_check_prevents_approval(context):
    make_reference(context)
    data = assessment(listening=True)
    data["checks"]["single_speaker"]["status"] = "rejected"
    assert review_reference(context["output"] / "reference.json", data)["review"]["status"] == "rejected"


@pytest.mark.parametrize("field,value", [("source_start_seconds", 12), ("review", {"status": "reviewed"})])
def test_manifest_clock_or_review_tampering_is_detected(context, field, value):
    make_reference(context)
    path = context["output"] / "reference.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(MediaError, match="ledger|bound"):
        verify_reference(path)


def test_ledger_history_cannot_be_rewritten_silently(context):
    make_reference(context)
    ledger = context["output"] / "review-ledger.jsonl"
    entry = json.loads(ledger.read_text())
    entry["actor"] = "someone_else"
    ledger.write_text(json.dumps(entry) + "\n")
    with pytest.raises(MediaError, match="hash chain"):
        verify_reference(context["output"] / "reference.json")


def test_artifact_paths_cannot_escape_the_reference_directory(context):
    make_reference(context)
    path = context["output"] / "reference.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"]["reference_text"]["path"] = "../asr.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(MediaError, match="escapes"):
        verify_reference(path)


def test_candidate_output_is_never_overwritten(context):
    make_reference(context)
    with pytest.raises(MediaError, match="already exists"):
        make_reference(context)


def test_asr_optional_dependency_has_an_actionable_error(context, monkeypatch, tmp_path):
    monkeypatch.delenv("ORT_DISABLE_TELEMETRY", raising=False)
    model = tmp_path / "model"
    model.mkdir()
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (model / name).write_text("synthetic")
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == "faster_whisper":
            assert os.environ.get("ORT_DISABLE_TELEMETRY") == "1"
            raise ImportError("synthetic missing optional dependency")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(MediaError, match="isolated environment"):
        transcribe_reference_context(context["media_manifest"], model_directory=model, model_repo="test/model",
            revision="a" * 40, output=tmp_path / "asr-output.json")


def test_cli_summary_never_prints_reference_transcript(context, capsys):
    make_reference(context)
    assert main(["verify", str(context["output"] / "reference.json")]) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["review_status"] == "candidate"
    assert "alpha bravo" not in output
