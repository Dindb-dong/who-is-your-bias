from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from bias.motion import MotionError, MotionLibrary, MotionSource, clip_digest, validate_clip
from tools.motion.mining import build_library, describe, fit_library, geodesic, mine_clip


def clip(source_id="event-a", amplitude=0.1):
    frames = []
    for i in range(40):
        angle = amplitude * math.sin(i * 0.4)
        frames.append({"time_ms": i * 125, "face": {"jawOpen": 0.4 + amplitude * math.sin(i * 0.3), "browInnerUp": 0.3 + amplitude * math.cos(i * 0.17)}, "bones": {"leftForearm": [0, 0, math.sin(angle / 2), math.cos(angle / 2)], "head": [0, 0, 0, 1]}})
    return {"schema_version": "avatar-clip.v1", "duration_ms": 5000, "max_gap_ms": 250, "source": {"source_id": source_id, "source_timeline": "native_video", "source_start_seconds": 20, "video_sha256": "a" * 64, "observations_sha256": "b" * 64, "review_sha256": "c" * 64, "input_sha256": "d" * 64}, "review": {"status": "geometry_preview", "reviewer": "test", "track_id": "visible-track", "track_continuity_reviewed": False, "source_clock_reviewed": True, "speaker_association_reviewed": False, "attribution_basis": "Synthetic test only"}, "calibration": {"neutral_frame_ms": 0, "face_neutral": {"jawOpen": 0.4}, "bone_neutral": {"leftForearm": [0, 0, 0, 1], "head": [0, 0, 0, 1]}, "coordinate_basis": "right_handed_y_up", "bone_rotation_space": "local_delta"}, "frames": frames, "limitations": []}


def source(sid, split="train", event_id=None, corpus_split="train"):
    return MotionSource(id=sid, source_event_id=event_id or sid, url="https://example.org/" + sid, corpus_split=corpus_split, motion_split=split)


def library_files(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    records = []
    for sid, amplitude, summary in (("event-a", 0.1, "Electric cars use batteries for efficient energy storage."), ("event-b", 0.2, "Tunnels may reduce urban traffic congestion.")):
        path = assets / (sid + ".json")
        path.write_text(json.dumps(clip(sid, amplitude)))
        records.append({"id": sid + "-clip", "path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "source_id": sid, "context": {"id": sid + "-context", "summary": summary, "cue_ids": ["cue-00001"], "timeline": "publisher_caption", "start_seconds": 18, "end_seconds": 26, "alignment": "unverified_nearby_context", "alignment_note": "Test fixture, no speech alignment."}})
    manifest = {"schema_version": "motion-input.v1", "id": "test-motion", "version": "1.0.0", "sources": [source("event-a").model_dump(), source("event-b", "validation").model_dump()], "clips": records, "window_ms": 2500, "stride_ms": 2500, "clusters_per_family": 2}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "catalog.json"
    build_library(manifest_path, assets, output)
    return manifest_path, assets, output


def test_quaternion_sign_is_not_motion():
    assert geodesic([0, 0, 0, 1], [0, 0, 0, -1]) == 0
    q = [0, 0, math.sin(0.3), math.cos(0.3)]
    assert geodesic(q, [-value for value in q]) == pytest.approx(0, abs=1e-7)


def test_stationary_quaternion_normalization_error_is_not_angular_motion():
    data = clip()
    for frame in data["frames"]:
        frame["bones"]["leftForearm"] = [0, 0, 0, 0.999]
    validate_clip(data)
    descriptor, _ = describe(data["frames"], "upper_body", 250)
    assert descriptor["leftForearm:range_rad"] == 0
    assert descriptor["leftForearm:speed_rad_s"] == 0
    assert geodesic([0, 0, 0, 0.999], [0, 0, 0, -1]) == 0
    with pytest.raises(MotionError, match="nonzero"):
        geodesic([0, 0, 0, 0], [0, 0, 0, 1])


@pytest.mark.parametrize("field,value", [("review", None), ("review", []), ("calibration", None), ("calibration", []), ("limitations", None)])
def test_malformed_optional_clip_objects_raise_motion_error(field, value):
    data = clip()
    data[field] = value
    with pytest.raises(MotionError):
        validate_clip(data)


def test_malformed_neutral_channels_and_review_status_raise_motion_error():
    data = clip()
    data["calibration"]["bone_neutral"] = None
    with pytest.raises(MotionError, match="calibration"):
        validate_clip(data)
    data = clip()
    data["review"]["status"] = []
    with pytest.raises(MotionError, match="review status"):
        validate_clip(data)


def test_missing_channels_remain_absent_and_large_gaps_are_not_mined():
    data = clip()
    for frame in data["frames"][:20]:
        del frame["face"]["jawOpen"]
    features, coverage = describe(data["frames"], "face", 250)
    assert "jawOpen:range" not in features
    assert coverage["jawOpen"] == 0.5
    data["frames"] = [frame for frame in data["frames"] if not 750 <= frame["time_ms"] < 1500]
    windows = mine_clip(data, "gapped")
    assert windows and all(window["start_ms"] >= 2500 for window in windows)


def test_validation_cannot_change_features_scales_or_prototypes():
    train = mine_clip(clip(), "training") + mine_clip(clip(amplitude=0.3), "training-b")
    validation = mine_clip(clip("event-b", 0.2), "validation")
    sources = {"training": source("event-a"), "training-b": source("event-a"), "validation": source("event-b", "validation")}
    first, _ = fit_library(copy.deepcopy(train + validation), sources, clusters=2)
    contaminated = copy.deepcopy(validation)
    for row in contaminated:
        row["descriptor"] = {key: value * 1e6 for key, value in row["descriptor"].items()}
        row["descriptor"]["holdout_only:range"] = 1e9
    second, evaluation = fit_library(copy.deepcopy(train) + contaminated, sources, clusters=2)
    assert first == second
    assert evaluation["face"]["held_out_source_events"] == ["event-b"]
    assert "holdout_only:range" not in second["face"]["feature_names"]


def test_review_gates_and_source_holdout_are_enforced(tmp_path):
    _, assets, output = library_files(tmp_path)
    library = MotionLibrary.load(output, assets)
    assert library.select("electric cars battery") is None
    selection = library.select("electric cars battery", allow_geometry_preview=True)
    assert selection is not None and selection.source_event_id == "event-a"
    assert selection.selection_basis == "nearby_source_context_retrieval"
    assert selection.clip_sha256 == clip_digest(library.clip(selection.primitive_id, allow_geometry_preview=True))
    assert library.select("urban tunnels", allow_geometry_preview=True) is None
    assert library.select("What do you think?", allow_geometry_preview=True) is None
    with pytest.raises(MotionError, match="review gate"):
        library.clip(selection.primitive_id)
    validation_id = next(p.id for p in library.catalog.primitives if p.clip_id == "event-b-clip")
    with pytest.raises(MotionError, match="review gate"):
        library.clip(validation_id, allow_geometry_preview=True)


def test_window_slice_retains_parent_reference_and_source_clock(tmp_path):
    _, assets, output = library_files(tmp_path)
    library = MotionLibrary.load(output, assets)
    primitive = next(p for p in library.catalog.primitives if p.clip_id == "event-a-clip" and p.start_ms == 2500 and p.family == "upper_body")
    result = library.clip(primitive.id, allow_geometry_preview=True)
    assert result["duration_ms"] == 2500
    assert result["source"]["source_start_seconds"] == 22.5
    assert result["source"]["observation_offset_ms"] == 2500
    assert result["frames"][0]["time_ms"] == 0
    assert all("face" not in frame and "head" not in frame["bones"] for frame in result["frames"])
    assert result["calibration"]["neutral_frame_ms"] == -2500
    assert result["calibration"]["parent_reference"]["rotation_operation"] == "retained_parent_local_delta"
    assert library._clips[primitive.clip_id]["calibration"]["neutral_frame_ms"] == 0
    validate_clip(result)


@pytest.mark.parametrize("tamper", ["hash", "path", "review", "split"])
def test_catalog_rejects_untrusted_assets_and_leaking_events(tmp_path, tamper):
    _, assets, output = library_files(tmp_path)
    catalog = json.loads(output.read_text())
    if tamper == "hash":
        (assets / "event-a.json").write_text("{}")
    elif tamper == "path":
        catalog["clips"][0]["path"] = "../outside.json"
    elif tamper == "review":
        path = assets / "event-a.json"
        data = json.loads(path.read_text())
        data["review"]["status"] = "speaker_reviewed"
        path.write_text(json.dumps(data))
        catalog["clips"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        catalog["sources"][1]["source_event_id"] = "event-a"
    output.write_text(json.dumps(catalog))
    with pytest.raises(MotionError):
        MotionLibrary.load(output, assets)


def test_corpus_holdout_cannot_be_used_for_motion_training(tmp_path):
    manifest_path, assets, output = library_files(tmp_path)
    data = json.loads(manifest_path.read_text())
    data["sources"][0]["corpus_split"] = "test"
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(MotionError, match="corpus holdout"):
        build_library(manifest_path, assets, output)


def test_shipped_catalog_has_real_source_holdout_and_closed_review_gate():
    root = Path(__file__).resolve().parents[2]
    library = MotionLibrary.load(root / "data/motion/elon-musk-ted-v1.json", root / "apps/web/public")
    assert library.summary()["reviewed_runtime_primitives"] == 0
    assert library.summary()["observed_frames"] > 100
    train_events = {source.source_event_id for source in library.catalog.sources if source.motion_split == "train"}
    held_out_events = {source.source_event_id for source in library.catalog.sources if source.motion_split != "train"}
    assert train_events and held_out_events and train_events.isdisjoint(held_out_events)
    for fitting in library.catalog.fitting["families"].values():
        assert set(fitting["training_source_events"]) <= train_events
    for primitive in library.catalog.primitives:
        if library._eligible(primitive, True):
            validate_clip(library.clip(primitive.id, allow_geometry_preview=True))
    assert library.select("electric transport") is None
    assert library.select("지하 터널 교통 정체에 대해서 알려줘", allow_geometry_preview=True) is None


@pytest.mark.parametrize("forged_timing,offset_shift", [(False, 0), (True, 0), (False, 2)])
def test_word_binding_requires_reproducible_hash_bound_matches(tmp_path, forged_timing, offset_shift):
    from tools.motion.contexts import bind_contexts
    from tools.references.workflow import align_transcripts

    manifest_path, assets, output = library_files(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["clips"] = manifest["clips"][:1]
    manifest["sources"] = manifest["sources"][:1]
    manifest_path.write_text(json.dumps(manifest))
    text = "Electric cars use batteries for efficient energy storage"
    publisher = {"schema_version": "publisher_transcript.v1", "source_id": "event-a", "source_timeline": "publisher_caption", "segments": [{"id": "cue-00001", "start": 18, "end": 26, "speaker": None, "speaker_spans": [{"kind": "caption_text", "text": text}]}]}
    publisher_file = tmp_path / "publisher.json"
    publisher_file.write_text(json.dumps({"transcript": publisher}))
    directory = tmp_path / "asr"
    directory.mkdir()
    audio = directory / "audio.wav"
    audio.write_bytes(b"synthetic hash fixture; not an audio inference test")
    audio_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
    asr = {"schema_version": "transcript.v1", "source_id": "event-a", "source_timeline": "native_video", "source_start_seconds": 20 + offset_shift, "audio_sha256": audio_hash, "engine": {"name": "synthetic-test"}, "segments": [{"words": [{"word": word, "start": i * 0.5, "end": i * 0.5 + 0.4, "confidence": 1} for i, word in enumerate(text.split())]}]}
    asr_file = directory / "asr.json"
    asr_file.write_text(json.dumps(asr))
    alignment = align_transcripts(publisher, asr)
    alignment.update(asr_artifact_sha256=hashlib.sha256(asr_file.read_bytes()).hexdigest(), publisher_artifact_sha256=hashlib.sha256(publisher_file.read_bytes()).hexdigest())
    if forged_timing:
        alignment["matches"][0]["source_start_seconds"] += 0.2
    (directory / "alignment.json").write_text(json.dumps(alignment))
    media = {"source_id": "event-a", "source_timeline": "native_video", "source_start_seconds": 20, "source_end_seconds": 25, "input": {"sha256": "d" * 64}, "artifacts": {"audio": {"path": "audio.wav", "sha256": audio_hash}}}
    (directory / "manifest.json").write_text(json.dumps(media))
    arguments = (manifest_path, assets, assets / "context-evidence.json")
    kwargs = {"publishers": {"event-a": publisher_file}, "alignment_directories": [directory]}
    if forged_timing:
        with pytest.raises(MotionError, match="could not be reproduced"):
            bind_contexts(*arguments, **kwargs)
        return
    if offset_shift:
        with pytest.raises(MotionError, match="source offsets differ"):
            bind_contexts(*arguments, **kwargs)
        return
    report = bind_contexts(*arguments, **kwargs)
    assert report["entries"][0]["status"] == "asr_word_matches"
    build_library(manifest_path, assets, output)
    library = MotionLibrary.load(output, assets)
    selection = library.select("electric cars", allow_geometry_preview=True)
    assert selection.context_alignment == "asr_word_matches"
    assert library.clip(selection.primitive_id, allow_geometry_preview=True)["review"]["speaker_association_reviewed"] is False
    catalog = json.loads(output.read_text())
    for changes in ({"cue_ids": ["cue-99999"]}, {"timeline": "another_caption_clock"},
        {"start_seconds": 80000, "end_seconds": 80010}, {"summary": "Unrelated replacement context."}):
        changed = copy.deepcopy(catalog)
        changed["clips"][0]["context"].update(changes)
        output.write_text(json.dumps(changed))
        with pytest.raises(MotionError, match="declaration changed"):
            MotionLibrary.load(output, assets)
    output.write_text(json.dumps(catalog))
    (assets / "context-evidence.json").write_text("{}")
    with pytest.raises(MotionError, match="evidence SHA-256"):
        MotionLibrary.load(output, assets)
