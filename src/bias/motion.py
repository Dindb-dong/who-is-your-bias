"""Source-context retrieval of measured geometry, with explicit review gates.

This module does not infer emotion, identify a person, or learn a person's
conditional gesture policy. Geometry clusters are fitted offline; the runtime
retrieves their measured windows using documented source context.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ID_PATTERN = r"^[a-z0-9][a-z0-9_.-]{0,119}$"
HASH_PATTERN = r"^[a-f0-9]{64}$"
REVIEW_FLAGS = ("track_continuity_reviewed", "source_clock_reviewed", "speaker_association_reviewed")
FINGER_CHANNELS = {side + finger + joint for side in ("left", "right")
                   for finger in ("Thumb", "Index", "Middle", "Ring", "Little")
                   for joint in (("Metacarpal", "Proximal", "Distal") if finger == "Thumb" else ("Proximal", "Intermediate", "Distal"))}
BODY_CHANNELS = FINGER_CHANNELS | {"chest", "leftUpperArm", "rightUpperArm", "leftForearm", "rightForearm", "leftHand", "rightHand"}
MAX_CATALOG_BYTES = 8 * 1024**2
MAX_CLIP_BYTES = 8 * 1024**2


class MotionError(ValueError):
    """A catalog, asset, or requested selection could not be verified."""


class MotionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class MotionContext(MotionRecord):
    id: str = Field(pattern=ID_PATTERN)
    summary: str = Field(min_length=1, max_length=1000)
    summary_kind: Literal["original_paraphrase"] = "original_paraphrase"
    cue_ids: list[str] = Field(max_length=100)
    timeline: str = Field(min_length=1, max_length=120)
    start_seconds: float = Field(ge=0, le=86400)
    end_seconds: float = Field(gt=0, le=86400)
    alignment: Literal["reviewed_word_anchors", "publisher_metadata_mapping", "asr_word_matches", "unverified_nearby_context"]
    alignment_note: str = Field(min_length=1, max_length=2000)
    alignment_evidence_id: str | None = None
    alignment_evidence_path: str | None = None
    alignment_evidence_sha256: str | None = Field(default=None, pattern=HASH_PATTERN)


class MotionSource(MotionRecord):
    id: str = Field(pattern=ID_PATTERN)
    source_event_id: str = Field(pattern=ID_PATTERN)
    url: str = Field(min_length=1, max_length=2000)
    corpus_split: Literal["train", "validation", "test"]
    motion_split: Literal["train", "validation", "test"]


class MotionClip(MotionRecord):
    id: str = Field(pattern=ID_PATTERN)
    path: str = Field(min_length=1, max_length=250)
    sha256: str = Field(pattern=HASH_PATTERN)
    source_id: str = Field(pattern=ID_PATTERN)
    context: MotionContext | None = None


class MotionPrimitive(MotionRecord):
    id: str = Field(pattern=ID_PATTERN)
    clip_id: str = Field(pattern=ID_PATTERN)
    family: Literal["face", "upper_body"]
    start_ms: float = Field(ge=0, le=120000)
    end_ms: float = Field(gt=0, le=120000)
    descriptor: dict[str, float]
    feature_coverage: dict[str, float]
    frame_count: int = Field(ge=3, le=3601)
    geometry_cluster_id: str | None = Field(default=None, pattern=ID_PATTERN)
    prototype_distance: float | None = Field(default=None, ge=0)


class MotionCatalog(MotionRecord):
    schema_version: Literal["motion-library.v1"]
    id: str = Field(pattern=ID_PATTERN)
    # Optional for frozen historical artifacts; new libraries declare a subject.
    persona_id: str | None = Field(default=None, pattern=ID_PATTERN)
    version: str = Field(min_length=1, max_length=60)
    created_at: str
    sources: list[MotionSource] = Field(min_length=1, max_length=1000)
    clips: list[MotionClip] = Field(min_length=1, max_length=2000)
    primitives: list[MotionPrimitive] = Field(min_length=1, max_length=20000)
    fitting: dict[str, Any]
    evaluation: dict[str, Any]
    limitations: list[str]


class MotionSelection(MotionRecord):
    schema_version: Literal["motion-selection.v1"] = "motion-selection.v1"
    library_id: str
    library_version: str
    primitive_id: str
    family: Literal["face", "upper_body"]
    source_id: str
    source_event_id: str
    source_timeline: str
    source_start_seconds: float
    source_end_seconds: float
    duration_ms: float
    review_status: Literal["geometry_preview", "speaker_reviewed"]
    selection_basis: Literal["source_context_retrieval", "nearby_source_context_retrieval"]
    context_summary: str
    context_alignment: str
    score: float = Field(ge=0)
    geometry_cluster_id: str | None = None
    clip_url: str
    clip_sha256: str
    parent_clip_sha256: str


def _read_json(path: Path, maximum: int) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > maximum:
        raise MotionError("Missing or oversized motion artifact")
    try:
        data = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(MotionError("Nonfinite motion JSON")))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise MotionError("Cannot read motion JSON") from error
    if not isinstance(data, dict):
        raise MotionError("Motion artifact must contain an object")
    return data


def _number(value: Any, label: str, minimum: float = 0, maximum: float = 120000) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise MotionError(f"Invalid {label}")
    return float(value)


def validate_clip(clip: dict[str, Any]) -> None:
    """Validate the animation subset consumed by mining and runtime playback."""
    if not isinstance(clip, dict) or clip.get("schema_version") != "avatar-clip.v1":
        raise MotionError("Expected avatar-clip.v1")
    duration = _number(clip.get("duration_ms"), "clip duration", 1)
    _number(clip.get("max_gap_ms"), "maximum gap", 1, 1000)
    source = clip.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("source_id"), str) or not isinstance(source.get("source_timeline"), str):
        raise MotionError("Clip source identity and timeline are required")
    _number(source.get("source_start_seconds"), "source start", 0, 86400)
    for name in ("video_sha256", "observations_sha256", "review_sha256"):
        if not re.fullmatch(HASH_PATTERN, str(source.get(name, ""))):
            raise MotionError(f"Missing clip provenance: {name}")
    review = clip.get("review", {})
    if not isinstance(review, dict):
        raise MotionError("Motion review must be an object")
    if not isinstance(review.get("status"), str) or review["status"] not in {"geometry_preview", "speaker_reviewed"}:
        raise MotionError("Unknown motion review status")
    for flag in REVIEW_FLAGS:
        if not isinstance(review.get(flag), bool):
            raise MotionError("Motion review flags must be explicit booleans")
        if review["status"] == "speaker_reviewed" and not review[flag]:
            raise MotionError("Speaker-reviewed geometry requires all review flags")
    calibration = clip.get("calibration", {})
    if not isinstance(calibration, dict):
        raise MotionError("Motion calibration must be an object")
    if calibration.get("coordinate_basis") != "right_handed_y_up" or calibration.get("bone_rotation_space") != "local_delta":
        raise MotionError("Unsupported motion coordinate calibration")
    for name in ("face_neutral", "bone_neutral"):
        if not isinstance(calibration.get(name, {}), dict):
            raise MotionError("Neutral calibration channels must be objects")
    limitations = clip.get("limitations", [])
    if not isinstance(limitations, list) or any(not isinstance(item, str) for item in limitations):
        raise MotionError("Clip limitations must be a list of text entries")
    neutral = _number(calibration.get("neutral_frame_ms"), "neutral reference timestamp", -120000, 120000)
    if not 0 <= neutral <= duration and not isinstance(calibration.get("parent_reference"), dict):
        raise MotionError("An external neutral reference requires parent calibration provenance")
    frames = clip.get("frames")
    if not isinstance(frames, list) or not 3 <= len(frames) <= 3601:
        raise MotionError("A motion clip needs 3–3601 frames")
    previous = -1.0
    for frame in frames:
        if not isinstance(frame, dict):
            raise MotionError("Invalid motion frame")
        time = _number(frame.get("time_ms"), "frame timestamp", 0, duration)
        if time <= previous or time >= duration:
            raise MotionError("Frame timestamps must increase inside the clip")
        previous = time
        for category in ("face", "bones"):
            values = frame.get(category, {})
            if not isinstance(values, dict) or len(values) > 80:
                raise MotionError("Invalid geometry channels")
            for name, value in values.items():
                if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,79}", name):
                    raise MotionError("Invalid geometry channel name")
                if category == "face":
                    _number(value, "face coefficient", 0, 1)
                else:
                    if not isinstance(value, list) or len(value) != 4:
                        raise MotionError("Bone rotation must be an XYZW quaternion")
                    components = [_number(v, "quaternion component", -1, 1) for v in value]
                    if abs(sum(v * v for v in components) - 1) > 0.01:
                        raise MotionError("Bone rotation must be normalized")


def _tokens(text: str) -> list[str]:
    # Transparent lexical retrieval; no sentiment or personality classification.
    ignored = {"the", "and", "are", "for", "with", "that", "this", "from", "how", "what", "was", "were", "will", "you", "your", "about", "into", "have", "has", "not", "can", "why", "does", "would", "could", "should", "his", "its", "than", "their", "they", "them", "these", "those", "is", "to", "of", "in", "on", "at", "an", "as", "be", "by", "or", "it", "do", "me", "my", "we", "us", "so", "if"}
    words = [token for token in re.findall(r"[^\W_]{2,}", text.casefold(), flags=re.UNICODE) if token not in ignored]
    # Korean particles often attach to nouns. Literal Hangul bigrams offer a
    # transparent partial-word match without claiming semantic understanding.
    for word in re.findall(r"[가-힣]{2,}", text):
        words.extend("ko:" + word[index:index + 2] for index in range(len(word) - 1) if word[index:index + 2] not in {"니다", "까요", "해줘", "알려", "세요", "하는", "있는"})
    return words


def clip_digest(clip: dict[str, Any]) -> str:
    """Digest of the canonical response payload, independent of JSON spacing."""
    return hashlib.sha256(json.dumps(clip, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def context_digest(context: MotionContext) -> str:
    """Bind authored context to its alignment without hashing circular references.

    Changing a paraphrase, cue selection, or publisher interval requires a new
    binding run. The hash records that declaration; it does not validate the
    paraphrase's meaning or identify a speaker.
    """
    return clip_digest(context.model_dump(include={"id", "summary", "summary_kind",
        "cue_ids", "timeline", "start_seconds", "end_seconds"}))


class MotionLibrary:
    def __init__(self, catalog: MotionCatalog, clips: dict[str, dict[str, Any]]):
        self.catalog = catalog
        self._clips = clips
        self._clip_records = {clip.id: clip for clip in catalog.clips}
        self._sources = {source.id: source for source in catalog.sources}
        self._primitives = {primitive.id: primitive for primitive in catalog.primitives}

    @classmethod
    def load(cls, catalog_path: str | Path, asset_root: str | Path) -> "MotionLibrary":
        catalog = MotionCatalog.model_validate(_read_json(Path(catalog_path), MAX_CATALOG_BYTES))
        root = Path(asset_root).resolve(strict=True)
        source_ids = [source.id for source in catalog.sources]
        clip_ids = [clip.id for clip in catalog.clips]
        primitive_ids = [primitive.id for primitive in catalog.primitives]
        if len(set(source_ids)) != len(source_ids) or len(set(clip_ids)) != len(clip_ids) or len(set(primitive_ids)) != len(primitive_ids):
            raise MotionError("Duplicate source, clip, or primitive identity")
        events: dict[str, tuple[str, str]] = {}
        for source in catalog.sources:
            split = (source.corpus_split, source.motion_split)
            if source.source_event_id in events and events[source.source_event_id] != split:
                raise MotionError("A source event crosses dataset splits")
            if source.motion_split == "train" and source.corpus_split != "train":
                raise MotionError("Corpus held-out events cannot fit motion prototypes")
            events[source.source_event_id] = split
        clips = {}
        contexts: dict[str, tuple[str, dict]] = {}
        for record in catalog.clips:
            relative = Path(record.path)
            path = (root / relative).resolve()
            if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
                raise MotionError("Motion asset path escapes its asset root")
            if record.source_id not in source_ids:
                raise MotionError("Motion clip references an unknown source")
            data = _read_json(path, MAX_CLIP_BYTES)
            if hashlib.sha256(path.read_bytes()).hexdigest() != record.sha256:
                raise MotionError("Motion clip SHA-256 mismatch")
            validate_clip(data)
            if data["source"]["source_id"] != record.source_id:
                raise MotionError("Clip and catalog source identities differ")
            if record.context and record.context.end_seconds <= record.context.start_seconds:
                raise MotionError("Invalid context interval")
            if record.context:
                identity = (record.source_id, record.context.model_dump())
                if record.context.id in contexts and contexts[record.context.id] != identity:
                    raise MotionError("A context identity has conflicting source attribution")
                contexts[record.context.id] = identity
                if record.context.alignment == "asr_word_matches":
                    relative_evidence = Path(record.context.alignment_evidence_path or "")
                    evidence = (root / relative_evidence).resolve()
                    if relative_evidence.is_absolute() or ".." in relative_evidence.parts or not evidence.is_relative_to(root) or not evidence.is_file() or evidence.stat().st_size > MAX_CATALOG_BYTES:
                        raise MotionError("Invalid source-context evidence path")
                    if hashlib.sha256(evidence.read_bytes()).hexdigest() != record.context.alignment_evidence_sha256:
                        raise MotionError("Source-context evidence SHA-256 mismatch")
                    report = _read_json(evidence, MAX_CATALOG_BYTES)
                    entries = report.get("entries")
                    if not isinstance(entries, list):
                        raise MotionError("Source-context evidence entries must be a list")
                    matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("id") == record.context.alignment_evidence_id]
                    if report.get("schema_version") != "motion-context-alignment.v1" or len(matches) != 1 or matches[0].get("source_id") != record.source_id or matches[0].get("clip_sha256") != record.sha256 or matches[0].get("status") != "asr_word_matches":
                        raise MotionError("Source-context evidence does not support this clip")
                    if matches[0].get("context_sha256") != context_digest(record.context):
                        raise MotionError("Source-context declaration changed after alignment")
                    source = data["source"]
                    if (matches[0].get("source_timeline") != source["source_timeline"]
                        or matches[0].get("start_seconds") != source["source_start_seconds"]
                        or matches[0].get("end_seconds") != source["source_start_seconds"] + data["duration_ms"] / 1000):
                        raise MotionError("Source-context evidence uses a different clip clock")
            clips[record.id] = data
        for primitive in catalog.primitives:
            clip = clips.get(primitive.clip_id)
            if clip is None or primitive.end_ms <= primitive.start_ms or primitive.end_ms > clip["duration_ms"]:
                raise MotionError("Primitive falls outside its clip")
            if not primitive.descriptor or any(not 0 <= coverage <= 1 for coverage in primitive.feature_coverage.values()):
                raise MotionError("Invalid primitive feature coverage")
        return cls(catalog, clips)

    def _eligible(self, primitive: MotionPrimitive, allow_geometry_preview: bool) -> bool:
        record = self._clip_records[primitive.clip_id]
        source = self._sources[record.source_id]
        review = self._clips[primitive.clip_id]["review"]
        return source.motion_split == "train" and source.corpus_split == "train" and (
            review["status"] == "speaker_reviewed" or allow_geometry_preview
        )

    def select(self, query: str, *, allow_geometry_preview: bool = False) -> MotionSelection | None:
        """BM25 over source paraphrases; abstain on absent lexical support.

        Only motion-training events participate. Geometry distance breaks ties;
        it is never an estimate of the target person's behavioral likelihood.
        """
        if not isinstance(query, str) or len(query) > 12000:
            raise MotionError("Motion query must be bounded text")
        eligible = [p for p in self.catalog.primitives if self._eligible(p, allow_geometry_preview) and self._clip_records[p.clip_id].context is not None]
        contexts = {self._clip_records[p.clip_id].context.id: self._clip_records[p.clip_id].context for p in eligible}
        if not contexts:
            return None
        documents = {key: Counter(_tokens(context.summary)) for key, context in contexts.items()}
        query_terms = set(_tokens(query))
        frequency = Counter(token for document in documents.values() for token in document)
        average_length = sum(sum(document.values()) for document in documents.values()) / len(documents)
        scores = {}
        for key, document in documents.items():
            score = 0.0
            supported = query_terms.intersection(document)
            # This is an engineering abstention gate, not a learned confidence
            # score. A generic transport word must not select a tunnel gesture
            # from an unrelated electric-car passage.
            if not query_terms or len(supported) / len(query_terms) < 0.2:
                scores[key] = 0.0
                continue
            for term in supported:
                tf = document[term]
                idf = math.log(1 + (len(documents) - frequency[term] + 0.5) / (frequency[term] + 0.5))
                score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * sum(document.values()) / max(1, average_length)))
            scores[key] = score
        ranked = sorted(eligible, key=lambda primitive: (-scores[self._clip_records[primitive.clip_id].context.id], primitive.prototype_distance if primitive.prototype_distance is not None else math.inf, primitive.id))
        chosen = ranked[0]
        record = self._clip_records[chosen.clip_id]
        assert record.context is not None
        score = scores[record.context.id]
        if score <= 0:
            return None
        source = self._sources[record.source_id]
        clip = self._clips[record.id]
        start = clip["source"]["source_start_seconds"] + chosen.start_ms / 1000
        return MotionSelection(
            library_id=self.catalog.id, library_version=self.catalog.version,
            primitive_id=chosen.id, family=chosen.family, source_id=record.source_id, source_event_id=source.source_event_id,
            source_timeline=clip["source"]["source_timeline"], source_start_seconds=start,
            source_end_seconds=start + (chosen.end_ms - chosen.start_ms) / 1000,
            duration_ms=chosen.end_ms - chosen.start_ms, review_status=clip["review"]["status"],
            selection_basis="nearby_source_context_retrieval" if record.context.alignment == "unverified_nearby_context" else "source_context_retrieval",
            context_summary=record.context.summary, context_alignment=record.context.alignment,
            score=round(score, 6), geometry_cluster_id=chosen.geometry_cluster_id,
            clip_url=f"/api/motion/clip/{chosen.id}", clip_sha256=clip_digest(self.clip(chosen.id, allow_geometry_preview=allow_geometry_preview)), parent_clip_sha256=record.sha256,
        )

    def clip(self, primitive_id: str, *, allow_geometry_preview: bool = False) -> dict[str, Any]:
        primitive = self._primitives.get(primitive_id)
        if primitive is None or not self._eligible(primitive, allow_geometry_preview):
            raise MotionError("Motion primitive is unavailable under the requested review gate")
        original = self._clips[primitive.clip_id]
        result = copy.deepcopy(original)
        result["id"] = primitive.id
        result["duration_ms"] = primitive.end_ms - primitive.start_ms
        result["source"]["source_start_seconds"] += primitive.start_ms / 1000
        result["source"]["observation_offset_ms"] = original["source"].get("observation_offset_ms", 0) + primitive.start_ms
        result["frames"] = [{**copy.deepcopy(frame), "time_ms": round(frame["time_ms"] - primitive.start_ms, 4)} for frame in original["frames"] if primitive.start_ms <= frame["time_ms"] < primitive.end_ms]
        allowed_bones = {"head"} if primitive.family == "face" else BODY_CHANNELS
        for frame in result["frames"]:
            if primitive.family == "upper_body":
                frame.pop("face", None)
            if "bones" in frame:
                frame["bones"] = {name: value for name, value in frame["bones"].items() if name in allowed_bones}
        if primitive.family == "upper_body":
            result["calibration"]["face_neutral"] = {}
        result["calibration"]["bone_neutral"] = {name: value for name, value in result["calibration"].get("bone_neutral", {}).items() if name in allowed_bones}
        # Keep the original calibration transform: these are deltas relative to
        # its reference pose, not newly calibrated anatomical joint rotations.
        result["calibration"]["neutral_frame_ms"] -= primitive.start_ms
        result["calibration"]["parent_reference"] = {"clip_sha256": self._clip_records[primitive.clip_id].sha256, "neutral_frame_ms": original["calibration"]["neutral_frame_ms"], "window_start_ms": primitive.start_ms, "rotation_operation": "retained_parent_local_delta"}
        result.setdefault("limitations", []).append("Primitive window retains its parent clip calibration; the reference frame may precede this window.")
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.catalog.id, "version": self.catalog.version,
            "source_events": sorted({source.source_event_id for source in self.catalog.sources}),
            "primitives": len(self.catalog.primitives),
            "unique_temporal_windows": len({(p.clip_id, p.start_ms, p.end_ms) for p in self.catalog.primitives}),
            "observed_frames": sum(len(clip["frames"]) for clip in self._clips.values()),
            "observed_seconds": sum(clip["duration_ms"] for clip in self._clips.values()) / 1000,
            "reviewed_runtime_primitives": sum(self._eligible(p, False) for p in self.catalog.primitives),
            "preview_runtime_primitives": sum(self._eligible(p, True) for p in self.catalog.primitives),
            "retrieval": {"algorithm": "bm25_source_paraphrases", "minimum_query_token_overlap": 0.2, "korean_matching": "literal_hangul_bigrams", "personal_gesture_policy": False},
            "fitting": self.catalog.fitting, "evaluation": self.catalog.evaluation,
            "limitations": self.catalog.limitations,
        }

