"""Extract public TED caption data without asserting audio diarization.

The original caption text and ordering remain available for question/answer
review. Speaker labels are textual publisher annotations. Cue end times are
explicitly marked interval proxies, never claimed to be forced alignment.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
MAX_CUES = 30_000
MAX_DURATION_SECONDS = 86_400
TIMELINE = "ted_transcript_cue"
VERSION = "1.0.0"

# Full names must also occur as explicit labels in this particular transcript.
# Page subject, title, speaker metadata, and alternating turns are never used.
DEFAULT_LABELS = {"Chris Anderson": "Chris Anderson", "Elon Musk": "Elon Musk"}
DEFAULT_ALIASES = {"CA": "Chris Anderson", "EM": "Elon Musk"}
NAME_TOKEN = r"(?:[A-Z]{1,8}|[A-Z][a-z]+(?: [A-Z][a-z]+){0,3})"
POSSIBLE_LABEL = re.compile(rf"(?<![\w])(?P<label>{NAME_TOKEN}):\s*")
STAGE_DIRECTION = re.compile(r"^(?:\s*\([^()]{1,100}\)\s*)+$")


class TranscriptError(ValueError):
    """Invalid or unsupported publisher input, with no automatic fallback."""


def _number(value: Any, label: str, *, high: float = MAX_DURATION_SECONDS) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TranscriptError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= high:
        raise TranscriptError(f"{label} must be between 0 and {high}")
    return result


def _json(raw: str) -> dict[str, Any]:
    def invalid_constant(value: str) -> None:
        raise TranscriptError(f"Nonfinite JSON number: {value}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TranscriptError("Duplicate JSON object key")
            result[key] = value
        return result

    try:
        result = json.loads(raw, parse_constant=invalid_constant, object_pairs_hook=unique_pairs)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise TranscriptError("Input does not contain valid publisher JSON") from exc
    if not isinstance(result, dict):
        raise TranscriptError("Publisher JSON must be an object")
    return result


class _NextDataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.active = False
        self.documents: list[str] = []
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script" and dict(attrs).get("id") == "__NEXT_DATA__":
            self.active = True
            self.parts = []

    def handle_data(self, data: str) -> None:
        if self.active:
            self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self.active:
            self.documents.append("".join(self.parts))
            self.active = False


def read_document(raw: bytes) -> tuple[dict[str, Any], str]:
    """Read either saved __NEXT_DATA__ JSON or the ordinary public HTML page."""
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise TranscriptError("Publisher document exceeds the 16 MiB limit")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError as exc:
        raise TranscriptError("Publisher input must be UTF-8") from exc
    if text.lstrip().startswith("{"):
        return _json(text), "saved_next_data"
    parser = _NextDataParser()
    parser.feed(text)
    parser.close()
    if len(parser.documents) != 1:
        raise TranscriptError("Expected exactly one public __NEXT_DATA__ script")
    return _json(parser.documents[0]), "publisher_html"


def validate_ted_url(value: str) -> str:
    if not isinstance(value, str):
        raise TranscriptError("Use an HTTPS TED talk page on ted.com or www.ted.com")
    try:
        parsed = urlparse(value)
        port = parsed.port
    except ValueError as exc:
        raise TranscriptError("Invalid TED talk URL") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"ted.com", "www.ted.com"}
        or parsed.username
        or parsed.password
        or port not in {None, 443}
        or not re.fullmatch(r"/talks/[a-z0-9_]+(?:/transcript)?/?", parsed.path)
        or parsed.fragment
    ):
        raise TranscriptError("Use an HTTPS TED talk page on ted.com or www.ted.com")
    return value


class _TEDRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        validate_ted_url(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def fetch_ted_page(url: str) -> bytes:
    """Fetch one bounded public page; no login, cookies, media, or bypasses."""
    validate_ted_url(url)
    request = Request(url, headers={"User-Agent": "who-is-your-bias-transcript-research/1.0", "Accept": "text/html"})
    try:
        with build_opener(_TEDRedirects()).open(request, timeout=25) as response:
            validate_ted_url(response.geturl())
            raw = response.read(MAX_DOCUMENT_BYTES + 1)
    except (OSError, ValueError) as exc:
        if isinstance(exc, TranscriptError):
            raise
        raise TranscriptError(f"Public TED fetch failed: {type(exc).__name__}") from exc
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise TranscriptError("Publisher document exceeds the 16 MiB limit")
    return raw


def _media_url(value: Any, *, hosts: set[str] | None = None) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = urlparse(value)
        port = parsed.port
    except ValueError:
        return None
    hostname = parsed.hostname or ""
    if parsed.scheme != "https" or parsed.username or parsed.password or port not in {None, 443}:
        return None
    allowed = hostname in hosts if hosts is not None else (
        hostname in {"ted.com", "tedcdn.com"} or hostname.endswith((".ted.com", ".tedcdn.com"))
    )
    return value if allowed else None


def _label_matches(text: str) -> list[re.Match[str]]:
    """Recognize textual labels while treating other colon phrases cautiously."""
    result = []
    for match in POSSIBLE_LABEL.finditer(text):
        prefix = text[:match.start()].rstrip()
        known = match["label"] in DEFAULT_LABELS or match["label"] in DEFAULT_ALIASES
        boundary = not prefix or prefix[-1] in ".!?—–)"
        if known or boundary:
            result.append(match)
    return result


def _speaker_spans(
    text: str,
    cue_id: str,
    label_map: dict[str, str],
    state: dict[str, str | None],
) -> tuple[list[dict[str, Any]], list[str]]:
    matches = _label_matches(text)
    spans: list[dict[str, Any]] = []
    explicit_labels = [m["label"] for m in matches]

    def append(start: int, end: int, explicit: bool) -> None:
        value = text[start:end].strip()
        if not value:
            return
        stage = bool(STAGE_DIRECTION.fullmatch(value))
        spans.append({
            "text": value, "start_character": start, "end_character": end,
            "speaker": None if stage else state["speaker"],
            "speaker_label": state["label"],
            "speaker_label_cue_id": state["anchor"],
            "attribution": "non_speech_marker" if stage else (
                "explicit_publisher_label" if explicit and state["speaker"] else
                "same_paragraph_label_continuation" if state["speaker"] else "unknown"
            ),
            "kind": "non_speech_marker" if stage else "caption_text",
        })

    cursor = 0
    explicit = False
    for match in matches:
        append(cursor, match.start(), explicit)
        state.update(label=match["label"], speaker=label_map.get(match["label"]), anchor=cue_id)
        cursor = match.end()
        explicit = True
    append(cursor, len(text), explicit)
    return spans, explicit_labels


def extract_ted(document: dict[str, Any], *, source_id: str, input_sha256: str) -> dict[str, Any]:
    """Return canonical captions, conservative utterance groups, and metadata."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,119}", source_id):
        raise TranscriptError("source_id must be a safe lowercase identifier")
    if not re.fullmatch(r"[0-9a-f]{64}", input_sha256):
        raise TranscriptError("input_sha256 must be a SHA-256 hex digest")
    try:
        page = document["props"]["pageProps"]
        video = page["videoData"]
        translation = page["transcriptData"]["translation"]
        paragraphs = translation["paragraphs"]
        language = translation["language"]["internalLanguageCode"]
        canonical_url = validate_ted_url(video["canonicalUrl"])
    except (KeyError, TypeError) as exc:
        raise TranscriptError("Missing TED videoData or public transcriptData translation") from exc
    if not isinstance(video, dict) or not isinstance(translation, dict):
        raise TranscriptError("Invalid TED video or translation object")
    if not isinstance(language, str) or not language.strip():
        raise TranscriptError("Publisher translation language is missing")
    if not isinstance(paragraphs, list) or not paragraphs:
        raise TranscriptError("Publisher transcript has no paragraphs")
    player = video.get("videoPlayerData") or {}
    if not isinstance(player, dict):
        raise TranscriptError("Invalid video player metadata")
    duration_value = player.get("duration", video.get("duration"))
    duration = _number(duration_value, "Publisher duration") if duration_value is not None else None

    flat: list[dict[str, Any]] = []
    last_time = -1.0
    for paragraph_index, paragraph in enumerate(paragraphs):
        if not isinstance(paragraph, dict) or not isinstance(paragraph.get("cues"), list):
            raise TranscriptError("Every publisher paragraph must contain a cues array")
        for cue_index, cue in enumerate(paragraph["cues"]):
            if not isinstance(cue, dict) or not isinstance(cue.get("text"), str) or not cue["text"].strip():
                raise TranscriptError("Every publisher cue needs nonempty text")
            if len(cue["text"]) > 100_000:
                raise TranscriptError("Publisher cue text is too large")
            time_ms = _number(cue.get("time"), "Publisher cue milliseconds", high=MAX_DURATION_SECONDS * 1000)
            if time_ms < last_time:
                raise TranscriptError("Publisher cue times must be nondecreasing")
            last_time = time_ms
            flat.append({"paragraph_index": paragraph_index, "cue_index": cue_index, "raw": cue, "start": time_ms / 1000})
            if len(flat) > MAX_CUES:
                raise TranscriptError("Publisher transcript exceeds 30,000 cues")
    if not flat:
        raise TranscriptError("Publisher transcript has no cues")
    caption_content = {"language": language, "paragraphs": [
        [{"time": cue["time"], "text": cue["text"]} for cue in paragraph["cues"]]
        for paragraph in paragraphs
    ]}
    caption_sha256 = hashlib.sha256(json.dumps(caption_content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()

    observed_labels = {m["label"] for item in flat for m in _label_matches(item["raw"]["text"])}
    label_map = {name: name for name in DEFAULT_LABELS if name in observed_labels}
    label_map.update({alias: name for alias, name in DEFAULT_ALIASES.items() if name in label_map})
    cue_times = sorted({item["start"] for item in flat})
    following_times = dict(zip(cue_times, cue_times[1:]))
    time_counts = Counter(item["start"] for item in flat)
    segments: list[dict[str, Any]] = []
    paragraph_rows = []
    cursor = 0
    for paragraph_index, paragraph in enumerate(paragraphs):
        paragraph_id = f"paragraph-{paragraph_index + 1:04d}"
        state: dict[str, str | None] = {"speaker": None, "label": None, "anchor": None}
        ids = []
        for cue_index, cue in enumerate(paragraph["cues"]):
            item = flat[cursor]
            cue_id = f"cue-{cursor + 1:05d}"
            ids.append(cue_id)
            spans, explicit_labels = _speaker_spans(cue["text"], cue_id, label_map, state)
            spoken = [span for span in spans if span["kind"] == "caption_text"]
            names = sorted({span["speaker"] for span in spoken if span["speaker"] is not None})
            entirely_known = bool(spoken) and all(span["speaker"] is not None for span in spoken)
            speaker = names[0] if entirely_known and len(names) == 1 else None
            mixed = len(names) > 1 or bool(names) and not entirely_known
            basis = (
                "mixed_speakers" if mixed else "non_speech_marker" if not spoken else
                "explicit_publisher_label" if speaker and explicit_labels else
                "same_paragraph_label_continuation" if speaker else "unknown"
            )
            end = following_times.get(item["start"])
            end_basis = "next_distinct_cue_start_proxy" if end is not None else "missing"
            if end is None and duration is not None and duration > item["start"]:
                end, end_basis = duration, "publisher_duration_proxy"
            anchors = {span["speaker_label_cue_id"] for span in spoken}
            context_id = next(iter(anchors)) if len(anchors) == 1 else cue_id
            segments.append({
                "id": cue_id, "start": item["start"], "end": end,
                "source_start": item["start"], "source_end": end,
                "text": cue["text"], "speaker": speaker, "speaker_basis": basis,
                "speaker_names": names, "explicit_labels": explicit_labels,
                "speaker_context_id": context_id or paragraph_id,
                "speaker_spans": spans, "paragraph_id": paragraph_id,
                "publisher_paragraph_index": paragraph_index, "publisher_cue_index": cue_index,
                "publisher_time_ms": cue["time"], "end_basis": end_basis,
                "shared_start_timestamp": time_counts[item["start"]] > 1,
                "previous_cue_id": f"cue-{cursor:05d}" if cursor else None,
                "next_cue_id": f"cue-{cursor + 2:05d}" if cursor + 1 < len(flat) else None,
                "review_status": "draft", "words": [],
                "review_required": ["publisher_speaker_labels", "transcript_accuracy", "source_timeline_alignment", "cue_end_proxy"],
            })
            cursor += 1
        paragraph_rows.append({
            "id": paragraph_id, "publisher_paragraph_index": paragraph_index,
            "text": " ".join(cue["text"] for cue in paragraph["cues"]), "cue_ids": ids,
            "previous_paragraph_id": f"paragraph-{paragraph_index:04d}" if paragraph_index else None,
            "next_paragraph_id": f"paragraph-{paragraph_index + 2:04d}" if paragraph_index + 1 < len(paragraphs) else None,
        })

    utterances: list[dict[str, Any]] = []
    last_key = None
    for segment in segments:
        key = (segment["paragraph_id"], segment["speaker"], segment["speaker_context_id"], segment["speaker_basis"] == "non_speech_marker")
        if key != last_key:
            utterances.append({
                "id": f"utterance-{len(utterances) + 1:04d}", "paragraph_id": segment["paragraph_id"],
                "speaker": segment["speaker"], "cue_ids": [], "text": "", "start": segment["start"],
                "end": None, "grouping": "consecutive_cues_in_same_paragraph_and_label_context", "review_status": "draft",
            })
            last_key = key
        utterance = utterances[-1]
        utterance["cue_ids"].append(segment["id"])
        utterance["text"] = (utterance["text"] + " " + segment["text"]).strip()
        utterance["end"] = segment["end"]
        utterance["end_basis"] = segment["end_basis"]

    media_links = []
    resources = player.get("resources") or {}
    if isinstance(resources, dict):
        hls = resources.get("hls") or {}
        if isinstance(hls, dict) and (url := _media_url(hls.get("stream"))):
            media_links.append({"kind": "publisher_hls", "url": url, "observed_field": "videoPlayerData.resources.hls.stream", "alignment": "unverified"})
        for resource in resources.get("h264") or []:
            if isinstance(resource, dict) and (url := _media_url(resource.get("file"))):
                media_links.append({"kind": "publisher_mp4", "url": url, "observed_field": "videoPlayerData.resources.h264.file", "alignment": "unverified"})
    external = player.get("external") or {}
    if isinstance(external, dict) and external.get("service") == "YouTube" and re.fullmatch(r"[A-Za-z0-9_-]{11}", str(external.get("code", ""))):
        media_links.append({"kind": "youtube", "url": "https://www.youtube.com/watch?v=" + external["code"], "observed_field": "videoPlayerData.external.code", "alignment": "unverified"})

    known = Counter(segment["speaker"] for segment in segments if segment["speaker"] is not None)
    bases = Counter(segment["speaker_basis"] for segment in segments)
    interval_seconds: Counter[str | None] = Counter()
    for segment in segments:
        if segment["end"] is not None:
            interval_seconds[segment["speaker"]] += segment["end"] - segment["start"]
    summary = {
        "schema_version": "transcript_acquisition_summary.v1", "extractor_version": VERSION,
        "source_id": source_id, "source_url": canonical_url, "input_sha256": input_sha256,
        "publisher_caption_sha256": caption_sha256,
        "publisher_talk_id": str(video.get("id", "")), "translation_id": str(translation.get("id", "")),
        "language": language, "source_timeline": TIMELINE,
        "publisher_duration_seconds": duration, "first_cue_seconds": segments[0]["start"],
        "last_cue_seconds": segments[-1]["start"], "paragraph_count": len(paragraph_rows),
        "cue_count": len(segments), "utterance_count": len(utterances),
        "word_count": sum(len(segment["text"].split()) for segment in segments),
        "known_speaker_cues": sum(known.values()), "unknown_speaker_cues": len(segments) - sum(known.values()),
        "known_speakers": dict(sorted(known.items())), "speaker_basis_counts": dict(sorted(bases.items())),
        "cue_interval_proxy_seconds_by_speaker": {
            name: round(interval_seconds[name], 3) for name in sorted(known)
        },
        "unassigned_cue_interval_proxy_seconds": round(interval_seconds[None], 3),
        "cue_interval_proxy_may_overlap": any(count > 1 for count in time_counts.values()),
        "duration_measurement_note": "Sums of caption onset-to-next-onset intervals, not measured speech duration; gaps and stage directions are included, and shared onsets may overlap.",
        "explicit_label_occurrences": sum(len(segment["explicit_labels"]) for segment in segments),
        "shared_timestamp_cues": sum(segment["shared_start_timestamp"] for segment in segments),
        "whisperx_exportable_cues": sum(segment["end"] is not None for segment in segments),
        "missing_end_cues": sum(segment["end"] is None for segment in segments),
        "speaker_identity_verified": False, "audio_alignment_verified": False,
        "review_status": "draft", "full_text_committed": False,
    }
    transcript = {
        "schema_version": "publisher_transcript.v1", "extractor": {"name": "ted_public_next_data", "version": VERSION},
        "source_id": source_id, "source_url": canonical_url, "source_timeline": TIMELINE,
        "source_start_seconds": 0, "language": language, "title": video.get("title"),
        "published_at": video.get("publishedAt"), "publisher_recorded_on": video.get("recordedOn"),
        "input_sha256": input_sha256, "publisher_caption_sha256": caption_sha256,
        "review_status": "draft", "confidence": "unverified_transcript",
        "speaker_attribution": "publisher_text_labels_only", "resolved_label_map": label_map,
        "speaker_policy": "No carry across paragraphs. Abbreviations resolve only when the full name appears as a label in this transcript. Mixed-speaker cues remain unknown; textual spans retain labels.",
        "timing_policy": "Only cue onsets are publisher timestamps. Ends are interval proxies from the next distinct onset or publisher duration; no word alignment or acoustic turn boundary is asserted.",
        "context_policy": "All questions, answers, unknown speakers, and stage directions are retained in source order. Paragraph and cue links preserve adjacent context without inferring question/answer roles.",
        "segments": segments, "paragraphs": paragraph_rows, "utterances": utterances,
        "behavioral_observations": [], "summary": summary,
    }
    whisperx = {
        "schema_version": "publisher_whisperx_draft.v1", "source_id": source_id,
        "source_timeline": TIMELINE, "source_start_seconds": 0, "language": language,
        "review_status": "draft", "confidence": "unverified_transcript",
        "speaker_attribution": "publisher_text_labels_only", "context_artifact": "transcript.json",
        "timing_policy": transcript["timing_policy"],
        "segments": [{key: segment[key] for key in (
            "id", "start", "end", "text", "speaker", "speaker_basis", "end_basis", "paragraph_id",
            "previous_cue_id", "next_cue_id", "review_status", "words",
        )} for segment in segments if segment["end"] is not None],
        "omitted_cue_ids": [segment["id"] for segment in segments if segment["end"] is None],
    }
    manifest = {
        "schema_version": "publisher_media.v1", "source_id": source_id, "source_url": canonical_url,
        "source_timeline": TIMELINE, "source_start_seconds": 0,
        "publisher_duration_seconds": duration, "input_sha256": input_sha256,
        "publisher_caption_sha256": caption_sha256,
        "media_links": media_links, "media_downloaded": False, "alignment_verified": False,
        "review_status": "draft", "artifacts": {},
        "note": "Publisher media inventory, not a prepared media.v1 clip. Downloading and verifying alignment are separate steps.",
    }
    return {"transcript": transcript, "whisperx": whisperx, "manifest": manifest, "summary": summary}


def write_bundle(bundle: dict[str, Any], output: str | Path) -> Path:
    """Write a new directory exclusively; never overwrite existing material."""
    path = Path(output).expanduser()
    if path.exists() or path.is_symlink():
        raise TranscriptError("Output already exists; choose a new output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    artifacts = {}
    for key in ("transcript", "whisperx", "summary"):
        name = key + ".json"
        raw = (json.dumps(bundle[key], ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        with (path / name).open("xb") as stream:
            stream.write(raw)
        artifacts[key] = {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    manifest = dict(bundle["manifest"], artifacts=artifacts)
    with (path / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return path.resolve()
