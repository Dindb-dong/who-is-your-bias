"""Read known publisher formats; raw source text stays in ignored local storage."""

from __future__ import annotations

import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from tools.transcripts.ted import extract_ted, read_document

MAX_BYTES = 16 * 1024 * 1024
LEX_PATHS = {"/elon-musk-4-transcript/", "/elon-musk-and-neuralink-team-transcript/"}


class AcquisitionError(ValueError):
    pass


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class _LexParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[dict] = []
        self.row: dict | None = None
        self.div_depth = 0
        self.field: str | None = None
        self.span_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set((attrs.get("class") or "").split())
        if tag == "div" and self.row is not None:
            self.div_depth += 1
        elif tag == "div" and "ts-segment" in classes:
            self.row = {"speaker": "", "stamp": "", "text": ""}
            self.div_depth = 1
        if self.row is not None and tag == "span":
            if self.field is not None:
                self.span_depth += 1
            else:
                matches = classes.intersection({"ts-name", "ts-timestamp", "ts-text"})
                if len(matches) == 1:
                    self.field = {"ts-name": "speaker", "ts-timestamp": "stamp", "ts-text": "text"}[matches.pop()]
                    self.span_depth = 1
        if self.row is not None and self.field == "text" and tag == "br":
            self.row["text"] += "\n"

    def handle_data(self, data):
        if self.row is not None and self.field is not None:
            self.row[self.field] += data

    def handle_endtag(self, tag):
        if self.row is not None and tag == "span" and self.field is not None:
            self.span_depth -= 1
            if self.span_depth == 0:
                self.field = None
        if self.row is not None and tag == "div":
            self.div_depth -= 1
            if self.div_depth == 0:
                self.rows.append({key: " ".join(value.split()) for key, value in self.row.items()})
                self.row = None
                self.field = None


def canonical_lex(raw: bytes, *, source_id: str, source_url: str) -> dict:
    if len(raw) > MAX_BYTES:
        raise AcquisitionError("Publisher input exceeds 16 MiB")
    parser = _LexParser()
    parser.feed(raw.decode("utf-8"))
    parser.close()
    if not parser.rows or len(parser.rows) > 30000 or parser.row is not None:
        raise AcquisitionError("No complete, bounded publisher transcript found")
    segments = []
    previous = -1
    for index, row in enumerate(parser.rows):
        match = re.fullmatch(r"\((\d{2}):(\d{2}):(\d{2})\)", row["stamp"])
        if not match or not row["text"]:
            raise AcquisitionError(f"Publisher row {index + 1} lacks a timestamp or text")
        hour, minute, second = map(int, match.groups())
        if minute >= 60 or second >= 60:
            raise AcquisitionError("Invalid publisher timestamp")
        start = hour * 3600 + minute * 60 + second
        if start > 86400:
            raise AcquisitionError("Publisher timestamps must be bounded")
        out_of_order = start < previous
        previous = start
        segments.append({"id": f"cue-{index + 1:05d}", "speaker": row["speaker"] or None, "text": row["text"],
                         "start": float(start), "end": None,
                         "speaker_basis": "explicit_publisher_label" if row["speaker"] else "unknown_continuation",
                         "timestamp_order_anomaly": out_of_order})
    for index, segment in enumerate(segments):
        following = segments[index + 1]["start"] if index + 1 < len(segments) and segments[index + 1]["start"] > segment["start"] else None
        segment["end"] = following
    content_hash = digest([{key: row[key] for key in ("id", "speaker", "text", "start")} for row in segments])
    return {"schema_version": "qa-publisher-transcript.v1", "source_id": source_id, "source_url": source_url,
            "source_timeline": "transcript_video", "language": "en", "segments": segments,
            "input_sha256": hashlib.sha256(raw).hexdigest(), "publisher_caption_sha256": content_hash,
            "speaker_attribution": "explicit_publisher_text_labels_only", "audio_alignment_verified": False}


def canonical_ted(raw: bytes, *, source_id: str) -> dict:
    document, _ = read_document(raw)
    return extract_ted(document, source_id=source_id, input_sha256=hashlib.sha256(raw).hexdigest())["transcript"]


def validate_source_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in {None, 443} or parsed.fragment:
        raise AcquisitionError("Use an ordinary HTTPS publisher URL")
    if parsed.hostname == "lexfridman.com" and parsed.path in LEX_PATHS and not parsed.query:
        return
    if parsed.hostname in {"ted.com", "www.ted.com"} and re.fullmatch(r"/talks/[a-z0-9_]+(?:/transcript)?/?", parsed.path) and parsed.query in {"", "view=transcript"}:
        return
    raise AcquisitionError("Only the reviewed TED and Lex publisher transcript URLs are supported")


class _Redirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        validate_source_url(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def acquire(source: dict, *, raw_path: Path | None = None) -> dict:
    url = source["url"]
    validate_source_url(url)
    if raw_path is not None:
        if not raw_path.is_file() or raw_path.stat().st_size > MAX_BYTES:
            raise AcquisitionError("Input must be a regular file no larger than 16 MiB")
        raw = raw_path.read_bytes()
    else:
        request = Request(url, headers={"User-Agent": "who-is-your-bias-qa-research/1.0", "Accept": "text/html"})
        with build_opener(_Redirects).open(request, timeout=45) as response:
            raw = response.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise AcquisitionError("Publisher input exceeds 16 MiB")
    return canonical_lex(raw, source_id=source["id"], source_url=url) if "lexfridman.com" in url else canonical_ted(raw, source_id=source["id"])
