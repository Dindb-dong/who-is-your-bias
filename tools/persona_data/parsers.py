"""Publisher-specific adapters. Unknown text never acquires a guessed speaker.

Original wording is emitted only to the caller's local directory. These adapters
do not equate publisher timestamps with acoustic boundaries.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path

from tools.qna.acquire import canonical_lex, canonical_ted


PARSER_VERSION = "original-dialogue-adapters.v1"


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest(value: object) -> str:
    return sha_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def clean(text: str) -> str:
    return " ".join(text.split())


class _Blocks(HTMLParser):
    """Retain paragraph text plus its exact publisher bold text, in source order."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[dict] = []
        self.tag: str | None = None
        self.parts: list[str] = []
        self.bold_parts: list[str] = []
        self.bold_depth = 0
        self.skip = 0

    def flush(self):
        if self.tag and clean("".join(self.parts)):
            self.rows.append({"tag": self.tag, "text": clean("".join(self.parts)),
                              "bold": clean("".join(self.bold_parts))})
        self.tag = None
        self.parts = []
        self.bold_parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "svg"}:
            self.skip += 1
        if not self.skip and tag in {"p", "h2", "h3"}:
            self.flush()
            self.tag = tag
        if tag in {"b", "strong"}:
            self.bold_depth += 1
        if tag == "br" and self.tag and not self.skip:
            self.parts.append(" ")
            if self.bold_depth:
                self.bold_parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "svg"}:
            self.skip = max(0, self.skip - 1)
        if tag in {"b", "strong"}:
            self.bold_depth = max(0, self.bold_depth - 1)
        if tag == self.tag:
            self.flush()

    def handle_data(self, data):
        if self.tag and not self.skip:
            self.parts.append(data)
            if self.bold_depth:
                self.bold_parts.append(data)


def html_blocks(raw: bytes) -> list[dict]:
    parser = _Blocks()
    parser.feed(raw.decode("utf-8"))
    parser.close()
    parser.flush()
    return parser.rows


def _row(index: int, speaker: str | None, text: str, basis: str, *, start=None, locator=None):
    return {"id": f"turn-{index:05d}", "speaker": speaker, "text": clean(text),
            "speaker_basis": basis, "start": start, "end": None,
            "publisher_locator": locator}


def _consecutive(rows: list[dict]) -> list[dict]:
    """Merge only contiguous same-speaker rows, never across an unknown block."""
    output: list[dict] = []
    for row in rows:
        row = dict(row)
        row["cue_ids"] = row.get("cue_ids", [row["id"]])
        if (output and row["speaker"] and output[-1]["speaker"] == row["speaker"]
                and output[-1]["speaker_basis"] == row["speaker_basis"]):
            previous = output[-1]
            previous["text"] = clean(previous["text"] + " " + row["text"])
            previous["cue_ids"].extend(row["cue_ids"])
            previous["end"] = row.get("end")
        else:
            output.append(row)
    return output


def _canonical_turns(document: dict) -> list[dict]:
    rows = []
    for segment in document["segments"]:
        speaker = segment.get("speaker")
        text = segment["text"]
        spans = segment.get("speaker_spans")
        if spans is not None:
            actual = [span for span in spans if span["kind"] == "caption_text"]
            if not actual or any(span["speaker"] != speaker for span in actual):
                speaker = None
            else:
                text = " ".join(span["text"] for span in actual)
        if segment.get("timestamp_order_anomaly"):
            speaker = None
        rows.append({"id": segment["id"], "speaker": speaker, "text": clean(text),
                     "speaker_basis": "explicit_publisher_text_labels" if speaker else "unknown",
                     "start": segment.get("start"), "end": segment.get("end"),
                     "publisher_locator": segment["id"]})
    return _consecutive(rows)


def _dwarkesh(raw: bytes) -> list[dict]:
    rows = html_blocks(raw)
    starts = [i for i, row in enumerate(rows) if row["tag"] == "h2" and row["text"] == "Transcript"]
    if len(starts) != 1:
        raise ValueError("Expected one Dwarkesh Transcript heading")
    names = {"Elon Musk", "Dwarkesh Patel", "John Collison"}
    speaker = None
    result = []
    for i, row in enumerate(rows[starts[0] + 1:], starts[0] + 1):
        text = row["text"]
        if row["tag"] == "p" and text in names and row["bold"] == text:
            speaker = text
        elif row["tag"] == "p" and speaker is not None:
            # Actual post content ends before the comments/subscribe section.
            if text.startswith(("Subscribe to", "Thanks for reading", "Discussion about this post", "Comments")):
                break
            if row["bold"] == text and text not in names:
                speaker = None
                result.append(_row(i, None, text, "unknown_bold_block"))
            else:
                result.append(_row(i, speaker, text, "publisher_name_heading"))
        elif row["tag"] in {"h2", "h3"}:
            # Chapter headings require a new name heading before more text.
            speaker = None
            result.append(_row(i, None, text, "chapter_heading"))
    return _consecutive(result)


def _yc(raw: bytes) -> list[dict]:
    rows = []
    pattern = re.compile(r"^(Sam Altman|Elon Musk)\s*\[(\d{1,2}):(\d{2})\]\s*[–—-]\s*(.+)$")
    for i, block in enumerate(html_blocks(raw)):
        match = pattern.match(block["text"])
        if not match:
            rows.append(_row(i, None, block["text"], "unattributed_block"))
            continue
        name, minute, second, text = match.groups()
        if int(second) >= 60:
            raise ValueError("Invalid YC publisher timestamp")
        rows.append(_row(i, name, text, "explicit_publisher_name_timestamp", start=int(minute)*60+int(second)))
    return _consecutive(rows)


def _wired(raw: bytes) -> list[dict]:
    names = {"Chris Anderson": "Chris Anderson", "Anderson": "Chris Anderson",
             "Elon Musk": "Elon Musk", "Musk": "Elon Musk"}
    rows = []
    anchored = set()
    for i, block in enumerate(html_blocks(raw)):
        match = re.match(r"^(Chris Anderson|Anderson|Elon Musk|Musk):\s*(.+)$", block["text"])
        if match:
            label, text = match.groups()
            if label in {"Chris Anderson", "Elon Musk"}:
                anchored.add(names[label])
            if names[label] not in anchored:
                raise ValueError("WIRED shortened speaker label precedes full-name anchor")
            rows.append(_row(i, names[label], text, "explicit_publisher_speaker_label"))
        else:
            # Pull quotes and surrounding narration are not original new turns.
            rows.append(_row(i, None, block["text"], "unattributed_or_pullquote"))
    return _consecutive(rows)


def _recode(raw: bytes) -> list[dict]:
    blocks = html_blocks(raw)
    start = next((i for i, row in enumerate(blocks) if row["text"].startswith("Kara Swisher:")), None)
    stop = next((i for i, row in enumerate(blocks) if row["text"] == "This article originally appeared on Recode.net."), None)
    if start is None or stop is None or stop <= start:
        raise ValueError("Recode transcript boundary markers changed")
    rows = []
    musk_anchor = False
    for i in range(start, stop):
        block = blocks[i]
        text = block["text"]
        if block["tag"] != "p":
            rows.append(_row(i, None, text, "chapter_heading"))
            continue
        if text.startswith("Kara Swisher:"):
            speaker, text = "Kara Swisher", text[len("Kara Swisher:"):].strip()
        elif text.startswith("Elon Musk:"):
            musk_anchor = True
            speaker, text = "Elon Musk", text[len("Elon Musk:"):].strip()
        elif clean(block["bold"]) == text:
            speaker = "Kara Swisher"
        elif musk_anchor and not block["bold"]:
            speaker = "Elon Musk"
        else:
            speaker = None
        rows.append(_row(i, speaker, text, "publisher_qa_typography_checked" if speaker else "ambiguous_typography"))
    return _consecutive(rows)


def _pdf_text(path: Path) -> str:
    # Argument vector, no shell; the supplied path is an acquired source file.
    result = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, check=True, timeout=60)
    return result.stdout.decode("utf-8")


def _lex_pdf(text: str) -> list[dict]:
    rows = []
    active = None
    for line in text.splitlines():
        text_line = line.strip()
        if (not text_line or text_line.startswith("This transcript was exported")
                or text_line.startswith("Transcript by Rev.com") or re.search(r"Page \d+ of \d+$", text_line)):
            continue
        match = re.match(r"^(Lex Fridman|Elon Musk):\s+(\d{2}):(\d{2})\s+(.+)$", text_line)
        if match:
            if active:
                rows.append(active)
            name, minute, second, value = match.groups()
            active = _row(len(rows), name, value, "explicit_pdf_speaker_timestamp", start=int(minute)*60+int(second))
        elif active:
            active["text"] = clean(active["text"] + " " + text_line)
    if active:
        rows.append(active)
    return _consecutive(rows)


def _ford_pdf(text: str) -> list[dict]:
    rows = []
    active = None
    raw_clock = None
    for line in text.splitlines():
        value = line.strip()
        if not value or re.fullmatch(r"PG\.\d+", value) or value.startswith("MEDIA ID:"):
            continue
        if value in {"BARRY HURD:", "ELON MUSK:"}:
            if active:
                rows.append(active)
            active = _row(len(rows), "Barry Hurd" if value == "BARRY HURD:" else "Elon Musk", "", "explicit_pdf_speaker_label")
            raw_clock = None
        elif active:
            stamp = re.match(r"^(\d{2}:\d{2}:\d{2};\d{2})\s*(.*)$", value)
            if stamp:
                raw_clock = raw_clock or stamp[1]
                value = stamp[2]
                active["publisher_locator"] = raw_clock
            active["text"] = clean(active["text"] + " " + value)
    if active:
        rows.append(active)
    # SMPTE frame rate/recording offsets are unknown; no numeric media time.
    return _consecutive(rows)


def parse(source: dict, artifact: Path) -> dict:
    raw = artifact.read_bytes()
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("Source artifact exceeds 16 MiB")
    identifier = source["id"]
    timeline = None
    metadata = {}
    if identifier.startswith("ted-"):
        canonical = canonical_ted(raw, source_id=identifier)
        rows = _canonical_turns(canonical)
        content_hash = canonical["publisher_caption_sha256"]
        timeline = canonical["source_timeline"]
    elif identifier in {"lex-400-2023", "lex-438-2024"}:
        canonical = canonical_lex(raw, source_id=identifier, source_url=source["url"])
        rows = _canonical_turns(canonical)
        content_hash = canonical["publisher_caption_sha256"]
        timeline = canonical["source_timeline"]
    else:
        adapters = {"dwarkesh-2026": _dwarkesh, "yc-2016": _yc, "wired-2012": _wired, "recode-2018": _recode}
        if identifier in adapters:
            rows = adapters[identifier](raw)
        elif identifier == "lex-049-2019":
            rows = _lex_pdf(_pdf_text(artifact))
        elif identifier == "henry-ford-2008":
            rows = _ford_pdf(_pdf_text(artifact))
        elif identifier in {"senate-launch-2014", "house-space-cooperation-2004", "senate-future-launch-2004",
                            "house-commercial-space-2005", "senate-advanced-energy-2007", "house-commercial-crew-2011"}:
            from .congress import parse_congress
            metadata = parse_congress(raw, identifier)
            rows = metadata.pop("turns")
        else:
            raise ValueError("No reviewed original-dialogue adapter; narrative/excerpts are not fabricated Q&A")
        content_hash = digest(rows)
        if identifier in {"yc-2016", "lex-049-2019"}:
            timeline = "publisher_transcript_video"
    return {"schema_version": "persona-original-turns.v1", "parser_version": PARSER_VERSION,
            "source_id": identifier, "input_sha256": hashlib.sha256(raw).hexdigest(),
            "transcript_sha256": content_hash, "source_timeline": timeline, "turns": rows,
            "audio_alignment_verified": False, **metadata}
