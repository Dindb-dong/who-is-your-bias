"""Bounded Jensen publisher adapters; no guessed speaker turns or media clocks."""
from __future__ import annotations

import hashlib
import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path

from .parsers import _consecutive, _row, _canonical_turns, _pdf_text, canonical_lex, clean, digest, html_blocks

VERSION = "jensen-publisher-dialogue.v2"


def colon_turns(blocks: list[dict], names: dict[str, str], *, start: str | None = None,
                stops: tuple[str, ...] = (), stop_at_heading: bool = True) -> list[dict]:
    """Explicit speaker labels plus contiguous publisher paragraphs, bounded body."""
    rows, active = [], None
    begun = start is None
    pattern = re.compile(r"^(" + "|".join(re.escape(x) for x in sorted(names, key=len, reverse=True)) + r"):\s*(.*)$")
    for i, block in enumerate(blocks):
        text = block["text"]
        if not begun:
            if not text.startswith(start):
                continue
            begun = True
        if any(text.startswith(stop) for stop in stops):
            break
        if block["tag"] != "p":
            if active and stop_at_heading:
                break
            active = None
            continue
        match = pattern.match(text)
        if match:
            name, value = match.groups()
            active = names[name]
            if value:
                rows.append(_row(i, active, value, "explicit_publisher_speaker_label"))
        elif active:
            # A new unregistered speaker-like label is an attribution barrier.
            if re.match(r"^[A-Z][\w .'-]{1,50}:\s", text):
                active = None
                rows.append(_row(i, None, text, "unknown_speaker_label"))
            else:
                rows.append(_row(i, active, text, "explicit_publisher_speaker_label"))
    if not any(row["speaker"] == "Jensen Huang" for row in rows):
        raise ValueError("No explicitly attributed Jensen turns in bounded publisher body")
    return _consecutive(rows)


def dwarkesh(raw: bytes) -> list[dict]:
    blocks = html_blocks(raw)
    starts = [i for i, block in enumerate(blocks) if block["tag"] == "h2" and block["text"] == "Transcript"]
    if len(starts) != 1:
        raise ValueError("Expected one Dwarkesh transcript heading")
    names = {"Dwarkesh Patel", "Jensen Huang"}
    rows, speaker = [], None
    for i, block in enumerate(blocks[starts[0]+1:], starts[0]+1):
        text = block["text"]
        if text.startswith(("Ready for more?", "Subscribe to", "Thanks for reading", "Discussion about this post")):
            break
        if block["tag"] == "p" and text in names and block["bold"] == text:
            speaker = text
        elif block["tag"] in {"h2", "h3"}:
            speaker = None
            rows.append(_row(i, None, text, "chapter_heading"))
        elif block["tag"] == "p" and speaker:
            if block["bold"] == text:
                speaker = None
                rows.append(_row(i, None, text, "unknown_bold_block"))
            else:
                rows.append(_row(i, speaker, text, "publisher_name_heading"))
    return _consecutive(rows)


def strange_loop(raw: bytes) -> tuple[list[dict], list[dict]]:
    rows, corrections = [], []
    speaker, locator = None, None
    pattern = re.compile(r"^(Joel|Jensen) \| (\d{2}:\d{2}:\d{2}:\d{2}) - (\d{2}:\d{2}:\d{2}:\d{2})$")
    blocks = html_blocks(raw)
    for i, block in enumerate(blocks):
        text = block["text"]
        match = pattern.match(text)
        if match:
            name, begin, end = match.groups()
            speaker = {"Joel": "Joel Hellermark", "Jensen": "Jensen Huang"}[name]
            locator = begin + " - " + end
            continue
        if speaker and (text in {"🍪", "Cookies"} or text.startswith("We and selected partners")):
            break
        if speaker and block["tag"] == "p":
            # Publisher accidentally appends its editorial taxonomy to the last
            # response. Remove that exact terminal block, retaining a hash audit.
            boundary = text.find(" AI - Multimodality AI - physical feedback model")
            if boundary != -1:
                original = text
                text = text[:boundary].rstrip()
                corrections.append({"block": i, "reason": "publisher_trailing_topic_tags",
                                    "before_sha256": hashlib.sha256(original.encode()).hexdigest(),
                                    "after_sha256": hashlib.sha256(text.encode()).hexdigest()})
            row = _row(i, speaker, text, "explicit_publisher_name_timecode", locator=locator)
            # Unknown timecode frame-rate: keep literal locator, never seconds.
            rows.append(row)
    return _consecutive(rows), corrections


class _YCData(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.transcripts: list[str] = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {"data-props", "data-page"} and value:
                try:
                    doc = json.loads(value)
                except json.JSONDecodeError:
                    continue
                def visit(node):
                    if isinstance(node, dict):
                        if isinstance(node.get("transcript"), str):
                            self.transcripts.append(node["transcript"])
                        for child in node.values():
                            visit(child)
                    elif isinstance(node, list):
                        for child in node:
                            visit(child)
                visit(doc)


def yc(raw: bytes) -> list[dict]:
    parser = _YCData()
    parser.feed(raw.decode())
    texts = list(dict.fromkeys(parser.transcripts))
    if len(texts) != 1:
        raise ValueError("Expected exactly one publisher embedded YC transcript")
    pattern = re.compile(r"(?m)^(Garry Tan(?: \(host\))?|Jensen Huang(?: \(guest\))?):\s*")
    matches = list(pattern.finditer(texts[0]))
    rows = []
    for i, match in enumerate(matches):
        text = texts[0][match.end():matches[i+1].start() if i+1<len(matches) else len(texts[0])]
        name = "Jensen Huang" if match.group(1).startswith("Jensen") else "Garry Tan"
        rows.append(_row(i, name, text, "explicit_publisher_embedded_speaker_label"))
    if not rows:
        raise ValueError("YC transcript contains no registered speaker labels")
    return _consecutive(rows)


def goldman(text: str) -> list[dict]:
    # PDF page furniture is excluded before label segmentation; no speech clock.
    text = text.split("The opinions and views expressed in this program", 1)[0]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    filtered = []
    for line in lines:
        if (re.fullmatch(r"\d+", line) or line == "Goldman Sachs Talks" or line.startswith(("©", "All rights reserved", "Date of recording:"))):
            continue
        filtered.append(line)
    joined = "\n".join(filtered)
    pattern = re.compile(r"(?m)^(David Solomon|Jensen Huang|DAVID SOLOMON|JENSEN HUANG):\s*")
    matches = list(pattern.finditer(joined));rows=[]
    for i, match in enumerate(matches):
        text = joined[match.end():matches[i+1].start() if i+1<len(matches) else len(joined)]
        if "This transcript" in text:
            text = text.split("This transcript",1)[0]
        name = "Jensen Huang" if match.group(1).lower().startswith("jensen") else "David Solomon"
        rows.append(_row(i,name,text,"explicit_pdf_speaker_label"))
    if not rows:
        raise ValueError("Goldman PDF contains no registered speaker labels")
    return _consecutive(rows)


def parse_jensen(source: dict, artifact: Path) -> dict:
    raw = artifact.read_bytes()
    if len(raw) > 16*1024*1024:
        raise ValueError("Source artifact exceeds 16 MiB")
    identifier = source["id"]
    timeline, corrections = None, []
    if identifier == "lex-494":
        document = canonical_lex(raw, source_id=identifier, source_url=source["url"])
        rows = _canonical_turns(document)
        timeline = document["source_timeline"]
    elif identifier == "dwarkesh-2026":
        rows = dwarkesh(raw)
    elif identifier == "acquired-2023":
        rows = colon_turns(html_blocks(raw), {"Ben":"Ben Gilbert","David":"David Rosenthal","Jensen":"Jensen Huang"}, start="Jensen, this is Acquired.")
    elif identifier == "stripe-sessions-2024":
        rows = colon_turns(html_blocks(raw), {"PATRICK COLLISON":"Patrick Collison","JENSEN HUANG":"Jensen Huang"},
                           start="PATRICK COLLISON:", stops=("Join us ","Something went wrong"))
    elif identifier == "stanford-vftt-2024":
        rows = colon_turns(html_blocks(raw), {"Shantam Jain":"Shantam Jain","Jensen Huang":"Jensen Huang"},
                           start="Shantam Jain: Jensen, this is such an honor.",stops=("Shantam Jain: You’ve been listening",))
    elif identifier == "csis-2025":
        rows = colon_turns(html_blocks(raw), {"John J. Hamre":"John J. Hamre","Dr. Hamre":"John J. Hamre",
                           "Jensen Huang":"Jensen Huang","Mr. Huang":"Jensen Huang"},start="John J. Hamre:",stops=("(END.)",))
    elif identifier == "strange-loop-2025":
        rows, corrections = strange_loop(raw)
    elif identifier == "yc-startup-school-2026":
        rows = yc(raw)
    elif identifier == "goldman-sachs-2024":
        rows = goldman(_pdf_text(artifact))
    else:
        raise ValueError("No reviewed Jensen adapter for this publisher/event")
    if not any(row["speaker"] == "Jensen Huang" for row in rows):
        raise ValueError("Publisher content does not contain attributed Jensen speech")
    return {"schema_version":"persona-original-turns.v1","parser_version":VERSION,
            "source_id":identifier,"input_sha256":hashlib.sha256(raw).hexdigest(),
            "transcript_sha256":digest(rows),"source_timeline":timeline,"turns":rows,
            "audio_alignment_verified":False,"corrections":corrections,
            "adapter_version":VERSION,"adapter_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "extraction_audit":{"speaker_basis":"explicit_publisher_labels_only","corrections":corrections,
                                "numeric_speech_clock_verified":False}}
