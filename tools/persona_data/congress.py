"""Reviewed GovInfo oral-discussion snapshots, separate from written testimony.

Only the six exact publisher artifacts reviewed for this adapter are accepted.
The frozen source registry is unchanged. New publisher bytes require another
explicitly reviewed adapter version; no speaker or section boundary is guessed.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path


ADAPTER_VERSION = "govinfo-oral-discussion.v1"


@dataclass(frozen=True)
class Hearing:
    sha256: str
    start: str
    stop: str
    committee: tuple[str, ...]
    witnesses: tuple[str, ...]
    headings: tuple[str, ...] = ()


HEARINGS = {
    "senate-launch-2014": Hearing(
        "50d63a0e68e16c9b8850fc9d8805698971789ea65820b399b0be619d7289e7ce",
        "Senator Durbin. Dr. Pace, thank you very much. I think you",
        "ADDITIONAL COMMITTEE QUESTION",
        ("Senator Durbin", "Senator Cochran", "Senator Feinstein", "Senator Shelby"),
        ("Dr. Pace", "Mr. Gass", "Mr. Musk", "Ms. Chaplain"),
        ("RD-180 ENGINE", "SPACEX LAUNCHES", "EELV LAUNCH CAPABILITY ACCOUNT (ELC)",
         "EELV COMPETITION", "AIR FORCE CERTIFICATION", "AUDITING AND OVERHEAD COSTS",
         "CONTRACT COST", "SUCCESSFUL LAUNCHES", "BUDGET CONSTRAINTS AND COMPETITION"),
    ),
    "house-space-cooperation-2004": Hearing(
        "d10833cf833e41f5abdd1afef678b2fa0b4f2fcdbc027bb565e7b9bdabc55b68",
        "Chairman Rohrabacher. Thank you very much, Mr. Musk.",
        "[Whereupon, at 3:50 p.m., the Subcommittee was adjourned.]",
        ("Chairman Rohrabacher", "Mr. Feeney", "Mr. Lampson"),
        ("Dr. Sega", "Mr. Musk", "Ms. Poniatowski", "Rear Admiral Steidle"),
        ("Impacts of the President's Space Exploration Initiative on",
         "NASA's Space Launch Initiative", "NASA Contribution to the National Aerospace Initiative",
         "Future of Space Launch Initiative Projects", "National Security Interests in Space Exploration",
         "NASA and DOD Cross-Certification of Launch Vehicles", "Difference Between Military and Civil Spacecraft",
         "Crew Exploration Vehicle", "Cooperation in Near-Earth Objects Detection", "Role of Industry",
         "Impediments to NASA-DOD Cooperation", "National Space Transportation Policy",
         "NASA-DOD Test Facilities", "Cross-Certification of Launch Vehicles",
         "Review of Space Treaties Affecting DOD and Commercial Use of", "the Moon"),
    ),
    "senate-future-launch-2004": Hearing(
        "afe10e5b07293c05edf212f1e29df22b0e02f04fdf5488d16cf60102752b4700",
        "Senator Brownback. Thank you very much.",
        "[Whereupon, at 4:25 p.m., the hearing was adjourned.]",
        ("Senator Brownback",), ("Dr. Karas", "Mr. Hickman", "Mr. Kahn", "Mr. Musk"),
    ),
    "house-commercial-space-2005": Hearing(
        "58d2de1eefb2966c940b89a8ea71af9a1a2448276b3d0be85ff69e1e033db1b4",
        "Chairman Calvert. Thank you. Thank you for your testimony.",
        "[Whereupon, at 11:45 a.m., the Subcommittee was adjourned.]",
        ("Chairman Calvert", "Mr. Rohrabacher"),
        ("Dr. Macauley", "Mr. Demisch", "Mr. Musk", "Mr. Vinter"),
        ("Emerging Space-based Markets",),
    ),
    "senate-advanced-energy-2007": Hearing(
        "b0133085394b7e04c7bf1831c8c24bb0b84b6f51e7c40570a78bd3fd92e17232",
        "The Chairman. Thank you, and thank you all for being here.",
        "[Whereupon, at 11:40 a.m., the hearing was adjourned.]",
        ("The Chairman", "Senator Bunning", "Senator Cantwell", "Senator Corker", "Senator Craig",
         "Senator Domenici", "Senator Salazar", "Senator Thomas"),
        ("Mr. Denniston", "Mr. Liebreich", "Mr. Musk", "Mr. Peters", "Mr. Reicher"),
    ),
    "house-commercial-crew-2011": Hearing(
        "c2bee57ee1f9d8fffadca86fa231b032b6669e62247e4034499a91a5dbd09954",
        "Chairman Hall. And I thank you, and I thank all of you for",
        "Okay. At this time we will get our very patient second",
        ("Chairman Hall", "Mr. Brooks", "Mr. Clarke", "Mr. Hultgren", "Mr. Palazzo", "Mr. Rohrabacher",
         "Mr. Tonko", "Mrs. Adams", "Ms. Johnson", "Ms. Lofgren"),
        ("Colonel Lindsey", "Colonel Precourt", "Dr. Sowers", "Mr. Elbon", "Mr. Musk"),
    ),
}

SOURCE_SCOPE = (
    "Original live oral discussion from the registered congressional hearing; "
    "only directly adjacent, explicitly labeled committee questions and Elon Musk responses are projected. "
    "Opening statements, prepared testimony, biographies, other panels and post-hearing written answers "
    "are excluded. Other witnesses retain their own labels in canonical context and are not attributed "
    "to Musk. The printed transcript is not acoustically verified; historical witness assertions and "
    "forecasts remain attributed to their hearing date."
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class _Pre(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.count = 0
        self.active = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "pre":
            if self.active:
                raise ValueError("Nested GovInfo transcript block")
            self.count += 1
            self.active = True

    def handle_endtag(self, tag):
        if tag == "pre":
            self.active = False

    def handle_data(self, value):
        if self.active:
            self.parts.append(value)


def _pre_text(raw: bytes) -> str:
    parser = _Pre()
    parser.feed(raw.decode("utf-8"))
    parser.close()
    if parser.count != 1 or parser.active:
        raise ValueError("Expected one complete GovInfo preformatted transcript")
    return "".join(parser.parts)


LABEL = re.compile(
    r"^    ((?:(?:Mr|Mrs|Ms|Dr)\. [A-Z][A-Za-z'-]+|"
    r"(?:Chairman|Senator|Colonel|Admiral|Rear Admiral) [A-Z][A-Za-z'-]+|The Chairman))\.(.*)$"
)
INSERTIONS = {
    "[The information follows:]", "[GRAPHIC(S) NOT AVAILABLE IN TIFF FORMAT]",
    "[The information appears in Appendix II:]",
}


def _oral_turns(text: str, hearing: Hearing) -> tuple[list[dict], dict]:
    """Parse only the explicitly reviewed discussion window and participants."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == hearing.start]
    stops = [i for i, line in enumerate(lines) if line.strip() == hearing.stop]
    if len(starts) != 1 or len(stops) != 1 or starts[0] >= stops[0]:
        raise ValueError("Reviewed GovInfo oral-discussion boundaries changed or repeated")
    start, stop = starts[0], stops[0]
    if not re.search(r"STATEMENT OF (?:MR\. )?ELON MUSK", " ".join(lines[:start])):
        raise ValueError("Musk speaker label lacks its full-name testimony anchor")
    names = {*hearing.committee, *hearing.witnesses}
    rows, excluded = [], []
    active = None

    def flush():
        nonlocal active
        if active is not None:
            active["text"] = " ".join(active.pop("parts")).strip()
            delegated_name_only = bool(re.fullmatch(r"(?:All right\. )?(?:Mr\. Musk|Elon)\?", active["text"]))
            active["question_eligible"] = (active["publisher_speaker_label"] in hearing.committee
                                           and "?" in active["text"] and not delegated_name_only)
            # The next speaker interrupted the printed answer. Preserve the
            # complete attributed turn, but do not label it a complete response.
            active["response_complete"] = not active["text"].endswith(("--", "—"))
            rows.append(active)
            active = None

    for index in range(start, stop):
        line, number = lines[index], index + 1
        value = line.strip()
        if not value:
            continue
        if re.match(r"(?:Prepared Statement|Biography for|Answers to Post-Hearing|Responses? to Written|APPENDIX\b)", value, re.I):
            raise ValueError("Written material crossed the reviewed oral-discussion boundary")
        if value in hearing.headings or value == "[Laughter.]":
            excluded.append({"pre_line": number, "kind": "editorial_heading" if value in hearing.headings else "stage_direction",
                             "text_sha256": _sha(value.encode())})
            continue
        if value in INSERTIONS:
            flush()
            identifier = f"turn-l{number:05d}"
            rows.append({"id": identifier, "cue_ids": [identifier], "speaker": None,
                         "speaker_basis": "unspoken_written_insertion", "text": value,
                         "start": None, "end": None, "publisher_locator": f"govinfo-pre-line:{number}",
                         "question_eligible": False})
            excluded.append({"pre_line": number, "kind": "written_insertion_boundary", "text_sha256": _sha(value.encode())})
            continue
        match = LABEL.match(line)
        if match:
            flush()
            label, words = match.groups()
            if label not in names:
                raise ValueError("Unreviewed speaker label in GovInfo discussion")
            identifier = f"turn-l{number:05d}"
            active = {"id": identifier, "cue_ids": [identifier],
                      "speaker": "Elon Musk" if label == "Mr. Musk" else label,
                      "publisher_speaker_label": label, "speaker_basis": "explicit_congressional_hearing_label",
                      "parts": [" ".join(words.split())], "start": None, "end": None,
                      "publisher_locator": f"govinfo-pre-line:{number}"}
        elif active is not None and not value.startswith("[") and len(line) - len(line.lstrip()) <= 4:
            active["parts"].append(" ".join(value.split()))
        else:
            raise ValueError("Unattributed or unreviewed material in GovInfo oral discussion")
    flush()
    if not any(row["speaker"] == "Elon Musk" for row in rows):
        raise ValueError("Reviewed GovInfo discussion has no attributed Musk response")
    return rows, {
        "window_pre_lines": {"first_included": start + 1, "last_included": stop, "stop_exclusive": stop + 1},
        "window_sha256": _sha("\n".join(lines[start:stop]).encode()),
        "excluded_prefix_sha256": _sha("\n".join(lines[:start]).encode()),
        "excluded_suffix_sha256": _sha("\n".join(lines[stop:]).encode()),
        "excluded_blocks": excluded,
        "projection_rule": "adjacent_direct_committee_question_with_question_mark_then_labeled_musk_turn",
        "interrupted_answers": "retained_as_excerpts_ineligible_for_style",
        "text_normalization": "collapse_whitespace_only; retain_published_spelling_and_hard_wrap_hyphens",
        "written_testimony_and_appendices_projected": False,
        "human_review_completed": False,
    }


def parse_congress(raw: bytes, source_id: str) -> dict:
    hearing = HEARINGS.get(source_id)
    if hearing is None:
        raise ValueError("No reviewed congressional source adapter")
    if _sha(raw) != hearing.sha256:
        raise ValueError("GovInfo publisher snapshot changed; review a new immutable adapter version")
    rows, audit = _oral_turns(_pre_text(raw), hearing)
    return {"turns": rows, "adapter_version": ADAPTER_VERSION,
            "adapter_sha256": _sha(Path(__file__).read_bytes()), "source_scope": SOURCE_SCOPE,
            "extraction_audit": audit}
