"""Synthetic publisher-shaped fixtures; actual artifacts remain local and hashed."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from tools.persona_data.congress import ADAPTER_VERSION, HEARINGS, Hearing, _oral_turns, parse_congress
from tools.persona_data.parsers import PARSER_VERSION, parse
from tools.persona_data.pipeline import extract_exchanges, project_records, runtime_source


SOURCE_ID = "senate-future-launch-2004"
START = "Senator Example. Why is the first approach useful?"
STOP = "[Whereupon, the hearing was adjourned.]"


def document(body=None):
    return """<html><pre>
STATEMENT OF ELON MUSK, WITNESS
    Mr. Musk. This opening statement must never become a Q&amp;A answer.
Prepared Statement of Elon Musk
    This is separately submitted written testimony.
""" + (body or """    Senator Example. Why is the first approach useful?
    Mr. Musk. The complete answer includes a qualification about its present limits.
    A second paragraph preserves the original explanation without truncation.
    Senator Example. Is that a forecast or an achieved result?
    Mr. Musk. It is a forecast; we cannot yet claim that it happened.
""") + """    [Whereupon, the hearing was adjourned.]
APPENDIX
    Senator Example. This written question is not live dialogue?
    Mr. Musk. This written answer must never enter the oral projection.
</pre></html>"""


def config(raw):
    return Hearing(hashlib.sha256(raw).hexdigest(), START, STOP,
                   ("Senator Example",), ("Mr. Musk", "Dr. Other"), ("TECHNICAL SECTION",))


def source():
    return {"id": SOURCE_ID, "source_event_id": SOURCE_ID, "event_family_id": SOURCE_ID,
            "split": "train", "url": "https://www.govinfo.gov/content/pkg/synthetic/html/synthetic.htm",
            "published_at": "2004-05-05", "recorded_at": "2004-05-05",
            "source_type": "prepared_congressional_testimony", "attribution": "Inventory prepared statement only."}


def parsed(monkeypatch, tmp_path, body=None):
    raw = document(body).encode()
    monkeypatch.setitem(HEARINGS, SOURCE_ID, config(raw))
    artifact = tmp_path / "publisher.html"
    artifact.write_bytes(raw)
    return parse(source(), artifact)


def exchanges(transcript):
    return extract_exchanges(source(), transcript, acquired_at="2026-11-11T00:00:00Z")


def test_prepared_opening_and_written_appendix_never_become_live_qa(monkeypatch, tmp_path):
    transcript = parsed(monkeypatch, tmp_path)
    rows, rejected = exchanges(transcript)
    assert len(rows) == 2 and not rejected
    texts = " ".join(row["question_answer"]["answer"] for row in rows)
    assert "second paragraph" in texts
    assert "opening statement" not in texts and "written answer" not in texts
    assert "forecast" in rows[1]["question_answer"]["answer"]
    assert transcript["parser_version"] == PARSER_VERSION
    assert transcript["adapter_version"] == ADAPTER_VERSION
    assert len(transcript["adapter_sha256"]) == 64
    assert transcript["source_timeline"] is None
    assert transcript["audio_alignment_verified"] is False
    assert transcript["extraction_audit"]["written_testimony_and_appendices_projected"] is False
    assert ADAPTER_VERSION in rows[0]["transformation_history"]


def test_other_witness_is_retained_but_not_invented_as_committee_question(monkeypatch, tmp_path):
    transcript = parsed(monkeypatch, tmp_path, """    Senator Example. Why is the first approach useful?
    Dr. Other. My company makes an independent claim; can I add a point?
    Mr. Musk. I disagree with that other witness on this specific matter.
    Senator Example. What do you independently expect next year?
    Mr. Musk. We expect a test next year, but its success is not certain.
""")
    rows, rejected = exchanges(transcript)
    assert len(rows) == 1
    assert rows[0]["question_answer"]["questioner"] == "Senator Example"
    assert "independently expect" in rows[0]["question_answer"]["question"]
    assert rejected[0]["reason"] == "not_direct_committee_question"
    assert transcript["turns"][1]["speaker"] == "Dr. Other"


def test_delegation_without_the_original_question_is_not_a_complete_pair(monkeypatch, tmp_path):
    transcript = parsed(monkeypatch, tmp_path, """    Senator Example. Why is the first approach useful?
    Dr. Other. My own answer comes first and must keep its own speaker.
    Senator Example. Mr. Musk?
    Mr. Musk. I agree with the preceding witness on this particular point.
""")
    rows, rejected = exchanges(transcript)
    assert rows == []
    assert rejected[0]["reason"] == "not_direct_committee_question"


def test_interrupted_answer_keeps_all_printed_words_but_is_not_style_complete(monkeypatch, tmp_path):
    transcript = parsed(monkeypatch, tmp_path, """    Senator Example. Why is the first approach useful?
    Mr. Musk. It is useful in this particular situation, although we must distinguish--
    Senator Example. Please stop so another witness can speak.
    Dr. Other. My views differ and cannot be attached to the preceding answer.
""")
    rows, rejected = exchanges(transcript)
    assert len(rows) == 1 and not rejected
    assert rows[0]["question_answer"]["answer"].endswith("distinguish--")
    assert rows[0]["question_answer"]["completeness"] == "excerpt"
    projections, _ = project_records(rows, {SOURCE_ID: source()})
    assert len(projections["factual"]) == 1
    assert projections["text_style"] == []
    assert projections["aligned_speech"] == [] and projections["visible_behavior"] == []


def test_audited_editorial_heading_does_not_become_spoken_text(monkeypatch, tmp_path):
    transcript = parsed(monkeypatch, tmp_path, """    Senator Example. Why is the first approach useful?
                      TECHNICAL SECTION
    Does it retain its original qualification?
    Mr. Musk. Yes, the qualified explanation remains part of the original answer.
    [Laughter.]
    Senator Example. Thank you.
""")
    rows, _ = exchanges(transcript)
    assert "TECHNICAL SECTION" not in rows[0]["question_answer"]["question"]
    assert "original qualification" in rows[0]["question_answer"]["question"]
    assert "Laughter" not in rows[0]["question_answer"]["answer"]
    assert [entry["kind"] for entry in transcript["extraction_audit"]["excluded_blocks"]] == ["editorial_heading", "stage_direction"]


def test_written_insertion_breaks_completeness_instead_of_becoming_speech(monkeypatch, tmp_path):
    transcript = parsed(monkeypatch, tmp_path, """    Senator Example. Why is the first approach useful?
    Mr. Musk. Here is the spoken portion with enough words for factual projection.
    [The information follows:]
    [GRAPHIC(S) NOT AVAILABLE IN TIFF FORMAT]
    Senator Example. Thank you.
""")
    rows, _ = exchanges(transcript)
    assert rows[0]["question_answer"]["completeness"] == "excerpt"
    assert "GRAPHIC" not in rows[0]["question_answer"]["answer"]


@pytest.mark.parametrize("change, error", [
    (lambda text: text.replace(STOP, "A different closing boundary"), "boundaries"),
    (lambda text: text.replace(STOP, STOP + "\n" + STOP), "boundaries"),
    (lambda text: text.replace("STATEMENT OF ELON MUSK", "UNATTRIBUTED TESTIMONY"), "full-name"),
    (lambda text: text.replace("Mr. Musk. The complete", "Dr. Unreviewed. The complete"), "Unreviewed speaker"),
    (lambda text: text.replace("    A second paragraph", "    Prepared Statement of Another Witness\n    A second paragraph"), "Written material"),
])
def test_structural_uncertainty_fails_closed(change, error):
    raw = document().encode()
    text = document().split("<pre>", 1)[1].split("</pre>", 1)[0]
    with pytest.raises(ValueError, match=error):
        _oral_turns(change(text), config(raw))


def test_unreviewed_publisher_bytes_and_unregistered_source_are_rejected():
    with pytest.raises(ValueError, match="snapshot changed"):
        parse_congress(document().encode(), SOURCE_ID)
    with pytest.raises(ValueError, match="No reviewed"):
        parse_congress(document().encode(), "unregistered-congressional-volume")


def test_runtime_source_preserves_inventory_scope_and_reports_actual_oral_projection(monkeypatch, tmp_path):
    transcript = parsed(monkeypatch, tmp_path)
    result = runtime_source(source(), transcript, acquired_at="2026-11-11T00:00:00Z")
    assert result["summary_inventory_attribution"] == source()["attribution"]
    assert result["source_type"] == "congressional_hearing_transcript"
    assert "live oral discussion" in result["attribution"]
    assert result["transcript_adapter_version"] == ADAPTER_VERSION
    assert result["source_artifact_sha256"] == transcript["input_sha256"]
    assert result["transcript_sha256"] == transcript["transcript_sha256"]
