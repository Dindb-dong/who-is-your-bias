"""Synthetic-only fixtures for publisher provenance and speaker boundaries."""

from __future__ import annotations

import hashlib
import json

import pytest

from bias.ingest import import_transcript, parse_whisperx
from tools.transcripts.__main__ import main
from tools.transcripts.ted import (
    TranscriptError,
    _TEDRedirects,
    extract_ted,
    read_document,
    validate_ted_url,
    write_bundle,
)


def document(paragraphs=None, *, duration=10):
    if paragraphs is None:
        paragraphs = [
            [(0, "Chris Anderson: What should we build?"), (1000, "Why is that useful?")],
            [(2000, "Elon Musk: A small example."), (3000, "It could help people.")],
            [(4000, "This paragraph has no speaker label."), (5000, "EM: Another example."),
             (6000, "That is my answer.CA: What happens next?"), (7000, "Please explain.")],
            [(8000, "(Applause)")],
        ]
    player = {
        "resources": {
            "hls": {"stream": "https://hls.ted.com/example/manifest.m3u8"},
            "h264": [{"file": "https://py.tedcdn.com/example.mp4"}],
        },
        "external": {"service": "YouTube", "code": "abcdefghijk"},
    }
    if duration is not None:
        player["duration"] = duration
    return {"props": {"pageProps": {
        "videoData": {"id": "123", "title": "Elon Musk subject title is not speaker evidence",
                      "canonicalUrl": "https://www.ted.com/talks/synthetic_example",
                      "publishedAt": "2020-01-01T12:00:00Z", "videoPlayerData": player},
        "transcriptData": {"translation": {"id": "456", "language": {"internalLanguageCode": "en"},
            "paragraphs": [{"cues": [{"time": t, "text": text} for t, text in paragraph]} for paragraph in paragraphs]}},
    }}}


def extract(data=None):
    return extract_ted(data or document(), source_id="synthetic", input_sha256="a" * 64)


def test_questions_answers_and_unknown_context_are_preserved():
    bundle = extract()
    transcript = bundle["transcript"]
    rows = transcript["segments"]
    assert len(rows) == 9
    assert rows[0]["speaker"] == "Chris Anderson"
    assert rows[1]["speaker_basis"] == "same_paragraph_label_continuation"
    assert rows[2]["speaker"] == rows[3]["speaker"] == "Elon Musk"
    assert rows[4]["speaker"] is None  # Never carry across an unlabeled paragraph.
    assert rows[5]["speaker"] == "Elon Musk"
    assert rows[7]["speaker"] == "Chris Anderson"
    assert transcript["paragraphs"][1]["previous_paragraph_id"] == "paragraph-0001"
    assert transcript["paragraphs"][1]["next_paragraph_id"] == "paragraph-0003"
    assert rows[2]["previous_cue_id"] == rows[1]["id"]
    assert "What should we build?" in transcript["paragraphs"][0]["text"]
    assert bundle["summary"]["known_speaker_cues"] == 6
    assert bundle["summary"]["unknown_speaker_cues"] == 3
    assert bundle["summary"]["utterance_count"] == 7


def test_mixed_speaker_cue_does_not_invent_a_turn_timestamp():
    bundle = extract()
    mixed = bundle["transcript"]["segments"][6]
    assert mixed["speaker"] is None
    assert mixed["speaker_basis"] == "mixed_speakers"
    assert mixed["speaker_names"] == ["Chris Anderson", "Elon Musk"]
    assert [span["speaker"] for span in mixed["speaker_spans"]] == ["Elon Musk", "Chris Anderson"]
    assert all("start" not in span for span in mixed["speaker_spans"])
    assert mixed["start"] == 6
    assert mixed["end"] == 7
    assert mixed["words"] == []


def test_subject_title_and_abbreviations_alone_cannot_assign_identity():
    bundle = extract(document([[(0, "EM: A response."), (1000, "Still speaking?")], [(2000, "CA: A question?")]]))
    assert bundle["transcript"]["resolved_label_map"] == {}
    assert bundle["summary"]["known_speaker_cues"] == 0
    assert all(row["speaker"] is None for row in bundle["whisperx"]["segments"])


def test_unknown_label_clears_prior_attribution():
    data = document([[(0, "Elon Musk: An answer."), (1000, "Host: A question?"), (2000, "Please continue.")]])
    rows = extract(data)["transcript"]["segments"]
    assert rows[0]["speaker"] == "Elon Musk"
    assert rows[1]["speaker"] is None
    assert rows[2]["speaker"] is None


def test_stage_directions_are_not_identity_or_avatar_evidence():
    data = document([[(0, "Elon Musk: A response."), (1000, "(Laughter)"), (2000, "(Applause)CA: Next?")],
                     [(3000, "Chris Anderson: Let us continue.")]])
    transcript = extract(data)["transcript"]
    assert transcript["segments"][1]["speaker"] is None
    assert transcript["segments"][1]["speaker_basis"] == "non_speech_marker"
    assert transcript["segments"][2]["speaker"] == "Chris Anderson"
    assert transcript["segments"][2]["speaker_spans"][0]["kind"] == "non_speech_marker"
    assert transcript["behavioral_observations"] == []


def test_shared_times_keep_original_order_and_explicit_proxy_ends():
    data = document([[(0, "Chris Anderson: First?"), (0, "Elon Musk: Second."), (2000, "More.")]], duration=4)
    rows = extract(data)["transcript"]["segments"]
    assert [row["start"] for row in rows] == [0, 0, 2]
    assert [row["end"] for row in rows] == [2, 2, 4]
    assert rows[0]["end_basis"] == "next_distinct_cue_start_proxy"
    assert rows[-1]["end_basis"] == "publisher_duration_proxy"
    assert rows[0]["shared_start_timestamp"] and rows[1]["shared_start_timestamp"]
    assert len(parse_whisperx(extract(data)["whisperx"])) == 3
    assert extract(data)["summary"]["cue_interval_proxy_may_overlap"] is True


def test_duration_statistics_remain_explicit_caption_interval_proxies():
    summary = extract()["summary"]
    assert summary["cue_interval_proxy_seconds_by_speaker"] == {"Chris Anderson": 3.0, "Elon Musk": 3.0}
    assert summary["unassigned_cue_interval_proxy_seconds"] == 4.0
    assert summary["cue_interval_proxy_may_overlap"] is False
    assert "not measured speech duration" in summary["duration_measurement_note"]


def test_missing_tail_duration_keeps_canonical_text_without_fabricating_end():
    bundle = extract(document([[(0, "Elon Musk: First."), (1000, "Final sentence.")]], duration=None))
    assert bundle["transcript"]["segments"][-1]["end"] is None
    assert bundle["whisperx"]["omitted_cue_ids"] == ["cue-00002"]
    assert len(bundle["whisperx"]["segments"]) == 1
    assert bundle["summary"]["missing_end_cues"] == 1


def test_media_urls_are_inventory_only_and_untrusted_hosts_are_not_exported():
    data = document()
    resources = data["props"]["pageProps"]["videoData"]["videoPlayerData"]["resources"]
    resources["h264"].extend([{"file": "https://localhost/private.mp4"}, {"file": "https://py.tedcdn.com:bad/example.mp4"}])
    manifest = extract(data)["manifest"]
    assert manifest["schema_version"] == "publisher_media.v1"
    assert manifest["media_downloaded"] is False
    assert manifest["alignment_verified"] is False
    assert len(manifest["media_links"]) == 3
    assert all(item["alignment"] == "unverified" for item in manifest["media_links"])


@pytest.mark.parametrize("time", [-1, float("nan"), float("inf"), True, "1000"])
def test_malformed_cue_times_are_rejected(time):
    data = document([[(time, "Elon Musk: Text.")]])
    with pytest.raises(TranscriptError):
        extract(data)


def test_reversed_times_and_empty_transcript_are_rejected():
    with pytest.raises(TranscriptError, match="nondecreasing"):
        extract(document([[(1000, "First."), (500, "Second.")]]))
    with pytest.raises(TranscriptError, match="no paragraphs"):
        extract(document([]))
    with pytest.raises(TranscriptError, match="nonempty text"):
        extract(document([[(0, " ")]]))
    with pytest.raises(TranscriptError, match="Missing TED"):
        extract_ted({}, source_id="test", input_sha256="a" * 64)


@pytest.mark.parametrize("source_id", ["../../outside", "UPPERCASE", "space here", ""])
def test_source_identity_cannot_be_used_as_a_path(source_id):
    with pytest.raises(TranscriptError, match="identifier"):
        extract_ted(document(), source_id=source_id, input_sha256="a" * 64)


def test_html_and_json_input_preserve_identical_publisher_data():
    data = document()
    raw = json.dumps(data).encode()
    html = b'<html><script>ignore this()</script><script type="application/json" id="__NEXT_DATA__">' + raw + b"</script></html>"
    assert read_document(raw) == (data, "saved_next_data")
    assert read_document(html) == (data, "publisher_html")
    with pytest.raises(TranscriptError, match="exactly one"):
        read_document(html + html)
    with pytest.raises(TranscriptError, match="exactly one"):
        read_document(b"<html>No transcript data</html>")
    with pytest.raises(TranscriptError, match="Duplicate"):
        read_document(b'{"props": {}, "props": {}}')
    with pytest.raises(TranscriptError, match="Nonfinite"):
        read_document(b'{"value": NaN}')
    with pytest.raises(TranscriptError, match="UTF-8"):
        read_document(b"\xff")


def test_caption_hash_is_independent_of_changing_page_metadata():
    first = document()
    changed = document()
    changed["props"]["pageProps"]["videoData"]["viewedCount"] = 99999
    assert extract(first)["summary"]["publisher_caption_sha256"] == extract(changed)["summary"]["publisher_caption_sha256"]
    changed["props"]["pageProps"]["transcriptData"]["translation"]["paragraphs"][0]["cues"][0]["text"] = "Chris Anderson: A changed question?"
    assert extract(first)["summary"]["publisher_caption_sha256"] != extract(changed)["summary"]["publisher_caption_sha256"]


@pytest.mark.parametrize("url", [
    "http://www.ted.com/talks/test", "https://127.0.0.1/talks/test", "https://www.ted.com.evil.test/talks/test",
    "https://name:password@www.ted.com/talks/test", "https://www.ted.com:1234/talks/test",
    "https://www.ted.com:bad/talks/test", "https://www.ted.com/account", "https://www.ted.com/talks/../account",
])
def test_fetch_url_must_remain_on_public_ted_talk_paths(url):
    with pytest.raises(TranscriptError):
        validate_ted_url(url)


def test_redirect_cannot_leave_the_publisher():
    with pytest.raises(TranscriptError):
        _TEDRedirects().redirect_request(None, None, 302, "redirect", {}, "https://localhost/private")
    assert validate_ted_url("https://www.ted.com/talks/synthetic_example?view=transcript")


def test_output_hashes_context_and_existing_importer_remain_drafts(tmp_path):
    output = tmp_path / "bundle"
    write_bundle(extract(), output)
    manifest = json.loads((output / "manifest.json").read_text())
    for item in manifest["artifacts"].values():
        raw = (output / item["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        assert len(raw) == item["bytes"]
    records = import_transcript(output / "whisperx.json", source_id="synthetic", timeline="ignored", manifest=output / "manifest.json")
    assert len(records) == 9
    assert all(record.review_status == "draft" and record.confidence == "unverified_transcript" for record in records)
    assert all(record.behavioral_observation is None for record in records)
    assert all(record.locator.timeline == "ted_transcript_cue" for record in records)
    assert records[4].transcript_speaker is None
    musk = import_transcript(output / "whisperx.json", source_id="synthetic", timeline="ted_transcript_cue", speaker="Elon Musk")
    assert len(musk) == 3
    assert (output / "transcript.json").exists()  # All original interviewer context remains available.


def test_existing_output_or_symlink_is_never_overwritten(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    keep = output / "important.txt"
    keep.write_text("keep")
    with pytest.raises(TranscriptError, match="exists"):
        write_bundle(extract(), output)
    assert keep.read_text() == "keep"
    link = tmp_path / "link"
    link.symlink_to(output, target_is_directory=True)
    with pytest.raises(TranscriptError, match="exists"):
        write_bundle(extract(), link)


def test_cli_prints_counts_not_transcript_text(tmp_path, capsys):
    input_path = tmp_path / "publisher.json"
    input_path.write_text(json.dumps(document()))
    code = main(["ted", "--input", str(input_path), "--source-id", "synthetic", "--output", str(tmp_path / "out")])
    assert code == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["cue_count"] == 9
    assert "What should we build?" not in captured.out
    assert main(["ted", "--input", str(input_path), "--source-id", "synthetic", "--output", str(tmp_path / "out")]) == 2
    assert "already exists" in capsys.readouterr().err
