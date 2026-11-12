"""Prepare candidate speech intervals by matching an untimed publisher transcript.

Sentence segmentation retains source turn IDs and character spans. Acoustic
times always come from the independently decoded audio; no PDF clock is made up.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tools.media.common import MediaError, sha256, source_identity, write_json
from .workflow import align_transcripts, asr_words, prepare_reference, publisher_words, tokens


def publisher_from_turns(canonical: Path, *, source_url: str) -> dict:
    source = json.loads(canonical.read_text())
    if source.get("schema_version") != "persona-original-turns.v1" or source.get("source_timeline") is not None:
        raise MediaError("Supply canonical original turns explicitly lacking a media clock")
    source_identity(source.get("source_id"), source_url, "untimed_text")
    segments = []
    for turn in source.get("turns", []):
        if turn.get("start") is not None or turn.get("end") is not None:
            raise MediaError("Untimed conversion must not discard existing timing")
        text = turn.get("text", "")
        speaker = turn.get("speaker")
        basis = turn.get("speaker_basis")
        attributed = isinstance(speaker, str) and bool(speaker) and basis in {
            "explicit_pdf_speaker_label", "explicit_label", "explicit_publisher_label", "explicit_publisher_speaker_label"}
        for index, match in enumerate(re.finditer(r"\S.*?(?:[.!?](?=\s|$)|$)", text, flags=re.DOTALL)):
            passage = match.group().strip()
            if not passage:
                continue
            segments.append({"id": f"{turn['id']}-sentence-{index:04d}", "start": None,
                "speaker": speaker if attributed else None,
                "speaker_basis": "explicit_publisher_label" if attributed else "unknown",
                "text": passage, "source_turn_id": turn["id"], "source_character_span": [match.start(), match.end()],
                "source_speaker_basis": basis,
                "speaker_spans": [{"kind": "caption_text", "text": passage, "speaker": speaker if attributed else None}]})
    result = {"schema_version": "publisher_transcript.v1", "source_id": source["source_id"], "source_url": source_url,
        "source_timeline": None, "timing_kind": "untimed_text", "segments": segments,
        "canonical_sha256": sha256(canonical), "original_input_sha256": source.get("input_sha256"),
        "segmentation": "Original punctuation sentence spans; text retained from canonical publisher turns.",
        "audio_alignment_verified": False}
    publisher_words(result)
    return result


def candidate_intervals(publisher: dict, asr: dict, *, speaker: str, limit: int = 10) -> list[dict]:
    if not 1 <= limit <= 100:
        raise MediaError("Candidate limit must be between one and one hundred")
    alignment = align_transcripts(publisher, asr)
    matches = alignment["matches"]
    words = asr_words(asr)
    segments = publisher["segments"]
    result = []
    latest_end = -1.0
    for first in range(len(segments)):
        for last in range(first, min(first+6, len(segments))):
            selected = segments[first:last+1]
            if any(cue.get("speaker") != speaker for cue in selected):
                break
            ids = [cue["id"] for cue in selected]
            subset = [item for item in matches if item["cue_id"] in ids]
            expected = [token for cue in selected for token in tokens(cue["text"])]
            if len(expected) < 10 or len(subset) != len(expected):
                continue
            start = min(item["source_start_seconds"] for item in subset)
            end = max(item["source_end_seconds"] for item in subset)
            if start < latest_end or not 8 <= end-start <= 14.9:
                continue
            observed = [word for word in words if word["source_start"] < end and word["source_end"] > start]
            if any(word["source_start"] < start or word["source_end"] > end for word in observed):
                continue
            if [token for word in observed for token in word["tokens"]] != expected:
                continue
            result.append({"cue_ids": ids, "start_seconds": start, "duration_seconds": round(end-start, 6),
                           "words": len(expected), "minimum_confidence": min((word["confidence"] for word in observed if word["confidence"] is not None), default=None)})
            latest_end = end
            break
        if len(result) >= limit:
            break
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--asr", type=Path, required=True)
    parser.add_argument("--source-media", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new candidate output directory")
    publisher = publisher_from_turns(args.canonical, source_url=args.source_url)
    asr = json.loads(args.asr.read_text())
    intervals = candidate_intervals(publisher, asr, speaker=args.speaker, limit=args.limit)
    args.output.mkdir(parents=True)
    publisher_path = args.output / "publisher.json"
    write_json(publisher_path, publisher)
    write_json(args.output / "alignment.json", align_transcripts(publisher, asr))
    receipts = []
    for index, interval in enumerate(intervals):
        output = args.output / f"reference-{index:03d}"
        result = prepare_reference(args.media_manifest, publisher_path, args.asr, args.source_media, output,
            speaker=args.speaker, cue_ids=interval["cue_ids"], start_seconds=interval["start_seconds"],
            duration_seconds=interval["duration_seconds"])
        receipts.append({"reference_manifest": str(output / "reference.json"), "sha256": sha256(output / "reference.json"),
                         "source_id": result["source_id"], "interval": interval, "review_status": result["review"]["status"]})
    write_json(args.output / "candidates.json", {"references": receipts, "candidate_count": len(receipts),
        "human_listening_completed": False, "independent_speaker_recognition_performed": False})
    print(json.dumps({"candidate_count": len(receipts), "intervals": intervals}, indent=2))


if __name__ == "__main__":
    main()
