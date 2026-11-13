"""Conservative, unprompted second-pass ASR screening of voice candidates."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from tools.media.common import artifact_path, sha256, write_json
from tools.media.prepare import prepare_clip
from .asr import transcribe_reference_context
from .workflow import asr_words, tokens, verify_reference


def _checkpoint(path: Path, value: dict) -> None:
    fd, name = tempfile.mkstemp(prefix=".screen-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def screen(candidates: list[Path], output: Path, *, model_directory: Path, model_repo: str,
           revision: str, min_word_confidence: float = 0.5, cpu_threads: int = 4) -> dict:
    if output.exists() or not 0 <= min_word_confidence <= 1:
        raise ValueError("Choose new screening output and a confidence threshold in [0,1]")
    output.mkdir(parents=True)
    protocol = {"schema_version": "voice-reference-screen.v1", "min_word_confidence": min_word_confidence,
        "require_exact_second_pass": True, "context_edge_margin_seconds": 0.05,
        "model_repo": model_repo, "model_revision": revision, "human_listening_completed": False,
        "limitations": ["This second decoding uses the same ASR model on a cropped input, not an independent human review.",
                        "ASR agreement does not establish voice similarity or the absence of overlapping speakers/music."]}
    write_json(output / "protocol.json", protocol)
    results = []
    for index, candidate in enumerate(candidates):
        manifest = verify_reference(candidate)
        row = {"reference_manifest": str(candidate.resolve()), "reference_manifest_sha256": sha256(candidate),
               "source_id": manifest["source_id"], "review_status": manifest["review"]["status"],
               "seconds": manifest["source_end_seconds"]-manifest["source_start_seconds"], "selected": False}
        confidence = manifest["automatic_checks"].get("minimum_asr_word_confidence")
        parent_asr = json.loads(artifact_path(candidate, manifest["artifacts"]["asr_context"]["path"]).read_text())
        context_words = asr_words(parent_asr)
        at_edge = (manifest["source_start_seconds"] <= parent_asr["source_start_seconds"]+0.05
                   or manifest["source_end_seconds"] >= max(word["source_end"] for word in context_words)-0.05)
        if manifest["automatic_checks"].get("exact_normalized_text_agreement") is not True:
            row["reason"] = "publisher_word_disagreement"
        elif confidence is None or confidence < min_word_confidence:
            row["reason"] = "low_or_missing_word_confidence"
        elif at_edge:
            row["reason"] = "context_edge_without_acoustic_padding"
        else:
            context = output / f"context-{index:03d}"
            audio = artifact_path(candidate, manifest["artifacts"]["audio"]["path"])
            prepare_clip(audio, context, source_id=manifest["source_id"], source_url=manifest["source_url"],
                source_timeline=manifest["source_timeline"], start=0, duration=row["seconds"],
                input_source_offset=manifest["source_start_seconds"], sample_rate=16000, include_video=False)
            asr_path = context / "asr.json"
            result = transcribe_reference_context(context / "manifest.json", model_directory=model_directory,
                model_repo=model_repo, revision=revision, output=asr_path, cpu_threads=cpu_threads)
            observed = " ".join(item["text"] for item in result["segments"])
            expected = artifact_path(candidate, manifest["artifacts"]["reference_text"]["path"]).read_text()
            row.update(second_asr_sha256=sha256(asr_path), second_asr_elapsed_seconds=result["elapsed_seconds"],
                expected_tokens=len(tokens(expected)), observed_tokens=len(tokens(observed)),
                selected=tokens(observed) == tokens(expected),
                reason="exact_second_decoding" if tokens(observed) == tokens(expected) else "second_decoding_disagreement")
        results.append(row)
        summary = {"schema_version": "voice-reference-screen-result.v1", "protocol_sha256": sha256(output / "protocol.json"),
            "references": results, "candidate_count": len(candidates), "processed": len(results),
            "selected": sum(item["selected"] for item in results), "listening_reviewed": 0}
        _checkpoint(output / "results.json", summary)
        print(json.dumps({"processed": len(results), "selected": summary["selected"], "source_id": row["source_id"], "reason": row["reason"]}), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", dest="candidates", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--min-word-confidence", type=float, default=0.5)
    parser.add_argument("--cpu-threads", type=int, default=4)
    screen(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
