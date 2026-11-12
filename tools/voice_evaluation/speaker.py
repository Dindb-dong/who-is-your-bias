"""Frozen WavLM speaker-embedding diagnostics, without a likeness pass threshold.

The reference set is explicitly training-exposed; these measurements do not
establish unseen-event generalization or independent human-perceived likeness.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

from tools.adaptation.voice_data import load_dataset
from .metrics import EvaluationError, wav_metrics
from .run import checkpoint, digest, read_json, synthesis_rows, timestamp, verified_audio


def cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right) or not all(math.isfinite(x) for x in left + right):
        raise EvaluationError("Speaker embeddings must have equal finite dimensions")
    denominator = math.sqrt(sum(x*x for x in left) * sum(x*x for x in right))
    if denominator <= 1e-12:
        raise EvaluationError("Speaker embeddings must have nonzero norm")
    return round(max(-1.0, min(1.0, sum(a*b for a, b in zip(left, right)) / denominator)), 6)


def compare(probe: list[float], references: list[dict]) -> dict:
    rows, events = [], defaultdict(list)
    for reference in references:
        score = cosine(probe, reference["embedding"])
        rows.append({"audio_sha256": reference["audio_sha256"], "event_family_id": reference["event_family_id"], "cosine": score})
        events[reference["event_family_id"]].append(score)
    if not rows:
        raise EvaluationError("Speaker comparison needs reference embeddings")
    means = {event: round(statistics.fmean(scores), 6) for event, scores in sorted(events.items())}
    return {"references": rows, "event_means": means, "event_balanced_mean": round(statistics.fmean(means.values()), 6),
            "minimum": min(row["cosine"] for row in rows), "maximum": max(row["cosine"] for row in rows)}


def _audio(path: Path):
    import io
    import wave
    import numpy as np
    from scipy.signal import resample_poly
    raw = path.read_bytes()
    signal = wav_metrics(raw, max_seconds=30)
    if signal["duration_seconds"] < 1 or not signal["signal_present"]:
        raise EvaluationError("Speaker diagnostics require 1–30 seconds of nonempty speech")
    with wave.open(io.BytesIO(raw), "rb") as wav:
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32)
        samples = samples.reshape(-1, wav.getnchannels()).mean(axis=1) / 32768.0
        rate = wav.getframerate()
    divisor = math.gcd(rate, 16000)
    if rate != 16000:
        samples = resample_poly(samples, 16000//divisor, rate//divisor).astype(np.float32)
    return samples, signal


def evaluate(synthesis_paths: list[Path], *, dataset: Path, persona_id: str, model_directory: Path,
             model_receipt: Path, output: Path, allow_candidates=False, threads=4) -> dict:
    if output.exists() or not 1 <= threads <= 4:
        raise EvaluationError("Choose a new speaker result file and 1–4 CPU threads")
    manifest, data = load_dataset(dataset, persona_id=persona_id, allow_candidates=allow_candidates)
    receipt = read_json(model_receipt)
    if receipt.get("repo") != "microsoft/wavlm-base-plus-sv" or not re.fullmatch(r"[a-f0-9]{40}", receipt.get("revision", "")):
        raise EvaluationError("Speaker evaluation requires a pinned WavLM verification model")
    files = {row["path"]: row for row in receipt["files"]}
    if not {"config.json", "preprocessor_config.json", "pytorch_model.bin"}.issubset(files):
        raise EvaluationError("Speaker model receipt is incomplete")
    for name in ("config.json", "preprocessor_config.json", "pytorch_model.bin"):
        if digest(model_directory / name) != files[name]["sha256"]:
            raise EvaluationError("Speaker model differs from its frozen receipt")
    cases = []
    for path in synthesis_paths:
        run = read_json(path)
        for row in synthesis_rows(run):
            if row.get("status") == "succeeded":
                if row.get("service_after", {}).get("persona_id") != persona_id:
                    raise EvaluationError("Generated speech does not establish the requested subject binding")
                cases.append((path, run, row, verified_audio(path, row)))
    if not cases:
        raise EvaluationError("No generated speech cases are available")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    torch.set_num_threads(threads)
    report = {"schema_version": "voice-speaker-diagnostics.v1", "status": "loading", "created_at": timestamp(),
        "persona_id": persona_id, "model": {"repo": receipt["repo"], "revision": receipt["revision"],
            "receipt_sha256": digest(model_receipt), "weights_sha256": files["pytorch_model.bin"]["sha256"],
            "torch": importlib.metadata.version("torch"), "transformers": importlib.metadata.version("transformers"),
            "device": "cpu", "dtype": "float32", "threads": threads},
        "dataset_manifest_sha256": digest(dataset / "manifest.json"), "implementation_sha256": digest(Path(__file__)),
        "reference_set_training_exposed": True, "independent_of_tts_training_model": True,
        "independent_of_reference_screen_asr_model": True, "heldout_generalization_evaluated": False,
        "human_listening_completed": False, "decision_threshold": None,
        "protocol": {"audio": "complete 1–30s clips, PCM16 decoded and polyphase-resampled to mono 16kHz",
            "aggregation": "cosine per clip; mean within event, then equal-weight mean over events",
            "sources": ["https://huggingface.co/microsoft/wavlm-base-plus-sv",
                        "https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/wavlm"]},
        "limitations": ["Cosines are descriptive diagnostics, not calibrated probabilities or human likeness scores.",
            "References were exposed to adaptation; no unseen-event fidelity claim is made.",
            "Reference speaker labels come from publisher transcripts; this model does not establish identity.",
            "Korean outputs are cross-language comparisons against English sources; no language-invariant calibration is claimed."],
        "reference_pairs": [], "cases": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint(output, report)
    started = time.monotonic()
    try:
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(model_directory), local_files_only=True)
        model = WavLMForXVector.from_pretrained(str(model_directory), local_files_only=True, weights_only=True).eval()
    except Exception as exc:
        report.update(status="incomplete", error_type=type(exc).__name__, stage="model_load",
            elapsed_seconds=round(time.monotonic()-started, 6))
        checkpoint(output, report)
        raise
    model.requires_grad_(False)
    report["load_seconds"] = round(time.monotonic()-started, 6)

    def embed(path):
        samples, signal = _audio(path)
        inputs = extractor(samples, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            vector = model(**inputs).embeddings[0].cpu().tolist()
        return vector, signal

    references = []
    for item, lineage in zip(data, manifest["lineage"]):
        vector, _ = embed(Path(item["audio"]))
        references.append({"embedding": vector, "audio_sha256": lineage["audio_sha256"], "event_family_id": lineage["event_family_id"]})
    for index, left in enumerate(references):
        for right in references[index+1:]:
            report["reference_pairs"].append({"left_sha256": left["audio_sha256"], "right_sha256": right["audio_sha256"],
                "same_event": left["event_family_id"] == right["event_family_id"], "cosine": cosine(left["embedding"], right["embedding"])})
    report["status"] = "running"
    checkpoint(output, report)
    for path, run, row, audio in cases:
        result = {"condition": run["condition"], "id": row["id"], "synthesis_sha256": digest(path),
            "audio_sha256": digest(audio), "asr_language": row.get("asr_language"), "status": "running"}
        report["cases"].append(result)
        checkpoint(output, report)
        then = time.monotonic()
        try:
            vector, signal = embed(audio)
            result.update(status="succeeded", audio_seconds=signal["duration_seconds"], comparisons=compare(vector, references))
        except (ValueError, RuntimeError, OSError) as exc:
            result.update(status="failed", error_type=type(exc).__name__)
        result["elapsed_seconds"] = round(time.monotonic()-then, 6)
        checkpoint(output, report)
    report.update(status="completed" if all(row["status"] == "succeeded" for row in report["cases"]) else "incomplete",
        completed_at=timestamp(), elapsed_seconds=round(time.monotonic()-started, 6))
    checkpoint(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("synthesis_paths", type=Path, nargs="+")
    for name in ("dataset", "model-directory", "model-receipt", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--persona-id", required=True)
    parser.add_argument("--allow-candidates", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = vars(parser.parse_args(argv))
    report = evaluate(**args)
    print(report["status"])


if __name__ == "__main__":
    main()
