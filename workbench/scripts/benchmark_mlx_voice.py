"""Measure full-answer Qwen3-TTS synthesis with existing weights on Apple Silicon."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import wave
from pathlib import Path
from time import perf_counter


def benchmark(model_dir: Path, reference_dir: Path, fixture: Path, output: Path,
              *, runs: int, seed: int, temperature: float) -> None:
    import mlx.core as mx
    import numpy as np
    from mlx_audio.tts.utils import load_model
    from mlx_audio.utils import load_audio

    data = json.loads(fixture.read_text(encoding="utf-8"))
    transcript = (reference_dir / "reference.txt").read_text(encoding="utf-8").strip()
    output.mkdir(parents=True)
    started = perf_counter()
    model = load_model(str(model_dir.resolve()))
    reference = load_audio(str(reference_dir / "reference.wav"), sample_rate=model.sample_rate)
    mx.eval(reference)
    report = {"device": str(mx.default_device()), "load_seconds": round(perf_counter()-started, 3), "runs": []}

    for index in range(runs):
        mx.random.seed(seed)
        started = perf_counter()
        chunks = list(model.generate(
            data["text"], ref_audio=reference, ref_text=transcript, lang_code=data.get("language", "English"),
            top_k=50, top_p=1.0, temperature=temperature, repetition_penalty=1.05,
            max_tokens=1000, stream=False, split_pattern="",
        ))
        if len(chunks) != 1:
            raise ValueError("Expected exactly one audio segment for the full answer")
        chunk = chunks[0]
        audio = np.asarray(chunk.audio.astype(mx.float32)).reshape(-1)
        elapsed = perf_counter() - started
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("Model returned invalid or empty audio")
        pcm = np.rint(np.clip(audio, -1, 1) * 32767).astype("<i2")
        path = output / f"qwen-mlx-{index + 1}.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setparams((1, 2, chunk.sample_rate, 0, "NONE", "not compressed"))
            wav.writeframes(pcm.tobytes())
        report["runs"].append({
            "seconds": round(elapsed, 3), "audio_seconds": len(audio) / chunk.sample_rate,
            "wav": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    report["warm_median_seconds"] = statistics.median(row["seconds"] for row in report["runs"][1:])
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.75)
    options = parser.parse_args()
    if options.runs < 2:
        parser.error("Use at least two runs to separate first and warm inference")
    benchmark(options.model_dir, options.reference_dir, options.fixture, options.output,
              runs=options.runs, seed=options.seed, temperature=options.temperature)


if __name__ == "__main__":
    main()
