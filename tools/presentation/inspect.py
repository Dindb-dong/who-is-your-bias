"""Run the existing independent ASR evaluator on prepared presentation speech."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .common import digest, read_plan, save, wav_info


def inspect(*, plan_path: Path, speech_dir: Path, output: Path, model_directory: Path,
            revision: str, threads: int = 1) -> dict:
    from tools.voice_evaluation.asr import transcribe_run
    plan = read_plan(plan_path)
    source = json.loads((speech_dir / "speech.json").read_text())
    audio = speech_dir / "speech.wav"
    if source.get("status") != "completed" or source["plan_sha256"] != digest(plan_path):
        raise ValueError("ASR needs a completed matching prepared-speech run")
    if source["audio"]["sha256"] != digest(audio):
        raise ValueError("The generated audio changed before ASR")
    text = " ".join(segment["text"] for segment in plan["segments"])
    if len(text) > 2000:
        raise ValueError("Use a shorter presentation for the bounded ASR check")
    if output.exists():
        raise ValueError("Choose a new ASR output directory")
    output.mkdir(parents=True)
    shutil.copy2(audio, output / "speech.wav")
    synthesis = {"schema_version": "voice_synthesis.v1", "status": "completed",
        "condition": "prepared-presentation", "execution": "existing_prepared_audio",
        "prepared_speech_sha256": digest(speech_dir / "speech.json"),
        "cases": [{"id": "presentation", "status": "succeeded", "text": text,
            "language": "English" if plan["language"] == "en" else "Korean", "asr_language": plan["language"],
            "audio": {"path": "speech.wav", "sha256": digest(audio), "bytes": audio.stat().st_size},
            "signal": {"duration_seconds": wav_info(audio)["seconds"]}}]}
    save(output / "synthesis.json", synthesis)
    return transcribe_run(output / "synthesis.json", model_directory=model_directory,
        model_repo="Systran/faster-whisper-small", revision=revision,
        output=output / "asr.json", cpu_threads=threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--speech-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    result = inspect(plan_path=args.plan, speech_dir=args.speech_dir, output=args.output,
        model_directory=args.model_directory, revision=args.revision, threads=args.threads)
    print(result["status"])
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
