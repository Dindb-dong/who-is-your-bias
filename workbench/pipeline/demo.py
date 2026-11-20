"""Synthetic fixtures only. No public figure's speech or online source is used."""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from . import dataset, experiments, handoff
from .common import PipelineError, read_csv, write_json


def create(root):
    root = Path(root)
    if list((root / "sources").glob("*/source.json")):
        raise PipelineError("예제 폴더에 이미 자료가 있습니다. 기존 예제를 사용하거나 새 폴더를 지정하세요.")
    raw = root / "example_inputs"
    raw.mkdir(parents=True, exist_ok=True)
    examples = [
        ("How do you handle project failure?", "I start with one small experiment and learn from the result.", "failure", "advice", "train"),
        ("How do you handle project failure?", "Thank you for asking. We discuss the failure together and decide what to try next.", "failure", "reflection", "train"),
        ("What would you tell a student whose project failed?", "Give yourself time, write down the lesson, and test one change tomorrow.", "failure", "advice", "eval"),
    ]
    for i, (q, a, topic, situation, split) in enumerate(examples, 1):
        subtitle = raw / f"demo-{i}.srt"
        subtitle.write_text(f"1\n00:00:00,000 --> 00:00:02,000\n{q}\n\n2\n00:00:02,000 --> 00:00:07,000\n{a}\n", encoding="utf-8")
        audio = raw / f"demo-{i}.wav"
        with wave.open(str(audio), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(b"".join(struct.pack("<h", int(500 * math.sin(2 * math.pi * (200 + i * 30) * t / 24000))) for t in range(24000 * 8)))
        sid = f"demo-{i}"
        dataset.register(root, sid, f"synthetic-recording-{i}", split, "synthetic-persona", "en",
                         audio, subtitle, notes="형식 시험용 가상 문답. 음성은 말소리가 아닌 테스트 신호입니다.", synthetic=True)
        rows = read_csv(dataset.source_dir(root, sid) / "review.csv")
        for j, row in enumerate(rows):
            row.update(speaker="interviewer" if j == 0 else "target", turn_id="q1", reviewed=True,
                       voice_clean=(j == 1), topic=topic, situation=situation)
        dataset.save_review(root, sid, rows)
    data = dataset.build(root)
    card = root / "persona.json"
    write_json(card, experiments.persona_card("synthetic-persona", "Synthetic Example"))
    experiment = experiments.prepare(data, card, k=1)
    ref = dataset.reference(root, "demo-1", "seg-00002")
    package = handoff.package(root, "This is a synthetic pipeline test, not a statement by a real person.",
                              target="synthetic-persona", reference_path=ref)
    result = {"dataset": str(data.resolve()), "experiment": str(experiment.resolve()),
              "reference": str(ref.resolve()), "handoff": str(package.resolve()),
              "note": "가상 데이터이며 LLM·음성 합성 결과가 아닙니다. WAV는 테스트 신호입니다."}
    write_json(root / "demo_result.json", result)
    return result
