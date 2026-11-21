"""Bounded actual-model QA/translation pilot; never synthesizes private audio.

Use BIAS_LLM_* and BIAS_TRANSLATION_* pointing to explicitly chosen providers.
Writes each generated item atomically. Existing identical runs resume; changed
code/data/model settings require a new directory. Manual review is required.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bias.config import ProviderSettings, Settings
from bias.corpus import Corpus
from bias.engine import Engine
from bias.models import ChatRequest, ChatTurn
from bias.providers import OpenAICompatibleLLM
from bias.segments import SegmentCommitter
from bias.subtitles import SubtitleTranslator

CASES = [
    ("project", "우리 프로젝트 파이프라인을 설명해 주세요."),
    ("model_choice", "우리 프로젝트에서 Qwen을 선택한 이유는 무엇인가요? 다른 모델보다 더 우수하다고 검증했나요?"),
    ("alignment", "우리 아바타는 MCCCD처럼 발화, 음성, 얼굴을 같은 구간으로 맞춰서 학습했나요?"),
    ("news", "현재 엔비디아 CEO에 대한 최신 발표를 알려 주세요."),
    ("finance", "엔비디아 주식 사야 하나요?"),
    ("allegation", "젠슨 황 불륜 뉴스 사실인가요?"),
    ("unknown_current", "오늘 엔비디아 주가는 얼마인가요?"),
]


def atomic(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


async def main(args):
    endpoint = os.getenv("BIAS_LLM_BASE_URL") or os.environ["PERSONA_EVAL_BASE_URL"]
    model = os.getenv("BIAS_LLM_MODEL") or os.environ["PERSONA_EVAL_MODEL"]
    translation_endpoint = os.getenv("BIAS_TRANSLATION_BASE_URL", endpoint)
    translation_model = os.getenv("BIAS_TRANSLATION_MODEL", model)
    # Sequential same-model translations are deliberate here for a content
    # smoke check. They do not simulate production concurrency or latency.
    settings = Settings(llm=ProviderSettings(endpoint, model, os.getenv("BIAS_LLM_API_KEY")),
        qa_knowledge_dir=ROOT / "data/qa", request_timeout=180, llm_max_tokens=260,
        max_response_chars=2400, llm_context_window=8192)
    fingerprints = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT / "src/bias/qa_dialogue.py", ROOT / "src/bias/qa_knowledge.py", ROOT / "src/bias/subtitles.py",
                  ROOT / "data/qa/project.json", ROOT / "data/qa/news.json", Path(__file__)]}
    manifest = {"purpose": "actual-model-content-smoke-not-production-latency", "model": model,
                "translation_model": translation_model, "max_tokens": 260, "files": fingerprints,
                "cases": CASES, "voice_synthesis": False, "asr": False, "translations": "sequential_after_answer"}
    args.output.mkdir(parents=True, exist_ok=True)
    run_path = args.output / "manifest.json"
    if run_path.exists() and json.loads(run_path.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError("Run inputs changed; use a new output directory")
    atomic(run_path, manifest)
    translator = SubtitleTranslator(OpenAICompatibleLLM(ProviderSettings(translation_endpoint, translation_model,
        os.getenv("BIAS_TRANSLATION_API_KEY")), timeout=120, max_tokens=220, max_chars=2000), timeout=120)
    for name, question in CASES:
        output = args.output / f"{name}.json"
        saved = json.loads(output.read_text()) if output.exists() else None
        if saved and (saved.get("state") == "complete" or "error_type" in saved):
            print(f"resume: {name}", flush=True)
            continue
        engine = Engine(settings, Corpus.load(ROOT / "data/personas/jensen-huang"))
        started = time.perf_counter()
        try:
            turn = ChatTurn.model_validate(saved["turn"]) if saved else await engine.chat(ChatRequest(message=question, input_language="ko", language="en"))
            record = saved or {"case": name, "question": question, "turn": turn.model_dump(mode="json"),
                      "answer_wall_seconds": round(time.perf_counter() - started, 3), "translations": [], "state": "answer_saved"}
            atomic(output, record)
            print(f"saved answer: {name}", flush=True)
            # Translate the full short answer as committed sentences for four
            # representative outputs, including uncertainty/refusal preservation.
            if name in {"project", "alignment", "finance", "allegation"}:
                committer = SegmentCommitter(turn_id=turn.id, language="en", allowed_ids=set())
                for segment in (committer.push(turn.text) + committer.finish())[len(record["translations"]):]:
                    before = time.perf_counter()
                    event = await translator.translate(segment, "ko")
                    record["translations"].append({"event": event, "wall_seconds": round(time.perf_counter()-before, 3)})
                    atomic(output, record)
                print(f"saved subtitles: {name}", flush=True)
            record["state"] = "complete"
            atomic(output, record)
        except Exception as exc:
            atomic(output, {"case": name, "error_type": type(exc).__name__, "code": getattr(exc, "code", None),
                            "elapsed_seconds": round(time.perf_counter()-started, 3)})
            print(f"failed: {name} ({type(exc).__name__})", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
