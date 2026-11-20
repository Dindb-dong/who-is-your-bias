from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

from . import batch, dataset, demo, experiments, handoff, media
from .common import PipelineError, load_env, read_csv, write_json


def parser():
    p = argparse.ArgumentParser(description="Who's your bias? 데이터 파이프라인")
    p.add_argument("--root", default="data/local", help="작업 자료 폴더")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("organize-sources", help="기존 자료 폴더를 제목과 출처 ID로 정리")
    sub.add_parser("doctor", help="설치 상태 확인")
    sub.add_parser("demo", help="가상 데이터로 전체 흐름 실행 (기본 data/demo)")
    add = sub.add_parser("ingest", help="로컬 파일 또는 유튜브 URL 등록")
    add.add_argument("--source-id", required=True)
    add.add_argument("--recording-id", required=True)
    add.add_argument("--split", choices=["train", "eval"], default="train")
    add.add_argument("--target", default="jensen-huang")
    add.add_argument("--language", default="en")
    add.add_argument("--media")
    add.add_argument("--subtitles")
    add.add_argument("--url", default="")
    add.add_argument("--notes", default="")
    tr = sub.add_parser("transcribe", help="자막 없는 미디어 자동 전사")
    tr.add_argument("source_id")
    tr.add_argument("--model", default="small")
    tr.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    tr.add_argument("--language", default="en")
    review = sub.add_parser("review-import", help="편집한 검수 CSV 저장")
    review.add_argument("source_id")
    review.add_argument("csv")
    sub.add_parser("build", help="검수 자료를 버전별 데이터셋으로 저장")
    ref = sub.add_parser("reference", help="검수한 단독 음성 구간 추출")
    ref.add_argument("source_id")
    ref.add_argument("segment_id")
    card = sub.add_parser("init-card", help="비어 있는 인물 카드 생성")
    card.add_argument("path")
    ex = sub.add_parser("experiment", help="A/B/C 프롬프트와 검색 결과 저장")
    ex.add_argument("--dataset", required=True)
    ex.add_argument("--card", required=True)
    ex.add_argument("--question", default="")
    ex.add_argument("--situation", default="")
    ex.add_argument("--language", default="en")
    ex.add_argument("--k", type=int, default=3)
    ex.add_argument("--max-chars", type=int, default=800)
    run = sub.add_parser("run-llm", help="설정한 API로 실제 답변 생성 (과금 가능)")
    run.add_argument("experiment")
    run.add_argument("--temperature", type=float, default=0.3)
    rate = sub.add_parser("ratings", help="수동 입력 답변으로 조건을 가린 평가표 생성")
    rate.add_argument("experiment")
    rate.add_argument("answers_csv")
    pack = sub.add_parser("handoff", help="동욱에게 넘길 한 턴 묶음 생성")
    pack.add_argument("--text-file", required=True)
    pack.add_argument("--reference")
    pack.add_argument("--language", default="en")
    pack.add_argument("--target", default="jensen-huang")
    voice = sub.add_parser("tts", help="ElevenLabs 음성 합성 (과금 가능)")
    voice.add_argument("package")
    attach = sub.add_parser("attach-audio", help="다른 모델에서 생성한 음성 연결")
    attach.add_argument("package")
    attach.add_argument("audio")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    load_env(Path(__file__).resolve().parents[1] / ".env")
    try:
        c = args.command
        if c == "doctor":
            result = {"python": sys.version.split()[0], "ffmpeg": media.ffmpeg(),
                      "javascript_runtime": str(Path('.tools/deno').resolve()) if Path('.tools/deno').exists() else shutil.which("deno") or shutil.which("node"),
                      **{name: importlib.util.find_spec(name) is not None for name in ("streamlit", "yt_dlp", "faster_whisper")},
                      "note": "유튜브 일부 영상은 별도 JS 실행기(Deno 등)가 필요합니다. 실제 URL 다운로드·모델 실행 여부와 별개입니다."}
        elif c == "organize-sources":
            result = batch.organize_saved_sources(args.root)
        elif c == "demo":
            result = demo.create("data/demo" if args.root == "data/local" else args.root)
        elif c == "ingest":
            result = dataset.register(args.root, args.source_id, args.recording_id, args.split, args.target,
                                      args.language, args.media, args.subtitles, args.url, args.notes)
        elif c == "transcribe":
            result = dataset.transcribe_source(args.root, args.source_id, args.model, args.device, args.language)
        elif c == "review-import":
            result = dataset.save_review(args.root, args.source_id, read_csv(args.csv))
        elif c == "build":
            result = dataset.build(args.root)
        elif c == "reference":
            result = dataset.reference(args.root, args.source_id, args.segment_id)
        elif c == "init-card":
            if Path(args.path).exists():
                raise PipelineError("이미 인물 카드가 있습니다.")
            write_json(args.path, experiments.persona_card())
            result = Path(args.path)
        elif c == "experiment":
            result = experiments.prepare(args.dataset, args.card, args.question, args.situation, args.language, args.k, args.max_chars)
        elif c == "run-llm":
            result = experiments.run_llm(args.experiment, args.temperature)
        elif c == "ratings":
            result = experiments.blind_sheet(args.experiment, read_csv(args.answers_csv), args.experiment)
        elif c == "handoff":
            result = handoff.package(args.root, Path(args.text_file).read_text(encoding="utf-8"), args.language, args.target, args.reference)
        elif c == "tts":
            result = handoff.elevenlabs(args.package)
        elif c == "attach-audio":
            result = handoff.attach_audio(args.package, args.audio)
        print(json.dumps(result, ensure_ascii=False, indent=2) if isinstance(result, dict) else result)
        return 0
    except (PipelineError, OSError, ValueError, KeyError) as exc:
        print(f"확인 필요: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
