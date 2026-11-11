"""Command line entry points for a reproducible local research workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings, repository_root
from .corpus import Corpus
from .evaluate import evaluate_retrieval, read_cases
from .ingest import import_transcript, write_evidence
from .retrieval import Retriever


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="bias", description="Who Is Your Bias? research engine")
    command.add_argument("--corpus", type=Path, default=None, help="Persona corpus directory")
    sub = command.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="Validate corpus schemas, provenance, and split references")
    snapshot = sub.add_parser("snapshot", help="Register an exact persona cutoff and independent modality versions")
    snapshot.add_argument("--id", required=True)
    snapshot.add_argument("--as-of", required=True, help="ISO timestamp with UTC offset")
    snapshot.add_argument("--ingested-at", required=True, help="Actual acquisition timestamp with UTC offset")
    snapshot.add_argument("--current-since", default="2026-01-01")
    snapshot.add_argument("--output", required=True, type=Path)
    search = sub.add_parser("search", help="Search reviewed training evidence")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=4)
    search.add_argument("--behavior", action="store_true")
    evaluate = sub.add_parser("evaluate", help="Run retrieval checks and report ungraded persona tests")
    evaluate.add_argument("--cases", type=Path, nargs="+")
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--limit", type=int, default=4)
    evaluate.add_argument("--fail-on-regression", action="store_true")
    ingest = sub.add_parser("ingest", help="Import a local SRT/VTT/WhisperX transcript as draft evidence")
    ingest.add_argument("input", type=Path)
    ingest.add_argument("--source-id", required=True)
    ingest.add_argument("--timeline", default="source_video")
    ingest.add_argument("--source-offset", type=float, default=0)
    ingest.add_argument("--speaker")
    ingest.add_argument("--manifest", type=Path)
    ingest.add_argument("--output", type=Path, required=True)
    ingest.add_argument("--overwrite", action="store_true")
    serve = sub.add_parser("serve", help="Start the local API and built studio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "serve":
            if args.corpus:
                import os
                os.environ["BIAS_CORPUS_DIR"] = str(args.corpus.resolve())
            import uvicorn
            uvicorn.run("bias.api:app", host=args.host, port=args.port, reload=args.reload)
            return 0
        if args.command == "ingest":
            records = import_transcript(args.input, source_id=args.source_id, timeline=args.timeline, source_offset=args.source_offset, speaker=args.speaker, manifest=args.manifest)
            write_evidence(records, args.output, overwrite=args.overwrite)
            print(json.dumps({"output": str(args.output), "records": len(records), "review_status": "draft", "indexed": False}))
            return 0
        corpus = Corpus.load(args.corpus or Settings.from_env().corpus_dir)
        if args.command == "snapshot":
            from datetime import date, datetime
            from .snapshots import register_snapshot
            snapshot = register_snapshot(corpus, identifier=args.id, as_of=datetime.fromisoformat(args.as_of),
                ingested_at=datetime.fromisoformat(args.ingested_at), current_evidence_since=date.fromisoformat(args.current_since))
            if args.output.exists():
                raise ValueError("Snapshot already exists; use a new versioned output path")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(snapshot.model_dump_json(indent=2) + "\n", encoding="utf-8")
            print(json.dumps(snapshot.summary(), ensure_ascii=False, indent=2))
        elif args.command == "validate":
            print(json.dumps({"valid": True, "persona": corpus.persona.id, **corpus.stats()}, ensure_ascii=False, indent=2))
        elif args.command == "search":
            result = Retriever(corpus).search(args.query, limit=args.limit, behavior=args.behavior)
            print(json.dumps({"confidence": result.confidence, "reason": result.reason, "hits": [{"score": hit.score, "matched_terms": hit.matched_terms, "citation": corpus.citation(hit.evidence.id).model_dump(mode="json")} for hit in result.hits]}, ensure_ascii=False, indent=2))
        elif args.command == "evaluate":
            files = args.cases or sorted((repository_root() / "data/evaluation").glob("*.json"))
            if not files:
                raise ValueError("No evaluation files found")
            report = evaluate_retrieval(corpus, read_cases(files), limit=args.limit)
            serialized = json.dumps(report, ensure_ascii=False, indent=2)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(serialized + "\n", encoding="utf-8")
            print(serialized)
            if args.fail_on_regression and report["summary"]["failed"]:
                return 1
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
