"""Package a measured run without redistributing original publisher passages."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

from .run import digest, file_identity, now, save


def public_turn(turn: dict) -> dict:
    value = copy.deepcopy(turn)
    for citation in value.get("citations", []):
        if "text" in citation:
            text = citation.pop("text")
            citation["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            citation["text_distribution"] = "Original publisher passage omitted; retained in the local raw response."
    return value


def publish(report_path: Path, failed_paths: list[Path], output: Path) -> dict:
    report = json.loads(report_path.read_text())
    if report.get("schema_version") != "demo-http-latency.v1" or not report.get("completed"):
        raise ValueError("A completed actual transport report is required")
    if output.exists():
        raise ValueError("Use a new publication directory")
    for row in report["runs"]:
        if row["status"] != "completed" or row["turn"]["provider"] != "openai":
            raise ValueError("Only complete model/voice turns can be published as successful")
        if any(value is not None for key, value in row["browser"].items() if key != "status") or row["acoustic_first_audible_ms"] is not None:
            raise ValueError("A transport run cannot claim browser or acoustic measurements")
        audio = report_path.parent / row["audio"]["path"]
        if audio.parent != report_path.parent or digest(audio) != row["audio"]["sha256"]:
            raise ValueError("The preserved generated audio changed")
    value = copy.deepcopy(report)
    value["schema_version"] = "demo-http-latency-summary.v1"
    value["publication"] = {"created_at": now(), "raw_report_sha256": digest(report_path),
        "publisher_source_passages": "Omitted from citation.text; exact UTF-8 text hashes and source IDs/URLs/locators remain.",
        "generated_outputs": "Unchanged", "timings": "Unchanged", "raw_report_retention": "Local execution output directory",
        "packager_sha256": digest(Path(__file__))}
    value["raw_artifact_identities"] = [{**item, "path": Path(item["path"]).name} for item in value.pop("artifacts")]
    for row in value["runs"]:
        row["turn"] = public_turn(row["turn"])
    value["earlier_attempts"] = []
    for path in failed_paths:
        failed = json.loads(path.read_text())
        if failed.get("schema_version") != "demo-http-latency.v1" or failed.get("completed"):
            raise ValueError("Earlier failed attempts must be actual incomplete transport reports")
        value["earlier_attempts"].append({"raw_report_sha256": digest(path), "started_at": failed["started_at"],
            "failure": failed.get("failure"), "runs": [{"request": row["request"], "status": row["status"],
                "transport": row.get("transport"), "generated_or_fallback_text": row.get("turn", {}).get("text"),
                "provider": row.get("turn", {}).get("provider"), "failure": row.get("failure"),
                "speech_http_status": row.get("speech_http_status")} for row in failed["runs"]]})
    output.mkdir(parents=True)
    for row in value["runs"]:
        shutil.copy2(report_path.parent / row["audio"]["path"], output / row["audio"]["path"])
    for name in ("llm.log", "voice.log", "app.log", "process-memory.jsonl", "llama-metrics.txt"):
        source = report_path.parent / name
        if source.is_file():
            expected = next(item["sha256"] for item in value["raw_artifact_identities"] if item["path"] == name)
            if digest(source) != expected:
                raise ValueError("An execution log changed after measurement")
            shutil.copy2(source, output / name)
    value["published_artifacts"] = [{**file_identity(path), "path": path.name} for path in sorted(output.iterdir())]
    save(output / "summary.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--failed-report", action="append", default=[], type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    publish(args.report.resolve(), [path.resolve() for path in args.failed_report], args.output.resolve())
    print(args.output / "summary.json")


if __name__ == "__main__":
    main()
