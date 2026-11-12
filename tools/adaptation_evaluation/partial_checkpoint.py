"""Capture a finalized attempt prefix without changing a running condition.

Snapshots are local, partial and deliberately incompatible with collection.
The coordinator archives, uploads and freshly restores them separately.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

from .local_condition import SUPERVISOR_SCHEMA, _identity_hashes, _validate_observed, validate_execution_contract
from .protocol import CONDITIONS, digest
from .run import _expected_request, _validate_report_identity


SCHEMA = "text-condition-partial-prefix.v1"
MANIFEST_SCHEMA = "text-condition-partial-checkpoint.v1"
METADATA = ("launcher.json", "warmup.json", "observed-before.json")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def _read_json(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value, raw


def _check_seal(value: dict, key: str) -> None:
    unsigned = dict(value)
    seal = unsigned.pop(key, None)
    if digest(unsigned) != seal:
        raise ValueError(f"Saved {key} seal differs")


def _seal(value: dict, key: str) -> dict:
    value[key] = digest(value)
    return value


def _jsonl_prefix(path: Path, count: int) -> tuple[list[dict], bytes]:
    rows, lines = [], []
    with path.open("rb") as stream:
        for _ in range(count):
            line = stream.readline()
            if not line.endswith(b"\n"):
                raise ValueError("The requested JSONL prefix is not completely written")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("JSONL rows must be objects")
            rows.append(row)
            lines.append(line)
    return rows, b"".join(lines)


def _match_logs(rows: list[dict], attempts: list[dict], raw: list[dict]) -> None:
    expected_raw = [{"case_id": row["case_id"], "condition": row["condition"],
                     "provider_generation": row["provider_generation"]} for row in rows]
    if digest(attempts) != digest(rows) or digest(raw) != digest(expected_raw):
        raise ValueError("Attempt/raw prefix case identity or exact contents differ from the sealed report")


def _validate_report(report: dict, plan: dict, contract: dict) -> int:
    _check_seal(report, "run_sha256")
    condition = report.get("condition")
    cases = plan["cases"]
    rows = report.get("runs")
    if (report.get("schema_version") != "text-adaptation-condition-run.v1" or condition not in CONDITIONS
            or report.get("plan_sha256") != plan["plan_sha256"] or report.get("policy") != plan["policy"]
            or report.get("planned_case_ids") != [case["id"] for case in cases]
            or not isinstance(rows, list) or len(rows) > len(cases)
            or report.get("status") not in {"running", "completed", "failed", "deadline_exhausted"}
            or report.get("human_review") != "ungraded" or report.get("efficacy_claim") is not False):
        raise ValueError("Condition report does not match the frozen execution")
    _validate_report_identity(plan, report["model"], condition, report.get("requested_adapter_run_sha256"))
    _validate_observed(plan, contract, report["model"], condition)
    finalized = 0
    for index, row in enumerate(rows):
        case = cases[index]
        raw, generated = row.get("provider_generation", {}), row.get("generation", {})
        identity = row.get("identity_check", {})
        if (row.get("case_id") != case["id"] or row.get("condition") != condition
                or row.get("prompt_sha256") != case["prompt_sha256"][condition]
                or row.get("factual_sha256") != case["factual_sha256"]
                or row.get("factual_ids") != case["factual_ids"]
                or row.get("style_ids") != ([] if condition == "facts" else case["style_ids"])
                or not isinstance(raw.get("text"), str) or type(generated.get("completed")) is not bool
                or digest({k: v for k, v in raw.items() if k not in {"completed", "error"}})
                   != digest({k: v for k, v in generated.items() if k not in {"completed", "error"}})
                or identity.get("before") != "verified"):
            raise ValueError("Report rows are not the exact frozen case prefix")
        if identity.get("after") == "pending":
            if index != len(rows) - 1 or generated["completed"]:
                raise ValueError("Only the last report row may await post-generation identity")
            continue
        if identity.get("after") not in {"verified", "changed", "unverified"}:
            raise ValueError("Attempt has no finalized identity outcome")
        if generated["completed"] and (raw.get("completed") is not True or raw.get("error") is not None
                or generated.get("error") is not None or identity["after"] != "verified"
                or row.get("validation_errors") or raw.get("response_model") != report["served_model"]
                or raw.get("request_sha256") != digest(_expected_request(plan, condition, report["served_model"], case))):
            raise ValueError("A completed attempt lacks verified generation identity")
        finalized += 1
    return finalized


def _validate_metadata(supervisor: dict, metadata: dict, report: dict, plan: dict, contract: dict,
                       plan_sha: str, contract_sha: str) -> None:
    _check_seal(supervisor, "supervisor_sha256")
    warmup = metadata["warmup.json"]
    if (supervisor.get("schema_version") != SUPERVISOR_SCHEMA
            or supervisor.get("condition") != report["condition"]
            or supervisor.get("plan_sha256") != plan["plan_sha256"]
            or supervisor.get("plan_file_sha256") != plan_sha
            or supervisor.get("contract_sha256") != contract["contract_sha256"]
            or supervisor.get("contract_file_sha256") != contract_sha
            or supervisor.get("wrapper_sha256") != contract["wrapper_sha256"]
            or supervisor.get("launcher_sha256") != digest(metadata["launcher.json"])
            or metadata["observed-before.json"] != report["model"]
            or supervisor.get("observed_before") != _identity_hashes(report["model"])
            or warmup.get("status_code") != 200 or warmup.get("excluded_from_condition_metrics") is not True
            or warmup.get("response", {}).get("model") != report["served_model"]
            or not warmup.get("response", {}).get("choices")):
        raise ValueError("Launcher, warmup, observed identity or supervisor differs from the condition")


def _capture(source: Path, plan: dict, contract: dict, plan_file_sha: str, contract_file_sha: str,
             count: int | None, include_server_log: bool) -> tuple[dict, dict[str, bytes]]:
    report, report_bytes = _read_json(source / "condition/run.json")
    finished = _validate_report(report, plan, contract)
    count = finished if count is None else count
    if not 1 <= count <= finished:
        raise ValueError("Choose a positive prefix no larger than the finalized attempts")
    metadata, files = {}, {}
    for name in METADATA:
        metadata[name], files["metadata/" + name] = _read_json(source / name)
    supervisor, files["metadata/supervisor-observed.json"] = _read_json(source / "supervisor.json")
    _validate_metadata(supervisor, metadata, report, plan, contract, plan_file_sha, contract_file_sha)
    attempts, files["attempts.prefix.jsonl"] = _jsonl_prefix(source / "condition/attempts.jsonl", count)
    raw, files["raw-generations.prefix.jsonl"] = _jsonl_prefix(source / "condition/raw-generations.jsonl", count)
    rows = copy.deepcopy(report["runs"][:count])
    _match_logs(rows, attempts, raw)
    server_observation = None
    if include_server_log:
        with (source / "server.log").open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            log = stream.read(size)
        if len(log) != size:
            raise ValueError("Server log shrank while its bounded bytes were captured")
        files["server.log.snapshot"] = log
        server_observation = {"captured_bytes": size, "final_log": False,
            "scope": "Exact first size-at-open bytes of a mutable log; later writes are not included."}
    # The report can advance while being read. Its already-finalized prefix and
    # immutable invocation metadata must stay identical, not its mutable status.
    after, after_bytes = _read_json(source / "condition/run.json")
    _validate_report(after, plan, contract)
    mutable = {"runs", "run_sha256", "status", "elapsed_seconds", "skipped_case_ids", "failure_type"}
    if (digest(after["runs"][:count]) != digest(rows)
            or digest({k: v for k, v in report.items() if k not in mutable})
               != digest({k: v for k, v in after.items() if k not in mutable})):
        raise ValueError("The captured prefix or invocation identity changed during the read")
    for name in METADATA:
        if (source / name).read_bytes() != files["metadata/" + name]:
            raise ValueError("Immutable launcher/warmup/identity metadata changed during the read")
    for name, target in (("attempts.jsonl", "attempts.prefix.jsonl"), ("raw-generations.jsonl", "raw-generations.prefix.jsonl")):
        if _jsonl_prefix(source / "condition" / name, count)[1] != files[target]:
            raise ValueError("An already-written JSONL prefix changed during the read")
    prefix = _seal({"schema_version": SCHEMA, "status": "partial", "collectible": False,
        "condition": report["condition"], "plan_sha256": plan["plan_sha256"], "contract_sha256": contract["contract_sha256"],
        "source_report": {"file_sha256": _sha(report_bytes), "run_sha256": report["run_sha256"],
            "observed_rows": len(report["runs"]), "finalized_prefix": finished,
            "recheck_file_sha256": _sha(after_bytes), "recheck_run_sha256": after["run_sha256"]},
        "source_run_metadata": {k: v for k, v in report.items() if k not in {"runs", "run_sha256"}},
        "captured_attempts": count, "captured_completed": sum(row["generation"]["completed"] for row in rows),
        "captured_failed": sum(not row["generation"]["completed"] for row in rows), "runs": rows,
        "uncaptured_case_ids": report["planned_case_ids"][count:], "server_log_observation": server_observation,
        "human_review": "ungraded", "efficacy_claim": False}, "partial_sha256")
    files["partial-run.json"] = _json_bytes(prefix)
    return prefix, files


def snapshot_condition(plan_path: Path, contract_path: Path, source: Path, output: Path, *,
                       attempts: int | None = None, include_server_log: bool = False, read_attempts: int = 3) -> dict:
    if (output.exists() or output.resolve().is_relative_to(source.resolve())
            or source.resolve().is_relative_to(output.resolve())):
        raise ValueError("Choose a new output directory outside the active condition directory")
    if ((attempts is not None and (type(attempts) is not int or attempts < 1))
            or type(read_attempts) is not int or not 1 <= read_attempts <= 10):
        raise ValueError("Use a positive prefix count and between one and ten bounded read attempts")
    plan, plan_bytes = _read_json(plan_path)
    contract, contract_bytes = _read_json(contract_path)
    validate_execution_contract(contract, plan)
    for attempt in range(read_attempts):
        try:
            prefix, files = _capture(source, plan, contract, _sha(plan_bytes), _sha(contract_bytes), attempts, include_server_log)
            break
        except (ValueError, OSError, KeyError, TypeError):
            if attempt + 1 == read_attempts:
                raise
            time.sleep(.05)
    manifest = _seal({"schema_version": MANIFEST_SCHEMA, "status": "partial_local_snapshot", "collectible": False,
        "created_at": datetime.now(timezone.utc).isoformat(), "source_condition_directory": str(source.resolve()),
        "plan_sha256": plan["plan_sha256"], "plan_file_sha256": _sha(plan_bytes),
        "contract_sha256": contract["contract_sha256"], "contract_file_sha256": _sha(contract_bytes),
        "condition": prefix["condition"], "captured_attempts": prefix["captured_attempts"],
        "captured_completed": prefix["captured_completed"], "captured_failed": prefix["captured_failed"],
        "case_ids": [row["case_id"] for row in prefix["runs"]], "tool_sha256": _sha(Path(__file__).read_bytes()),
        "remote_persistence_performed": False, "remote_readback_verified": False,
        "source_report_included": False,
        "limitations": "Only the finalized prefix and captured metadata are saved; report seals are local observations. This is not a final condition, collector input, remote receipt or resumable generation cursor.",
        "next_action": "Archive, upload and freshly restore this snapshot externally; retain the final full-condition save and collector unchanged.",
        "human_review": "ungraded", "efficacy_claim": False,
        "files": {name: {"bytes": len(raw), "sha256": _sha(raw)} for name, raw in sorted(files.items())}}, "manifest_sha256")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name, raw in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with target.open("xb") as stream:
            stream.write(raw)
    # No manifest is published until every captured byte is written.
    with (output / "manifest.json").open("xb") as stream:
        stream.write(_json_bytes(manifest))
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--condition-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts", type=int, help="Capture exactly the first N finalized attempts; default is all currently finalized")
    parser.add_argument("--include-server-log", action="store_true")
    parser.add_argument("--read-attempts", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        result = snapshot_condition(args.plan, args.contract, args.condition_directory, args.output,
            attempts=args.attempts, include_server_log=args.include_server_log, read_attempts=args.read_attempts)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"Partial condition snapshot failed: {exc}\n")
    print(json.dumps({key: result[key] for key in ("status", "captured_attempts", "captured_completed", "captured_failed",
        "collectible", "manifest_sha256", "remote_persistence_performed", "remote_readback_verified")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
