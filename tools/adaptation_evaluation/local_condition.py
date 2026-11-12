"""Run one frozen condition, stop its server, then allow a durable save boundary.

This additive supervisor does not change the original comparison protocol or
its sealed plans. Collection requires separate verified persistence receipts.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

import httpx

from bias.config import ProviderSettings
from services.language.adapters import immutable_revision, repository_id, validate_adapter_run
from .protocol import CONDITIONS, _context, digest, validate_plan
from .run import IDENTITY_FIELDS, _validate_base, compare_reports, export_review, read_identity, run_condition

CONTRACT_SCHEMA = "text-single-condition-execution-contract.v1"
SUPERVISOR_SCHEMA = "text-single-condition-supervisor.v1"
DURABILITY_SCHEMA = "text-condition-durability.v1"
SHA256 = re.compile(r"[a-f0-9]{64}")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _valid_base_identity(identity: dict) -> bool:
    return (isinstance(identity, dict) and set(identity) == set(IDENTITY_FIELDS)
        and repository_id(identity.get("base_model")) and immutable_revision(identity.get("base_revision"))
        and identity.get("chat_template_options") == {"enable_thinking": False}
        and all(isinstance(identity.get(key), str) and SHA256.fullmatch(identity[key])
                for key in ("base_checkpoint_manifest_sha256", "tokenizer_contract_sha256")))


def _valid_runtime(runtime: dict) -> bool:
    return (isinstance(runtime, dict) and set(runtime) == {"device", "dtype", "threads"}
        and runtime.get("device") == "cpu" and runtime.get("dtype") in {"float32", "bfloat16"}
        and type(runtime.get("threads")) is int and 1 <= runtime["threads"] <= 16)


def make_execution_contract(plan_hashes: list[str], base_identity: dict, *, dtype="float32", threads=8) -> dict:
    if not plan_hashes or len(set(plan_hashes)) != len(plan_hashes) or any(not SHA256.fullmatch(x) for x in plan_hashes):
        raise ValueError("Register unique canonical frozen plan hashes")
    runtime = {"device": "cpu", "dtype": dtype, "threads": threads}
    if not _valid_base_identity(base_identity) or not _valid_runtime(runtime):
        raise ValueError("Register independently verified base bytes/tokenizer and one common runtime")
    contract = {"schema_version": CONTRACT_SCHEMA,
        "wrapper_file": "tools/adaptation_evaluation/local_condition.py", "wrapper_sha256": _sha(Path(__file__)),
        "plan_sha256": sorted(plan_hashes), "conditions": list(CONDITIONS),
        "base_identity": base_identity, "runtime": runtime,
        "decoding": "unchanged frozen plan policy", "warmup": {"max_tokens": 12, "excluded_per_condition": True},
        "persistence_boundary": "one condition per process invocation; save and freshly verify its complete directory before another condition",
        "collection": "requires three sealed condition reports and matching fresh-readback file receipts",
        "timing_scope": "separate model launch and excluded warmup for each condition; sequential order is not counterbalanced"}
    contract["contract_sha256"] = digest(contract)
    return contract


def validate_execution_contract(contract: dict, plan: dict) -> None:
    unsigned = dict(contract)
    seal = unsigned.pop("contract_sha256", None)
    if (contract.get("schema_version") != CONTRACT_SCHEMA or digest(unsigned) != seal
            or contract.get("wrapper_file") != "tools/adaptation_evaluation/local_condition.py"
            or contract.get("wrapper_sha256") != _sha(Path(__file__))
            or plan["plan_sha256"] not in contract.get("plan_sha256", [])
            or contract.get("conditions") != list(CONDITIONS)
            or not _valid_base_identity(contract.get("base_identity"))
            or any(contract["base_identity"].get(key) != value for key, value in (plan.get("base_identity") or {}).items())
            or not _valid_runtime(contract.get("runtime"))
            or contract.get("warmup") != {"max_tokens": 12, "excluded_per_condition": True}):
        raise ValueError("Execution wrapper, frozen plan or registered runtime contract changed")
    validate_plan(plan)


def _sealed_supervisor(path: Path, report: dict) -> None:
    report.pop("supervisor_sha256", None)
    report["supervisor_sha256"] = digest(report)
    _write(path, report)


def _identity_hashes(observed: dict) -> dict:
    return {name + "_sha256": digest(observed[name]) for name in ("identity", "adapter", "runtime")}


def _validate_observed(plan: dict, contract: dict, observed: dict, condition: str) -> None:
    _validate_base(plan, observed, condition)
    if (observed["identity"] != contract["base_identity"]
            or any(observed["runtime"].get(key) != value for key, value in contract["runtime"].items())):
        raise ValueError("Observed base bytes, tokenizer or runtime differs from the execution contract")


def _review_data(plan: dict, report: dict) -> dict:
    cases = {case["id"]: case for case in plan["cases"]}
    return {"schema_version": "text-single-condition-review-data.v1", "plan_sha256": plan["plan_sha256"],
        "condition": report["condition"], "blind": False, "sharing_scope": "private condition data; use collection for blind export",
        "human_review": "ungraded", "samples": [{"case_id": row["case_id"],
            "question": cases[row["case_id"]]["question"], "reference_answer": cases[row["case_id"]].get("reference_answer"),
            "factual_evidence": _context(cases[row["case_id"]]["messages"]["facts"])["factual_evidence"],
            "generation": row["generation"], "judgments": row["judgments"]} for row in report["runs"]]}


def run_local_condition(plan_path: Path, contract_path: Path, condition: str, output: Path, *,
                        adapter_path: Path | None = None, adapter_sha256: str | None = None,
                        dtype="float32", threads=8, timeout=180., total_seconds: float | None = None) -> dict:
    plan = json.loads(plan_path.read_text())
    contract = json.loads(contract_path.read_text())
    validate_execution_contract(contract, plan)
    identity = plan.get("base_identity") or {}
    if (condition not in CONDITIONS or output.exists() or dtype != contract["runtime"]["dtype"]
            or type(threads) is not int or threads != contract["runtime"]["threads"]
            or not repository_id(identity.get("base_model")) or not immutable_revision(identity.get("base_revision"))
            or isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout)
            or not 0 < timeout <= 600):
        raise ValueError("Choose a new output, one condition and the registered bounded CPU runtime")
    total_seconds = min(14400., timeout * len(plan["cases"]) + 120.) if total_seconds is None else total_seconds
    if (isinstance(total_seconds, bool) or not isinstance(total_seconds, (int, float))
            or not math.isfinite(total_seconds) or not 0 < total_seconds <= 14400):
        raise ValueError("Invalid condition time budget")
    if condition == "adapter":
        if adapter_path is None or not isinstance(adapter_sha256, str) or not SHA256.fullmatch(adapter_sha256):
            raise ValueError("Adapter condition requires the exact completed run and receipt SHA-256")
        adapter = validate_adapter_run(adapter_path, base_model=identity["base_model"],
            base_revision=identity["base_revision"], expected_run_sha256=adapter_sha256)
        if adapter.run_sha256 != adapter_sha256:
            raise ValueError("Validated adapter receipt changed")
    elif adapter_path is not None or adapter_sha256 is not None:
        raise ValueError("Base conditions must not configure an adapter")

    output.mkdir(parents=True)
    started = time.monotonic()
    supervisor = {"schema_version": SUPERVISOR_SCHEMA, "condition": condition, "status": "starting",
        "plan_sha256": plan["plan_sha256"], "plan_file_sha256": _sha(plan_path),
        "contract_sha256": contract["contract_sha256"], "contract_file_sha256": _sha(contract_path),
        "wrapper_sha256": _sha(Path(__file__)), "started_at": datetime.now(timezone.utc).isoformat(),
        "requested_runtime": {"device": "cpu", "dtype": dtype, "threads": threads,
                              "input_limit": plan["policy"]["context_window"], "output_limit": plan["policy"]["max_tokens"]},
        "time_budget": {"per_case_seconds": timeout, "condition_seconds": total_seconds,
                        "startup_seconds": 30, "warmup_seconds": timeout},
        "adapter_run_sha256": adapter_sha256, "server_stopped": False,
        "timing_scope": contract["timing_scope"], "warmup_included_in_condition_metrics": False,
        "next_action": "save the complete condition directory and verify fresh readback before starting another condition",
        "human_review": "ungraded", "efficacy_claim": False}
    _sealed_supervisor(output / "supervisor.json", supervisor)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    settings = {"MODEL": identity["base_model"], "REVISION": identity["base_revision"],
        "SERVED_MODEL": "comparison-local", "DTYPE": dtype, "DEVICE": "cpu", "THREADS": str(threads),
        "LOCAL_FILES_ONLY": "true", "GENERATION_TIMEOUT": str(timeout),
        "MAX_INPUT_TOKENS": str(plan["policy"]["context_window"]),
        "MAX_OUTPUT_TOKENS": str(max(32, plan["policy"]["max_tokens"]))}
    if condition == "adapter":
        settings.update(ADAPTER_RUN=str(adapter_path.resolve()), ADAPTER_RUN_SHA256=adapter_sha256)
    declared_env = {"BIAS_LANGUAGE_" + key: value for key, value in settings.items()}
    declared_env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                        OPENBLAS_NUM_THREADS=str(threads), TOKENIZERS_PARALLELISM="false")
    env = {key: value for key, value in os.environ.items() if not key.startswith("BIAS_LANGUAGE_")}
    env.update(declared_env)
    # Preserve a virtualenv interpreter's spelling; resolve() can dereference
    # its symlink and silently select a base environment without dependencies.
    command = [str(Path(sys.executable).absolute()), "-m", "uvicorn", "services.language.app:app", "--host", "127.0.0.1", "--port", str(port)]
    launcher = {"command": command, "declared_environment": declared_env,
        "transport": "server and HTTP client in this one invocation", "inherited_environment_distributed": False}
    _write(output / "launcher.json", launcher)
    supervisor["launcher_sha256"] = digest(launcher)
    provider = ProviderSettings(f"http://127.0.0.1:{port}/v1", "comparison-local")
    process, report = None, None
    try:
        with (output / "server.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            supervisor["server_pid"] = process.pid
            while time.monotonic() - started < 30:
                if process.poll() is not None:
                    raise ValueError("Language server exited before listening; inspect server.log")
                try:
                    response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1, follow_redirects=False)
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.1)
            else:
                raise TimeoutError("Language server did not start within 30 seconds")
            supervisor["startup_ready_ms"] = round((time.monotonic() - started) * 1000, 3)
            supervisor["status"] = "warming"
            _sealed_supervisor(output / "supervisor.json", supervisor)
            warm_started = time.monotonic()
            warm = httpx.post(provider.base_url + "/chat/completions", timeout=timeout, follow_redirects=False,
                json={"model": provider.model, "messages": [{"role": "user", "content": "Reply with one word: Ready"}],
                      "stream": False, "max_tokens": 12, "temperature": 0, "seed": 42})
            try:
                warm_body = warm.json()
            except ValueError:
                warm_body = {"unparsed_response": warm.text}
            _write(output / "warmup.json", {"status_code": warm.status_code, "response": warm_body,
                                          "excluded_from_condition_metrics": True})
            if warm.status_code != 200:
                raise ValueError("Language warmup failed; inspect server.log")
            if not isinstance(warm_body, dict) or warm_body.get("model") != provider.model or not warm_body.get("choices"):
                raise ValueError("Warmup returned an unexpected model or missing completion")
            supervisor["warmup_request_ms"] = round((time.monotonic() - warm_started) * 1000, 3)
            supervisor["startup_and_warmup_ms"] = round((time.monotonic() - started) * 1000, 3)
            before = asyncio.run(read_identity(provider, condition=condition, adapter_sha256=adapter_sha256,
                                                 deadline=time.monotonic() + timeout))
            _validate_observed(plan, contract, before, condition)
            _write(output / "observed-before.json", before)
            supervisor["observed_before"] = _identity_hashes(before)
            supervisor["status"] = "running"
            _sealed_supervisor(output / "supervisor.json", supervisor)
            report = asyncio.run(run_condition(plan, condition, provider, output / "condition",
                adapter_sha256=adapter_sha256, timeout=timeout, total_seconds=total_seconds))
            after = asyncio.run(read_identity(provider, condition=condition, adapter_sha256=adapter_sha256,
                                                deadline=time.monotonic() + timeout))
            _write(output / "observed-after.json", after)
            supervisor["observed_after"] = _identity_hashes(after)
            if before != after:
                raise ValueError("Model/runtime identity changed across the condition invocation")
            supervisor["status"] = report["status"]
    except BaseException as exc:
        supervisor.update(status="failed", failure_type=type(exc).__name__)
        raise
    finally:
        active_exception = sys.exc_info()[1]
        finalization_error = None
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired) as exc:
                finalization_error = exc
                supervisor.update(status="failed", cleanup_failure_type=type(exc).__name__)
            supervisor["server_returncode"] = process.returncode
            supervisor["server_stopped"] = process.poll() is not None
        else:
            supervisor["server_stopped"] = True
        run_path = output / "condition/run.json"
        if report is None and run_path.is_file():
            try:
                report = json.loads(run_path.read_text())
            except (OSError, ValueError):
                supervisor["partial_report_unreadable"] = True
        if report is not None:
            try:
                _write(output / "condition-review.json", _review_data(plan, report))
                supervisor["condition_report_sha256"] = _sha(run_path)
                supervisor["condition_run_sha256"] = report["run_sha256"]
                supervisor["attempted"] = len(report["runs"])
                supervisor["completed"] = sum(row["generation"]["completed"] for row in report["runs"])
            except (OSError, KeyError, TypeError, ValueError) as exc:
                finalization_error = finalization_error or exc
                supervisor.update(status="failed", review_export_failure_type=type(exc).__name__)
        supervisor["elapsed_seconds"] = round(time.monotonic() - started, 3)
        supervisor["finished_at"] = datetime.now(timezone.utc).isoformat()
        supervisor["files_sha256"] = {str(p.relative_to(output)): _sha(p) for p in sorted(output.rglob("*"))
            if p.is_file() and p != output / "supervisor.json" and not p.name.endswith(".part")}
        _sealed_supervisor(output / "supervisor.json", supervisor)
        if finalization_error is not None and active_exception is None:
            raise finalization_error
    return supervisor


def collect_conditions(plan_path: Path, contract_path: Path, directories: list[Path], receipts_path: Path, output: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    contract = json.loads(contract_path.read_text())
    validate_execution_contract(contract, plan)
    receipts = json.loads(receipts_path.read_text())
    if (output.exists() or len(directories) != 3 or len(set(p.resolve() for p in directories)) != 3
            or receipts.get("schema_version") != DURABILITY_SCHEMA):
        raise ValueError("Choose a new collection and three distinct condition directories with verified receipts")
    entries = receipts.get("conditions", [])
    by_condition = {row.get("condition"): row for row in entries}
    if len(entries) != 3 or set(by_condition) != set(CONDITIONS):
        raise ValueError("Persistence receipts must cover exactly facts, style and adapter")
    reports, supervisors = [], []
    for directory in directories:
        supervisor = json.loads((directory / "supervisor.json").read_text())
        unsigned = dict(supervisor)
        seal = unsigned.pop("supervisor_sha256", None)
        if (digest(unsigned) != seal or supervisor.get("schema_version") != SUPERVISOR_SCHEMA
                or supervisor.get("plan_sha256") != plan["plan_sha256"]
                or supervisor.get("contract_sha256") != contract["contract_sha256"]
                or supervisor.get("wrapper_sha256") != contract["wrapper_sha256"]
                or supervisor.get("server_stopped") is not True):
            raise ValueError("Condition supervision identity, completion or seal changed")
        actual = {str(p.relative_to(directory)): _sha(p) for p in sorted(directory.rglob("*")) if p.is_file()}
        expected = {**supervisor["files_sha256"], "supervisor.json": _sha(directory / "supervisor.json")}
        receipt = by_condition.get(supervisor["condition"], {})
        archive = receipt.get("archive", {})
        if (actual != expected or receipt.get("status") != "uploaded_and_fresh_readback_verified"
                or receipt.get("plan_sha256") != plan["plan_sha256"] or receipt.get("files_sha256") != actual
                or not isinstance(archive.get("bytes"), int) or isinstance(archive.get("bytes"), bool) or archive["bytes"] <= 0
                or not isinstance(archive.get("sha256"), str) or not SHA256.fullmatch(archive["sha256"])
                or any(not isinstance(archive.get(key), str) or not archive[key].strip() for key in ("file_id", "library_file_id"))):
            raise ValueError("Every condition file requires matching fresh-readback persistence evidence")
        report = json.loads((directory / "condition/run.json").read_text())
        if (report["condition"] != supervisor["condition"] or report["run_sha256"] != supervisor.get("condition_run_sha256")
                or _sha(directory / "condition/run.json") != supervisor.get("condition_report_sha256")):
            raise ValueError("The condition report differs from its saved supervisor")
        _validate_observed(plan, contract, report["model"], report["condition"])
        reports.append(report)
        supervisors.append(supervisor)
    summary = compare_reports(plan, reports)
    summary["local_execution"] = {"supervision":"one real server per condition invocation",
        "conditions_by_start_time":[s["condition"] for s in sorted(supervisors,key=lambda s:s["started_at"])],
        "warmup_excluded_per_condition":True, "order_counterbalanced":False,
        "contract_sha256":contract["contract_sha256"], "durability_receipts_file_sha256":_sha(receipts_path),
        "supervisors":[{k:s.get(k) for k in ("condition","status","failure_type","supervisor_sha256","startup_ready_ms","warmup_request_ms")} for s in supervisors],
        "persistence_validation":"Matches supplied verified receipts to every local file; this collector does not itself contact remote storage."}
    output.mkdir(parents=True)
    _write(output / "comparison.json", summary)
    review = export_review(plan, reports, output / "blind-review")
    _write(output / "review-summary.json", review)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--condition", choices=CONDITIONS, required=True)
    run.add_argument("--adapter-run", type=Path)
    run.add_argument("--adapter-run-sha256")
    run.add_argument("--dtype", choices=["float32","bfloat16"], default="float32")
    run.add_argument("--threads", type=int, default=8)
    run.add_argument("--timeout", type=float, default=180)
    run.add_argument("--total-seconds", type=float)
    collect = commands.add_parser("collect")
    collect.add_argument("--condition-directory", type=Path, action="append", required=True)
    collect.add_argument("--durability-receipts", type=Path, required=True)
    for command in [run,collect]:
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--contract", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "run":
            result = run_local_condition(args.plan,args.contract,args.condition,args.output,
                adapter_path=args.adapter_run,adapter_sha256=args.adapter_run_sha256,dtype=args.dtype,
                threads=args.threads,timeout=args.timeout,total_seconds=args.total_seconds)
            print(json.dumps({key:result.get(key) for key in ("condition","status","attempted","completed","server_stopped","next_action")}))
        else:
            result = collect_conditions(args.plan,args.contract,args.condition_directory,args.durability_receipts,args.output)
            print(json.dumps({"plan_sha256":result["plan_sha256"],"conditions":list(result["conditions"]),"human_review":result["human_review"]}))
    except (OSError,ValueError,TimeoutError,httpx.HTTPError) as exc:
        parser.exit(2,f"Single-condition evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
