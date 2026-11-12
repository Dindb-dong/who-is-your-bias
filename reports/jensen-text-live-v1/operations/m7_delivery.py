"""Operational M7 launch/preservation; frozen repository implementations stay unchanged.

The verify and archive commands load no model. The run command requires a
coordinator release note and owns one condition through the shared supervisor.
It never starts the second condition or retries a failed neural invocation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile


HERE = Path(__file__).absolute().parent


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def bindings():
    value = json.loads((HERE / "bindings.json").read_text())
    root = Path(value["repository"])
    for name, expected in value["repository_file_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("Frozen repository input changed: " + name)
    if sha(Path(value["adapter_run"]) / "run.json") != value["adapter_run_sha256"]:
        raise ValueError("Completed adapter run.json changed")
    return value


def command(value, condition, *, output=None, verify_only=False):
    args = [value["inference_python"], "-m", "tools.runtime_checks.text_live",
        "--repository", value["repository"], "--artifacts", value["artifacts"],
        "--cache-dir", value["cache_dir"], "--inference-python", value["inference_python"],
        "--protocol", str(Path(value["repository"]) / value["protocol"]),
        "--protocol-sha256", value["protocol_sha256"], "--condition", condition,
        "--allow-candidates"]
    if condition == "adapter":
        args += ["--adapter-run", value["adapter_run"],
            "--adapter-run-sha256", value["adapter_run_sha256"]]
    if verify_only:
        args.append("--verify-only")
    elif output is not None:
        args += ["--output", str(output)]
    else:
        raise ValueError("Actual execution requires a new output directory")
    return args


def environment(value):
    env = os.environ.copy()
    root = Path(value["repository"])
    env.update(PYTHONPATH=str(root / "src") + os.pathsep + str(root),
        OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
        TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def verify():
    value = bindings()
    results = {}
    for condition in ("base", "adapter"):
        path = HERE / ("preflight-" + condition + ".json")
        if path.exists():
            raise ValueError("Preserve the existing preflight; choose a new preparation directory")
        invocation = command(value, condition, verify_only=True)
        process = subprocess.run(invocation, cwd=value["repository"], env=environment(value),
            capture_output=True, text=True, timeout=90)
        if process.returncode:
            write_new(HERE / ("preflight-" + condition + "-failure.json"),
                {"command": invocation, "returncode": process.returncode,
                 "stdout": process.stdout, "stderr": process.stderr})
            raise RuntimeError("No-model preflight failed: " + condition)
        result = json.loads(process.stdout)
        if result.get("weights_loaded") is not False or result.get("voice_enabled") is not False:
            raise ValueError("Preflight is not explicitly model-free and voice-off")
        write_new(path, result)
        results[condition] = {"file": path.name, "sha256": sha(path),
            "weights_loaded": result["weights_loaded"], "returncode": process.returncode}
    receipt = {"schema_version": "m7-completed29-preparation.v1",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": "no_model_preflights_passed_awaiting_coordinator_release",
        "bindings_sha256": sha(HERE / "bindings.json"),
        "operational_helper_sha256": sha(Path(__file__)), "conditions": results,
        "model_loaded": False, "generation_executed": False,
        "adapter_run_sha256": value["adapter_run_sha256"],
        "protocol_sha256": value["protocol_sha256"]}
    write_new(HERE / "preparation.json", receipt)
    return receipt


def make_bundle(files, archive, metadata):
    archive = Path(archive).absolute()
    receipt_path = archive.with_name(archive.name + ".package.json")
    if archive.exists() or receipt_path.exists():
        raise ValueError("Keep previous archives/receipts; choose a new name")
    payloads = {}
    for name, path in sorted(files.items()):
        path = Path(path)
        if path.is_symlink() or not path.is_file() or name.startswith("/") or ".." in Path(name).parts:
            raise ValueError("Bundle inputs must be regular files with safe member names")
        payloads[name] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    manifest = {"schema_version": "m7-operational-artifact.v1", "metadata": metadata,
        "payloads": payloads}
    encoded = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as tar:
        member = tarfile.TarInfo("artifact-manifest.json")
        member.size = len(encoded)
        member.mode = 0o600
        tar.addfile(member, io.BytesIO(encoded))
        for name, path in sorted(files.items()):
            tar.add(path, arcname=name, recursive=False)
    for name, path in files.items():
        if sha(path) != payloads[name]["sha256"]:
            raise ValueError("Input changed while packaging: " + name)
    receipt = {"schema_version": "m7-operational-package-receipt.v1",
        "archive": str(archive), "bytes": archive.stat().st_size, "sha256": sha(archive),
        "manifest": manifest, "upload_status": "not_uploaded", "fresh_readback_verified": False}
    write_new(receipt_path, receipt)
    return {"archive": str(archive), "receipt": str(receipt_path),
        "bytes": receipt["bytes"], "sha256": receipt["sha256"], "payload_files": len(payloads)}


def package_preparation():
    value = bindings()
    preparation = json.loads((HERE / "preparation.json").read_text())
    if preparation["model_loaded"] or preparation["generation_executed"]:
        raise ValueError("Expected a no-model preparation record")
    files = {"preparation/" + p.name: p for p in HERE.iterdir() if p.is_file()}
    archive = Path(value["artifacts"]) / "runtime-checks/jensen-m7-preparation-completed29-v1.tar.gz"
    return make_bundle(files, archive, {"role": "m7-completed29-operational-preparation",
        "model_loaded": False, "generation_executed": False,
        "protocol_sha256": value["protocol_sha256"],
        "adapter_run_sha256": value["adapter_run_sha256"]})


def run_paths(value, name):
    if not re.fullmatch(r"m7-(base|adapter)-[a-z0-9_.-]{1,80}", name):
        raise ValueError("Use an explicit bounded M7 condition invocation name")
    root = Path(value["artifacts"]) / "runtime-checks"
    return {"output": root / name, "log": root / (name + ".log"),
        "launch": root / (name + ".launch.json"),
        "owner": Path(value["supervisor_directory"]) / (name + ".json")}


def package_run(name):
    value = bindings()
    paths = run_paths(value, name)
    owner = json.loads(paths["owner"].read_text())
    if owner.get("child_stopped") is not True or not owner.get("finished_at"):
        raise ValueError("Preserve only a stopped invocation; inspect an interrupted owner first")
    files = {"control/owner.json": paths["owner"], "control/launch.json": paths["launch"],
        "control/process.log": paths["log"], "preparation/bindings.json": HERE / "bindings.json",
        "preparation/m7_delivery.py": Path(__file__), "preparation/preparation.json": HERE / "preparation.json"}
    if paths["output"].is_dir():
        for path in sorted(paths["output"].rglob("*")):
            if path.is_file():
                files["run/" + path.relative_to(paths["output"]).as_posix()] = path
    summary = paths["output"] / "public-summary.json"
    public = json.loads(summary.read_text()) if summary.exists() else {}
    metadata = {"role": "actual-m7-neural-functional-evidence", "name": name,
        "invocation_state": owner["invocation_state"], "child_returncode": owner.get("child_returncode"),
        "child_stopped": True, "functional_status": public.get("status", "no_functional_report"),
        "passed_checks": public.get("passed_checks"), "total_checks": public.get("total_checks"),
        "protocol_sha256": value["protocol_sha256"],
        "adapter_run_sha256": value["adapter_run_sha256"],
        "failure_preserved": owner.get("child_returncode") != 0 or public.get("status") != "passed"}
    archive = Path(value["artifacts"]) / "runtime-checks" / ("jensen-" + name + ".tar.gz")
    return make_bundle(files, archive, metadata)


def run(condition, name, release_note):
    value = bindings()
    if not release_note.strip() or not name.startswith("m7-" + condition + "-"):
        raise ValueError("Record the coordinator release for this exact condition invocation")
    paths = run_paths(value, name)
    if any(path.exists() for path in paths.values()):
        raise ValueError("Never overwrite/retry an existing M7 invocation")
    cmd = command(value, condition, output=paths["output"])
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, text = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            meminfo[key] = text.strip()
    write_new(paths["launch"], {"condition": condition, "name": name, "command": cmd,
        "release_note": release_note, "started_at": datetime.now(timezone.utc).isoformat(),
        "repository_head": subprocess.check_output(["git", "-C", value["repository"], "rev-parse", "HEAD"], text=True).strip(),
        "bindings_sha256": sha(HERE / "bindings.json"), "cpu_count": os.cpu_count(),
        "memory_before_launch": meminfo, "free_disk_bytes": shutil.disk_usage(value["artifacts"]).free,
        "supervisor_timeout_seconds": value["supervisor_timeout_seconds"]})
    os.environ.update(environment(value))
    os.chdir(value["repository"])
    sys.path[:0] = [value["repository"], str(Path(value["repository"]) / "src")]
    from tools.adaptation.supervise import run_supervised
    state = run_supervised(cmd, Path(value["supervisor_directory"]), name, paths["log"],
        timeout=value["supervisor_timeout_seconds"], interval=1.)
    package = package_run(name)
    return {"owner": str(paths["owner"]), "child_returncode": state["child_returncode"],
        "child_stopped": state["child_stopped"], "package": package,
        "next": "Save this archive and verify fresh readback before ROOT publication or the next condition."}


def verify_archive(archive, receipt_path, file_id, library_file_id, output):
    receipt = json.loads(Path(receipt_path).read_text())
    archive = Path(archive)
    if archive.stat().st_size != receipt["bytes"] or sha(archive) != receipt["sha256"]:
        raise ValueError("Readback archive byte count/SHA-256 differs")
    expected = receipt["manifest"]
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        actual = {m.name: m for m in members}
        if (len(actual) != len(members) or any(not m.isfile() for m in members)
                or set(actual) != {"artifact-manifest.json", *expected["payloads"]}):
            raise ValueError("Readback archive has missing, duplicate or unexpected members")
        if json.load(tar.extractfile(actual["artifact-manifest.json"])) != expected:
            raise ValueError("Readback embedded manifest differs")
        for name, item in expected["payloads"].items():
            member = actual[name]
            with tar.extractfile(member) as stream:
                value = hashlib.file_digest(stream, "sha256").hexdigest()
            if member.size != item["bytes"] or value != item["sha256"]:
                raise ValueError("Readback payload differs: " + name)
    result = {"schema_version": "m7-operational-fresh-readback.v1",
        "verified_at": datetime.now(timezone.utc).isoformat(), "file_id": file_id,
        "library_file_id": library_file_id, "file_name": archive.name,
        "bytes": receipt["bytes"], "sha256": receipt["sha256"],
        "verified_payload_files": len(expected["payloads"]), "metadata": expected["metadata"],
        "status": "fresh_readback_archive_and_every_payload_verified"}
    write_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("verify")
    sub.add_parser("package-preparation")
    execute = sub.add_parser("run")
    execute.add_argument("--condition", choices=("base", "adapter"), required=True)
    execute.add_argument("--name", required=True)
    execute.add_argument("--release-note", required=True)
    package = sub.add_parser("package-run")
    package.add_argument("--name", required=True)
    restore = sub.add_parser("verify-archive")
    for arg in ("archive", "receipt", "file-id", "library-file-id", "output"):
        restore.add_argument("--" + arg, required=True)
    args = parser.parse_args()
    if args.action == "verify":
        result = verify()
    elif args.action == "package-preparation":
        result = package_preparation()
    elif args.action == "run":
        result = run(args.condition, args.name, args.release_note)
    elif args.action == "package-run":
        result = package_run(args.name)
    else:
        result = verify_archive(args.archive, args.receipt, args.file_id, args.library_file_id, args.output)
    print(json.dumps(result, indent=2))
    return 1 if args.action == "run" and result["child_returncode"] != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
