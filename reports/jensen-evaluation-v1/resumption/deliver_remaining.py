"""Bind existing M6 delivery helpers to this checkout and one new artifact root.

No inference or transfer occurs here. Original helpers retain their byte
identities and validation; this wrapper changes only explicit workspace paths.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
OPERATIONS = ROOT / "reports/jensen-evaluation-v1/operations"
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(OPERATIONS)]


def load(name, artifact_root):
    expected = json.loads((OPERATIONS / "source-manifest.json").read_text())["files"][name + ".py"]
    source = OPERATIONS / (name + ".py")
    raw = source.read_bytes()
    if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        raise ValueError(f"Original operational helper changed: {name}")
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != source.resolve():
        raise ValueError(f"Unexpected imported helper: {name}")
    module.ROOT, module.ART = ROOT, artifact_root
    return module


def bindings(artifact_root, inference_python):
    return {
        "schema_version": "jensen-m6-relocated-delivery.v1",
        "repository": "Dindb-dong/who-is-your-bias", "checkout": str(ROOT),
        "artifact_root": str(artifact_root), "inference_python": str(inference_python),
        "delivery_wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "launch_wrapper_sha256": hashlib.sha256(Path(__file__).with_name("launch_remaining.py").read_bytes()).hexdigest(),
        "original_helper_manifest_sha256": hashlib.sha256((OPERATIONS / "source-manifest.json").read_bytes()).hexdigest(),
        "matched_output": "text-evaluation-v1/execution-heldout-bf16-v2/adapter",
        "rag_output": "text-evaluation-v1/execution-heldout-rag-bf16-v2",
        "shared_process_state": "text-process",
        "matched_process_name": "m6-heldout-adapter-bf16-v2",
        "rag_process_name": "m6-heldout-rag-bf16-v2",
        "relocation_only": True, "frozen_inference_or_prompt_changes": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bindings", "matched-validate", "matched-package", "matched-finalize",
        "partial-package", "partial-finalize", "collect", "rag-package", "rag-verify"))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--inference-python", type=Path, default=ROOT / ".venv-inference/bin/python")
    parser.add_argument("--attempts", type=int)
    parser.add_argument("--mode", choices=("partial3", "final", "failure"))
    parser.add_argument("--artifact-receipt", type=Path)
    args = parser.parse_args()
    artifact_root = args.artifact_root.absolute()
    inference_python = args.inference_python.absolute()
    partial_command = args.command.startswith("partial-")
    rag_command = args.command.startswith("rag-")
    if partial_command != (args.attempts is not None) or (partial_command and not 1 <= args.attempts <= 6):
        parser.error("Only partial commands require --attempts from 1 through 6")
    if rag_command != (args.mode is not None):
        parser.error("Only RAG commands require an explicit --mode")
    if (args.command == "rag-verify") != (args.artifact_receipt is not None):
        parser.error("Only rag-verify requires --artifact-receipt")
    try:
        base = load("m6_partial_delivery", artifact_root)
        bound = bindings(artifact_root, inference_python)
        raw_bindings = base.encoded(bound)
        if args.command == "bindings":
            print(raw_bindings.decode(), end="")
            return 0
        base.write_new(artifact_root / "m6-relocation-bindings.json", raw_bindings)
        if args.command.startswith("matched-"):
            module = load("m6_condition_delivery", artifact_root)
            action = args.command.removeprefix("matched-")
            if action == "validate":
                result = module.validate("heldout", "adapter", artifact_root / bound["matched_output"])
                print(json.dumps({"validated_condition_files": len(result[4]), "status": result[3]["status"]}))
            else:
                getattr(module, action)("heldout", "adapter")
        elif partial_command:
            getattr(base, args.command.removeprefix("partial-"))("heldout", "adapter", args.attempts)
        elif args.command == "collect":
            load("collect_package_m6", artifact_root).collect_package("heldout")
        elif rag_command:
            launch = load("launch_m6_rag", artifact_root)
            launch.PLAN = artifact_root / "text-evaluation-v1/heldout-rag-v1/plan.json"
            launch.CONTRACT = ROOT / "reports/jensen-evaluation-v1/heldout-rag-v1/execution-contract-bf16-v2.json"
            launch.OUTPUT = artifact_root / bound["rag_output"]
            launch.STATE_DIRECTORY = artifact_root / "text-process"
            launch.LOG = launch.STATE_DIRECTORY / f"{launch.NAME}.log"
            launch.INFERENCE_PYTHON = inference_python
            module = load("m6_rag_delivery", artifact_root)
            original_members = module.fixed_members

            def relocated_members():
                return {**original_members(),
                    "source/resumption/deliver_remaining.py": Path(__file__).read_bytes(),
                    "source/resumption/launch_remaining.py": Path(__file__).with_name("launch_remaining.py").read_bytes(),
                    "inputs/relocation-bindings.json": raw_bindings}

            # Retain original helpers and their checks; archive the explicit
            # relocation wrapper/bindings alongside them for exact recovery.
            module.fixed_members = relocated_members
            if args.command == "rag-package":
                module.package(args.mode)
            else:
                module.verify(args.mode, args.artifact_receipt.absolute())
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
