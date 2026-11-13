"""Relocate the two remaining M6 invocations without changing their sealed code.

The default invocation validates local bytes and prints a command. It loads the
tokenizer for its contract check, but never language-model weights. Execution
requires --execute and always uses a new output plus the shared process lock.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from services.language.adapters import (  # noqa: E402
    checkpoint_manifest, json_digest, tokenizer_contract, validate_adapter_run,
)
from tools.adaptation.supervise import run_supervised  # noqa: E402
from tools.adaptation_evaluation.local_condition import validate_execution_contract  # noqa: E402
from tools.adaptation_evaluation.rag_diagnostic import validate_rag_plan  # noqa: E402


ADAPTER_RUN_SHA256 = "e1102328f6cc85e9c096dc554e6255465d67870706df5d3a4f03bf091c6e3f42"
REGISTRATIONS = {
    "adapter": {
        "plan_sha256": "7fd35de01ea1a364aa7b5a2bf8547bf3a4a6cd3ebee4b3764b8ce7c8957e5c7c",
        "contract": "reports/jensen-evaluation-v1/execution-contract-bf16-v2.json",
        "contract_sha256": "9c79398b54118f284ffdc6e4592d501e1cc214e77f6a895152ae91019286b6c0",
    },
    "rag": {
        "plan_sha256": "3e2c6409b45a67b11798c17f23c6ce869aa9d71309775bf9af8f69afcc7d02d6",
        "contract": "reports/jensen-evaluation-v1/heldout-rag-v1/execution-contract-bf16-v2.json",
        "contract_sha256": "7c3c0bc535cc2f92efa582a909c01e7f44280f58fd21ae66f184408cd714464d",
    },
}
VERSIONS = {"torch": "2.11.0+cpu", "transformers": "4.57.3"}
CASE_IDS = ["heldout-qa26-no-priors-2023-turn-" + suffix for suffix in
            ("00072", "00078", "00186", "00190", "00253", "00297")]


def require(value, message):
    if not value:
        raise ValueError(message)


def unused(path):
    require(not path.exists() and not path.is_symlink(),
            f"Preserve prior evidence and choose a new path: {path}")


def prepare(args):
    registration = REGISTRATIONS[args.condition]
    plan_path = args.plan.absolute()
    contract_path = ROOT / registration["contract"]
    plan_raw, contract_raw = plan_path.read_bytes(), contract_path.read_bytes()
    plan, contract = json.loads(plan_raw), json.loads(contract_raw)
    validate_execution_contract(contract, plan)
    if args.condition == "rag":
        validate_rag_plan(plan)
        require(plan["rag_diagnostic"]["execution_conditions"] == ["facts"],
                "The separate RAG diagnostic remains base-only")
    require(plan["plan_sha256"] == registration["plan_sha256"]
            and contract["contract_sha256"] == registration["contract_sha256"]
            and plan["exposure"] == "heldout"
            and [case["id"] for case in plan["cases"]] == CASE_IDS
            and contract["runtime"] == {"device": "cpu", "dtype": "bfloat16", "threads": 8},
            "The registered six-case plan or CPU BF16 contract changed")
    require(all(plan["policy"].get(key) == value for key, value in {
        "max_tokens": 160, "context_window": 4096, "temperature": 0,
        "top_p": 1, "seed": 42, "reference_answer_supplied": False,
    }.items()), "The fixed prompt/output policy changed")

    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    require(versions == VERSIONS, "Restore Torch 2.11.0+cpu and Transformers 4.57.3 before execution")
    identity = contract["base_identity"]
    hf_home = args.hf_home.absolute()
    hub = hf_home / "hub"
    snapshot = hub / ("models--" + identity["base_model"].replace("/", "--")) / "snapshots" / identity["base_revision"]
    base_files = checkpoint_manifest(snapshot)
    require(json_digest(base_files) == identity["base_checkpoint_manifest_sha256"],
            "The local base snapshot differs from the original checkpoint manifest")

    # Read the actual local tokenizer; from_pretrained on this directory does
    # not load model weights and cannot retrieve another revision.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    require(json_digest(tokenizer_contract(tokenizer)) == identity["tokenizer_contract_sha256"],
            "The restored tokenizer contract differs from the frozen base")
    adapter_path = args.adapter_run.absolute() if args.adapter_run else None
    if args.condition == "adapter":
        require(adapter_path is not None, "The matched adapter condition needs the completed29 serving directory")
        verified = validate_adapter_run(adapter_path, base_model=identity["base_model"],
            base_revision=identity["base_revision"], expected_run_sha256=ADAPTER_RUN_SHA256)
        require(verified.report["optimizer_steps"] == 29, "Use the saved completed29 run")
        verified.verify_base_checkpoint(snapshot)
        verified.verify_tokenizer(tokenizer)
    else:
        require(adapter_path is None, "The registered RAG diagnostic cannot use an adapter")

    require(re.fullmatch(r"[a-zA-Z0-9_.-]{1,100}", args.name), "Use a simple process name")
    output, state = args.output.absolute(), args.state_dir.absolute()
    log = state / f"{args.name}.log"
    for path in (output, state / f"{args.name}.json", state / f"stop-{args.name}.json", log):
        unused(path)
    # Keep the interpreter's venv spelling: resolving the executable's symlink
    # can silently select a different base environment.
    command = [str(Path(sys.executable).absolute()), "-m"]
    if args.condition == "adapter":
        command += ["tools.adaptation_evaluation.local_condition", "run", "--condition", "adapter",
                    "--adapter-run", str(adapter_path), "--adapter-run-sha256", ADAPTER_RUN_SHA256,
                    "--dtype", "bfloat16", "--threads", "8"]
    else:
        command += ["tools.adaptation_evaluation.rag_diagnostic", "run"]
    command += ["--plan", str(plan_path), "--contract", str(contract_path), "--timeout", "180", "--output", str(output)]
    environment = {
        "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT),
        "HF_HOME": str(hf_home), "HF_HUB_CACHE": str(hub), "HUGGINGFACE_HUB_CACHE": str(hub),
        "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "false",
    }
    return {
        "schema_version": "jensen-m6-relocated-launch.v1", "condition": args.condition,
        "executed": False, "model_weights_loaded": False, "tokenizer_contract_checked": True,
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "command": command, "working_directory": str(ROOT), "environment": environment,
        "plan_sha256": plan["plan_sha256"], "plan_file_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "contract_sha256": contract["contract_sha256"], "contract_file_sha256": hashlib.sha256(contract_raw).hexdigest(),
        "base_identity": identity, "cached_checkpoint_files": len(base_files), "versions": versions,
        "adapter_run_sha256": ADAPTER_RUN_SHA256 if adapter_path else None,
        "shared_lock": str(state / "active.lock"), "process_receipt": str(state / f"{args.name}.json"),
        "log": str(log), "process_timeout_seconds": 3600,
        "note": "Relocation only; original prompts, cases, decoded-output budgets, and sealed inference code remain unchanged.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", choices=REGISTRATIONS)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--adapter-run", type=Path)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        prepared = prepare(args)
        if not args.execute:
            print(json.dumps(prepared, indent=2))
            return 0
        os.chdir(ROOT)
        os.environ.update(prepared["environment"])
        result = run_supervised(prepared["command"], args.state_dir.absolute(), args.name,
                                Path(prepared["log"]), timeout=prepared["process_timeout_seconds"])
        print(json.dumps({key: result.get(key) for key in (
            "name", "invocation_state", "child_returncode", "child_stopped", "elapsed_seconds")}))
        return 0 if (result.get("invocation_state") == "exited"
                     and result.get("child_returncode") == 0 and result.get("child_stopped") is True) else 1
    except (OSError, ValueError, importlib.metadata.PackageNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
