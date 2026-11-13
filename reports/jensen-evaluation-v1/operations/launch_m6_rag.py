"""Launch the sealed six-case, base-only heldout RAG diagnostic on explicit request.

Default invocation validates registered inputs and prints the proposed command.
It creates no output, claims no completed condition, and starts no model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from tools.adaptation.supervise import run_supervised
from tools.adaptation_evaluation.local_condition import validate_execution_contract
from tools.adaptation_evaluation.rag_diagnostic import validate_rag_plan

ROOT = Path('/workspace/scratch/6f64e1b53384/who-is-your-bias')
ART = Path('/workspace/scratch/6f64e1b53384/artifacts/jensen-2026-11-13')
NAME = 'm6-heldout-rag-bf16-v2'
PLAN = ART / 'text-evaluation-v1/heldout-rag-v1/plan.json'
CONTRACT = ROOT / 'reports/jensen-evaluation-v1/heldout-rag-v1/execution-contract-bf16-v2.json'
OUTPUT = ART / 'text-evaluation-v1/execution-heldout-rag-bf16-v2'
STATE_DIRECTORY = ART / 'text-process'
LOG = ART / f'{NAME}.log'
INFERENCE_PYTHON = ROOT / '.venv-inference/bin/python'
PLAN_FILE_SHA256 = '58338fe59a75eac2ff86e5e52fece7ea268145987957709d4df72d04ea2b7adb'
PLAN_SHA256 = '3e2c6409b45a67b11798c17f23c6ce869aa9d71309775bf9af8f69afcc7d02d6'
CONTRACT_FILE_SHA256 = '3aa4356d0466a9d3a60daaa007f164091ea42869a67689cf9112adbb77a86d07'
CONTRACT_SHA256 = '7c3c0bc535cc2f92efa582a909c01e7f44280f58fd21ae66f184408cd714464d'
CASE_IDS = [f'heldout-qa26-no-priors-2023-turn-{suffix}' for suffix in
            ('00072', '00078', '00186', '00190', '00253', '00297')]
DECLARED_ENVIRONMENT = {
    'PYTHONPATH': 'src:.',
    'HF_HOME': '/workspace/scratch/6f64e1b53384/model-cache',
    'OMP_NUM_THREADS': '8',
    'MKL_NUM_THREADS': '8',
    'OPENBLAS_NUM_THREADS': '8',
    'TOKENIZERS_PARALLELISM': 'false',
}


def verified_json(path: Path, expected_sha256: str) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f'Registered input bytes changed: {path}')
    return json.loads(raw)


def validate_inputs() -> None:
    plan = verified_json(PLAN, PLAN_FILE_SHA256)
    contract = verified_json(CONTRACT, CONTRACT_FILE_SHA256)
    validate_rag_plan(plan)
    validate_execution_contract(contract, plan)
    if (plan['plan_sha256'] != PLAN_SHA256
            or contract['contract_sha256'] != CONTRACT_SHA256
            or plan['exposure'] != 'heldout'
            or [row['id'] for row in plan['cases']] != CASE_IDS
            or plan['rag_diagnostic']['registered_case_ids'] != CASE_IDS
            or plan['rag_diagnostic']['generation_case_ids'] != CASE_IDS
            or plan['rag_diagnostic']['execution_conditions'] != ['facts']
            or contract['runtime'] != {'device': 'cpu', 'dtype': 'bfloat16', 'threads': 8}
            or any(plan['policy'].get(key) != value for key, value in {
                'max_tokens': 160, 'context_window': 4096,
                'temperature': 0, 'top_p': 1, 'seed': 42,
                'reference_answer_supplied': False,
            }.items())):
        raise ValueError('The six-case base-only BF16 registration changed')
    if not INFERENCE_PYTHON.is_file():
        raise ValueError('The established inference Python is unavailable')
    for path in (OUTPUT, STATE_DIRECTORY / f'{NAME}.json',
                 STATE_DIRECTORY / f'stop-{NAME}.json', LOG):
        if path.exists() or path.is_symlink():
            raise ValueError(f'Preserve prior invocation evidence; path already exists: {path}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true',
                        help='Actually acquire the shared supervisor lock and launch the model')
    args = parser.parse_args()
    try:
        validate_inputs()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    command = [str(INFERENCE_PYTHON), '-m', 'tools.adaptation_evaluation.rag_diagnostic',
               'run', '--plan', str(PLAN), '--contract', str(CONTRACT),
               '--timeout', '180', '--output', str(OUTPUT)]
    if not args.execute:
        print(json.dumps({
            'name': NAME, 'executed': False, 'sealed_inputs_validated': True,
            'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'proposed_command': command, 'environment': DECLARED_ENVIRONMENT,
            'supervisor_timeout_seconds': 3600,
            'shared_lock': str(STATE_DIRECTORY / 'active.lock'),
            'supervisor_receipt': str(STATE_DIRECTORY / f'{NAME}.json'),
            'supervisor_log': str(LOG), 'condition_output': str(OUTPUT),
            'plan_sha256': PLAN_SHA256, 'plan_file_sha256': PLAN_FILE_SHA256,
            'contract_sha256': CONTRACT_SHA256, 'contract_file_sha256': CONTRACT_FILE_SHA256,
            'execution_conditions': ['facts'], 'registered_case_ids': CASE_IDS,
            'runtime': {'device': 'cpu', 'dtype': 'bfloat16', 'threads': 8},
            'model_generations': 0,
            'scope': 'Separate offline heldout retrieval-to-generation diagnostic; no matched comparison',
        }, indent=2))
        return 0
    os.chdir(ROOT)
    os.environ.update(DECLARED_ENVIRONMENT)
    result = run_supervised(command, STATE_DIRECTORY, NAME, LOG, timeout=3600)
    summary = {key: result.get(key) for key in (
        'name', 'invocation_state', 'child_returncode', 'child_stopped',
        'elapsed_seconds', 'max_waited_child_rss_bytes')}
    summary['condition_supervisor'] = str(OUTPUT / 'supervisor.json')
    summary['condition_completion_claimed'] = False
    print(json.dumps(summary))
    return 0 if (result.get('invocation_state') == 'exited'
                 and result.get('child_returncode') == 0
                 and result.get('child_stopped') is True) else 1


if __name__ == '__main__':
    raise SystemExit(main())
