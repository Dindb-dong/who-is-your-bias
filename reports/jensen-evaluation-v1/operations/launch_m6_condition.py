"""Operational entry point for the existing frozen single-condition runner."""
import argparse
import json
import os
from pathlib import Path
import sys
from tools.adaptation.supervise import run_supervised

ROOT = Path('/workspace/scratch/6f64e1b53384/who-is-your-bias')
ART = Path('/workspace/scratch/6f64e1b53384/artifacts/jensen-2026-11-13')
parser = argparse.ArgumentParser()
parser.add_argument('exposure', choices=['development','heldout'])
parser.add_argument('condition', choices=['facts','style','adapter'])
parser.add_argument('--adapter-run', type=Path)
parser.add_argument('--adapter-run-sha256')
parser.add_argument('--execute', action='store_true')
args = parser.parse_args()
if args.condition == 'adapter':
    if args.adapter_run is None or not args.adapter_run_sha256:
        parser.error('Adapter requires the fresh completed serving run and exact run.json SHA-256')
elif args.adapter_run is not None or args.adapter_run_sha256 is not None:
    parser.error('A base condition cannot use an adapter')
name = f'm6-{args.exposure}-{args.condition}-bf16-v2'
output = ART / f'text-evaluation-v1/execution-{args.exposure}-bf16-v2/{args.condition}'
if output.exists() or (ART / f'text-process/{name}.json').exists():
    parser.error('The output or supervision receipt already exists; do not rerun or overwrite this attempt')
command = [str(ROOT / '.venv-inference/bin/python'), '-m', 'tools.adaptation_evaluation.local_condition', 'run',
    '--condition', args.condition, '--plan', str(ART / f'text-evaluation-v1/frozen-v2/{args.exposure}/plan.json'),
    '--contract', 'reports/jensen-evaluation-v1/execution-contract-bf16-v2.json',
    '--dtype', 'bfloat16', '--threads', '8', '--timeout', '180', '--output', str(output)]
if args.condition == 'adapter':
    command += ['--adapter-run', str(args.adapter_run), '--adapter-run-sha256', args.adapter_run_sha256]
declared = {'PYTHONPATH': 'src:.', 'HF_HOME': '/workspace/scratch/6f64e1b53384/model-cache',
    'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8', 'TOKENIZERS_PARALLELISM': 'false'}
if not args.execute:
    print(json.dumps({'proposed_command': command, 'environment': declared, 'timeout_seconds': 3600,
        'name': name, 'executed': False}))
    sys.exit(0)
os.chdir(ROOT)
os.environ.update(declared)
result = run_supervised(command, ART / 'text-process', name, ART / f'{name}.log', timeout=3600)
print(json.dumps({key: result.get(key) for key in ('name','invocation_state','child_returncode','child_stopped','elapsed_seconds','max_waited_child_rss_bytes')}))
sys.exit(result.get('child_returncode') or (0 if result.get('child_stopped') else 1))
