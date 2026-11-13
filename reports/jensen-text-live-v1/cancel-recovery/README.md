# Separate cancellation and recovery diagnostic

This two-request diagnostic is registered separately from the frozen four-turn
M7 functional probe. It runs **once per eligible condition**, only after that
condition's preserved M7 attempt failed before reaching any `disconnect.*`
check. A passing M7 run or an M7 run that already reached cancellation is
ineligible. The original failure remains a failure; these outputs never enter
M6 scores or replace M7 conversation, follow-up, correction or topic-switch
results.

The [protocol](protocol.json) has SHA-256
`bd0fd90916bfc07e52600aef0479960a1e0e2b562a46c7ff8e74403e82c03c7a`.
It was authored before the initial M7 identity-envelope failure was disclosed
to this diagnostic's author. No diagnostic model output existed at registration.
The separately recorded harness correction requires the actual
`language-model-identity.v1` envelope plus every unchanged frozen identity
field; it does not change a model, prompt, snapshot, budget or decoding setting.
The [preparation record](preparation.json) identifies the one corrected source
hash and the other 49 unchanged inputs from the original M7 inventory.

The parent BF16 protocol remains byte-identical:
`6c699aaa81a5f90919be7c78579f400c58590db5924abd045e6c76bcf9cd331e`.
Its model, completed29 adapter, corpus, condition-specific snapshots, CPU BF16,
eight threads, 4,096-token context, 160 output tokens, 240-second request limit,
60-second idle recovery limit and deterministic decoding are reused unchanged.
Both original prompts are copied exactly into the diagnostic seal.

## What is measured

1. Start a new empty main-API session with the original long CUDA disconnect
   request. At the first nonempty SSE token, inspect the actual provider's exact
   identity and busy state and the session's empty context; then close TCP.
2. Check that the provider becomes idle and that the partial answer never enters
   conversation memory.
3. Send the original short CUDA recovery question once in the same session.
   Independently record actual busy admission, first token, complete result/done,
   errors, elapsed times, and token/final-response hashes.
4. Inspect memory, delete the session and stop both servers even when recovery
   truncates or fails. A length error remains an incomplete response. Successful
   cancellation or re-admission cannot turn it into a completed recovery.

The implementation reuses `prepare`, `observed_identity`, `StreamClient`,
`language_child` and `main_server` from the M7 harness. A short independent health
observer records the recovery worker's actual busy admission. Raw generated
text, source context, errors and HTTP traces stay in private output files;
`public-summary.json` contains only identities, hashes, timings and outcomes.
Human review and semantic/persona quality remain ungraded.

Eligibility additionally requires the original report's source hash to match
the current corrected M7 harness. The initial identity-envelope failure is
preserved separately; it does not substitute for running the corrected original
probe. The diagnostic writes an exclusive condition attempt marker under the
artifact root and never retries a request or overwrites an earlier output.

## Exact current-workspace launch and preservation

Run this only after the coordinator has published these files, saved the
corrected original M7 result, and released the shared model slot. Set the
condition and the **actual preserved corrected-harness summary path** explicitly.
The helper exits without generation when eligibility does not match.

From `/workspace/scratch/31a6dc706f63/who-is-your-bias`:

```bash
PYTHONPATH=src:. .venv/bin/python - base /absolute/path/to/corrected-m7/public-summary.json <<'PY'
import importlib.util
import json
import os
from pathlib import Path
import sys
from tools.adaptation.supervise import run_supervised
from tools.runtime_checks.runtime import digest

root = Path.cwd().resolve()
art = Path('/workspace/scratch/31a6dc706f63/artifacts/jensen-pr19-completion-v1')
condition, eligibility = sys.argv[1], Path(sys.argv[2]).resolve()
assert condition in ('base', 'adapter')
name = 'm7-cancel-recovery-' + condition + '-20261113-v1'
output = art / 'runtime-checks' / name
log = output.with_name(name + '.log')
launch = output.with_name(name + '.launch.json')
command = [str(root / '.venv-inference/bin/python'), '-m',
    'tools.runtime_checks.text_cancel_recovery', '--repository', str(root),
    '--artifacts', str(art), '--cache-dir',
    '/workspace/scratch/31a6dc706f63/model-cache/hub',
    '--inference-python', str(root / '.venv-inference/bin/python'),
    '--protocol', str(root / 'reports/jensen-text-live-v1/protocol-bf16-v2.json'),
    '--protocol-sha256', '6c699aaa81a5f90919be7c78579f400c58590db5924abd045e6c76bcf9cd331e',
    '--diagnostic-protocol', str(root / 'reports/jensen-text-live-v1/cancel-recovery/protocol.json'),
    '--diagnostic-protocol-sha256', 'bd0fd90916bfc07e52600aef0479960a1e0e2b562a46c7ff8e74403e82c03c7a',
    '--condition', condition, '--eligibility-report', str(eligibility),
    '--output', str(output), '--allow-candidates']
if condition == 'adapter':
    command += ['--adapter-run',
        '/workspace/scratch/31a6dc706f63/restored/jensen-text-adapter-completed29-v1/text-adapter-v1',
        '--adapter-run-sha256', 'e1102328f6cc85e9c096dc554e6255465d67870706df5d3a4f03bf091c6e3f42']
os.environ.update(PYTHONPATH=str(root / 'src') + os.pathsep + str(root),
    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='8',
    MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false')
with launch.open('x') as stream:
    json.dump({'command': command, 'eligibility_sha256': digest(eligibility),
        'condition': condition, 'purpose': 'separate diagnostic, no original-result replacement'}, stream, indent=2)
state = run_supervised(command, art / 'process', name, log, timeout=1860, interval=1)
# A nonzero child return code still reaches packaging and remains a failure.
spec = importlib.util.spec_from_file_location('m7_pack',
    root / 'reports/jensen-text-live-v1/operations/m7_delivery.py')
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)
files = {'control/owner.json': art / 'process' / (name + '.json'),
    'control/launch.json': launch, 'control/process.log': log,
    'control/original-m7-public-summary.json': eligibility,
    'source/diagnostic.py': root / 'tools/runtime_checks/text_cancel_recovery.py',
    'source/parent-harness.py': root / 'tools/runtime_checks/text_live.py',
    'source/protocol.json': root / 'reports/jensen-text-live-v1/cancel-recovery/protocol.json'}
if output.is_dir():
    files.update({'run/' + path.relative_to(output).as_posix(): path
                  for path in output.rglob('*') if path.is_file()})
print(json.dumps(pack.make_bundle(files,
    output.with_name('jensen-' + name + '.tar.gz'),
    {'role': 'actual-separate-neural-cancellation-recovery-diagnostic',
     'condition': condition, 'child_returncode': state['child_returncode'],
     'child_stopped': state['child_stopped'], 'quality_claim': False}), indent=2))
PY
```

For the adapter condition, change only the first argument from `base` to
`adapter` and pass its own eligible original summary. Do not execute both in
parallel. Configuration-only preparation uses the same command with
`--verify-only`; before the original probe is available, omit
`--eligibility-report` and label eligibility pending.

Save the produced archive even if the child returned nonzero. Use the existing
`m7_delivery.py verify-archive` operation with its adjacent `.package.json`, the
genuine freshly downloaded archive and actual returned persistent IDs. Only
then publish the public summary and receipt. No artifact upload or successful
neural execution is claimed by the preparation record.
