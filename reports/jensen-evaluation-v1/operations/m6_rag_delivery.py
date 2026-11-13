"""Preserve and freshly verify the separate fixed six-case RAG invocation.

No model, remote transfer, central-manifest edit, Git write or matched counter
update occurs here. Verification requires a separately obtained genuine fresh
archive and its verified artifact record. All raw artifacts remain private.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import io
import json
import os
from pathlib import Path
import tarfile

import launch_m6_rag as launch
import m6_partial_delivery as base
from tools.adaptation_evaluation import partial_checkpoint as partial
from tools.adaptation_evaluation import rag_diagnostic as rag
from tools.adaptation_evaluation.local_condition import (
    SUPERVISOR_SCHEMA, _identity_hashes, _validate_observed, validate_execution_contract,
)
from tools.adaptation_evaluation.protocol import digest
from tools.adaptation_evaluation.run import _expected_request

ROOT, ART = launch.ROOT, launch.ART
HELPER = Path(__file__).resolve()
BASE_SHA = '33b244c32af6fd1ed89b794f5d295ea20f2a7ab97a160972fa81b09df93ac7fe'
LAUNCH_SHA = '5a78a311e46d07b3c3f48468335b2a8307ed440102f76b930b6e17b904c40ced'
require, raw_file, sha = base.require, base.raw_file, base.sha


def locations(mode):
    require(mode in {'partial3', 'final', 'failure'}, 'Unknown fixed RAG delivery mode')
    suffix = {'partial3': '-partial0003', 'final': '', 'failure': '-failure'}[mode]
    stem = launch.NAME + suffix
    return {'stem': stem, 'artifact_id': 'jensen-' + stem,
            'archive': ART / f'jensen-{stem}.tar.gz',
            'packed': ART / f'{stem}-packaged.json',
            'result': ART / f'{stem}-result.json',
            'restore': ART / 'text-evaluation-v1/rag-readbacks' / (mode + '-bf16-v2'),
            'snapshot': ART / 'text-evaluation-v1/rag-deliveries/partial0003'}


def tree_bytes(directory):
    base.safe_path(directory)
    require(directory.is_dir(), f'Expected directory: {directory}')
    payload = {}
    for path in sorted(directory.rglob('*')):
        base.safe_path(path)
        require(path.is_file() or path.is_dir(), f'Special file refused: {path}')
        if path.is_file():
            payload[path.relative_to(directory).as_posix()] = raw_file(path)
    return payload


def frozen_inputs():
    plan_raw, contract_raw = raw_file(launch.PLAN), raw_file(launch.CONTRACT)
    require(sha(plan_raw) == launch.PLAN_FILE_SHA256
            and sha(contract_raw) == launch.CONTRACT_FILE_SHA256, 'Frozen RAG input bytes changed')
    plan, contract = json.loads(plan_raw), json.loads(contract_raw)
    rag.validate_rag_plan(plan)
    validate_execution_contract(contract, plan)
    require(plan['plan_sha256'] == launch.PLAN_SHA256
            and contract['contract_sha256'] == launch.CONTRACT_SHA256
            and [row['id'] for row in plan['cases']] == launch.CASE_IDS
            and plan['rag_diagnostic']['execution_conditions'] == ['facts'], 'RAG registration differs')
    return plan_raw, contract_raw, plan, contract


def fixed_members():
    plan_raw, contract_raw, _, _ = frozen_inputs()
    require(sha(raw_file(Path(base.__file__))) == BASE_SHA
            and sha(raw_file(Path(launch.__file__))) == LAUNCH_SHA, 'Operational dependency changed')
    files = {
        'source/m6_rag_delivery.py': HELPER,
        'source/launch_m6_rag.py': Path(launch.__file__),
        'source/m6_partial_delivery.py': Path(base.__file__),
    }
    for relative in ('tools/adaptation_evaluation/partial_checkpoint.py',
                     'tools/adaptation_evaluation/rag_diagnostic.py',
                     'tools/adaptation_evaluation/local_condition.py',
                     'tools/adaptation_evaluation/run.py',
                     'tools/adaptation_evaluation/protocol.py', 'tools/adaptation/supervise.py'):
        files['source/' + relative] = ROOT / relative
    return {'inputs/plan.json': plan_raw, 'inputs/execution-contract-bf16-v2.json': contract_raw,
            **{name: raw_file(path) for name, path in files.items()}}


def row_metadata(row):
    fields = ('completed', 'error', 'finish_reason', 'response_model', 'http_statuses',
              'wall_ms', 'first_text_ms', 'request_sha256')
    return {'case_id': row['case_id'],
            'provider_generation': {key: row['provider_generation'].get(key) for key in fields},
            'generation': {key: row['generation'].get(key) for key in fields},
            'identity_check': row.get('identity_check'),
            'validation_errors': row.get('validation_errors')}


def condition_audit(payload, owner, plan, contract, plan_raw, contract_raw):
    supervisor = json.loads(payload['supervisor.json'])
    report = json.loads(payload['condition/run.json'])
    partial._check_seal(supervisor, 'supervisor_sha256')
    finalized = partial._validate_report(report, plan, contract)
    require(supervisor['schema_version'] == SUPERVISOR_SCHEMA
            and supervisor['condition'] == report['condition'] == 'facts'
            and report['exposure'] == plan['exposure'] == 'heldout'
            and supervisor['server_stopped'] is True, 'RAG invocation identity or stop state differs')
    require(supervisor['status'] in {'completed', 'failed', 'deadline_exhausted'}
            and report['status'] in {'completed', 'failed', 'deadline_exhausted'}
            and finalized == len(report['runs']), 'A final report has unfinished attempts')
    if report['status'] == 'completed':
        require(finalized == len(plan['cases']), 'Completed loop does not cover all six cases')
    if 'skipped_case_ids' in report:
        require(report['skipped_case_ids'] == report['planned_case_ids'][finalized:], 'Skipped suffix differs')
    require(supervisor['condition_report_sha256'] == sha(payload['condition/run.json'])
            and supervisor['condition_run_sha256'] == report['run_sha256'], 'Supervisor/run binding differs')
    require({name: sha(raw) for name, raw in payload.items()}
            == {**supervisor['files_sha256'], 'supervisor.json': sha(payload['supervisor.json'])},
            'Whole closed-condition inventory differs')
    metadata = {name: json.loads(payload[name]) for name in partial.METADATA}
    partial._validate_metadata(supervisor, metadata, report, plan, contract, sha(plan_raw), sha(contract_raw))
    after = json.loads(payload['observed-after.json'])
    _validate_observed(plan, contract, after, 'facts')
    require(after == metadata['observed-before.json'] == report['model']
            and supervisor['observed_after'] == _identity_hashes(after), 'Final health identity differs')
    partial._match_logs(report['runs'], base.jsonl(payload['condition/attempts.jsonl'], finalized),
                        base.jsonl(payload['condition/raw-generations.jsonl'], finalized))
    for row, case in zip(report['runs'], plan['cases']):
        expected = digest(_expected_request(plan, 'facts', report['served_model'], case))
        require(row['generation'].get('request_sha256') == row['provider_generation'].get('request_sha256')
                == expected, 'Actual RAG request identity differs')
    completed = sum(row['generation']['completed'] is True for row in report['runs'])
    require(supervisor['attempted'] == finalized and supervisor['completed'] == completed,
            'Supervisor counts differ')
    expected_command = [str(launch.INFERENCE_PYTHON), '-m', 'tools.adaptation_evaluation.rag_diagnostic',
                        'run', '--plan', str(launch.PLAN), '--contract', str(launch.CONTRACT),
                        '--timeout', '180', '--output', str(launch.OUTPUT)]
    require(owner['name'] == launch.NAME and owner['command'] == expected_command
            and owner['invocation_state'] == 'exited' and owner['child_returncode'] == 0
            and owner['child_stopped'] is True, 'Process owner did not exit cleanly with the registered command')
    return {'validation': 'closed_invocation_revalidated', 'condition_status': supervisor['status'],
            'run_status': report['status'], 'registered': len(plan['rag_diagnostic']['registered_case_ids']),
            'planned': len(plan['cases']), 'attempted': finalized, 'completed': completed,
            'failed': finalized - completed, 'unattempted': len(plan['cases']) - finalized,
            'run_sha256': report['run_sha256'], 'supervisor_sha256': supervisor['supervisor_sha256'],
            'model_identity_hashes': _identity_hashes(report['model']),
            'completed_case_ids': [row['case_id'] for row in report['runs'] if row['generation']['completed']],
            'case_metadata': [row_metadata(row) for row in report['runs']],
            'server_stopped': True, 'process_owner_exit_verified': True,
            'eligible_for_completed_output_review': supervisor['status'] == 'completed'}


def audit(mode, members):
    plan_raw, contract_raw = members['inputs/plan.json'], members['inputs/execution-contract-bf16-v2.json']
    plan, contract = json.loads(plan_raw), json.loads(contract_raw)
    rag.validate_rag_plan(plan)
    validate_execution_contract(contract, plan)
    if mode == 'partial3':
        payload = {name.removeprefix('partial0003/'): raw for name, raw in members.items()
                   if name.startswith('partial0003/')}
        manifest, prefix, _, report = base.validate_snapshot(payload, 'heldout', 'facts', 3, plan_raw, contract_raw)
        return {'validation': 'finalized_prefix_revalidated', 'captured_attempts': 3,
                'captured_completed': manifest['captured_completed'], 'captured_failed': manifest['captured_failed'],
                'registered': len(plan['rag_diagnostic']['registered_case_ids']), 'planned': len(plan['cases']),
                'uncaptured_case_ids': prefix['uncaptured_case_ids'], 'manifest_sha256': manifest['manifest_sha256'],
                'partial_sha256': prefix['partial_sha256'], 'snapshot_created_at': manifest['created_at'],
                'model_identity_hashes': _identity_hashes(report['model']),
                'case_metadata': [row_metadata(row) for row in prefix['runs']],
                'eligible_for_completed_output_review': False, 'final_full_condition_save_required': True}
    payload = {name.removeprefix('condition-output/'): raw for name, raw in members.items()
               if name.startswith('condition-output/')}
    if mode == 'final':
        require('process-owner.log' in members, 'Full invocation must retain its outer log')
        owner = json.loads(members['process-owner.json'])
        return condition_audit(payload, owner, plan, contract, plan_raw, contract_raw)
    # Explicit failure evidence may be incomplete. Archive bytes without promoting
    # missing/invalid receipts, pending rows or unknown process state to success.
    try:
        owner = json.loads(members['process-owner.json'])
        require(isinstance(owner, dict), 'Non-object owner receipt')
    except (KeyError, TypeError, ValueError):
        owner = {}
    observed = {'validation': 'failure_bytes_preserved_without_completed_condition_claim',
                'registered': len(plan['rag_diagnostic']['registered_case_ids']), 'planned': len(plan['cases']),
                'eligible_for_completed_output_review': False,
                'condition_payload_files': sorted(payload),
                'process_owner_observation': {key: owner.get(key) for key in
                    ('name', 'invocation_state', 'child_returncode', 'child_stopped', 'failure_type')}}
    try:
        supervisor = json.loads(payload['supervisor.json'])
        partial._check_seal(supervisor, 'supervisor_sha256')
        observed['sealed_supervisor_observation'] = {key: supervisor.get(key) for key in
            ('condition', 'status', 'server_stopped', 'failure_type', 'cleanup_failure_type', 'supervisor_sha256')}
    except (KeyError, TypeError, ValueError, AttributeError):
        observed['sealed_supervisor_observation'] = None
    try:
        report = json.loads(payload['condition/run.json'])
        finalized = partial._validate_report(report, plan, contract)
        observed['report_observation'] = {'status': report['status'], 'finalized_prefix': finalized,
            'observed_rows': len(report['runs']), 'run_sha256': report['run_sha256'],
            'case_metadata': [row_metadata(row) for row in report['runs']]}
    except (KeyError, TypeError, ValueError, AttributeError):
        observed['report_observation'] = None
    return observed


def package(mode):
    paths = locations(mode)
    for key in ('archive', 'packed', 'result'):
        base.safe_path(paths[key]); require(not paths[key].exists(), f'Preserve existing delivery: {paths[key]}')
    fixed = fixed_members()
    if mode == 'partial3':
        base.safe_path(paths['snapshot']); require(not paths['snapshot'].exists(), 'Preserve existing prefix snapshot')
        partial.snapshot_condition(launch.PLAN, launch.CONTRACT, launch.OUTPUT, paths['snapshot'],
                                   attempts=3, include_server_log=False, read_attempts=3)
        source_paths = {f'partial0003/{name}': paths['snapshot'] / name for name in tree_bytes(paths['snapshot'])}
    else:
        payload = tree_bytes(launch.OUTPUT) if launch.OUTPUT.exists() else {}
        source_paths = {f'condition-output/{name}': launch.OUTPUT / name for name in payload}
        for name, path in (('process-owner.json', launch.STATE_DIRECTORY / f'{launch.NAME}.json'),
                           ('process-owner.log', launch.LOG)):
            if path.exists(): source_paths[name] = path
        require(source_paths, 'No actual RAG invocation evidence exists')
    members = {**fixed, **{name: raw_file(path) for name, path in source_paths.items()}}
    observation = audit(mode, members)
    buffer = io.BytesIO()
    with gzip.GzipFile(filename='', fileobj=buffer, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as tar:
            for name, raw in sorted(members.items()):
                base.member_name(name)
                info = tarfile.TarInfo(name); info.size = len(raw); info.mode = 0o600
                info.uid = info.gid = info.mtime = 0; info.uname = info.gname = ''
                tar.addfile(info, io.BytesIO(raw))
    require(fixed_members() == fixed and all(raw_file(path) == members[name] for name, path in source_paths.items()),
            'Source bytes changed during archive capture; retain existing output and inspect')
    archive_raw = buffer.getvalue()
    packed = {'schema_version': 'm6-rag-private-package.v1', 'mode': mode, 'artifact_id': paths['artifact_id'],
              'archive': paths['archive'].name, 'bytes': len(archive_raw), 'sha256': sha(archive_raw),
              'recorded_at': datetime.now(timezone.utc).isoformat(), 'observation': observation,
              'archive_members': {name: {'bytes': len(raw), 'sha256': sha(raw)} for name, raw in sorted(members.items())},
              'plan_sha256': launch.PLAN_SHA256, 'contract_sha256': launch.CONTRACT_SHA256,
              'collectible': False, 'matched_counters_updated': False,
              'remote_upload_performed': False, 'remote_readback_verified': False}
    base.write_new(paths['archive'], archive_raw)
    base.write_new(paths['packed'], base.encoded(packed))
    print(json.dumps({key: packed[key] for key in ('mode', 'artifact_id', 'archive', 'bytes', 'sha256')}))
    return packed


def verify(mode, record_path):
    paths = locations(mode); packed = base.read(paths['packed'])
    record_raw = raw_file(record_path); record = json.loads(record_raw)
    require(packed['schema_version'] == 'm6-rag-private-package.v1' and packed['mode'] == mode
            and packed['artifact_id'] == paths['artifact_id'] and packed['archive'] == paths['archive'].name,
            'Package identity differs')
    require(record['artifact_id'] == paths['artifact_id'] and record['file_name'] == packed['archive']
            and record['upload_status'] == 'uploaded'
            and record['verification_status'] == 'fresh-materialization-byte-verified'
            and type(record['size_bytes']) is int and record['size_bytes'] > 0
            and all(isinstance(record.get(key), str) and record[key].strip() for key in ('file_id', 'library_file_id')),
            'Require the separately obtained actual upload/fresh-readback record')
    fresh = ART / 'readback' / packed['archive']; archive_raw = raw_file(fresh)
    if paths['archive'].exists():
        require(not os.path.samefile(fresh, paths['archive']), 'Fresh copy aliases the original archive')
    require(len(archive_raw) == packed['bytes'] == record['size_bytes']
            and sha(archive_raw) == packed['sha256'] == record['sha256'], 'Fresh archive bytes differ')
    members = {}
    with tarfile.open(fileobj=io.BytesIO(archive_raw), mode='r:gz') as tar:
        for entry in tar.getmembers():
            base.member_name(entry.name)
            require(entry.name not in members and entry.isfile() and not entry.issparse()
                    and entry.name in packed['archive_members'], 'Unexpected/unsafe archive member')
            identity = packed['archive_members'][entry.name]
            require(entry.size == identity['bytes'], 'Member size differs')
            with tar.extractfile(entry) as stream:
                raw = stream.read()
            require(len(raw) == identity['bytes'] and sha(raw) == identity['sha256'], 'Member bytes differ')
            members[entry.name] = raw
    require(set(members) == set(packed['archive_members']), 'Missing archive member')
    require(all(members.get(name) == raw for name, raw in fixed_members().items()), 'Archived source or input changed')
    observation = audit(mode, members)
    require(observation == packed['observation'], 'Fresh validation differs from captured evidence')
    base.safe_path(paths['restore']); paths['restore'].mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, raw in sorted(members.items()):
        base.write_new(paths['restore'] / name, raw)
    require(tree_bytes(paths['restore']) == members and sha(raw_file(fresh)) == packed['sha256']
            and raw_file(record_path) == record_raw, 'Restored inventory, archive or transfer record changed')
    artifact_keys = ('artifact_id', 'file_name', 'file_id', 'library_file_id', 'size_bytes',
                     'sha256', 'upload_status', 'verification_status')
    result = {'schema_version': 'm6-rag-persistence-result.v1', 'mode': mode,
              'status': 'saved_and_fresh_readback_verified', 'recorded_at': packed['recorded_at'],
              'exposure': 'heldout', 'execution_condition': 'facts', 'scope': 'separate_single_base_offline_rag',
              'plan_sha256': launch.PLAN_SHA256, 'contract_sha256': launch.CONTRACT_SHA256,
              'observation': observation, 'artifact': {key: record[key] for key in artifact_keys},
              'artifact_record_file_sha256': sha(record_raw),
              'archive_members': packed['archive_members'], 'private_restored_root': str(paths['restore']),
              'raw_errors_and_normalized_errors_retained_separately': True,
              'collectible': False, 'matched_counters_updated': False,
              'human_review': 'ungraded', 'efficacy_claim': False,
              'verification': {'fresh_archive_bytes_and_hash': True, 'every_regular_member_hash': True,
                               'exact_restored_inventory': True, 'mode_specific_evidence_revalidated': True},
              'publication': 'Scratch candidate only; coordinator owns central indexing and Git publication.'}
    base.write_new(paths['result'], base.encoded(result))
    print(json.dumps({'result': str(paths['result']), 'mode': mode, 'artifact_id': paths['artifact_id'],
                      'restored_member_count': len(members), 'collectible': False}))
    return result


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('package', 'verify'))
    parser.add_argument('mode', choices=('partial3', 'final', 'failure'))
    parser.add_argument('--artifact-receipt', type=Path)
    args = parser.parse_args()
    if args.command == 'verify' and args.artifact_receipt is None:
        parser.error('verify requires the real standalone --artifact-receipt')
    if args.command == 'package' and args.artifact_receipt is not None:
        parser.error('package does not consume a transfer receipt')
    if args.command == 'package': package(args.mode)
    else: verify(args.mode, args.artifact_receipt)


if __name__ == '__main__':
    main()
