"""Operational packaging/readback helper; never runs or alters a frozen condition."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
import tarfile

from tools.adaptation_evaluation.local_condition import SUPERVISOR_SCHEMA, validate_execution_contract
from tools.adaptation_evaluation.partial_checkpoint import _validate_report
from tools.adaptation_evaluation.protocol import digest
from tools.adaptation_evaluation.run import _expected_request

ROOT = Path('/workspace/scratch/6f64e1b53384/who-is-your-bias')
ART = Path('/workspace/scratch/6f64e1b53384/artifacts/jensen-2026-11-13')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def safe_path(path):
    for part in (path, *path.parents):
        require(not part.is_symlink(), f'Symlink is not allowed: {part}')
    return path

def file_bytes(path):
    safe_path(path)
    require(path.is_file(), f'Expected a regular file: {path}')
    return path.read_bytes()

def sha(path):
    safe_path(path)
    require(path.is_file(), f'Expected a regular file: {path}')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    return json.loads(file_bytes(path))

def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()

def prepare_json(path, value, *, mutable=False, expected_old=None):
    """Check the complete proposed document before any publication or staging."""
    safe_path(path)
    raw = json_bytes(value)
    old = file_bytes(path) if path.exists() else None
    if old is not None and digest(json.loads(old)) == digest(value):
        raw = old  # Reuse the existing bytes without reformatting them.
    elif old is not None:
        require(mutable and old == expected_old, f'Refuse a different existing result: {path}')
    elif mutable:
        require(expected_old is None, f'Expected document disappeared: {path}')
    if mutable:
        require(old == expected_old or old == raw, f'Document changed during preparation: {path}')
    temporary = path.with_name(path.name + '.delivery-part')
    safe_path(temporary)
    staged = file_bytes(temporary) if temporary.exists() else b''
    require(raw.startswith(staged), f'Conflicting interrupted document: {temporary}')
    return {'path': path, 'temporary': temporary, 'raw': raw, 'old': old, 'staged': staged, 'mutable': mutable}

def stage_json(operation):
    temporary, raw = operation['temporary'], operation['raw']
    existing = file_bytes(temporary) if temporary.exists() else b''
    require(existing == operation['staged'], f'Staged document changed: {temporary}')
    if temporary.exists() and len(existing) < len(raw):
        require(temporary.stat().st_nlink == 1, 'Refuse to append through a hard-linked document stage')
    with temporary.open('ab' if temporary.exists() else 'xb') as stream:
        stream.write(raw[len(existing):])
        stream.flush()
        os.fsync(stream.fileno())
    require(file_bytes(temporary) == raw, f'Staged document bytes differ: {temporary}')

def publish_json(operation):
    path, temporary, raw = operation['path'], operation['temporary'], operation['raw']
    require(file_bytes(temporary) == raw, f'Staged document changed before publication: {temporary}')
    current = file_bytes(path) if path.exists() else None
    if current == raw:
        temporary.unlink()
        return
    require(current == operation['old'], f'Document changed before publication: {path}')
    if current is None:
        os.link(temporary, path)  # Refuse to clobber an unexpectedly created file.
        temporary.unlink()
    else:
        require(operation['mutable'], f'Refuse a different existing result: {path}')
        temporary.replace(path)

def write_new(path, value):
    operation = prepare_json(path, value)
    stage_json(operation)
    publish_json(operation)

def inventory(directory):
    safe_path(directory)
    require(directory.is_dir(), f'Expected directory: {directory}')
    paths = sorted(directory.rglob('*'))
    require(not any(p.is_symlink() for p in paths), 'Symlink in result directory')
    require(all(p.is_file() or p.is_dir() for p in paths), 'Special file in result directory')
    return {str(p.relative_to(directory)): sha(p) for p in paths if p.is_file()}

def sealed(value, key):
    unsigned = dict(value)
    require(unsigned.pop(key, None) == digest(unsigned), f'Invalid {key} seal')

def validate(exposure, condition, directory):
    plan_path = ART / f'text-evaluation-v1/frozen-v2/{exposure}/plan.json'
    contract_path = ROOT / 'reports/jensen-evaluation-v1/execution-contract-bf16-v2.json'
    plan, contract = read(plan_path), read(contract_path)
    validate_execution_contract(contract, plan)
    supervisor = read(directory / 'supervisor.json')
    report = read(directory / 'condition/run.json')
    sealed(supervisor, 'supervisor_sha256')
    finalized = _validate_report(report, plan, contract)
    require(supervisor.get('schema_version') == SUPERVISOR_SCHEMA, 'Supervisor schema differs')
    require(supervisor['server_stopped'] is True, 'Condition server has not stopped')
    require(supervisor['condition'] == report['condition'] == condition, 'Condition identity differs')
    require(report.get('exposure') == plan['exposure'] == exposure, 'Exposure identity differs')
    require(supervisor['plan_sha256'] == report['plan_sha256'] == plan['plan_sha256'], 'Plan identity differs')
    require(supervisor['plan_file_sha256'] == sha(plan_path), 'Frozen plan file bytes differ')
    require(supervisor['contract_sha256'] == contract['contract_sha256'], 'Contract identity differs')
    require(supervisor['contract_file_sha256'] == sha(contract_path), 'Contract file bytes differ')
    require(supervisor['wrapper_sha256'] == contract['wrapper_sha256'], 'Wrapper identity differs')
    require(supervisor['condition_report_sha256'] == sha(directory / 'condition/run.json'), 'Report file bytes differ')
    require(supervisor['condition_run_sha256'] == report['run_sha256'], 'Report seal differs from supervisor')
    terminal = {'completed', 'failed', 'deadline_exhausted'}
    require(report['status'] in terminal and supervisor['status'] in terminal, 'Condition invocation is not terminal')
    if report['status'] == 'completed':
        require(finalized == len(report['runs']) == len(plan['cases']), 'Completed loop lacks its whole finalized case prefix')
    if 'skipped_case_ids' in report:
        require(report['skipped_case_ids'] == report['planned_case_ids'][len(report['runs']):], 'Skipped-case denominator differs')
    complete = sum(row['generation']['completed'] is True for row in report['runs'])
    require(supervisor.get('attempted') == len(report['runs']) and supervisor.get('completed') == complete,
            'Supervisor attempt/completion counts differ')
    for row, case in zip(report['runs'], plan['cases']):
        request_sha = digest(_expected_request(plan, condition, report['served_model'], case))
        require(row['generation'].get('request_sha256') == row['provider_generation'].get('request_sha256') == request_sha,
                'Actual request differs from frozen messages or decoding policy')
    files = inventory(directory)
    require(files == {**supervisor['files_sha256'], 'supervisor.json': sha(directory / 'supervisor.json')}, 'Closed condition file inventory differs')
    return plan, contract, supervisor, report, files

def member_name(value):
    path = PurePosixPath(value)
    require(value and path.parts and not path.is_absolute() and path.as_posix() == value and '..' not in path.parts
            and '\\' not in value and ':' not in path.parts[0], 'Unsafe archive member name')
    return path

def restore_archive(archive, destination, expected):
    """Resume only matching files or exact byte prefixes of verified members."""
    safe_path(destination)
    expected_dirs = set()
    allowed_parts = set()
    for name in expected:
        path = member_name(name)
        expected_dirs.update(str(parent) for parent in path.parents if str(parent) != '.')
        part = name + '.delivery-part'
        require(part not in expected, 'Archive member conflicts with a staging filename')
        allowed_parts.add(part)
    if destination.exists():
        present = inventory(destination)
        require(set(present) <= set(expected) | allowed_parts, 'Unexpected existing readback files')
        for name in set(present) & set(expected):
            require(present[name] == expected[name], f'Changed existing readback member: {name}')
        require(all(str(path.relative_to(destination)) in expected_dirs for path in destination.rglob('*') if path.is_dir()),
                'Unexpected existing readback directory')
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        require(len(names) == len(set(names)) and set(names) == set(expected), 'Archive member inventory differs')
        require(all(member.isfile() and not member.issparse() for member in members), 'Archive contains a link or special member')
        for member in members:
            member_name(member.name)
            stream = tar.extractfile(member)
            require(stream is not None, 'Unreadable archive member')
            with stream:
                require(hashlib.file_digest(stream, 'sha256').hexdigest() == expected[member.name], 'Archive member bytes differ')
        # All members are checked before any restoration write.
        destination.mkdir(parents=True, exist_ok=True)
        for member in members:
            target = destination / member.name
            temporary = target.with_name(target.name + '.delivery-part')
            safe_path(target); safe_path(temporary)
            if target.exists():
                require(sha(target) == expected[member.name], 'Existing restored file changed')
                if temporary.exists():
                    with target.open('rb') as full, temporary.open('rb') as part:
                        while block := part.read(1024 * 1024):
                            require(full.read(len(block)) == block, 'Conflicting interrupted restored file')
                    temporary.unlink()
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            staged_size = temporary.stat().st_size if temporary.exists() else 0
            require(staged_size <= member.size, 'Interrupted restored file is larger than its archive member')
            if temporary.exists():
                require(temporary.stat().st_nlink == 1, 'Refuse to append through a hard-linked restore stage')
            stream = tar.extractfile(member)
            with stream:
                if temporary.exists():
                    with temporary.open('rb') as prior:
                        while block := prior.read(1024 * 1024):
                            require(stream.read(len(block)) == block, 'Changed interrupted restored file bytes')
                with temporary.open('ab' if temporary.exists() else 'xb') as out:
                    while block := stream.read(1024 * 1024):
                        out.write(block)
                    out.flush(); os.fsync(out.fileno())
            require(temporary.stat().st_size == member.size and sha(temporary) == expected[member.name], 'Restored member bytes differ')
            os.link(temporary, target)
            temporary.unlink()
    require(inventory(destination) == expected, 'Restored archive inventory differs')

def timing_population(rows):
    result = {'population_count': len(rows)}
    for source, prefix in (('first_text_ms', 'first_text'), ('wall_ms', 'wall')):
        values = [row['generation'].get(source) for row in rows]
        values = [value for value in values if type(value) in (int, float) and math.isfinite(value)]
        result[prefix + '_numeric_count'] = len(values)
        result[prefix + '_median_ms'] = round(statistics.median(values), 3) if values else None
    return result

def package(exposure, condition):
    name = f'm6-{exposure}-{condition}-bf16-v2'
    directory = ART / f'text-evaluation-v1/execution-{exposure}-bf16-v2/{condition}'
    plan, contract, supervisor, report, files = validate(exposure, condition, directory)
    owner_path = ART / f'text-process/{name}.json'
    owner = read(owner_path)
    require(owner['invocation_state'] == 'exited' and owner['child_stopped'] is True, 'Process owner has not stopped')
    archive = ART / f'jensen-{name}.tar.gz'
    members = {f'{condition}/{rel}': directory / rel for rel in files}
    members['process-owner.json'] = owner_path
    owner_log = Path(owner['log'])
    if owner_log.is_file():
        members['process-owner.log'] = owner_log
    member_hashes = {name: sha(path) for name, path in members.items()}
    with archive.open('xb') as output:
        with tarfile.open(fileobj=output, mode='w:gz') as tar:
            for member, path in sorted(members.items()):
                tar.add(path, arcname=member, recursive=False)
    require(member_hashes == {name: sha(path) for name, path in members.items()}, 'Source files changed while packaging')
    packed = {'archive': archive.name, 'bytes': archive.stat().st_size, 'sha256': sha(archive),
        'condition': condition, 'exposure': exposure, 'plan_sha256': plan['plan_sha256'],
        'contract_sha256': contract['contract_sha256'], 'files_sha256': files,
        'archive_members_sha256': member_hashes, 'supervisor_status': supervisor['status'], 'owner': owner}
    write_new(ART / f'{name}-packaged.json', packed)
    print(json.dumps({k: packed[k] for k in ('archive','bytes','sha256','supervisor_status')}))

def finalize(exposure, condition):
    name = f'm6-{exposure}-{condition}-bf16-v2'
    packed = read(ART / f'{name}-packaged.json')
    require(packed['archive'] == f'jensen-{name}.tar.gz' and packed['condition'] == condition
            and packed['exposure'] == exposure, 'Packaged artifact identity differs')
    record = next(row for row in read(ROOT / 'data/runs/jensen-2026-11-13/artifacts.json')['artifacts']
                  if row['artifact_id'] == f'jensen-{name}')
    require(record['upload_status'] == 'uploaded' and record['verification_status'] == 'fresh-materialization-byte-verified',
            'External upload and fresh-materialization receipt is missing')
    require(type(record['size_bytes']) is int and record['size_bytes'] > 0
            and all(isinstance(record.get(key), str) and record[key].strip() for key in ('file_id', 'library_file_id')),
            'Artifact size or persistent locator is invalid')
    recovered = ART / 'readback' / packed['archive']
    require(sha(recovered) == packed['sha256'] == record['sha256'], 'Materialized archive hash differs')
    require(recovered.stat().st_size == packed['bytes'] == record['size_bytes'], 'Materialized archive size differs')
    destination = ART / f'text-evaluation-v1/condition-readbacks/{exposure}-{condition}-bf16-v2'
    restore_archive(recovered, destination, packed['archive_members_sha256'])
    directory = destination / condition
    plan, contract, supervisor, report, files = validate(exposure, condition, directory)
    require(files == packed['files_sha256'] and plan['plan_sha256'] == packed['plan_sha256']
            and contract['contract_sha256'] == packed['contract_sha256'], 'Restored packaged execution identity differs')
    owner = read(destination / 'process-owner.json')
    require(owner == packed['owner'] and owner['invocation_state'] == 'exited' and owner['child_stopped'] is True,
            'Archived process-owner evidence differs')
    require(sha(recovered) == packed['sha256'], 'Materialized archive changed during validation')
    receipt = {'condition': condition, 'plan_sha256': plan['plan_sha256'],
        'status': 'uploaded_and_fresh_readback_verified',
        'archive': {'file_id': record['file_id'], 'library_file_id': record['library_file_id'],
                    'bytes': record['size_bytes'], 'sha256': record['sha256']}, 'files_sha256': files}
    receipt_path = ART / f'text-evaluation-v1/execution-{exposure}-bf16-v2/durability-receipts.json'
    receipts_old = file_bytes(receipt_path) if receipt_path.exists() else None
    receipts = json.loads(receipts_old) if receipts_old is not None else {'schema_version': 'text-condition-durability.v1', 'conditions': []}
    require(receipts['schema_version'] == 'text-condition-durability.v1', 'Durability receipt schema differs')
    conditions = [row['condition'] for row in receipts['conditions']]
    require(len(conditions) == len(set(conditions)) and set(conditions) <= {'facts', 'style', 'adapter'}, 'Duplicate or foreign durability condition')
    previous = [row for row in receipts['conditions'] if row['condition'] == condition]
    if previous:
        require(digest(previous[0]) == digest(receipt), 'Refuse a different existing durability receipt')
    else:
        receipts['conditions'].append(receipt)
    complete = sum(row['generation']['completed'] is True for row in report['runs'])
    all_attempts = timing_population(report['runs'])
    completed_only = timing_population([row for row in report['runs'] if row['generation']['completed'] is True])
    cases = []
    for row in report['runs']:
        citation = row['citation_format']
        cases.append({'case_id': row['case_id'], **{key: row['generation'].get(key) for key in
            ('completed','error','finish_reason','first_text_ms','wall_ms','usage')},
            'validation_errors': row['validation_errors'], 'identity_check': row['identity_check'],
            'citation_format': {key: citation.get(key) for key in ('required','status','valid_ids','invalid_ids','malformed_markers','semantic_entailment')},
            'citation_coverage': {key: value for key,value in citation.get('coverage',{}).items() if key != 'units'},
            'judgments': row['judgments']})
    # A stable terminal observation time makes interrupted publication byte-reproducible.
    recorded_at = supervisor['finished_at']
    require(isinstance(recorded_at, str) and datetime.fromisoformat(recorded_at.replace('Z', '+00:00')).tzinfo is not None,
            'Terminal supervisor timestamp is invalid')
    result = {'schema_version': 'jensen-matched-condition-result.v1', 'recorded_at': recorded_at,
        'recorded_at_basis': 'terminal_supervisor_observation',
        'condition': condition, 'exposure': exposure, 'status': 'executed_saved_and_fresh_readback_verified',
        'loop_status': report['status'], 'plan_sha256': plan['plan_sha256'], 'contract_sha256': contract['contract_sha256'],
        'runtime': supervisor['requested_runtime'], 'planned': len(plan['cases']), 'attempted': len(report['runs']),
        'completed': complete, 'failed': len(report['runs'])-complete, 'unattempted': len(plan['cases'])-len(report['runs']),
        'all_failures_retained': True, 'timing': {
            **{key: completed_only[key] for key in ('first_text_median_ms', 'wall_median_ms', 'first_text_numeric_count', 'wall_numeric_count')},
            'generic_median_population': 'completed_only',
            'numeric_policy': 'Finite int/float values per metric; booleans and absent/non-finite values excluded.',
            'completed_only': completed_only, 'all_attempts': all_attempts,
            'startup_and_excluded_warmup_ms': supervisor['startup_and_warmup_ms'],
            'whole_process_elapsed_seconds': owner['elapsed_seconds'],
            'max_waited_child_rss_bytes': owner.get('max_waited_child_rss_bytes')},
        'observed_before': supervisor.get('observed_before'), 'observed_after': supervisor.get('observed_after'),
        'server_stopped': supervisor['server_stopped'], 'artifact': record,
        'verification': {'archive_sha256': True, 'condition_file_count': len(files), 'all_condition_files': True,
            'all_archive_members': True, 'supervisor_and_condition_seals': True,
            'fresh_readback_directory': str(directory.relative_to(ART))},
        'per_case': cases, 'full_raw_outputs': 'preserved privately in the verified artifact',
        'paired_comparison': 'pending collection', 'semantic_support_review': 'ungraded', 'human_review': 'ungraded', 'efficacy_claim': False}
    reports = ROOT / 'reports/jensen-evaluation-v1'
    result_name = f'{exposure}-{condition}-bf16-result.json'
    status_path = reports / 'execution-status.json'
    status_old = file_bytes(status_path)
    status = json.loads(status_old)
    entries = [row for row in status['conditions'] if row['exposure'] == exposure and row['condition'] == condition]
    require(len(entries) == 1, 'Execution status must identify exactly one matching condition')
    entry = entries[0]
    entry.update({key: result[key] for key in ('status','planned','attempted','completed','failed','unattempted','plan_sha256','contract_sha256')})
    entry.update(report=result_name, artifact_id=record['artifact_id'])
    for key in ('attempted','completed','failed'):
        require(all(type(row[key]) is int and row[key] >= 0 for row in status['conditions']), 'Invalid execution-status denominator')
        status['matched_'+key] = sum(row[key] for row in status['conditions'])
    timestamps = [result['recorded_at']]
    if status.get('recorded_at'):
        timestamps.append(status['recorded_at'])
    status['recorded_at'] = max(timestamps, key=lambda value: datetime.fromisoformat(value.replace('Z', '+00:00')))
    # Prepare and validate all public/receipt documents before changing any of
    # them. Each stage is prefix-resumable; different prior results are rejected.
    operations = [prepare_json(reports / result_name, result),
        prepare_json(receipt_path, receipts, mutable=True, expected_old=receipts_old),
        prepare_json(status_path, status, mutable=True, expected_old=status_old)]
    for operation in operations:
        stage_json(operation)
    for operation in operations:
        publish_json(operation)
    print(json.dumps({'report': result_name, 'files_verified': len(files), 'attempted': result['attempted'],
        'completed': complete, 'failed': result['failed'], 'first_text_median_ms': completed_only['first_text_median_ms'],
        'wall_median_ms': completed_only['wall_median_ms'], 'completed_only': completed_only, 'all_attempts': all_attempts}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['validate','package','finalize'])
    parser.add_argument('exposure', choices=['development','heldout'])
    parser.add_argument('condition', choices=['facts','style','adapter'])
    args = parser.parse_args()
    if args.command == 'validate':
        values = validate(args.exposure, args.condition, ART / f'text-evaluation-v1/execution-{args.exposure}-bf16-v2/{args.condition}')
        print(json.dumps({'validated_condition_files': len(values[4]), 'status': values[3]['status']}))
    elif args.command == 'package':
        package(args.exposure, args.condition)
    else:
        finalize(args.exposure, args.condition)
