"""Operational immutable partial-condition delivery; never runs a model or collector."""
from __future__ import annotations

import argparse
from datetime import datetime
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

from tools.adaptation_evaluation import partial_checkpoint as partial
from tools.adaptation_evaluation.local_condition import _identity_hashes, validate_execution_contract
from tools.adaptation_evaluation.protocol import digest

ROOT = Path('/workspace/scratch/6f64e1b53384/who-is-your-bias')
ART = Path('/workspace/scratch/6f64e1b53384/artifacts/jensen-2026-11-13')
HELPER = Path(__file__).resolve()
HELPER_MEMBER = 'm6_partial_delivery.py'
PAYLOADS = {'partial-run.json', 'attempts.prefix.jsonl', 'raw-generations.prefix.jsonl',
            'metadata/launcher.json', 'metadata/warmup.json', 'metadata/observed-before.json',
            'metadata/supervisor-observed.json'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe_path(path):
    for component in (path, *path.parents):
        require(not component.is_symlink(), f'Symlink refused: {component}')
    return path


def raw_file(path):
    safe_path(path)
    require(path.is_file(), f'Expected regular file: {path}')
    return path.read_bytes()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(raw_file(path))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def write_new(path, raw):
    """Exact existing bytes are idempotent; conflicting or partial outputs are retained."""
    safe_path(path)
    if path.exists():
        require(raw_file(path) == raw, f'Refuse different existing output: {path}')
        return
    temporary = path.with_name(path.name + '.partial-delivery-part')
    safe_path(temporary)
    require(not temporary.exists(), f'Interrupted output requires review: {temporary}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    require(raw_file(temporary) == raw, 'Staged bytes changed')
    os.link(temporary, path)
    temporary.unlink()


def names(exposure, condition, attempts):
    require(exposure in {'development', 'heldout'} and condition in {'facts', 'style', 'adapter'},
            'Unknown exposure or condition')
    require(type(attempts) is int and 1 <= attempts <= 9999, 'Use a positive four-digit bounded attempt count')
    stem = f'm6-{exposure}-{condition}-bf16-v2-partial{attempts:04d}'
    basename = f'{exposure}-{condition}-attempts{attempts:04d}'
    return stem, basename


def paths(exposure, condition, attempts):
    stem, basename = names(exposure, condition, attempts)
    return {'stem': stem, 'basename': basename,
            'plan': ART / f'text-evaluation-v1/frozen-v2/{exposure}/plan.json',
            'contract': ROOT / 'reports/jensen-evaluation-v1/execution-contract-bf16-v2.json',
            'source': ART / f'text-evaluation-v1/execution-{exposure}-bf16-v2/{condition}',
            'snapshot': ART / 'text-evaluation-v1/partial-deliveries' / basename,
            'archive': ART / f'jensen-{stem}.tar.gz',
            'packed': ART / f'{stem}-packaged.json',
            'public': ROOT / f'reports/jensen-evaluation-v1/{exposure}-{condition}-bf16-partial{attempts:04d}.json'}


def member_name(name):
    p = PurePosixPath(name)
    require(name and p.parts and not p.is_absolute() and p.as_posix() == name and '..' not in p.parts
            and '\\' not in name and ':' not in p.parts[0], 'Unsafe archive member name')


def jsonl(raw, count):
    lines = raw.splitlines(keepends=True)
    require(len(lines) == count and all(line.endswith(b'\n') for line in lines), 'Incomplete or excess JSONL prefix')
    rows = [json.loads(line) for line in lines]
    require(all(isinstance(row, dict) for row in rows), 'Non-object JSONL row')
    return rows


def validate_snapshot(payload, exposure, condition, attempts, plan_raw, contract_raw):
    manifest = json.loads(payload['manifest.json'])
    partial._check_seal(manifest, 'manifest_sha256')
    require(manifest.get('schema_version') == partial.MANIFEST_SCHEMA
            and manifest.get('status') == 'partial_local_snapshot' and manifest.get('collectible') is False
            and manifest.get('source_report_included') is False
            and manifest.get('remote_persistence_performed') is False
            and manifest.get('remote_readback_verified') is False
            and manifest.get('human_review') == 'ungraded' and manifest.get('efficacy_claim') is False,
            'Snapshot manifest is not a capture-time partial')
    require(set(manifest['files']) == PAYLOADS and set(payload) == PAYLOADS | {'manifest.json'},
            'Unexpected partial snapshot payload inventory')
    for name, identity in manifest['files'].items():
        member_name(name)
        require(identity == {'bytes': len(payload[name]), 'sha256': sha(payload[name])}, 'Snapshot payload hash or size differs')
    plan, contract = json.loads(plan_raw), json.loads(contract_raw)
    validate_execution_contract(contract, plan)
    require(plan['exposure'] == exposure and manifest['condition'] == condition, 'Snapshot exposure or condition differs')
    require(manifest['plan_file_sha256'] == sha(plan_raw) and manifest['plan_sha256'] == plan['plan_sha256']
            and manifest['contract_file_sha256'] == sha(contract_raw)
            and manifest['contract_sha256'] == contract['contract_sha256'], 'Frozen plan or execution contract differs')
    require(manifest['tool_sha256'] == sha(raw_file(Path(partial.__file__))), 'Snapshot implementation changed')
    prefix = json.loads(payload['partial-run.json'])
    partial._check_seal(prefix, 'partial_sha256')
    require(prefix.get('schema_version') == partial.SCHEMA and prefix.get('status') == 'partial'
            and prefix.get('collectible') is False and prefix.get('human_review') == 'ungraded'
            and prefix.get('efficacy_claim') is False and prefix.get('server_log_observation') is None,
            'Captured prefix identity or public-claim boundary differs')
    for key in ('condition', 'plan_sha256', 'contract_sha256'):
        require(prefix[key] == manifest[key], 'Prefix and manifest identity differ')
    rows = prefix['runs']
    case_ids = [row['case_id'] for row in rows]
    require(manifest['captured_attempts'] == prefix['captured_attempts'] == attempts == len(rows), 'Attempt count differs')
    require(case_ids == manifest['case_ids'] == [case['id'] for case in plan['cases'][:attempts]], 'Frozen prefix case order differs')
    require(prefix['uncaptured_case_ids'] == [case['id'] for case in plan['cases'][attempts:]], 'Uncaptured denominator differs')
    # Revalidate only the captured rows, not the deliberately omitted live report.
    report = dict(prefix['source_run_metadata'])
    require(report.get('condition') == condition and report.get('exposure', exposure) == exposure,
            'Observed report exposure or condition differs')
    report['runs'] = rows
    report['run_sha256'] = digest(report)
    require(partial._validate_report(report, plan, contract) == attempts, 'Prefix includes an unfinished identity outcome')
    complete = sum(row['generation']['completed'] is True for row in rows)
    require(prefix['captured_completed'] == manifest['captured_completed'] == complete
            and prefix['captured_failed'] == manifest['captured_failed'] == attempts - complete, 'Completion counts differ')
    partial._match_logs(rows, jsonl(payload['attempts.prefix.jsonl'], attempts),
                        jsonl(payload['raw-generations.prefix.jsonl'], attempts))
    metadata = {name: json.loads(payload['metadata/' + name]) for name in partial.METADATA}
    supervisor = json.loads(payload['metadata/supervisor-observed.json'])
    partial._validate_metadata(supervisor, metadata, report, plan, contract, sha(plan_raw), sha(contract_raw))
    observed = prefix['source_report']
    require(type(observed['observed_rows']) is int and type(observed['finalized_prefix']) is int
            and attempts <= observed['finalized_prefix'] <= observed['observed_rows'] <= len(plan['cases']),
            'Source observation counts differ')
    require(datetime.fromisoformat(manifest['created_at'].replace('Z', '+00:00')).tzinfo is not None,
            'Snapshot timestamp must include timezone')
    return manifest, prefix, plan, report


def package(exposure, condition, attempts):
    p = paths(exposure, condition, attempts)
    for key in ('snapshot', 'archive', 'packed', 'public'):
        safe_path(p[key]); require(not p[key].exists(), f'Refuse existing delivery: {p[key]}')
    partial.snapshot_condition(p['plan'], p['contract'], p['source'], p['snapshot'],
                               attempts=attempts, include_server_log=False, read_attempts=3)
    files = sorted(p['snapshot'].rglob('*'))
    require(all(not x.is_symlink() and (x.is_dir() or x.is_file()) for x in files), 'Snapshot contains a special path')
    payload = {str(x.relative_to(p['snapshot'])): raw_file(x) for x in files if x.is_file()}
    manifest, prefix, _, _ = validate_snapshot(payload, exposure, condition, attempts, raw_file(p['plan']), raw_file(p['contract']))
    helper_raw = raw_file(HELPER)
    members = {p['basename'] + '/' + name: raw for name, raw in payload.items()}
    members[HELPER_MEMBER] = helper_raw
    buffer = io.BytesIO()
    with gzip.GzipFile(filename='', fileobj=buffer, mode='wb', mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w', format=tarfile.PAX_FORMAT) as tar:
            for name, raw in sorted(members.items()):
                member_name(name)
                info = tarfile.TarInfo(name); info.size = len(raw); info.mode = 0o600
                info.uid = info.gid = info.mtime = 0; info.uname = info.gname = ''
                tar.addfile(info, io.BytesIO(raw))
    archive_raw = buffer.getvalue()
    require(raw_file(HELPER) == helper_raw
            and all(raw_file(p['snapshot'] / name) == raw for name, raw in payload.items()), 'Immutable package input changed')
    packed = {'schema_version': 'jensen-partial-condition-package.v1', 'exposure': exposure, 'condition': condition,
              'captured_attempts': attempts, 'collectible': False, 'snapshot_basename': p['basename'],
              'archive': p['archive'].name, 'bytes': len(archive_raw), 'sha256': sha(archive_raw),
              'plan_sha256': manifest['plan_sha256'], 'contract_sha256': manifest['contract_sha256'],
              'manifest_sha256': manifest['manifest_sha256'], 'partial_sha256': prefix['partial_sha256'],
              'helper_source_sha256': sha(helper_raw),
              'archive_members': {name: {'bytes': len(raw), 'sha256': sha(raw)} for name, raw in sorted(members.items())}}
    write_new(p['archive'], archive_raw)
    write_new(p['packed'], encoded(packed))
    result = {key: packed[key] for key in ('archive', 'bytes', 'sha256', 'captured_attempts', 'collectible')}
    print(json.dumps(result))
    return packed


def finalize(exposure, condition, attempts):
    p = paths(exposure, condition, attempts); packed = read(p['packed'])
    require(packed['schema_version'] == 'jensen-partial-condition-package.v1'
            and packed['exposure'] == exposure and packed['condition'] == condition
            and packed['captured_attempts'] == attempts and packed['snapshot_basename'] == p['basename']
            and packed['archive'] == p['archive'].name and packed['collectible'] is False, 'Packaged identity differs')
    matches = [row for row in read(ROOT / 'data/runs/jensen-2026-11-13/artifacts.json')['artifacts']
               if row['artifact_id'] == 'jensen-' + p['stem']]
    require(len(matches) == 1, 'Require exactly one verified central artifact')
    record = matches[0]
    require(record['upload_status'] == 'uploaded' and record['verification_status'] == 'fresh-materialization-byte-verified'
            and record['file_name'] == packed['archive'], 'External upload and fresh-readback receipt missing or mismatched')
    require(type(record['size_bytes']) is int and record['size_bytes'] > 0
            and all(isinstance(record.get(k), str) and record[k].strip() for k in ('file_id', 'library_file_id')),
            'Persistent identity or size is invalid')
    fresh = ART / 'readback' / packed['archive']; raw = raw_file(fresh)
    if p['archive'].exists():
        require(not os.path.samefile(fresh, p['archive']), 'Readback aliases the original archive')
    require(len(raw) == packed['bytes'] == record['size_bytes'] and sha(raw) == packed['sha256'] == record['sha256'],
            'Fresh archive bytes or hash differs')
    members = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as tar:
        for member in tar.getmembers():
            member_name(member.name)
            require(member.name not in members and member.isfile() and not member.issparse(), 'Duplicate or nonregular tar member')
            require(member.name in packed['archive_members'], 'Unexpected archive member')
            expected = packed['archive_members'][member.name]
            require(member.size == expected['bytes'], 'Archive member size differs')
            stream = tar.extractfile(member); require(stream is not None, 'Unreadable archive member')
            with stream:
                content = stream.read()
            require(len(content) == expected['bytes'] and sha(content) == expected['sha256'], 'Archive member bytes differ')
            members[member.name] = content
    require(set(members) == set(packed['archive_members']), 'Missing archive member')
    helper = members.pop(HELPER_MEMBER)
    require(sha(helper) == packed['helper_source_sha256'] == sha(raw_file(HELPER)), 'Exact operational helper identity differs')
    archive_root = p['basename'] + '/'
    require(all(name.startswith(archive_root) for name in members), 'Snapshot archive root differs')
    payload = {name[len(archive_root):]: content for name, content in members.items()}
    manifest, prefix, plan, report = validate_snapshot(payload, exposure, condition, attempts, raw_file(p['plan']), raw_file(p['contract']))
    for key in ('plan_sha256', 'contract_sha256', 'manifest_sha256'):
        require(manifest[key] == packed[key], 'Archived manifest differs from package receipt')
    require(prefix['partial_sha256'] == packed['partial_sha256'], 'Archived prefix differs from package receipt')
    require(sha(raw_file(fresh)) == packed['sha256'], 'Fresh archive changed during verification')
    artifact_keys = ('artifact_id', 'file_name', 'file_id', 'library_file_id', 'size_bytes', 'sha256', 'upload_status', 'verification_status')
    rows = prefix['runs']
    result = {'schema_version': 'jensen-matched-condition-partial-result.v1', 'recorded_at': manifest['created_at'],
              'recorded_at_basis': 'immutable_snapshot_creation', 'exposure': exposure, 'condition': condition,
              'status': 'partial_saved_and_fresh_readback_verified', 'collectible': False,
              'final_full_condition_save_required': True, 'planned_case_count': len(plan['cases']),
              **{key: manifest[key] for key in ('captured_attempts', 'captured_completed', 'captured_failed',
                                               'case_ids', 'plan_sha256', 'plan_file_sha256', 'contract_sha256', 'contract_file_sha256')},
              'completed_case_ids': [row['case_id'] for row in rows if row['generation']['completed'] is True],
              'failed_case_ids': [row['case_id'] for row in rows if row['generation']['completed'] is False],
              'uncaptured_case_ids': prefix['uncaptured_case_ids'],
              'model_identity_hashes': _identity_hashes(report['model']),
              'snapshot_manifest_sha256': manifest['manifest_sha256'],
              'snapshot_manifest_file_sha256': sha(payload['manifest.json']), 'partial_sha256': prefix['partial_sha256'],
              'helper_source_sha256': packed['helper_source_sha256'], 'snapshot_tool_sha256': manifest['tool_sha256'],
              'payload_files_sha256': {name: sha(content) for name, content in sorted(payload.items())},
              'artifact': {key: record[key] for key in artifact_keys},
              'verification': {'fresh_archive_bytes_and_sha256': True, 'exact_regular_member_inventory': True,
                               'all_archive_member_hashes': True, 'snapshot_manifest_and_prefix_seals': True,
                               'captured_rows_and_metadata_revalidated': True, 'original_full_report_revalidated': False},
              'scope': 'Immutable finalized attempt prefix only. The active condition may have advanced. This is not a final condition, collector input or generation resume cursor.',
              'full_raw_outputs': 'Preserved only inside the private verified partial artifact.',
              'human_review': 'ungraded', 'efficacy_claim': False}
    write_new(p['public'], encoded(result))
    print(json.dumps({'report': p['public'].name, 'captured_attempts': attempts,
                      'captured_completed': manifest['captured_completed'], 'captured_failed': manifest['captured_failed'],
                      'collectible': False}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['package', 'finalize'])
    parser.add_argument('exposure', choices=['development', 'heldout'])
    parser.add_argument('condition', choices=['facts', 'style', 'adapter'])
    parser.add_argument('attempts', type=int)
    args = parser.parse_args()
    try:
        globals()[args.command](args.exposure, args.condition, args.attempts)
    except (ValueError, KeyError, TypeError, OSError, tarfile.TarError) as exc:
        parser.exit(2, f'Partial delivery failed: {type(exc).__name__}: {exc}\n')
