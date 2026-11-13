"""Collect three durable M6 conditions and privately archive their blind review packet.

Only agent-review/reviewer/ may be shared with blinded reviewers. This helper
does not run models, contact storage, change final counters or publish files.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile

from tools.adaptation_evaluation.blind_review import DEFAULT_RUBRIC, package_reviews
from tools.adaptation_evaluation.local_condition import (
    DURABILITY_SCHEMA, collect_conditions, validate_execution_contract,
)
from tools.adaptation_evaluation.protocol import CONDITIONS

ROOT = Path('/workspace/scratch/6f64e1b53384/who-is-your-bias')
ART = Path('/workspace/scratch/6f64e1b53384/artifacts/jensen-2026-11-13')
HELPER = Path(__file__).resolve()
PLAN_SEALS = {
    'development': 'cd2bd7340c3eb7d9c1ddc248b78e25d69b8e00c9d246454cf5925e082a1dd030',
    'heldout': '7fd35de01ea1a364aa7b5a2bf8547bf3a4a6cd3ebee4b3764b8ce7c8957e5c7c',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe(path):
    require(not any(part.is_symlink() for part in (path, *path.parents)),
            f'Symlink refused: {path}')
    return path


def read_bytes(path):
    safe(path)
    require(path.is_file(), f'Expected regular file: {path}')
    return path.read_bytes()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def inventory(directory):
    safe(directory)
    require(directory.is_dir(), f'Expected directory: {directory}')
    result = {}
    for path in sorted(directory.rglob('*')):
        safe(path)
        require(path.is_file() or path.is_dir(), f'Special file refused: {path}')
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = sha(read_bytes(path))
    return result


def unused(path):
    safe(path)
    require(not path.exists(), f'Preserve existing or interrupted output: {path}')


def new_stream(path):
    safe(path)
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb')


def publish_new(path, raw):
    temporary = path.with_name(path.name + '.collection-part')
    unused(path)
    unused(temporary)
    with new_stream(temporary) as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    require(read_bytes(temporary) == raw, 'Staged receipt bytes changed')
    os.link(temporary, path)
    temporary.unlink()


def collect_package(exposure):
    stem = f'm6-{exposure}-collection-review-bf16-v2'
    output = ART / f'text-evaluation-v1/collections/{exposure}-bf16-v2'
    collection, review = output / 'collection', output / 'agent-review'
    archive = ART / f'jensen-{stem}.tar.gz'
    archive_part = archive.with_name(archive.name + '.collection-part')
    packaged = ART / f'{stem}-packaged.json'
    for path in (output, archive, archive_part, packaged,
                 packaged.with_name(packaged.name + '.collection-part')):
        unused(path)

    plan_path = ART / f'text-evaluation-v1/frozen-v2/{exposure}/plan.json'
    contract_path = ROOT / 'reports/jensen-evaluation-v1/execution-contract-bf16-v2.json'
    receipts_path = ART / f'text-evaluation-v1/execution-{exposure}-bf16-v2/durability-receipts.json'
    inputs = {'inputs/plan.json': plan_path,
              'inputs/execution-contract-bf16-v2.json': contract_path,
              'inputs/durability-receipts.json': receipts_path,
              'collect_package_m6.py': HELPER}
    original = {name: read_bytes(path) for name, path in inputs.items()}
    rubric_bytes = read_bytes(DEFAULT_RUBRIC)
    plan, contract, receipts = (json.loads(original[name]) for name in (
        'inputs/plan.json', 'inputs/execution-contract-bf16-v2.json',
        'inputs/durability-receipts.json'))
    validate_execution_contract(contract, plan)
    require(plan['exposure'] == exposure and plan['plan_sha256'] == PLAN_SEALS[exposure]
            and contract['runtime'] == {'device': 'cpu', 'dtype': 'bfloat16', 'threads': 8},
            'Use the registered matched plan and common BF16 contract')
    entries = receipts.get('conditions', [])
    require(receipts.get('schema_version') == DURABILITY_SCHEMA
            and isinstance(entries, list) and len(entries) == 3
            and all(isinstance(row, dict) for row in entries), 'Require all three full-condition receipts')
    by_condition = {row.get('condition'): row for row in entries}
    require(set(by_condition) == set(CONDITIONS), 'Receipts must cover facts, style and adapter exactly')
    directories, condition_hashes = [], {}
    for condition in CONDITIONS:
        receipt = by_condition[condition]
        source_archive = receipt.get('archive', {})
        require(receipt.get('status') == 'uploaded_and_fresh_readback_verified'
                and receipt.get('plan_sha256') == plan['plan_sha256']
                and type(source_archive.get('bytes')) is int and source_archive['bytes'] > 0
                and isinstance(source_archive.get('sha256'), str)
                and re.fullmatch(r'[0-9a-f]{64}', source_archive['sha256'])
                and all(isinstance(source_archive.get(key), str) and source_archive[key].strip()
                        for key in ('file_id', 'library_file_id')),
                f'Missing complete fresh-readback receipt for {condition}')
        directory = ART / f'text-evaluation-v1/condition-readbacks/{exposure}-{condition}-bf16-v2/{condition}'
        actual = inventory(directory)
        require(actual == receipt.get('files_sha256'), f'Readback files differ for {condition}')
        directories.append(directory)
        condition_hashes[condition] = actual

    # No output is created before all three receipts and readback inventories pass.
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    summary = collect_conditions(plan_path, contract_path, directories, receipts_path, collection)
    manifest = package_reviews([(plan_path, collection)], review, rubric=DEFAULT_RUBRIC)
    require(manifest['rubric_sha256'] == sha(rubric_bytes)
            and read_bytes(review / 'reviewer/rubric.md') == rubric_bytes,
            'The registered rubric changed while packaging')
    review_hashes = inventory(review)
    require(review_hashes == {**manifest['files_sha256'],
                              'manifest.json': sha(read_bytes(review / 'manifest.json'))},
            'Review manifest does not cover the exact finished packet')
    counts = {condition: {'planned': len(plan['cases']),
                         'attempted': row['attempted'], 'completed': row['completed'],
                         'failed': row['attempted'] - row['completed'],
                         'unattempted': len(plan['cases']) - row['attempted']}
              for condition, row in summary['conditions'].items()}
    totals = {key: sum(row[key] for row in counts.values()) for key in
              ('planned', 'attempted', 'completed', 'failed', 'unattempted')}
    require(manifest['denominators'] == totals and manifest['review_samples'] == totals['completed'],
            'Review sample and failure denominators differ')

    files = inventory(output)
    members = {f'{output.name}/{relative}': output / relative for relative in files}
    members.update(inputs)
    identities = {name: {'bytes': len(raw), 'sha256': sha(raw)}
                  for name, path in members.items() for raw in [read_bytes(path)]}
    with new_stream(archive_part) as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as tar:
                for name, path in sorted(members.items()):
                    raw = read_bytes(path)
                    require(identities[name] == {'bytes': len(raw), 'sha256': sha(raw)},
                            f'Archive source changed: {name}')
                    info = tarfile.TarInfo(name)
                    info.size, info.mode = len(raw), 0o600
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ''
                    tar.addfile(info, io.BytesIO(raw))
        stream.flush()
        os.fsync(stream.fileno())

    require(files == inventory(output), 'Collection or realized review changed during archiving')
    require(all(read_bytes(path) == original[name] for name, path in inputs.items())
            and read_bytes(DEFAULT_RUBRIC) == rubric_bytes, 'Frozen inputs or helper changed')
    require(all(inventory(directory) == condition_hashes[condition]
                for condition, directory in zip(CONDITIONS, directories)), 'A condition readback changed')
    archive_raw = read_bytes(archive_part)
    os.link(archive_part, archive)
    archive_part.unlink()
    receipt = {
        'schema_version': 'm6-collection-review-package.v1', 'status': 'packaged_locally_ungraded',
        'exposure': exposure, 'archive': archive.name, 'bytes': len(archive_raw),
        'sha256': sha(archive_raw), 'archive_member_count': len(identities),
        'archive_members': identities,
        'archive_members_sha256': {name: row['sha256'] for name, row in identities.items()},
        'collection_root': str(output), 'collection_files_sha256': files,
        'plan_sha256': plan['plan_sha256'], 'plan_file_sha256': sha(original['inputs/plan.json']),
        'contract_sha256': contract['contract_sha256'],
        'contract_file_sha256': sha(original['inputs/execution-contract-bf16-v2.json']),
        'durability_receipts_file_sha256': sha(original['inputs/durability-receipts.json']),
        'helper_sha256': sha(original['collect_package_m6.py']),
        'rubric_sha256': manifest['rubric_sha256'], 'review_packager_sha256': manifest['packager_sha256'],
        'review_manifest_file_sha256': review_hashes['manifest.json'],
        'conditions': counts, 'denominators': totals, 'review_samples': manifest['review_samples'],
        'source_pools': manifest['source_pools'], 'archive_access': 'private_coordinator_only',
        'share_only': str(review / 'reviewer'),
        'private_key_included': True, 'realized_random_review_mapping_preserved': True,
        'human_review': 'ungraded', 'efficacy_claim': False,
        'remote_upload_performed': False, 'remote_readback_verified': False,
        'validation_scope': 'Existing collector and review packager unchanged; supplied durable receipts and local exact bytes checked. Root must upload and genuinely freshly verify this new archive.',
    }
    publish_new(packaged, (json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())
    print(json.dumps({key: receipt[key] for key in (
        'archive', 'bytes', 'sha256', 'archive_member_count', 'exposure',
        'denominators', 'review_samples', 'rubric_sha256', 'share_only')}))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exposure', choices=('development', 'heldout'))
    args = parser.parse_args()
    collect_package(args.exposure)


if __name__ == '__main__':
    main()
