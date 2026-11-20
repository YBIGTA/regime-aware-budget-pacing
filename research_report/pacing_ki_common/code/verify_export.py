"""Read-only integrity checks for the Ki-only publication package.

Historical experiment audit records are preserved, not rewritten or reissued.
This verifies the export and available frozen inputs, not omitted NSGA-II history.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
COMMON = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify():
    manifest = json.loads((COMMON / 'export_manifest.json').read_text(encoding='utf8'))
    records = manifest['files']
    for rel, record in records.items():
        data = (ROOT / rel).read_bytes()
        if digest(data) != record['export_sha256']:
            raise ValueError(f'Export file changed: {rel}')
        if record['change'] == 'shared_runtime_path_only':
            data = data.replace(b'research_report/pacing_ki_common', b'research_report/pacing_nsga2')
            if digest(data) != record['source_sha256']:
                raise ValueError(f'Unexpected code change beyond import path: {rel}')
        if record['change'] in ('unchanged', 'shared_replay_utility_relocated_unchanged'):
            if digest(data) != record['source_sha256']:
                raise ValueError(f'Original content changed: {rel}')
    code_checks = 0
    simulator_checks = 0
    for bundle in manifest['bundles']:
        folder = ROOT / 'research_report' / bundle
        protocol = json.loads((folder / 'results/protocol.json').read_text(encoding='utf8'))
        for name, expected in protocol['sources'].items():
            rel = f'research_report/{bundle}/code/{name}'
            if records[rel]['source_sha256'] != expected:
                raise ValueError(f'Historical source hash mismatch: {rel}')
            code_checks += 1
        if 'plan_sha256' in protocol:
            if digest((folder / 'PLAN.md').read_bytes()) != protocol['plan_sha256']:
                raise ValueError(f'Historical plan changed: {bundle}')
        for rel, expected in protocol['simulator'].items():
            data = (ROOT / rel).read_bytes()
            if digest(data.replace(b'\r\n', b'\n')) != expected['lf_sha256']:
                raise ValueError(f'Frozen simulator/model differs: {rel}')
            simulator_checks += 1
    excluded = list((ROOT / 'research_report').glob('pacing_nsga2*'))
    if excluded:
        raise ValueError(f'Excluded experiment directories present: {excluded}')
    return dict(status='passed', check='ki_publication_integrity',
                exported_files=len(records), historical_source_hashes=code_checks,
                normalized_simulator_checks=simulator_checks,
                historical_results_unchanged=True, original_audits_reissued=False,
                nsga_experiment_folders_present=False, final_test_used=False)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
