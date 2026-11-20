"""Replay frozen Ki policies without selection, fitting or archive mutations."""
from pathlib import Path
import argparse
import csv
import importlib
import json
import sys
from verify_export import verify

ROOT = Path(__file__).resolve().parents[3]
CHOICES = {
    'pacing_sensitivity_pi': ('bridge', 'fast_sensitivity', 'SensitivityPI', 'selection.json', 'selected_specs'),
    'pacing_sensitivity_refinement': ('env', 'fast_refined', 'RefinedPI', 'selection.json', 'selected_specs'),
    'pacing_sensitivity_hypotheses': ('env', 'fast_hypotheses', 'HypothesisPI', 'calibration.json', 'specs'),
    'pacing_deadline_pi': ('env', 'fast_deadline', 'DeadlinePI', 'protocol.json', 'design'),
    'pacing_asymmetric_pi': ('env', 'fast_asymmetric', 'AsymmetricPI', 'protocol.json', 'design'),
}


def replay(bundle, count):
    verify()
    env_name, fast_name, class_name, spec_file, spec_key = CHOICES[bundle]
    folder = ROOT / 'research_report' / bundle
    sys.path.insert(0, str(folder / 'code'))
    env = importlib.import_module(env_name)
    fast = importlib.import_module(fast_name)
    controller = getattr(importlib.import_module('controller'), class_name)
    specs = env.rt.read_json(folder / 'results' / spec_file)[spec_key]
    if spec_key == 'design':
        specs = specs['policies']
    if not 1 <= count <= len(env.FRESH):
        raise ValueError(f'--days must be between 1 and {len(env.FRESH)}')
    indices = env.np.linspace(0, len(env.FRESH) - 1, count, dtype=int)
    days = [env.FRESH[int(i)] for i in indices]
    with (folder / 'results/fresh_daily.csv').open(encoding='utf8', newline='') as handle:
        recorded = {(r['model'], int(r['day'])): r for r in csv.DictReader(handle)}
    max_archived = max_fast_reference = 0.0
    for day in days:
        for name, spec in specs.items():
            saved = recorded[name, day]
            calculated, _ = fast.run(day, spec)
            slow, _ = env.rt.core.rollout(env.rt.episode(day), env.rt.cfg, controller(spec))
            if calculated['tape_sha256'] != saved['tape_sha256']:
                raise ValueError(f'Auction tape mismatch: {bundle}, {name}, {day}')
            max_archived = max(max_archived, *(abs(calculated[m] - float(saved[m])) for m in env.METRICS))
            max_fast_reference = max(max_fast_reference, *(abs(calculated[m] - slow[m]) for m in env.METRICS))
    if max(max_archived, max_fast_reference) >= 1e-9:
        raise ValueError(f'Replay differs: {max_archived}, {max_fast_reference}')
    for forbidden in (200000, 200499):
        try:
            env.rt.episode(forbidden)
        except ValueError:
            pass
        else:
            raise ValueError('Final test was not blocked')
    verify()  # Includes every original result byte: replay must not overwrite it.
    result = dict(status='passed', check='ki_publication_replay', bundle=bundle,
                  days=days, policies=len(specs), fast_rollouts=count*len(specs),
                  reference_rollouts=count*len(specs),
                  max_archived_metric_difference=max_archived,
                  max_fast_reference_difference=max_fast_reference,
                  final_test_used=False, archived_results_unchanged=True,
                  optimizer_modules_loaded=sorted(set(sys.modules) & {'pymoo', 'cma'}))
    output = ROOT / 'outputs/ki-export-checks'
    output.mkdir(parents=True, exist_ok=True)
    (output / (bundle + '-replay.json')).write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', choices=CHOICES, required=True)
    parser.add_argument('--days', type=int, default=2,
                        help='Evenly spaced days from the existing confirmation cohort; no new model selection.')
    args = parser.parse_args()
    print(json.dumps(replay(args.bundle, args.days), indent=2))
