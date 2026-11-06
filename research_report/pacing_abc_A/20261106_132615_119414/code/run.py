"""Commands: init, collect, tune, evaluate. Test is accessed ONLY by evaluate."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
from core import Config, Gains, PI, VERSION, generate, rollout, features

HERE = Path(__file__).resolve().parent

def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text())

def source_hash():
    h = hashlib.sha256()
    for name in ('core.py', 'run.py'):
        h.update(name.encode())
        h.update((HERE/name).read_bytes())
    return h.hexdigest()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def csv_write(path, rows):
    if not rows:
        raise ValueError('Empty output')
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def load(out):
    meta = read_json(out/'protocol.json')
    cfg = Config(**meta['config'])
    if meta['source_sha256'] != source_hash() or meta['numpy'] != np.__version__:
        raise ValueError('Code/NumPy changed: create a new run; do not overwrite this experiment')
    if meta['version'] != VERSION or meta['config_sha256'] != cfg.digest():
        raise ValueError('Protocol changed')
    return cfg, meta

def init(out, cfg):
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Output must be empty; previous experiment is not overwritten')
    # No evaluation episodes generated here.
    write_json(out/'protocol.json', dict(version=VERSION, config=asdict(cfg),
        config_sha256=cfg.digest(), source_sha256=source_hash(),
        python=platform.python_version(), numpy=np.__version__,
        splits={s:list(cfg.ids(s)) for s in ('train','validation','test')},
        primary='Mean daily normalized absolute cumulative tracking error',
        selection='Mean daily tracking_error + unspent, ties by tracking_error then candidate index',
        action='Continuous [0,1]; B/C base uses common 101-point grid, then shared continuous PI correction',
        initial='All models start at initial_multiplier. No current interval information.',
        capacity='Necessary prefix-capacity diagnostic only; no episode filtering',
        inference='Fixed base A; B/C frozen predictors, identical past-only features, no online fitting',
        primary_comparison='Freeze A gains across ABC; change only base provider. Predictors trained and selected inside train split.',
        secondary_comparison='Retune all methods with equal end-to-end validation evaluation budgets; report separately.',
        final_test_policy='Freeze all ABC artifacts before evaluate; adapting BC after reading A test requires a new final holdout.',
        tuning='Same validation days and maximum end-to-end candidate evaluations for A/B/C',
        warning='Synthetic engineering benchmark, not calibrated real-market evidence'))
    print('Protocol created:', out/'protocol.json')

def collect(out):
    cfg, meta = load(out)
    destination = out/'train_prediction.csv'
    if destination.exists():
        raise FileExistsError(destination)
    rng_seed = cfg.master_seed+500
    manifests = []
    # Stream rows rather than retaining entire dataset. One real action per interval.
    temp = destination.with_suffix('.tmp')
    with temp.open('w', newline='') as f:
        writer = None
        for day in cfg.ids('train'):
            ep = generate(cfg, day)
            rng = np.random.default_rng(np.random.SeedSequence([rng_seed,day]))
            def provider(obs):
                return float(rng.uniform(cfg.multiplier_min, cfg.multiplier_max))
            def sink(obs, action, spend, censored):
                nonlocal writer
                x = features(obs, action)
                row = dict(day=day, tick=obs.tick, target_spend_fraction=spend/ep.budget,
                           budget_censored=int(censored))
                row.update({f'x{i}':float(v) for i,v in enumerate(x)})
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
            rollout(ep, cfg, PI(cfg, Gains(0,0), provider), sink=sink)
            manifests.append(dict(day=day,tape_sha256=ep.digest()))
    temp.replace(destination)
    write_json(out/'train_prediction_manifest.json', dict(episodes=manifests,
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        target='realized next interval spend / budget',
        fit_rule='B/C use all same rows including budget-censored outcomes; predict realized budget-constrained spend. Remaining budget is encoded by budget and spent fraction. Current-row censor flag is label metadata, NOT a model input',
        features='candidate, time fraction, log budget, spent fraction, error; then H past intervals: present mask, action, spend/B, log1p count, win rate, base mean, censor flag',
        row_count=cfg.train_days*cfg.ticks))
    print('Training dataset:', destination)

def candidates(cfg):
    rng = np.random.default_rng(np.random.SeedSequence([cfg.master_seed,777]))
    values = [Gains(0,0), Gains(4,0)]
    while len(values) < cfg.trials:
        values.append(Gains(float(10**rng.uniform(-1,2)),
                            0. if len(values)%4 == 0 else float(10**rng.uniform(-2,2))))
    return values

def tune(out):
    cfg, meta = load(out)
    if (out/'frozen_A.json').exists():
        raise ValueError('A is frozen; use a new experiment directory to change settings')
    episodes = [generate(cfg, day) for day in cfg.ids('validation')]
    results = []
    for index, gains in enumerate(candidates(cfg)):
        scores = [rollout(e,cfg,PI(cfg,gains))[0] for e in episodes]
        row = dict(candidate=index, **asdict(gains),
                   objective=float(np.mean([s['objective'] for s in scores])),
                   tracking_error=float(np.mean([s['tracking_error'] for s in scores])),
                   unspent=float(np.mean([s['unspent'] for s in scores])),
                   multiplier_tv=float(np.mean([s['multiplier_tv'] for s in scores])))
        results.append(row)
        print(f"candidate {index+1}/{cfg.trials}: J={row['objective']:.6f}", flush=True)
    selected = min(results,key=lambda r:(r['objective'],r['tracking_error'],r['candidate']))
    csv_write(out/'A_validation_search.csv',results)
    frozen = dict(model='A',protocol_sha256=fingerprint(meta), selection=selected,
                  gains={k:selected[k] for k in ('kp','ki')},
                  validation_tapes=[dict(day=e.day,sha256=e.digest()) for e in episodes])
    write_json(out/'frozen_A.json',frozen)
    print('Frozen A:',json.dumps(selected))

def summarize(scores, cfg):
    keys = ('tracking_error','unspent','objective','multiplier_tv','max_ahead','saturation_fraction')
    rng = np.random.default_rng(cfg.bootstrap_seed)
    indices = rng.integers(0,len(scores),size=(2000,len(scores)))
    result = {}
    for key in keys:
        values = np.array([s[key] for s in scores])
        lo,hi = np.quantile(values[indices].mean(axis=1),[.025,.975])
        result[key] = dict(mean=float(values.mean()),bootstrap95=[float(lo),float(hi)])
    result['days'] = len(scores)
    result['capacity_shortfall_days'] = sum(s['capacity_shortfall']>1e-12 for s in scores)
    result['uncertainty_scope'] = 'Independent synthetic days from this generator only; future ABC comparison must bootstrap paired day differences.'
    return result

def evaluate(out):
    cfg, meta = load(out)
    frozen = read_json(out/'frozen_A.json')
    if frozen['protocol_sha256'] != fingerprint(meta):
        raise ValueError('Frozen model/protocol mismatch')
    lock = dict(frozen_sha256=fingerprint(frozen),protocol_sha256=fingerprint(meta))
    target = out/'test_A'
    target.mkdir(exist_ok=True)
    lock_path = target/'TEST_ACCESSED.json'
    if lock_path.exists():
        if read_json(lock_path) != lock:
            raise ValueError('Test already consumed under different settings')
    else:
        write_json(lock_path,lock)  # before first test tape is generated
    if (target/'summary.json').exists():
        print('Already evaluated; returning saved report. Do not tune on this test.')
        print((target/'summary.json').read_text())
        return
    gains = Gains(**frozen['gains'])
    scores = []
    for day in cfg.ids('test'):
        path = target/f'day_{day}.json'
        if path.exists():
            score = read_json(path)
        else:
            ep = generate(cfg,day)
            score,rows = rollout(ep,cfg,PI(cfg,gains))
            csv_write(target/f'trace_{day}.csv',rows)
            write_json(path,score)
        scores.append(score)
    csv_write(target/'episodes.csv',scores)
    summary = summarize(scores,cfg)
    write_json(target/'summary.json',summary)
    print(json.dumps(summary,indent=2))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['init','collect','tune','evaluate'])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--config',type=Path,help='init only; complete or partial Config JSON')
    args = ap.parse_args()
    # Concurrent writes to the same run are rejected, including on macOS/Linux.
    import fcntl
    args.out.mkdir(parents=True,exist_ok=True)
    lock_file = args.out.parent/('.'+args.out.name+'.pacing.lock')
    with lock_file.open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This experiment is already running')
        if args.command == 'init':
            cfg = Config(**(read_json(args.config) if args.config else {}))
            init(args.out,cfg)
        else:
            if args.config:
                ap.error('--config is init-only')
            {'collect':collect,'tune':tune,'evaluate':evaluate}[args.command](args.out)

if __name__ == '__main__':
    main()
