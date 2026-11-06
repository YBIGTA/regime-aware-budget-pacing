"""Larger, resumable A-only development search. Never opens final test."""
import argparse
from dataclasses import asdict, replace
import hashlib
from functools import lru_cache
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from core import Config, Gains, PI, generate, rollout
from run import collect, init, read_json, write_json, source_hash, csv_write

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('outputs/pacing_abc_strong_v2'))
    parser.add_argument('--quick-check', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out/'strong.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Already running')
    cfg = Config(train_days=1000, validation_days=200, test_days=500, trials=768)
    stages = [(512,32),(128,64),(32,128),(8,256)]
    if args.quick_check:
        cfg = replace(cfg, minutes=60, train_days=3, validation_days=3, test_days=3)
        stages = [(8,2),(4,3),(2,4),(2,5)]
    counts = [n for n,d in stages]
    dev_days = list(range(30000,30000+stages[-1][1]))
    audit_days = list(range(150000,150000+(3 if args.quick_check else 200)))
    metadata = dict(config=asdict(cfg), stages=stages, development_days=dev_days,
        audit_days=audit_days, source=source_hash(), runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy=np.__version__, local_candidates=4 if args.quick_check else 256,
        ranges=dict(kp=[.01,1000],ki=[.001,1000],m0=[.02,.98]),
        selection='validation mean tracking_error+unspent; TV reported separately',
        fairness='Primary: freeze selected A gains and cold start in BC. Secondary: ABC same stage sizes, days, and total simulation budget; report model fitting time separately.',
        reuse='B/C may access exactly the same training/development/validation information; audit is development-only convergence diagnostic, never final test.',
        scope='Uncalibrated synthetic benchmark. No claim of global optimum or real-market validity.')
    meta_path = out/'strong_protocol.json'
    if meta_path.exists():
        if read_json(meta_path) != json.loads(json.dumps(metadata)):
            raise SystemExit('Protocol/code changed; use a new output directory')
    else:
        if any(p.name!='strong.lock' for p in out.iterdir()):
            raise SystemExit('Output is not empty')
        write_json(meta_path, metadata)
    proto_hash = hashlib.sha256(meta_path.read_bytes()).hexdigest()
    if (out/'DONE.json').exists():
        print((out/'DONE.json').read_text(), flush=True)
        return
    print('OUTPUT',out,flush=True)
    data = out/'common_data'
    if not (data/'protocol.json').exists():
        init(data,cfg)
    if not (data/'train_prediction.csv').exists():
        collect(data)
    if not (data/'train_prediction_manifest.json').exists():
        raise SystemExit('Training collection incomplete: use a new output directory')
    cache = out/'scores'
    cache.mkdir(exist_ok=True)
    @lru_cache(maxsize=256)
    def episode(day):
        return generate(cfg,day)
    def key(p):
        return hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()[:20]
    def score(p, days):
        values=[]
        for day in days:
            path=cache/f'{key(p)}_{day}.json'
            if path.exists():
                row=read_json(path)
            else:
                ep=episode(day)
                active=replace(cfg,initial_multiplier=p['m0'])
                row=rollout(ep,active,PI(active,Gains(p['kp'],p['ki'])))[0]
                write_json(path,row)
            values.append(row)
        return values
    def average(p, days):
        rows=score(p,days)
        return float(np.mean([r['objective'] for r in rows]))
    rng=np.random.default_rng(20261106)
    pool=[dict(kp=93.45690571935442,ki=20.019117873752368,m0=.5),
          dict(kp=0.,ki=0.,m0=.5),dict(kp=4.,ki=0.,m0=.5)]
    while len(pool)<counts[0]:
        pool.append(dict(kp=float(10**rng.uniform(-2,3)),
                         ki=0. if len(pool)%5==0 else float(10**rng.uniform(-3,3)),
                         m0=float(rng.uniform(.02,.98))))
    def successive(pool, schedule, name):
        for stage,(keep,days) in enumerate(schedule):
            ranked=[]
            for i,p in enumerate(pool):
                j=average(p,dev_days[:days])
                ranked.append((j,key(p),p))
                if i%16==0:
                    print(f'{name} stage={stage+1} candidate={i+1}/{len(pool)} days={days} J={j:.7f}',flush=True)
            ranked.sort(key=lambda x:(x[0],x[1]))
            pool=[p for _,_,p in ranked[:keep]]
            write_json(out/f'{name}_stage{stage+1}.json',ranked)
        return pool
    schedule=[(counts[i+1] if i+1<len(counts) else counts[-1],days)
              for i,(_,days) in enumerate(stages)]
    global_final=successive(pool,schedule,'global')
    anchor=min(global_final,key=lambda p:(average(p,dev_days),key(p)))
    local=[anchor]
    while len(local)<metadata['local_candidates']:
        local.append(dict(kp=float(np.clip(max(anchor['kp'],.01)*10**rng.normal(0,.35),.01,1000)),
            ki=0. if len(local)%5==0 else float(np.clip(max(anchor['ki'],.001)*10**rng.normal(0,.5),.001,1000)),
            m0=float(np.clip(anchor['m0']+rng.normal(0,.12),.02,.98))))
    if args.quick_check:
        local_schedule=[(2,2),(2,5)]
    else:
        local_schedule=[(64,32),(16,128),(8,256)]
    local_final=successive(local,local_schedule,'local')
    finalists={key(p):p for p in global_final+local_final}
    table=[]
    for p in finalists.values():
        rows=score(p,list(cfg.ids('validation')))
        table.append(dict(**p,objective=float(np.mean([r['objective'] for r in rows])),
            tracking_error=float(np.mean([r['tracking_error'] for r in rows])),
            unspent=float(np.mean([r['unspent'] for r in rows])),
            multiplier_tv=float(np.mean([r['multiplier_tv'] for r in rows]))))
    table.sort(key=lambda r:(r['objective'],r['tracking_error'],key({k:r[k] for k in ('kp','ki','m0')})))
    csv_write(out/'validation_finalists.csv',table)
    selected=table[0]
    params={k:selected[k] for k in ('kp','ki','m0')}
    old=score(anchor,audit_days)
    new=score(params,audit_days)
    delta=np.array([b['objective']-a['objective'] for a,b in zip(old,new)])
    boot=np.random.default_rng(813).choice(delta,(2000,len(delta)),replace=True).mean(axis=1)
    ci=list(map(float,np.quantile(boot,[.025,.975])))
    tolerance=max(.0002,.02*float(np.mean([r['objective'] for r in old])))
    stable=bool(ci[0]>=-tolerance and ci[1]<=tolerance)
    boundary=bool(params['kp']>=900 or params['ki']>=900 or params['m0']<=.025 or params['m0']>=.975)
    audit=dict(added_search_equivalent=stable,delta_mean=float(delta.mean()),bootstrap95=ci,
        practical_margin=tolerance,parameter_near_boundary=boundary,
        note='Development convergence diagnostic, not final-test significance or a proof of optimality.')
    write_json(out/'convergence_audit.json',audit)
    frozen=dict(model='A',protocol_sha256=proto_hash,parameters=params,validation=selected,
        convergence=audit,ready_for_BC=stable and not boundary,
        common_cold_start=params['m0'],test_opened=False)
    write_json(out/'frozen_A.json',frozen)
    status='SEARCH_STABLE' if frozen['ready_for_BC'] else 'MORE_DEVELOPMENT_REQUIRED'
    write_json(out/'DONE.json',dict(status=status,selected=selected,audit=audit,test_opened=False))
    print('RESULT',status,json.dumps(frozen,ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
