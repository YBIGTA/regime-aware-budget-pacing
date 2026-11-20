"""Fixed hypothesis contrasts; no fresh-data selection or parameter search."""
import argparse,csv,time,json
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
from env import *
from controller import HypothesisPI
from fast_hypotheses import run,TRACE
from fast_eval import pack
OUT=BUNDLE/'results'
MAIN=[('G','C_matched'),('R_neutral','R'),('L_noema','L'),('L_wide','L')]
PAIRS=sorted(set(MAIN+[(k,'CMA') for k in policies() if k!='CMA']+
  [('G','C_all'),('G','C_warm'),('G','C_rule'),('C_warm','C_all'),('C_rule','C_warm'),
   ('C_matched','C_rule'),('G_neutral','G'),('G_first','G'),('R_first','R')]))

def now():return datetime.now(timezone.utc).isoformat()

def sources():
    return {p.name:rt.sha(p) for p in Path(__file__).parent.glob('*.py') if p.name!='report.py'}

def init():
    OUT.mkdir(parents=True,exist_ok=True)
    design=dict(parent='5156715',calibration80=CAL,fresh800=FRESH,probe=PROBE,policies=policies(),
      contrasts=PAIRS,main_TE_contrasts=MAIN,bootstrap=20000,seed=20261121,final_test_used=False,
      no_selection=True,noninferiority_margin=0,mean_match_tolerance=1e-6)
    path=OUT/'protocol.json'
    if path.exists():
        p=rt.read_json(path);assert p['design']==json.loads(json.dumps(design))
        assert p['sources']==sources(),'Frozen source changed'
        assert p['plan_sha256']==rt.sha(BUNDLE/'PLAN.md');return
    refs=list((PARENT/'code').glob('*.py'))+list((OLD/'code').glob('*.py'))
    refs += [PARENT/'results'/n for n in ['selection.json','calibration.json','slope_model.json']]
    rt.write_json(path,dict(frozen_utc=now(),design=design,sources=sources(),
      reference_sources={str(p.relative_to(ROOT)):rt.sha(p) for p in refs},
      plan_sha256=rt.sha(BUNDLE/'PLAN.md'),simulator=rt.fingerprint()))
    print('Protocol frozen: 14 policies; 4 fixed main TE contrasts',flush=True)

def diagnose(tr,sp):
    err=np.abs(tr[:,TRACE.index('error')]);f=tr[:,TRACE.index('factor')]
    fallback=tr[:,TRACE.index('fallback')]>0
    eligible=np.flatnonzero(~fallback);first=f[eligible[0]] if len(eligible) else 1.
    return dict(mean_factor=float(f.mean()),mean_ki=float(tr[:,4].mean()),
      fallback_rate=float(fallback.mean()),late_fallback_rate=float(fallback[6:].mean()),
      factor_clip_rate=float(tr[:,15].mean()),ki_ceiling_rate=float((tr[:,4]>=6000-1e-10).mean()),
      first_adaptive_factor=float(first),initial_log_deviation=abs(math.log(first/sp['target'])),
      first2h_TE=float(err[:24].mean()),post2h_TE=float(err[24:].mean()),
      TE_0_30=float(err[:6].sum()/288),TE_30_120=float(err[6:24].sum()/288),
      TE_120_1200=float(err[24:240].sum()/288),TE_1200_1440=float(err[240:].sum()/288))

def cal_eval(sp):
    tr=np.concatenate([run(d,sp)[1] for d in CAL])
    return float(tr[:,10].mean()),float(tr[:,11].mean())

def calibrate():
    init()
    if (OUT/'calibration.json').exists():return
    specs=policies();target,_=cal_eval(specs['G']);iterations=[]
    for i in range(12):
        value,rate=cal_eval(specs['C_matched'])
        iterations.append(dict(iteration=i,constant=specs['C_matched']['constant'],mean=value,fallback_rate=rate,target=target))
        if abs(value-target)<1e-6:break
        c=(target-rate)/(1-rate);assert .5<=c<=1
        specs['C_matched']['constant']=float(c)
    assert abs(value-target)<1e-6,'Calibration did not converge; do not open confirmation'
    rt.write_json(OUT/'calibration.json',dict(frozen_utc=now(),protocol_sha256=rt.sha(OUT/'protocol.json'),
      specs=specs,target_mean_factor=target,iterations=iterations,rollouts=(1+len(iterations))*len(CAL)))
    print('Frozen matched constant',specs['C_matched']['constant'],'mean error',value-target,flush=True)

def fast_task(item):
    day,specs=item;rows=[];diag=[];profiles=[]
    for k,sp in specs.items():
        score,tr=run(day,sp);rows.append(dict(model=k,**score))
        diag.append(dict(model=k,day=day,**diagnose(tr,sp)))
        profiles.append(np.column_stack([np.abs(tr[:,1]),tr[:,10],tr[:,4],tr[:,11],tr[:,15]]))
    return rows,diag,np.stack(profiles)

def reference_task(item):
    day,specs=item;ep=rt.episode(day);rows=[];diag=[]
    for k,sp in specs.items():
        score,records=rt.core.rollout(ep,rt.cfg,rt.policy(sp) if k=='CMA' else HypothesisPI(sp))
        rows.append(dict(model=k,**{x:score[x] for x in ['day','tape_sha256']+METRICS}))
        if k!='CMA':
            tr=np.zeros((288,len(TRACE)))
            for col in ['error','factor','ki','fallback','factor_clipped']:
                tr[:,TRACE.index(col)]=[r[col] for r in records]
            diag.append(dict(model=k,day=day,**diagnose(tr,sp)))
    return rows,diag

def read_csv(path):
    with path.open(encoding='utf8') as f:return list(csv.DictReader(f))

def paired(rows,pairs,metrics):
    days=sorted({int(r['day']) for r in rows});idx={(r['model'],int(r['day'])):r for r in rows}
    columns=[(a,b,m) for a,b in pairs for m in metrics]
    d=np.array([[float(idx[a,day][m])-float(idx[b,day][m]) for a,b,m in columns] for day in days])
    rng=np.random.default_rng(20261121);boot=[]
    for _ in range(40):
        weights=rng.multinomial(len(days),np.full(len(days),1/len(days)),size=500)
        boot.append(weights@d/len(days))
    boot=np.concatenate(boot);out=[]
    for j,(a,b,m) in enumerate(columns):
        ci=np.quantile(boot[:,j],[.00625,.025,.95,.975,.99375])
        out.append(dict(new=a,baseline=b,metric=m,mean_difference=float(d[:,j].mean()),
          ci95_low=float(ci[1]),ci95_high=float(ci[3]),one_sided95_upper=float(ci[2]),
          main_TE=(a,b) in MAIN and m=='tracking_error',ci98_75_low=float(ci[0]),ci98_75_high=float(ci[4]),
          days_better=int((d[:,j]<-1e-12).sum()),days_worse=int((d[:,j]>1e-12).sum()),days_equal=int((np.abs(d[:,j])<=1e-12).sum())))
    return out

def confirm():
    init()
    if (OUT/'evaluation_done.json').exists():return
    cal=rt.read_json(OUT/'calibration.json');specs=cal['specs'];start=time.perf_counter()
    rt.write_json(OUT/'evaluation_started.json',dict(started_utc=now(),calibration_sha256=rt.sha(OUT/'calibration.json')))
    run(PROBE[0],specs['CMA'])
    for day in FRESH:pack(day)
    fast=[];diagnostics=[];profile=np.zeros((len(specs),288,5))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for rows,diag,tr in pool.map(fast_task,[(d,specs) for d in FRESH]):
            fast.extend(rows);diagnostics.extend(diag);profile+=tr
    fast.sort(key=lambda r:(r['model'],r['day']));diagnostics.sort(key=lambda r:(r['model'],r['day']))
    rt.write_csv(OUT/'fresh_daily_fast.csv',fast);rt.write_csv(OUT/'diagnostics_daily.csv',diagnostics)
    names=['absolute_error','factor','ki','fallback','factor_clip']
    rt.write_csv(OUT/'time_profiles.csv',[dict(model=k,tick=t,**dict(zip(names,profile[i,t]/len(FRESH)))) for i,k in enumerate(specs) for t in range(288)])
    print('Fast complete; independently replaying',len(fast),'days x policies',flush=True)
    rows=[];diag=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i,(batch,db) in enumerate(pool.map(reference_task,[(d,specs) for d in FRESH])):
            rows.extend(batch);diag.extend(db)
            if (i+1)%80==0:print('Reference days',i+1,'/',len(FRESH),flush=True)
    rows.sort(key=lambda r:(r['model'],r['day']))
    index={(r['model'],r['day']):r for r in fast};dindex={(r['model'],r['day']):r for r in diagnostics}
    errors=[dict(model=r['model'],day=r['day'],max_metric_difference=max(abs(r[m]-index[r['model'],r['day']][m]) for m in METRICS)) for r in rows]
    diag_error=max(abs(r[m]-dindex[r['model'],r['day']][m]) for r in diag for m in r if m not in ['day','model'])
    err=max(r['max_metric_difference'] for r in errors);assert err<1e-8 and diag_error<1e-8
    rt.write_csv(OUT/'evaluator_audit.csv',errors);rt.write_csv(OUT/'fresh_daily.csv',rows)
    scores=rt.summarize(rows);rt.write_csv(OUT/'fresh_summary.csv',[dict(model=k,**v) for k,v in scores.items()])
    rt.write_csv(OUT/'fresh_paired.csv',paired(rows,PAIRS,METRICS))
    dm=[m for m in diagnostics[0] if m not in ['model','day']]
    rt.write_csv(OUT/'diagnostics_paired.csv',paired(diagnostics,PAIRS,dm))
    rt.write_csv(OUT/'diagnostics_summary.csv',[dict(model=k,**{m:float(np.mean([r[m] for r in diagnostics if r['model']==k])) for m in dm}) for k in specs])
    rt.write_json(OUT/'evaluation_done.json',dict(finished_utc=now(),models=len(specs),days=len(FRESH),
      fast_rollouts=len(fast),reference_rollouts=len(rows),seconds=time.perf_counter()-start,
      max_fast_reference_difference=err,max_diagnostic_difference=diag_error,
      calibration_sha256=rt.sha(OUT/'calibration.json'),final_test_used=False))
    print('Confirmation complete; no policy selected',flush=True)

def audit():
    init();p=rt.read_json(OUT/'protocol.json');c=rt.read_json(OUT/'calibration.json')
    a=rt.read_json(OUT/'evaluation_started.json');z=rt.read_json(OUT/'evaluation_done.json')
    assert p['frozen_utc']<c['frozen_utc']<a['started_utc']<z['finished_utc']
    assert c['protocol_sha256']==rt.sha(OUT/'protocol.json')
    assert a['calibration_sha256']==z['calibration_sha256']==rt.sha(OUT/'calibration.json')
    assert all(rt.sha(ROOT/k)==v for k,v in p['reference_sources'].items())
    assert p['simulator']==rt.fingerprint()
    rows=read_csv(OUT/'fresh_daily.csv');tapes={}
    assert len(rows)==len(FRESH)*14 and len({(r['model'],r['day']) for r in rows})==len(rows)
    for r in rows:tapes.setdefault(int(r['day']),set()).add(r['tape_sha256'])
    assert set(tapes)==set(FRESH) and all(len(v)==1 for v in tapes.values())
    rt.write_json(OUT/'audit.json',dict(status='passed',audited_utc=now(),source_and_plan_unchanged=True,
      parents_unchanged=True,calibration_before_confirmation=True,shared_tapes=True,no_fresh_selection=True,
      final_test_used=False,max_metric_difference=z['max_fast_reference_difference'],max_diagnostic_difference=z['max_diagnostic_difference']))
    print('Audit passed',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['init','calibrate','confirm','audit','all'])
    stage=parser.parse_args().stage
    for fn in ([init,calibrate,confirm,audit] if stage=='all' else [globals()[stage]]):fn()
