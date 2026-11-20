"""Predeclared finite refinement study. No NSGA-II or CMA weight training."""
import argparse,csv,time,math
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
from env import *
from controller import RefinedPI,slope_features,predict_slope,usable
from fast_refined import run,evaluate,TRACE
OUT=BUNDLE/'results'
PRIMARY='centered_ratio'
FAMILIES={'centered_ratio':6,'gated_ratio':7,'learned_slope':8}
POWERS=[.25,.5,1.]
TARGETS=[.75,1.,1.25]
def now():return datetime.now(timezone.utc).isoformat()
def design():
    return dict(parent_commit='41b9c48',fit240=FIT,calibration80=CAL,selection200=SELECT,fresh400=FRESH,
      primary_family=PRIMARY,families=FAMILIES,powers=POWERS,targets=TARGETS,nsga2='closed',final_test_used=False,
      fixed='B predictor, CMA E2 weights/Kp map, cap1, weighted integral state, existing antiwindup and target',
      correction='clip(target*exp(gate*(center-power*log(max(gain,1e-8)))),0.5,2); invalid feedback falls back to factor1',
      calibration='Fit center on baseline trajectories so mean factor equals target including fallback. Execute all 27 adaptive policies on CAL80, then set each matched constant to its realized mean factor. Freeze before selection.',
      gate='1/(1+(residual_ratio_standard_error/max(ratio,1e-8)/0.25)^2); heuristic, not a calibrated confidence interval',
      estimator='Methods 6/7: trailing12 ratio + EMA 1-exp(-1/6), >=6 valid observations. Method8: same EMA of predicted local derivative. All start at previous prior.',
      model=dict(features=FEATURES,ridge_alpha=100.,target='log(max(offline finite difference at B base +/-0.05,1e-5))',
        label_scope='FIT240 current auction tape only for offline supervision; no current tape features',
        scaling='fit240 mean/std only; standardized features clipped +/-8 at prediction',prediction_clip=[1e-5,.2]),
      selection='Within each family min J among TE/UR <= CMA and V<CMA, otherwise CMA retained; also freeze minJ diagnostic. Primary family fixed in advance.',
      constant_comparison='All matched constants plus 0.75 and 1.25. Match on calibration closed-loop mean factor, not on confirmation or mean Ki.',
      attribution='Compare adaptive policies with their matched constants separately from CMA replacement. Mean success and uncertainty reported separately.',
      bootstrap='20000 paired day replicates seed20261120, unadjusted 95% descriptive intervals',
      limits='Synthetic one-day first-price episodes, independently coded Ki adaptation rather than full original paper. Fresh400 and final500 cannot be used for tuning.')
def init():
    OUT.mkdir(parents=True,exist_ok=True)
    hashes={p.name:rt.sha(p) for p in Path(__file__).parent.glob('*.py') if p.name!='report.py'}
    path=OUT/'protocol.json'
    if path.exists():
        p=rt.read_json(path);assert p['design']==design();assert p['sources']==hashes,'Frozen source changed';return
    oldfiles=list((FIRST/'code').glob('*.py'))+list((OLD/'code').glob('*.py'))
    rt.write_json(path,dict(frozen_utc=now(),design=design(),sources=hashes,simulator=rt.fingerprint(),
      reference_sources={str(p.relative_to(ROOT)):rt.sha(p) for p in oldfiles},plan_sha256=rt.sha(BUNDLE/'PLAN.md')))
    print('Protocol frozen',flush=True)
def dataset_day(day):
    ep=rt.episode(day);_,tr=run(day,spec())
    hist=[];spent=0.;xs=[];ys=[];ids=[]
    zero=rt.core.Feedback(-1,0,0,0,0,0,False)
    for t,tick in enumerate(ep.tape):
        obs=rt.core.Observation(t,288,ep.budget,spent,tuple(([zero]*12+hist)[-12:]))
        base=tr[t,6]
        if t>=12 and len(usable(obs))>=6:
            low=max(0,base-.05);high=min(1,base+.05)
            def spend(a):
                bid=tick.base*a
                return float(bid[(bid>tick.competitor)&(bid>0)].sum())/ep.budget
            xs.append(slope_features(obs,base));ys.append((spend(high)-spend(low))/(high-low));ids.append([day,t])
        action=tr[t,0];cost,wins,censored=rt.core.execute(tick,action,max(0.,ep.budget-spent));spent+=cost
        assert abs(spent-tr[t,2])<1e-7
        hist.append(rt.core.Feedback(t,action,cost,len(tick.base),wins,float(tick.base.mean()) if len(tick.base) else 0.,censored))
    return np.asarray(xs),np.asarray(ys),np.asarray(ids)
def dataset(days):
    run(PROBE[0],spec())
    with ProcessPoolExecutor(max_workers=8) as pool:batches=list(pool.map(dataset_day,days))
    return tuple(np.concatenate([b[i] for b in batches]) for i in range(3))
def fit():
    init()
    if (OUT/'slope_model.json').exists():return
    start=time.perf_counter();x,y,ids=dataset(FIT)
    mu=x.mean(axis=0);sd=x.std(axis=0);sd[sd<1e-10]=1.
    z=(x-mu)/sd;target=np.log(np.maximum(y,1e-5));intercept=float(target.mean())
    coef=np.linalg.solve(z.T@z+100*np.eye(z.shape[1]),z.T@(target-intercept))
    model=dict(mean=mu.tolist(),scale=sd.tolist(),coef=coef.tolist(),intercept=intercept,
      fitted_utc=now(),days=len(FIT),rows=len(y),features=FEATURES,alpha=100.,seconds=time.perf_counter()-start)
    rt.write_json(OUT/'slope_model.json',model)
    print('Sensitivity model fitted on',len(y),'development intervals',flush=True)
def save(name,rows):
    rt.write_csv(OUT/(name+'_daily.csv'),rows);scores=rt.summarize(rows)
    rt.write_csv(OUT/(name+'_summary.csv'),[dict(model=k,**v) for k,v in scores.items()]);return scores
def calibrate_center(gain,gate,valid,power,target):
    def mean(c):return float(np.where(valid,np.clip(target*np.exp(gate*(c-power*np.log(np.maximum(gain,1e-8)))),.5,2),1.).mean())
    lo,hi=-60.,60.
    for _ in range(90):
        mid=(lo+hi)/2
        if mean(mid)<target:lo=mid
        else:hi=mid
    center=(lo+hi)/2
    assert abs(mean(center)-target)<1e-9
    return center,mean(center)
def calibrate():
    init()
    if (OUT/'calibration.json').exists():return
    start=time.perf_counter();model=rt.read_json(OUT/'slope_model.json')
    specs={'S0_CMA':spec(),'S1_legacy':spec(4)};families={k:[] for k in FAMILIES};checks=[]
    for family,method in FAMILIES.items():
        # power=0, center=0, target=1 gives CMA's actions with passive estimator tracing.
        tr=np.concatenate([run(day,spec(method,power=0,model=model))[1] for day in CAL])
        for p in POWERS:
            for target in TARGETS:
                center,actual=calibrate_center(tr[:,9],tr[:,17],tr[:,11]==0,p,target)
                key=f'{family}_p{p:g}_m{target:g}'
                specs[key]=spec(method,p,target,center,model=model);families[family].append(key)
                checks.append(dict(model=key,power=p,target=target,center=center,baseline_mean_factor=actual))
    adaptive={k:v for k,v in specs.items() if k not in ['S0_CMA','S1_legacy']}
    rows=[];matched={}
    for key,sp in adaptive.items():
        factors=[]
        for day in CAL:
            score,tr=run(day,sp);rows.append(dict(model=key,**score));factors.append(tr[:,10])
        value=float(np.concatenate(factors).mean());control='matched_'+key
        specs[control]=spec(5,constant=value);matched[key]=control
        next(r for r in checks if r['model']==key)['closed_loop_mean_factor']=value
    specs.update({'constant_0.75':spec(5,constant=.75),'constant_1.25':spec(5,constant=1.25)})
    families['constant']=list(matched.values())+['constant_0.75','constant_1.25']
    save('calibration',rows);rt.write_csv(OUT/'centering_checks.csv',checks)
    x,y,ids=dataset(CAL);pred=np.array([predict_slope(row,model) for row in x])
    rt.write_csv(OUT/'calibration_slope_forecasts.csv',[dict(day=int(i[0]),tick=int(i[1]),actual=float(a),learned=float(b),B=B_SLOPE) for i,a,b in zip(ids,y,pred)])
    rt.write_json(OUT/'slope_validation.json',dict(split='CAL80, no fit or model selection on these labels',observations=len(y),
      B_mae=float(np.abs(y-B_SLOPE).mean()),learned_mae=float(np.abs(y-pred).mean()),
      B_rmse=float(np.sqrt(np.mean((y-B_SLOPE)**2))),learned_rmse=float(np.sqrt(np.mean((y-pred)**2))),
      learned_correlation=float(np.corrcoef(y,pred)[0,1]),actual_cv=float(y.std()/y.mean()),learned_cv=float(pred.std()/pred.mean())))
    rt.write_json(OUT/'calibration.json',dict(frozen_utc=now(),model_sha256=rt.sha(OUT/'slope_model.json'),
      specs=specs,families=families,matched=matched,seconds=time.perf_counter()-start,
      passive_rollouts=3*len(CAL),adaptive_rollouts=len(adaptive)*len(CAL),label_rollouts=len(CAL)))
    print('Centers and matched constants frozen;',len(specs),'policies',flush=True)
def succeeds(s,b):
    return s['tracking_error']<=b['tracking_error']+1e-15 and s['unspent']<=b['unspent']+1e-15 and s['multiplier_tv']<b['multiplier_tv']-1e-12
def choose(keys,scores):
    feasible=[k for k in keys if succeeds(scores[k],scores['S0_CMA'])]
    order=lambda k:(scores[k]['objective'],scores[k]['multiplier_tv'],k)
    return dict(primary=min(feasible,key=order) if feasible else 'S0_CMA',diagnostic=min(keys,key=order),eligible=len(feasible))
def select():
    init()
    if (OUT/'selection.json').exists():return
    cal=rt.read_json(OUT/'calibration.json');specs=cal['specs'];start=time.perf_counter()
    scores=save('selection',evaluate(specs,SELECT));picks={k:choose(v,scores) for k,v in cal['families'].items()}
    keys={'S0_CMA','S1_legacy'}|{p[r] for p in picks.values() for r in ['primary','diagnostic']}
    keys|={cal['matched'][k] for k in list(keys) if k in cal['matched']}
    # Equal-power neutral-mean ablation is fixed, independent of its score.
    for family in FAMILIES:
        key=f'{family}_p0.25_m1';keys|={key,cal['matched'][key]}
    rt.write_json(OUT/'selection.json',dict(frozen_utc=now(),calibration_sha256=rt.sha(OUT/'calibration.json'),
      families=picks,primary=picks[PRIMARY],selected_specs={k:specs[k] for k in sorted(keys)},scores=scores,
      seconds=time.perf_counter()-start,rollouts=len(specs)*len(SELECT)))
    print('Selection frozen',picks,'; confirmation models',len(keys),flush=True)
def reference_task(item):
    day,specs=item;ep=rt.episode(day);rows=[]
    for key,sp in specs.items():
        policy=rt.policy(sp) if key=='S0_CMA' else RefinedPI(sp)
        score,_=rt.core.rollout(ep,rt.cfg,policy)
        rows.append(dict(model=key,**{k:score[k] for k in ['day','tape_sha256']+METRICS}))
    return rows
def paired(rows,pairs):
    days=sorted({r['day'] for r in rows});index={(r['model'],r['day']):r for r in rows}
    sample=np.random.default_rng(20261120).integers(0,len(days),(20000,len(days)));out=[]
    for key,base in sorted(pairs):
        if key==base:continue
        for metric in METRICS:
            d=np.array([index[key,day][metric]-index[base,day][metric] for day in days]);ci=np.quantile(d[sample].mean(axis=1),[.025,.975])
            out.append(dict(new=key,baseline=base,metric=metric,mean_difference=float(d.mean()),
              ci95_low=float(ci[0]),ci95_high=float(ci[1]),days_better=int((d<-1e-12).sum()),days_worse=int((d>1e-12).sum()),days_equal=int((np.abs(d)<=1e-12).sum())))
    return out
def confirm():
    init()
    if (OUT/'evaluation_done.json').exists():return
    sel=rt.read_json(OUT/'selection.json');cal=rt.read_json(OUT/'calibration.json');specs=sel['selected_specs'];start=time.perf_counter()
    rt.write_json(OUT/'evaluation_started.json',dict(started_utc=now(),selection_sha256=rt.sha(OUT/'selection.json')))
    fast=evaluate(specs,FRESH);rt.write_csv(OUT/'fresh_daily_fast.csv',fast)
    print('Fast evaluation complete; original Python replay:',len(fast),'rollouts',flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i,batch in enumerate(pool.map(reference_task,[(d,specs) for d in FRESH])):
            rows.extend(batch)
            if (i+1)%50==0:print('Reference days',i+1,'/400',flush=True)
    rows.sort(key=lambda r:(r['model'],r['day']));index={(r['model'],r['day']):r for r in fast}
    errors=[dict(model=r['model'],day=r['day'],max_difference=max(abs(r[m]-index[r['model'],r['day']][m]) for m in METRICS)) for r in rows]
    rt.write_csv(OUT/'evaluator_audit.csv',errors);err=max(r['max_difference'] for r in errors);assert err<1e-8
    scores=save('fresh',rows);pairs={(k,'S0_CMA') for k in specs}
    pairs|={(k,'S1_legacy') for k in specs if not k.startswith('matched_')}
    pairs|={(k,cal['matched'][k]) for k in specs if k in cal['matched']}
    pairs|={('gated_ratio_p0.25_m1','centered_ratio_p0.25_m1'),('learned_slope_p0.25_m1','centered_ratio_p0.25_m1')}
    rt.write_csv(OUT/'fresh_paired.csv',paired(rows,pairs))
    comparisons=[]
    for family,pick in sel['families'].items():
        for role in ['primary','diagnostic']:
            key=pick[role];b=scores['S0_CMA'];s=scores[key]
            comparisons.append(dict(family=family,role=role,model=key,eligible=pick['eligible'],
              **{m+'_relative_pct':(s[m]/b[m]-1)*100 for m in METRICS},strict_CMA_success=key!='S0_CMA' and succeeds(s,b),
              strict_matched_success=succeeds(s,scores[cal['matched'][key]]) if key in cal['matched'] else False))
    rt.write_csv(OUT/'method_comparison.csv',comparisons)
    key=sel['primary']['primary'];ok=key!='S0_CMA' and succeeds(scores[key],scores['S0_CMA'])
    rt.write_json(OUT/'conclusion.json',dict(primary_family=PRIMARY,primary=sel['primary'],fresh_strict_mean_success=ok,
      decision='candidate_for_final_test' if ok else 'retain_CMA',final_test_used=False))
    rt.write_json(OUT/'evaluation_done.json',dict(finished_utc=now(),models=len(specs),rollouts=len(rows),reference_rollouts=len(rows),
      seconds=time.perf_counter()-start,max_fast_reference_difference=err,selection_sha256=rt.sha(OUT/'selection.json')))
    print('Confirmation complete:',rt.read_json(OUT/'conclusion.json'),flush=True)
def audit():
    init();p=rt.read_json(OUT/'protocol.json');m=rt.read_json(OUT/'slope_model.json');c=rt.read_json(OUT/'calibration.json')
    s=rt.read_json(OUT/'selection.json');e=rt.read_json(OUT/'evaluation_started.json');end=rt.read_json(OUT/'evaluation_done.json')
    assert p['frozen_utc']<m['fitted_utc']<c['frozen_utc']<s['frozen_utc']<e['started_utc']<end['finished_utc']
    assert rt.sha(OUT/'slope_model.json')==c['model_sha256']
    assert rt.sha(OUT/'calibration.json')==s['calibration_sha256']
    assert rt.sha(OUT/'selection.json')==e['selection_sha256']==end['selection_sha256']
    assert rt.sha(BUNDLE/'PLAN.md')==p['plan_sha256']
    assert all(rt.sha(ROOT/k)==v for k,v in p['reference_sources'].items());assert rt.fingerprint()==p['simulator']
    with (OUT/'fresh_daily.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    tapes={}
    for r in rows:tapes.setdefault(int(r['day']),set()).add(r['tape_sha256'])
    assert set(tapes)==set(FRESH) and all(len(v)==1 for v in tapes.values())
    rt.write_json(OUT/'audit.json',dict(status='passed',audited_utc=now(),source_and_plan_unchanged=True,
      previous_experiments_unchanged=True,model_calibration_selection_before_confirmation=True,
      shared_tapes=True,final_test_used=False,max_fast_reference_difference=end['max_fast_reference_difference']))
    print('Audit passed',flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['init','fit','calibrate','select','confirm','audit','all'])
    stage=parser.parse_args().stage
    for fn in ([init,fit,calibrate,select,confirm,audit] if stage=='all' else [globals()[stage]]):fn()
