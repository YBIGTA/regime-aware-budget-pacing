"""Ten fixed policies: isolate one-sided remaining-time spend targets."""
import argparse,csv,time,json
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
from env import *
from controller import AsymmetricPI
from fast_asymmetric import run,TRACE
from fast_eval import pack
OUT=BUNDLE/'results'
MAIN=[('CMA_A05','CMA'),('G_A05','G'),('CMA_A05','CMA_T05'),('G_A05','G_T05')]
PAIRS=sorted(set(MAIN+[(k,'CMA') for k in policies() if k!='CMA']+
  [(k,'G') for k in policies() if k.startswith('G_')]+
  [('CMA_A1','CMA_T1'),('G_A1','G_T1'),('G_A05','CMA_A05'),('G_A1','CMA_A1')]))
PROFILE=['error','factor','ki','base','target_rate','required_rate','forecast_rate','risk_weight',
         'restoration_applied','factor_before','integral_blocked','saturated','budget_censored','target_above_prediction']

def now():return datetime.now(timezone.utc).isoformat()
def sources():return {p.name:rt.sha(p) for p in Path(__file__).parent.glob('*.py') if p.name!='report.py'}
def read_csv(path):
    with path.open(encoding='utf8') as f:return list(csv.DictReader(f))

def cohort_separation():
    def ints(value):
        if isinstance(value,dict):return set().union(*(ints(v) for v in value.values())) if value else set()
        if isinstance(value,list):return set().union(*(ints(v) for v in value)) if value else set()
        return {value} if isinstance(value,int) and not isinstance(value,bool) else set()
    checked={}
    for path in (ROOT/'research_report').glob('pacing_*/results/protocol.json'):
        if path.parent.parent==BUNDLE:continue
        assert not set(FRESH+PROBE).intersection(ints(rt.read_json(path))),path
        checked[str(path.relative_to(ROOT))]=rt.sha(path)
    assert len(checked)>=7
    return checked

def init():
    OUT.mkdir(parents=True,exist_ok=True)
    design=dict(parent='cb8b0bb',fresh1000=FRESH,probe=PROBE,policies=policies(),
      contrasts=PAIRS,main_UR_contrasts=MAIN,primary='G_A05',bootstrap=20000,seed=20261123,
      final_test_used=False,no_fit_calibration_or_selection=True,noninferiority_margin=0,
      cohort_separation=cohort_separation())
    path=OUT/'protocol.json'
    if path.exists():
        p=rt.read_json(path);assert p['design']==json.loads(json.dumps(design))
        assert p['sources']==sources(),'Frozen source changed'
        assert p['plan_sha256']==rt.sha(BUNDLE/'PLAN.md');return
    refs=list((PARENT/'code').glob('*.py'))+list((OLD/'code').glob('*.py'))+list((DEADLINE/'code').glob('*.py'))+[PARENT/'results/calibration.json',DEADLINE/'results/protocol.json']
    rt.write_json(path,dict(frozen_utc=now(),design=design,sources=sources(),
      reference_sources={str(p.relative_to(ROOT)):rt.sha(p) for p in refs},
      plan_sha256=rt.sha(BUNDLE/'PLAN.md'),simulator=rt.fingerprint()))
    print('Protocol frozen: ten policies; four main UR comparisons',flush=True)

def diagnose(tr):
    def x(name):return tr[:,TRACE.index(name)]
    e=x('error');f=x('factor');before=x('factor_before');applied=x('restoration_applied')>0
    pre=np.r_[0.,e[:-1]];shift=x('target_rate')-1/288;db=x('base')-x('base_nominal')
    return dict(behind_rate=float((pre>1e-12).mean()),target_raise_rate=float((shift>1e-12).mean()),
      target_lower_rate=float((shift<-1e-12).mean()),base_raise_rate=float((db>1e-12).mean()),
      base_lower_rate=float((db<-1e-12).mean()),last2h_target_raise_rate=float((shift[-24:]>1e-12).mean()),
      mean_target_raise=float(np.maximum(shift,0).mean()),mean_target_lower=float(np.maximum(-shift,0).mean()),
      mean_positive_error=float(np.maximum(e,0).mean()),mean_ahead=float(np.maximum(-e,0).mean()),
      last2h_base_raise_rate=float((db[-24:]>1e-12).mean()),mean_factor=float(f.mean()),mean_factor_before=float(before.mean()),mean_ki=float(x('ki').mean()),
      mean_ki_original=float(x('ki_original').mean()),mean_target_rate=float(x('target_rate').mean()),
      mean_required_rate=float(x('required_rate').mean()),mean_risk_weight=float(x('risk_weight').mean()),
      risk_active_rate=float((x('risk_weight')>0).mean()),restoration_rate=float(applied.mean()),
      restore_and_blocked_rate=float((applied & (x('integral_blocked')>0)).mean()),
      restoration_last2h_rate=float(applied[-24:].mean()),mean_factor_last2h=float(f[-24:].mean()),
      base_change_rate=float((np.abs(x('base')-x('base_nominal'))>1e-12).mean()),
      target_above_prediction_rate=float(x('target_above_prediction').mean()),
      target_above_prediction_last2h=float(x('target_above_prediction')[-24:].mean()),
      fallback_rate=float(x('fallback').mean()),factor_clip_rate=float(x('factor_clipped').mean()),
      ki_ceiling_rate=float((x('ki')>=6000-1e-10).mean()),
      last2h_TE=float(np.abs(e[-24:]).mean()),last30m_TE=float(np.abs(e[-6:]).mean()),
      error20h=float(e[239]),error22h=float(e[263]),error23h=float(e[275]),
      remaining_pre_final=float(1/288+e[-2]),final_spend=float(1/288+e[-2]-e[-1]),
      last2h_saturated=float(x('saturated')[-24:].mean()),
      last2h_budget_censored=float(x('budget_censored')[-24:].mean()))

def fast_task(item):
    day,specs=item;rows=[];diag=[];profiles=[]
    for k,sp in specs.items():
        score,tr=run(day,sp);rows.append(dict(model=k,**score));diag.append(dict(model=k,day=day,**diagnose(tr)))
        profiles.append(np.column_stack([np.abs(tr[:,1])]+[tr[:,TRACE.index(c)] for c in PROFILE]))
    return rows,diag,np.stack(profiles)

def reference_task(item):
    day,specs=item;ep=rt.episode(day);rows=[];diag=[]
    for k,sp in specs.items():
        score,records=rt.core.rollout(ep,rt.cfg,AsymmetricPI(sp))
        rows.append(dict(model=k,**{x:score[x] for x in ['day','tape_sha256']+METRICS}))
        tr=np.zeros((288,len(TRACE)))
        for col in TRACE:
            if col=='g_ref':continue
            tr[:,TRACE.index(col)]=[r[col] for r in records]
        diag.append(dict(model=k,day=day,**diagnose(tr)))
    return rows,diag

def bootstrap_columns(d):
    rng=np.random.default_rng(20261123);boot=[];n=d.shape[0]
    for _ in range(40):
        weights=rng.multinomial(n,np.full(n,1/n),size=500);boot.append(weights@d/n)
    boot=np.concatenate(boot);out=[]
    for j in range(d.shape[1]):
        ci=np.quantile(boot[:,j],[.00625,.025,.95,.975,.99375])
        out.append(dict(mean_difference=float(d[:,j].mean()),ci95_low=float(ci[1]),ci95_high=float(ci[3]),
          one_sided95_upper=float(ci[2]),ci98_75_low=float(ci[0]),ci98_75_high=float(ci[4]),
          days_better=int((d[:,j]<-1e-12).sum()),days_worse=int((d[:,j]>1e-12).sum()),days_equal=int((np.abs(d[:,j])<=1e-12).sum())))
    return out

def paired(rows,pairs,metrics):
    days=sorted({int(r['day']) for r in rows});idx={(r['model'],int(r['day'])):r for r in rows}
    cols=[(a,b,m) for a,b in pairs for m in metrics]
    d=np.array([[float(idx[a,day][m])-float(idx[b,day][m]) for a,b,m in cols] for day in days])
    return [dict(new=a,baseline=b,metric=m,main_UR=(a,b) in MAIN and m=='unspent',**v)
            for (a,b,m),v in zip(cols,bootstrap_columns(d))]

def strict(s,b):
    return s['tracking_error']<=b['tracking_error'] and s['unspent']<=b['unspent'] and s['multiplier_tv']<b['multiplier_tv']

def summarize_and_compare(rows,diagnostics):
    scores=rt.summarize(rows);rt.write_csv(OUT/'fresh_summary.csv',[dict(model=k,**v) for k,v in scores.items()])
    paired_rows=paired(rows,PAIRS,METRICS);rt.write_csv(OUT/'fresh_paired.csv',paired_rows)
    dm=[m for m in diagnostics[0] if m not in ['model','day']]
    rt.write_csv(OUT/'diagnostics_summary.csv',[dict(model=k,**{m:float(np.mean([r[m] for r in diagnostics if r['model']==k])) for m in dm}) for k in policies()])
    selected_dm=['last2h_TE','last30m_TE','remaining_pre_final','final_spend','mean_positive_error','mean_ahead','mean_target_raise','mean_target_lower','base_raise_rate','base_lower_rate']
    rt.write_csv(OUT/'diagnostics_paired.csv',paired(diagnostics,PAIRS,selected_dm))
    split=[]
    for label,days in [('first500',FRESH[:500]),('last500',FRESH[500:])]:
        s=rt.summarize([r for r in rows if r['day'] in set(days)])
        split.extend(dict(split=label,model=k,**v) for k,v in s.items())
    rt.write_csv(OUT/'half_sample_summary.csv',split)
    comparisons=[]
    for k,s in scores.items():
        b=scores['CMA'];comparisons.append(dict(model=k,**{m+'_relative_pct':(s[m]/b[m]-1)*100 for m in METRICS},
          strict_mean_CMA_success=k!='CMA' and strict(s,b)))
    rt.write_csv(OUT/'CMA_comparisons.csv',comparisons)
    pi={(r['new'],r['baseline'],r['metric']):r for r in paired_rows}
    rt.write_json(OUT/'conclusion.json',dict(primary='G_A05',primary_strict_CMA_mean_success=strict(scores['G_A05'],scores['CMA']),
      main_UR=[dict(new=a,baseline=b,adjusted_improvement_supported=pi[a,b,'unspent']['ci98_75_high']<0,
        **{k:pi[a,b,'unspent'][k] for k in ['mean_difference','ci98_75_low','ci98_75_high']}) for a,b in MAIN],
      decision='candidate_for_final_validation' if strict(scores['G_A05'],scores['CMA']) else 'retain_CMA',
      no_fresh_selection=True,final_test_used=False))

def confirm():
    init()
    if (OUT/'evaluation_done.json').exists():return
    protocol=rt.read_json(OUT/'protocol.json');specs=protocol['design']['policies'];start=time.perf_counter()
    rt.write_json(OUT/'evaluation_started.json',dict(started_utc=now(),protocol_sha256=rt.sha(OUT/'protocol.json')))
    run(PROBE[0],specs['CMA'])
    for day in FRESH:pack(day)
    fast=[];diagnostics=[];profile=np.zeros((len(specs),288,1+len(PROFILE)))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for rows,diag,tr in pool.map(fast_task,[(d,specs) for d in FRESH]):
            fast.extend(rows);diagnostics.extend(diag);profile+=tr
    fast.sort(key=lambda r:(r['model'],r['day']));diagnostics.sort(key=lambda r:(r['model'],r['day']))
    rt.write_csv(OUT/'fresh_daily_fast.csv',fast);rt.write_csv(OUT/'diagnostics_daily.csv',diagnostics)
    rt.write_csv(OUT/'time_profiles.csv',[dict(model=k,tick=t,**dict(zip(['absolute_error']+PROFILE,profile[i,t]/len(FRESH)))) for i,k in enumerate(specs) for t in range(288)])
    print('Fast complete; original Python replay:',len(fast),'rollouts',flush=True)
    rows=[];diag=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i,(batch,db) in enumerate(pool.map(reference_task,[(d,specs) for d in FRESH])):
            rows.extend(batch);diag.extend(db)
            if (i+1)%100==0:print('Reference days',i+1,'/',len(FRESH),flush=True)
    rows.sort(key=lambda r:(r['model'],r['day']))
    index={(r['model'],r['day']):r for r in fast};dindex={(r['model'],r['day']):r for r in diagnostics}
    errors=[dict(model=r['model'],day=r['day'],max_metric_difference=max(abs(r[m]-index[r['model'],r['day']][m]) for m in METRICS)) for r in rows]
    diag_error=max(abs(r[m]-dindex[r['model'],r['day']][m]) for r in diag for m in r if m not in ['day','model'])
    err=max(r['max_metric_difference'] for r in errors);assert err<1e-8 and diag_error<1e-8,(err,diag_error)
    rt.write_csv(OUT/'evaluator_audit.csv',errors);rt.write_csv(OUT/'fresh_daily.csv',rows)
    summarize_and_compare(rows,diagnostics)
    rt.write_json(OUT/'evaluation_done.json',dict(finished_utc=now(),models=len(specs),days=len(FRESH),
      fast_rollouts=len(fast),reference_rollouts=len(rows),seconds=time.perf_counter()-start,
      max_fast_reference_difference=err,max_diagnostic_difference=diag_error,
      protocol_sha256=rt.sha(OUT/'protocol.json'),final_test_used=False))
    print('Confirmation complete:',rt.read_json(OUT/'conclusion.json'),flush=True)

def audit():
    init();p=rt.read_json(OUT/'protocol.json');a=rt.read_json(OUT/'evaluation_started.json');z=rt.read_json(OUT/'evaluation_done.json')
    assert p['frozen_utc']<a['started_utc']<z['finished_utc']
    assert a['protocol_sha256']==z['protocol_sha256']==rt.sha(OUT/'protocol.json')
    assert all(rt.sha(ROOT/k)==v for k,v in p['reference_sources'].items());assert p['simulator']==rt.fingerprint()
    rows=read_csv(OUT/'fresh_daily.csv');tapes={}
    assert len(rows)==len(FRESH)*len(policies()) and len({(r['model'],r['day']) for r in rows})==len(rows)
    for r in rows:tapes.setdefault(int(r['day']),set()).add(r['tape_sha256'])
    assert set(tapes)==set(FRESH) and all(len(v)==1 for v in tapes.values())
    rt.write_json(OUT/'audit.json',dict(status='passed',audited_utc=now(),source_and_plan_unchanged=True,
      parents_unchanged=True,protocol_before_confirmation=True,shared_tapes=True,no_fresh_selection=True,
      final_test_used=False,max_metric_difference=z['max_fast_reference_difference'],max_diagnostic_difference=z['max_diagnostic_difference']))
    print('Audit passed',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['init','confirm','audit','all'])
    stage=parser.parse_args().stage
    for fn in ([init,confirm,audit] if stage=='all' else [globals()[stage]]):fn()
