"""Frozen, finite sensitivity-PI pilot. No neural-network retraining or NSGA-II."""
import argparse
import csv
import json
import time
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from bridge import *
from fast_sensitivity import evaluate, run, TRACE
from controller import SensitivityPI

OUT=BUNDLE/'results'
PRIMARY='state_window_smooth'
PAPER='https://www.amazon.science/publications/adaptive-spend-control-for-programmatic-advertising-via-input-reparameterization'
PDF='https://cdn.amazon.science/eb/30/86a891d6459a9550880f82df1836/scipub-approval152129-49286955-adaptive-spend-control-for-programmatic-advertising-via-input-reparameterization.pdf'

def now(): return datetime.now(timezone.utc).isoformat()

def design():
    specs,families=candidates()
    return dict(parent_commit='ad090af',nsga2_status='closed_without_adoption',
        development80=TRAIN,selection160=SELECT,fresh300=FRESH,probe=PROBE,final_test_used=False,
        candidate_specs=specs,families=families,primary_family=PRIMARY,
        hypotheses=['B slope is nearly constant and may supply no state-dependent information',
          'Observed spend/input ratios provide additional causal information for inverse Ki scaling',
          'Trailing-hour estimation responds faster than first-day cumulative estimation; EMA reduces estimation noise',
          'Compare against constant Ki scaling to distinguish adaptation from scalar retuning'],
        scope='Ki correction only. Frozen B predictor, frozen CMA E2 weights and Kp map, fixed target/cap1/antiwindup. Weighted integral retained, never rescaled.',
        correction='Ki_new=clip(Ki_old*clip((g_ref/max(g,0.1*g_ref))**power,0.5,2),0,6000)',
        g_ref={'derivative':B_SLOPE,'observed_ratio':RATIO_PRIOR,'ratio_prior_formula':'(1/288)/initial_multiplier'},
        methods={'derivative':'Existing B slope around selected base, +/-0.05 with one-sided boundary. Negative predicted spend remains clipped as in B.',
          'cumulative':'First-day paper-inspired prior blend: (1-t/288)*g_ref+(t/288)*sum(past normalized spend)/(1e-6+sum(past action)).',
          'window_raw':'Previous 12 completed ticks; sum normalized spend/(1e-6+sum action).',
          'window_smooth':'Same trailing12 ratio, EMA alpha=1-exp(-1/6). This intraday EMA is our extension, not paper Eq7 daily smoothing.'},
        guards='Exclude budget-censored feedback and action<=1e-4 from ratio sums. Window needs >=6 valid samples. Invalid/nonfinite/negative estimate falls back to unmodified Ki; zero positive-input spend is valid low gain and bounded by factor cap.',
        powers=[.25,.5,1.],constant_controls=[.5,.75,1.25,1.5,2.],
        tuning='24 total policies including 2 unmodified baselines; 80 development days describe sensitivity, 160 new selection days choose each family. No optimizer seeds because no stochastic optimizer or weight training.',
        choice='Within each family, min J among TE<=family baseline and UR<=baseline and V<baseline; if none retain baseline. Also freeze min-J diagnostic for every family, never auto-adopt it.',
        family_baseline='Fixed families use F0_fixed, state families use S0_CMA. Overall replacement requires S0_CMA comparison.',
        success='Primary new model must pass strict selection AND fresh mean TE/UR non-worsening and V reduction vs S0_CMA. Accuracy-only and J-improvement are separate secondary findings.',
        uncertainty='20000 paired daily bootstrap replicates, seed20261119, 95% descriptive intervals. No multiple-comparison correction; no noninferiority claim from CI containing zero.',
        fresh_policy='Freeze ALL selected/diagnostic weights and specifications before first fresh300 evaluation; no fresh result can change primary family or powers.',
        diagnostic='Actual auction finite-difference slope uses current tape only AFTER rollout, never passed to controller. Budget uncensored potential response at base +/-0.05; not a deployable observation.',
        limitations='Uncalibrated synthetic first-price simulator; one-day episodes reset estimator; paper 30-day/2min pure-I/seasonality/input transform architecture is NOT reproduced.')

def init():
    OUT.mkdir(parents=True,exist_ok=True)
    sources={p.name:rt.sha(p) for p in Path(__file__).parent.glob('*.py') if p.name!='report.py'}
    path=OUT/'protocol.json'
    if path.exists():
        data=rt.read_json(path)
        assert data['design']==design()
        assert data['sources']==sources, 'Frozen experiment source changed'
        return
    rt.write_json(path,dict(frozen_utc=now(),design=design(),sources=sources,simulator=rt.fingerprint(),
        reference_sources={str(p.relative_to(ROOT)):rt.sha(p) for p in (OLD/'code').glob('*.py')}))
    rt.write_json(OUT/'paper_review.json',dict(checked_date='2026-11-19',title='Adaptive Spend Control for Programmatic Advertising via Input Reparameterization',
       authors=['Niklas Karlsson','Tina Fan','Dan Kirlin','Abhirup Mukherjee'],official_page=PAPER,pdf=PDF,conference='IEEE CDC 2026',
       public_code_status='No author-provided code repository or code-release link found in the checked sources; not proof none exists.',
       checks=['Official article has PDF download, no paper-specific repository link',
          'Full seven-page PDF: no occurrence of code or github; no software availability section',
          'Amazon Science Code and datasets listing: no title match',
          'Web title/author/GitHub-targeted searches: no matching author implementation found'],
       equations={'4':'u=1/(alpha+lambda); private value=u*v, not necessarily final first-price bid',
          '5-6':'First-day prior plus cumulative spend/input ratio estimate; estimate is explicitly biased relative to a derivative',
          '7':'Day2 onwards daily smoothing, not every-five-minute EMA',
          '8-9':'Pure incremental-spend I controller u=SAT[u_prev+(1-exp(-Delta/Tcl))/gain*error]'},
       reproduction='Independent project-specific implementation of inverse gain adaptation; no author code used and no full-paper reproduction claim.'))
    print('Protocol frozen',flush=True)

def save_summary(name,rows):
    rt.write_csv(OUT/f'{name}_daily.csv',rows)
    scores=rt.summarize(rows)
    rt.write_csv(OUT/f'{name}_summary.csv',[{'model':k,**v} for k,v in scores.items()])
    return scores

def develop():
    init()
    if (OUT/'development_done.json').exists(): return
    start=time.perf_counter();specs,_=candidates()
    rows=evaluate(specs,TRAIN)
    save_summary('development',rows)
    diagnostics=[]
    for day in TRAIN:
        ep=rt.episode(day)
        _,tr=run(day,specs['S0_CMA'])
        for t in range(1,288):
            base=tr[t,6];lo=max(0.,base-.05);hi=min(1.,base+.05)
            tick=ep.tape[t]
            def potential(a):
                bid=tick.base*a
                return float(bid[(bid>tick.competitor)&(bid>0)].sum())/ep.budget
            oracle=(potential(hi)-potential(lo))/(hi-lo)
            diagnostics.append(dict(day=day,tick=t,base=base,B_slope=tr[t,7],actual_fd_slope=oracle,
                saturated=tr[t,13],budget_censored=tr[t,14]))
    rt.write_csv(OUT/'B_slope_diagnostic.csv',diagnostics)
    pred=np.array([r['B_slope'] for r in diagnostics]); actual=np.array([r['actual_fd_slope'] for r in diagnostics])
    ratio=pred/B_SLOPE
    rt.write_json(OUT/'slope_diagnosis.json',dict(analytic_slope=B_SLOPE,observations=len(pred),
      B_slope_quantiles=np.quantile(pred,[0,.05,.5,.95,1]).tolist(),
      fraction_within_0_1_percent_of_constant=float(np.mean(np.abs(ratio-1)<=.001)),
      actual_slope_quantiles=np.quantile(actual,[0,.05,.5,.95,1]).tolist(),
      B_cv=float(pred.std()/pred.mean()),actual_cv=float(actual.std()/actual.mean()),
      mean_absolute_error=float(np.mean(np.abs(pred-actual))),
      caveat='Counterfactual diagnostics only, at B selected base. Empirical finite differences include auction win thresholds; ratio estimators are intentionally biased and are not scored as unbiased derivatives.'))
    rt.write_json(OUT/'development_done.json',dict(finished_utc=now(),rollouts=len(rows),seconds=time.perf_counter()-start))
    print('Development complete',rt.read_json(OUT/'slope_diagnosis.json'),flush=True)

def choose(pool,scores,baseline):
    b=scores[baseline]
    feasible=[k for k in pool if scores[k]['tracking_error']<=b['tracking_error']+1e-15
              and scores[k]['unspent']<=b['unspent']+1e-15 and scores[k]['multiplier_tv']<b['multiplier_tv']-1e-12]
    diagnostic=min(pool,key=lambda k:(scores[k]['objective'],scores[k]['multiplier_tv'],k))
    primary=min(feasible,key=lambda k:(scores[k]['objective'],scores[k]['multiplier_tv'],k)) if feasible else baseline
    return dict(primary=primary,diagnostic=diagnostic,eligible=len(feasible),baseline=baseline,
                status='selected' if feasible else 'retain_incumbent')

def select():
    init()
    if (OUT/'selection.json').exists(): return
    specs,families=candidates();start=time.perf_counter()
    scores=save_summary('selection',evaluate(specs,SELECT))
    picks={}
    for family,pool in families.items():
        baseline='F0_fixed' if family.startswith('fixed') or family=='F0_fixed' else 'S0_CMA'
        picks[family]=choose(pool,scores,baseline)
    keys={'S0_CMA','F0_fixed'}|{p[k] for p in picks.values() for k in ('primary','diagnostic')}
    rt.write_json(OUT/'selection.json',dict(frozen_utc=now(),families=picks,primary_family=PRIMARY,
         primary=picks[PRIMARY],selected_specs={k:specs[k] for k in sorted(keys)},
         all_specs=specs,scores=scores,seconds=time.perf_counter()-start,rollouts=len(specs)*len(SELECT)))
    print('Selection frozen',picks[PRIMARY],'; models:',len(keys),flush=True)

def reference_task(item):
    day,specs=item
    ep=rt.episode(day);rows=[]
    for key,sp in specs.items():
        policy=rt.policy(sp) if key=='S0_CMA' else SensitivityPI(sp)
        score,_=rt.core.rollout(ep,rt.cfg,policy)
        rows.append({'model':key,**{k:score[k] for k in ['day','tape_sha256']+METRICS}})
    return rows

def paired(rows,pairs):
    days=sorted({r['day'] for r in rows})
    index={(r['model'],r['day']):r for r in rows}
    rng=np.random.default_rng(20261119)
    sample=rng.integers(0,len(days),(20000,len(days)))
    result=[]
    for key,baseline in pairs:
        if key==baseline:continue
        for metric in METRICS:
            diff=np.array([index[key,d][metric]-index[baseline,d][metric] for d in days])
            ci=np.quantile(diff[sample].mean(axis=1),[.025,.975])
            result.append(dict(new=key,baseline=baseline,metric=metric,mean_difference=float(diff.mean()),
                ci95_low=float(ci[0]),ci95_high=float(ci[1]),days_better=int((diff<-1e-12).sum()),
                days_worse=int((diff>1e-12).sum()),days_equal=int((np.abs(diff)<=1e-12).sum())))
    return result

def confirm():
    init();sel=rt.read_json(OUT/'selection.json')
    if (OUT/'evaluation_done.json').exists():return
    started=time.perf_counter();specs=sel['selected_specs']
    rt.write_json(OUT/'evaluation_started.json',dict(started_utc=now(),selection_sha256=rt.sha(OUT/'selection.json')))
    rows=evaluate(specs,FRESH)
    rt.write_csv(OUT/'fresh_daily_fast.csv',rows)
    print('Fast confirmation complete; independent Python replay',len(rows),'rollouts',flush=True)
    reference=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i,batch in enumerate(pool.map(reference_task,[(d,specs) for d in FRESH])):
            reference.extend(batch)
            if (i+1)%50==0: print('Reference days',i+1,'/300',flush=True)
    reference.sort(key=lambda r:(r['model'],r['day']))
    fast={(r['model'],r['day']):r for r in rows}
    checks=[dict(model=r['model'],day=r['day'],max_metric_difference=max(abs(r[m]-fast[r['model'],r['day']][m]) for m in METRICS)) for r in reference]
    rt.write_csv(OUT/'evaluator_audit.csv',checks)
    assert max(r['max_metric_difference'] for r in checks)<1e-8
    scores=save_summary('fresh',reference)
    pairs={(k,b) for k in specs for b in ['S0_CMA','F0_fixed'] if k!=b}
    smooth=sel['families']['state_window_smooth']['diagnostic']
    for family in ['state_constant','state_cumulative','state_window_raw']:
        pairs.add((smooth,sel['families'][family]['diagnostic']))
    rt.write_csv(OUT/'fresh_paired.csv',paired(reference,sorted(pairs)))
    comparisons=[]
    for family,pick in sel['families'].items():
        for role in ['primary','diagnostic']:
            key=pick[role];s=scores[key]; b=scores['S0_CMA']
            comparisons.append(dict(family=family,role=role,model=key,selection_status=pick['status'],eligible=pick['eligible'],
              **{m+'_relative_pct':(s[m]/b[m]-1)*100 for m in METRICS},
              fresh_strict_mean_success=key!='S0_CMA' and s['tracking_error']<=b['tracking_error'] and s['unspent']<=b['unspent'] and s['multiplier_tv']<b['multiplier_tv']))
    rt.write_csv(OUT/'method_comparison.csv',comparisons)
    picked=sel['primary']['primary'];b=scores['S0_CMA'];s=scores[picked]
    strict=picked!='S0_CMA' and all(s[m]<=b[m] for m in METRICS[:2]) and s['multiplier_tv']<b['multiplier_tv']
    rt.write_json(OUT/'primary_model.json',dict(family=PRIMARY,selection=sel['primary'],spec=specs[picked],
       fresh_strict_mean_success=strict,adoption_decision='candidate_for_final_test' if strict else 'do_not_adopt',
       diagnostic_model=sel['primary']['diagnostic'],final_test_used=False))
    rt.write_json(OUT/'evaluation_done.json',dict(finished_utc=now(),models=len(specs),rollouts=len(rows),
       reference_rollouts=len(reference),seconds=time.perf_counter()-started,
       max_fast_reference_difference=max(r['max_metric_difference'] for r in checks),
       selection_sha256=rt.sha(OUT/'selection.json'),final_test_used=False))
    print('Confirmation complete',rt.read_json(OUT/'primary_model.json')['adoption_decision'],flush=True)

def audit():
    init();p=rt.read_json(OUT/'protocol.json');s=rt.read_json(OUT/'selection.json')
    e=rt.read_json(OUT/'evaluation_started.json');end=rt.read_json(OUT/'evaluation_done.json')
    assert p['frozen_utc']<s['frozen_utc']<e['started_utc']<end['finished_utc']
    assert rt.sha(OUT/'selection.json')==e['selection_sha256']==end['selection_sha256']
    assert all(rt.sha(ROOT/k)==v for k,v in p['reference_sources'].items())
    assert rt.fingerprint()==p['simulator']
    rows=list(csv.DictReader((OUT/'fresh_daily.csv').open(encoding='utf-8')))
    tapes={}
    for r in rows:tapes.setdefault(r['day'],set()).add(r['tape_sha256'])
    assert len(tapes)==300 and all(len(v)==1 for v in tapes.values())
    assert set(int(r['day']) for r in rows)==set(FRESH)
    rt.write_json(OUT/'audit.json',dict(status='passed',audited_utc=now(),selection_before_confirmation=True,
       frozen_sources_unchanged=True,shared_day_tapes=True,final_test_used=False,
       candidate_policies=len(candidates()[0]),development_rollouts=len(candidates()[0])*len(TRAIN),
       selection_rollouts=len(candidates()[0])*len(SELECT),reference_rollouts=end['reference_rollouts'],
       max_fast_reference_difference=end['max_fast_reference_difference']))
    print('Audit passed',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['init','develop','select','confirm','audit','all'])
    args=parser.parse_args()
    for fn in ([init,develop,select,confirm,audit] if args.stage=='all' else [globals()[args.stage]]):fn()
