"""Preflighted, resumable paired gain diagnostics and shallow-tree control.
Existing data only; deterministic expected-value replay; limited development validation.
"""
import os, sys, json, math, time, hashlib, traceback, subprocess, shutil, fcntl
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeRegressor
ROOT=Path(__file__).resolve().parent
LAB=ROOT/'auctionnet_lab'
sys.path.insert(0,str(LAB))
import gain_feasibility_v4 as g
CODE=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
OUT=LAB/'runs'/('gain_final_v9_'+CODE[:8])
ADS=[1,13,25,37]
TRAIN=[7,8,9,10]
TUNE=[11,12]
CHECK=[13]
ANCHORS=list(range(6,43,6))
F=['time_left','remaining_ratio','error','integral','log_alpha','last_spend_ratio','recent_spend_ratio',
   'log_volume','volume_change','volume_cv','value_median','value_spread','price_value_median',
   'price_value_q90','price_value_change','price_shift','value_shift','volume_shift',
   'reference_capacity','spend_sensitivity']


def jsave(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    tmp.replace(path)


def csvsave(path,frame):
    tmp=path.with_suffix(path.suffix+'.tmp');frame.to_csv(tmp,index=False);tmp.replace(path)


def state(status,stage,**extra):
    jsave(OUT/'STATUS.json',dict(status=status,stage=stage,time=time.strftime('%Y-%m-%d %H:%M:%S'),**extra))
    print(status,stage,extra,flush=True)


def cached(path,fn):
    if path.exists():
        try:return joblib.load(path)
        except Exception:
            path.rename(path.with_name(path.name+'.corrupt_'+str(time.time_ns())))
    error=None
    for attempt in range(2):
        try:
            result=fn();tmp=path.with_suffix(path.suffix+'.tmp');joblib.dump(result,tmp,compress=3);tmp.replace(path)
            return result
        except Exception as exc:
            error=exc
            with (OUT/'retry_errors.log').open('a') as log:log.write(traceback.format_exc()+'\n')
            if attempt==0:time.sleep(2)
    raise error


def context():
    c=json.loads((LAB/'runs/global_retune_v3/selected_controller.json').read_text())
    if c['antiwindup']!='clamp' or not c['calibrated']:raise ValueError('Expected calibrated clamp controller')
    if sorted(map(int,c['initial_bias']))!=ADS:raise ValueError('Unexpected advertiser set')
    p=c['parameters'];bias={int(k):float(np.clip(v+p['offset'],-6,4)) for k,v in c['initial_bias'].items()}
    pool=sorted(set((float(kp),float(ki)) for kp in [4.,16.,64.,p['kp']] for ki in [0.,.3,3.,p['ki']]))
    original=pool.index((p['kp'],p['ki']))
    return c,p,bias,pool,original


def file_inventory():
    files={}
    for period in TRAIN+TUNE+CHECK:
        for ad in ADS:
            paths=(list((LAB/'data').glob(f'pack-*/p{period}-a{ad}.parquet')) if period!=13
                   else [ROOT/f'auctionnet_paper_v7/data/eval_p13_a{ad}.parquet'])
            if len(paths)!=1 or not paths[0].is_file():raise FileNotFoundError(f'Expected one complete file for period={period}, ad={ad}: {paths}')
            files[(period,ad)]=paths[0]
    return files


def check_trace(f,B):
    if not np.isfinite(f[['alpha','spend','remaining','error','value','du']].to_numpy()).all():raise ValueError('Nonfinite replay')
    if (f.remaining< -1e-7).any() or f.spend.sum()>B+1e-6:raise ValueError('Budget invariant failed')
    if (f.spend< -1e-8).any() or f.du.max()>.50000001:raise ValueError('Output/spend invariant failed')
    if not f.alpha.between(math.exp(-6)-1e-10,math.exp(4)+1e-10).all():raise ValueError('Output bound failed')


def metric(f,B):
    check_trace(f,B)
    te=float(f.error.abs().mean());tv=float(np.abs(np.diff(np.log(f.alpha.to_numpy()))).mean())
    return dict(te=te,value=float(f.value.sum()),unspent=float(f.remaining.iloc[-1]/B),tv=tv,j=te+.01*tv,
                early_exhausted=int((f.remaining.iloc[:43]<.1).any()))


def feature(ep,t,s,history):
    B=ep['budget'];old=ep['ticks'][t-1]
    x=np.array([(48-t)/48,s[0]/B,t/48-(B-s[0])/B,s[1],s[2],float(history.spend.iloc[-1])/B,
                float(history.spend.iloc[-3:].mean())/B,*old['features'],
                old['ref_spend']/max(s[0]/(48-t),1e-8),old['sensitivity']],dtype=float)
    if x.shape!=(len(F),) or not np.isfinite(x).all():raise ValueError('Invalid causal feature')
    return x


def choose(model,x,threshold,base,K):
    indices=getattr(model,'feature_indices',list(range(x.shape[1])))
    prediction=np.asarray(model.predict(x[:,indices])).reshape(len(x),2*K)
    dj=prediction[:,:K]/20.;dv=prediction[:,K:]
    valid=(dj < -threshold)&(dv>=-.01)
    scores=np.where(valid,dj,np.inf);scores[:,base]=0
    return scores.argmin(axis=1)


def simulate(ep,pool,base,bias,model=None,threshold=.001):
    pieces=[];s=None;start=0
    while start<48:
        end=6 if start==0 else min(start+6,48);choice=base
        if model is not None and start>=6 and s[0]>=.1:
            choice=int(choose(model,feature(ep,start,s,pd.concat(pieces,ignore_index=True))[None,:],threshold,base,len(pool))[0])
        kp,ki=pool[choice];f,states=g.roll(ep,dict(kp=kp,ki=ki),bias,start,end,s)
        f['candidate']=choice;f['decision_tick']=start;f['kp']=kp;f['ki']=ki
        f['leaf']=(-1 if model is None or start==0 or s[0]<.1 else int(model.apply(feature(ep,start,s,pd.concat(pieces,ignore_index=True))[None,getattr(model,'feature_indices',list(range(len(F))))])[0]))
        pieces.append(f);s=states[end];start=end
    f=pd.concat(pieces,ignore_index=True);check_trace(f,ep['budget']);return f


def preflight():
    OUT.mkdir(parents=True,exist_ok=True)
    c,p,bias,pool,original=context();files=file_inventory()
    manifest=dict(code=CODE,controller=c,files={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files.values()},
        dependencies={name:hashlib.sha256((LAB/name).read_bytes()).hexdigest() for name in ['run.py','gain_feasibility_v4.py','controller_diagnostic_v2.py']})
    existing=OUT/'manifest.json'
    if existing.exists() and json.loads(existing.read_text())!=manifest:raise ValueError('Inputs changed; preserve this run and use a new output directory/code version')
    jsave(existing,manifest)
    cache=OUT/'cache';cache.mkdir(exist_ok=True)
    state('PREFLIGHT','modules_and_actual_data')
    lab=g.d.load_lab();g.d.self_test(lab)
    for (period,ad),path in files.items():
        def load(path=path,period=period,ad=ad):
            df=pd.read_parquet(path)
            if 'advertiserNumber' in df.columns:df=df.rename(columns={'advertiserNumber':'advertiser'})
            required=['deliveryPeriodIndex','advertiser','budget','CPAConstraint','timeStepIndex','pvIndex','pValue','pValueSigma','leastWinningCost']
            if set(required)-set(df.columns):raise ValueError(f'Missing required columns: {path}')
            df=df[required]
            if df.deliveryPeriodIndex.nunique()!=1 or df.advertiser.nunique()!=1:raise ValueError('Mixed episode identities')
            if not np.isfinite(df[required].to_numpy(dtype=float)).all():raise ValueError(f'Nonfinite required replay fields: {path}')
            if sorted(df.timeStepIndex.unique())!=list(range(48)):raise ValueError('Expected ticks 0..47')
            ep=lab.prepare_episode(df,42)
            if (ep['period'],ep['ad'])!=(period,ad):raise ValueError('Episode identity differs from filename')
            return ep
        ep=cached(cache/f'episode_{period}_{ad}.joblib',load)
        if period in TRAIN:
            f,states=g.roll(ep,p,bias[ad]);ref,_=g.d.replay(ep,p,bias[ad],'clamp')
            if not np.isclose(metric(f,ep['budget'])['te'],ref['te']) or not np.isclose(f.value.sum(),ref['expected_conversions']):raise ValueError('Real-data replay parity failed')
            for t in ANCHORS:
                part,_=g.roll(ep,p,bias[ad],t,t+6,states[t])
                if not np.allclose(part[['alpha','spend','remaining','error']],f.iloc[t:t+6][['alpha','spend','remaining','error']]):raise ValueError('State cloning parity failed')
                feature(ep,t,states[t],f.iloc[:t])
            segmented=simulate(ep,pool,original,bias[ad])
            if not np.allclose(segmented[['alpha','spend','remaining','error']],f[['alpha','spend','remaining','error']]):
                raise ValueError('Segmented continuous replay differs from fixed reference')
        print('CHECKED',period,ad,flush=True)
    tiny=DecisionTreeRegressor(max_depth=2,random_state=42).fit(np.zeros((20,len(F))),np.zeros((20,2*len(pool))))
    choose(tiny,np.zeros((1,len(F))),.001,original,len(pool))
    jsave(OUT/'protocol.json',dict(train_periods=TRAIN,tuning_periods=TUNE,report_periods=CHECK,advertisers=ADS,
        baseline='Best validation-selected fixed gain from the same 16 candidates; original initialization shared by all methods',
        prior_information='Original initialization/controller was developed using periods 7..10 (now training). Period 13 has already been inspected in other analyses; limited development holdout, not a pristine final test.',
        stage_A='Matched state cloning; six-tick lookahead; future oracle diagnostic versus a train-selected common pair',
        stage_B='Full and time-only depth 2/3 multioutput trees (four fits total), fixed minimum leaf; predict tracking-cost and value changes; thresholds .001/.003; select on whole-day periods 11/12 only',
        selection='Minimize TE+0.01*log-output-TV subject to >=98% baseline mean expected value. Activate only if validation TE also improves at least 2%. Raw tree results always retained.',
        observation='Past ticks only; platform-level access to losing prices assumed. Budget, integral and previous output carried across switches.',
        randomness='Deterministic expected clipped conversion value and common per-tick budget ordering; no repeated stochastic conversion draws needed',
        limitations='Fixed-price replay; 4 advertisers, one reporting period; windows overlap and are NOT independent samples; no significance or population-generalization claim',
        features=F,gains=pool,expected_data_files=len(files)))
    state('PREFLIGHT_PASSED','ready',output=str(OUT))


@lru_cache(maxsize=2)
def episode(period,ad):
    def restore():
        path=file_inventory()[(period,ad)]
        df=pd.read_parquet(path)
        if 'advertiserNumber' in df.columns:df=df.rename(columns={'advertiserNumber':'advertiser'})
        return g.d.load_lab().prepare_episode(df,42)
    return cached(OUT/'cache'/f'episode_{period}_{ad}.joblib',restore)


def fork_rows(ep,pool,base,bias):
    rows=[];bkp,bki=pool[base]
    seen_prefixes=set()
    for prefix,gains in [('fixed',(bkp,bki)),('P',(bkp,0.)),('mild',(.5*bkp,bki))]:
        if gains in seen_prefixes:continue
        seen_prefixes.add(gains)
        f,states=g.roll(ep,dict(kp=gains[0],ki=gains[1]),bias)
        for t in ANCHORS:
            if states[t][0]<.1:continue
            j=[];value=[]
            for kp,ki in pool:
                part,_=g.roll(ep,dict(kp=kp,ki=ki),bias,t,t+6,states[t]);check_trace(part,states[t][0])
                j.append(float(part.error.abs().mean()+.01*part.du.mean()));value.append(float(part.value.sum()))
            rows.append(dict(period=ep['period'],ad=ep['ad'],prefix=prefix,start=t,x=feature(ep,t,states[t],f.iloc[:t]).tolist(),j=j,value=value))
    return rows


def pmean(values,periods):return float(pd.DataFrame(dict(period=periods,value=values)).groupby('period').value.mean().mean())


def fit_tree(rows,depth,base,K,family):
    x=np.array([r['x'] for r in rows]);j=np.array([r['j'] for r in rows]);v=np.array([r['value'] for r in rows]);periods=np.array([r['period'] for r in rows])
    if len(rows)<20:raise ValueError('Too few usable states for a meaningful tree fit')
    # Equal weighting per period and per advertiser/prefix case, not per raw row.
    group=[(r['period'],r['ad'],r['prefix']) for r in rows];counts={k:group.count(k) for k in set(group)}
    w=np.array([1/counts[k] for k in group]);w/=w.mean()
    target=np.concatenate([20*(j-j[:,[base]]),np.clip((v-v[:,[base]])/np.maximum(v[:,[base]],1e-6),-2,2)],axis=1)
    if not np.isfinite(target).all():raise ValueError('Nonfinite learning target')
    indices=[0] if family=='time_only' else list(range(len(F)))
    model=DecisionTreeRegressor(max_depth=depth,min_samples_leaf=max(12,int(.05*len(rows))),random_state=42).fit(x[:,indices],target,sample_weight=w)
    model.feature_indices=indices
    return model


def rule_text(model,pool,base):
    tree=model.tree_;names=[F[i] for i in model.feature_indices];lines=[]
    def visit(node,path):
        if tree.children_left[node]==tree.children_right[node]:
            pred=tree.value[node].reshape(-1);K=len(pool);dj=pred[:K]/20;dv=pred[K:]
            candidates=[i for i in range(K) if dj[i]<-.001 and dv[i]>=-.01]
            winner=min(candidates,key=lambda i:dj[i]) if candidates else base
            lines.append(' AND '.join(path)+f' -> leaf={node}, kp/ki={pool[winner]}, predicted_delta_j={dj[winner]:.6f}, predicted_relative_value={dv[winner]:.6f}')
        else:
            name=names[tree.feature[node]];threshold=tree.threshold[node]
            visit(tree.children_left[node],path+[f'{name} <= {threshold:.6g}'])
            visit(tree.children_right[node],path+[f'{name} > {threshold:.6g}'])
    visit(0,[])
    return '\n'.join(lines)+'\n'



def evaluate_model(periods,pool,base,bias,model=None,threshold=.001):
    rows=[]
    for period in periods:
        for ad in ADS:
            ep=episode(period,ad);f=simulate(ep,pool,base,bias[ad],model,threshold)
            rows.append(dict(period=period,advertiser=ad,**metric(f,ep['budget'])))
    return pd.DataFrame(rows)


def aggregate(frame):return frame.groupby('period')[['te','value','unspent','tv','j','early_exhausted']].mean().mean()


def oracle_summary(rows,base,common):
    frame=[]
    for r in rows:
        j=np.array(r['j']);v=np.array(r['value']);eligible=v+1e-12>=.98*v[base]
        values=np.where(eligible,j,np.inf);best=int(values.argmin())
        if values[base]<=values[best]+1e-10:best=base
        frame.append(dict(period=r['period'],advertiser=r['ad'],prefix=r['prefix'],start=r['start'],
            baseline_j=j[base],common_j=j[common],oracle_j=j[best],oracle_pair=best,
            common_value_pass=int(eligible[common]),meaningful_gain=int(j[best]<=j[base]-max(.001,.02*j[base]))))
    return pd.DataFrame(frame)


def work():
    c,p,bias,pool,original=context();cache=OUT/'cache'
    state('RUNNING','fixed_baseline_selection')
    fixed=[]
    for period in TUNE:
        for ad in ADS:
            def compute(period=period,ad=ad):
                ep=episode(period,ad);records=[]
                for candidate,(kp,ki) in enumerate(pool):
                    f,_=g.roll(ep,dict(kp=kp,ki=ki),bias[ad]);records.append(dict(period=period,advertiser=ad,candidate=candidate,**metric(f,ep['budget'])))
                return records
            fixed.extend(cached(cache/f'fixed_{period}_{ad}.joblib',compute))
    fixed=pd.DataFrame(fixed);csvsave(OUT/'fixed_tuning_episodes.csv',fixed)
    stats=fixed.groupby(['candidate','period'])[['te','value','j']].mean().groupby('candidate').mean()
    passing=stats[stats.value+1e-12>=.98*stats.loc[original,'value']]
    base=int(passing.sort_values(['j','te'],kind='stable').index[0])
    fixed_ref=stats.loc[base];jsave(OUT/'baseline.json',dict(candidate=base,kp=pool[base][0],ki=pool[base][1],bias=bias))
    state('RUNNING','experiment_2_matched_forks',baseline=base)
    train=[]
    for period in TRAIN:
        for ad in ADS:
            part=cached(cache/f'fork_{period}_{ad}_base{base}.joblib',lambda period=period,ad=ad:fork_rows(episode(period,ad),pool,base,bias[ad]))
            train.extend(part);print('FORKS',period,ad,len(part),flush=True)
    if not train:raise ValueError('No usable fork states')
    jsave(OUT/'training_states.json',train)
    j=np.array([r['j'] for r in train]);v=np.array([r['value'] for r in train]);periods=np.array([r['period'] for r in train])
    # Shared local comparator selected from training only, under a fixed penalty.
    losses=j+np.maximum(0,.98*v[:,[base]]-v)/np.maximum(v[:,[base]],1.)
    common=int(np.argmin([pmean(losses[:,k],periods) for k in range(len(pool))]))
    csvsave(OUT/'experiment2_train_windows.csv',oracle_summary(train,base,common))
    state('RUNNING','experiment_3_tree_tuning')
    settings=[];models={};tuning=[]
    for family in ['full','time_only']:
        for depth in [2,3]:
            model=fit_tree(train,depth,base,len(pool),family);models[(family,depth)]=model
            joblib.dump(model,OUT/f'tree_{family}_depth{depth}.joblib')
            (OUT/f'tree_{family}_depth{depth}.txt').write_text(rule_text(model,pool,base))
            for threshold in [.001,.003]:
                evaluation=evaluate_model(TUNE,pool,base,bias,model,threshold);evaluation['family']=family;evaluation['depth']=depth;evaluation['threshold']=threshold;tuning.append(evaluation)
                a=aggregate(evaluation);ratio=float(a['value']/max(fixed_ref['value'],1e-12))
                settings.append(dict(family=family,depth=depth,threshold=threshold,te=float(a['te']),value=float(a['value']),j=float(a['j']),value_ratio=ratio))
                print('TREE TUNED',settings[-1],flush=True)
    candidates=pd.DataFrame(settings);csvsave(OUT/'tree_tuning_settings.csv',candidates);csvsave(OUT/'tree_tuning_episodes.csv',pd.concat(tuning,ignore_index=True))
    # Predeclared ordering; no check-period outcomes participate in selection.
    select_key=lambda z:(max(0,.98-z['value_ratio']),z['j'],z['depth'],-z['threshold'])
    choice=min([z for z in settings if z['family']=='full'],key=select_key)
    time_choice=min([z for z in settings if z['family']=='time_only'],key=select_key)
    enabled=bool(choice['value_ratio']+1e-12>=.98 and choice['te']<=.98*fixed_ref['te'] and choice['j']<fixed_ref['j'])
    choice.update(enabled=enabled,baseline=base,common_local=common,time_only_selection=time_choice)
    jsave(OUT/'FROZEN_SELECTION.json',choice)
    state('RUNNING','frozen_period13_evaluation',selection=choice)
    # Report all main comparisons, including the raw tree even if validation rejected it.
    model=models[('full',choice['depth'])];time_model=models[('time_only',time_choice['depth'])];threshold=choice['threshold'];results=[];trace_rows=[]
    for period in CHECK:
        for ad in ADS:
            ep=episode(period,ad)
            variants={}
            variants['original_PI']=simulate(ep,pool,original,bias[ad])
            variants['tuned_fixed']=simulate(ep,pool,base,bias[ad])
            variants['tree_raw']=simulate(ep,pool,base,bias[ad],model,threshold)
            variants['time_only']=simulate(ep,pool,base,bias[ad],time_model,time_choice['threshold'])
            variants['selected_policy']=variants['tree_raw'] if enabled else variants['tuned_fixed']
            matched,_=g.roll(ep,dict(kp=pool[base][0],ki=0.),bias[ad]);matched['candidate']=-1;variants['matched_P_ablation']=matched
            for name,f in variants.items():
                row=dict(period=period,advertiser=ad,method=name,**metric(f,ep['budget']));results.append(row)
                ff=f.copy();ff['method']=name;ff['advertiser']=ad;ff['period']=period;trace_rows.append(ff)
            csvsave(OUT/'experiment3_episodes.csv',pd.DataFrame(results))
            print('REPORT EPISODE DONE',period,ad,flush=True)
    e=pd.DataFrame(results);summary=e.groupby(['method','period'])[['te','value','unspent','tv','j','early_exhausted']].mean().groupby('method').mean().reset_index()
    csvsave(OUT/'experiment3_summary.csv',summary);csvsave(OUT/'experiment3_traces.csv',pd.concat(trace_rows,ignore_index=True))
    report_forks=[]
    for period in CHECK:
        for ad in ADS:
            report_forks.extend(cached(cache/f'fork_{period}_{ad}_base{base}.joblib',lambda period=period,ad=ad:fork_rows(episode(period,ad),pool,base,bias[ad])))
    if report_forks:
        windows=oracle_summary(report_forks,base,common);csvsave(OUT/'experiment2_report_windows.csv',windows)
        a=windows.groupby('period')[['baseline_j','common_j','oracle_j','common_value_pass','meaningful_gain']].mean().mean()
        gain=100*(a.baseline_j-a.oracle_j)/max(a.baseline_j,1e-12)
        jm=np.array([r['j'] for r in report_forks]);vm=np.array([r['value'] for r in report_forks]);ps=np.array([r['period'] for r in report_forks])
        allowed=(vm+1e-12>=.98*vm[:,[base]]).all(axis=0)
        future_common=min(np.flatnonzero(allowed),key=lambda k:pmean(jm[:,k],ps))
        future_common_j=pmean(jm[:,future_common],ps)
        extra_gain=100*(future_common_j-a.oracle_j)/max(future_common_j,1e-12)
        report_summary=pd.DataFrame([dict(a,oracle_improvement_pct=gain,windows=len(windows),
            future_best_common_pair=int(future_common),future_best_common_j=future_common_j,
            oracle_vs_future_common_gain_pct=extra_gain)])
    else:
        gain=None;extra_gain=None
        report_summary=pd.DataFrame([dict(oracle_improvement_pct=None,windows=0,note='All prefixes exhausted before eligible anchors; no oracle claim possible')])
    csvsave(OUT/'experiment2_summary.csv',report_summary)
    baseline_cases=e[e.method=='tuned_fixed'][['period','advertiser','te','value']].rename(columns={'te':'baseline_te','value':'baseline_value'})
    paired=e[e.method=='tree_raw'].merge(baseline_cases,on=['period','advertiser'],validate='one_to_one')
    paired['te_gain_pct']=100*(paired.baseline_te-paired.te)/paired.baseline_te.clip(lower=1e-12)
    paired['value_ratio']=paired.value/paired.baseline_value.clip(lower=1e-12)
    csvsave(OUT/'paired_advertisers.csv',paired)
    indexed=summary.set_index('method');b=indexed.loc['tuned_fixed'];t=indexed.loc['tree_raw']
    te_gain=100*(b.te-t.te)/max(b.te,1e-12);value_ratio=t['value']/max(b['value'],1e-12)
    conclusion='' 
    # Korean final report: descriptive conclusions, no window-level significance claim.
    if gain is None:conclusion='보고 기간의 모든 접두 궤적에서 비교 시점 이전에 예산이 소진되어, 구간별 oracle 이득을 평가할 수 없다.'
    elif gain<2:conclusion='이 후보 집합·3시간 구간에서는 상태별 선택의 추가 개선 여지가 작다.'
    elif te_gain>0 and value_ratio>=.98:conclusion='미래 정보 진단의 개선 여지가 있고, 선택된 트리도 보고 기간에서 TE 개선과 전환 보존을 함께 보였다. 일반화 검증은 별도로 필요하다.'
    else:conclusion='미래 정보 진단에서는 개선 여지가 있지만, 선택된 트리가 보고 기간에서 TE 개선과 전환 보존을 함께 달성하지 못했다.'
    extra_text='평가 불가' if extra_gain is None else f'{extra_gain:.2f}%'
    gain_text='평가 가능한 구간 없음' if gain is None else f'{gain:.2f}%'
    time_ref=indexed.loc['time_only']
    time_gain=100*(time_ref.te-t.te)/max(time_ref.te,1e-12)
    text=f'''# 두 실험 결과\n\n정상 완료. 학습 7~10, 설정 선택 11~12, 고정 후 보고 13. 광고주 {ADS}.\n\n실험 2: 미래를 아는 oracle의 구간 점수 개선 {gain_text}. 이는 실제 정책 성능이 아니다.\n미래를 아는 최선의 공통 계수보다 상태별 oracle이 추가로 개선한 비율: {extra_text}. 양쪽 모두 동일한 구간별 전환 하한을 적용했다.\n실험 3: 고정 기준 대비 원시 트리의 하루 TE 개선 {te_gain:.2f}%, 기대 전환 비율 {value_ratio:.4f}.\n시간 정보만 쓰는 트리 대비 전체 피처 트리의 TE 개선: {time_gain:.2f}%. 이것이 양수가 아니라면 환경 피처의 추가 효과를 주장하지 않는다.\n검증 기준에 따른 트리 활성화: {enabled}. 비활성화이면 selected_policy는 고정 기준과 동일하다.\n\n{conclusion}\n\n광고주별 원시 트리 대 고정 기준:\n{paired[['period','advertiser','te_gain_pct','value_ratio']].to_string(index=False)}\n\n{summary.to_string(index=False)}\n\n범위: period 13은 다른 분석에서 이미 확인한 개발 보고 기간이다. 광고주 4명·한 기간의 사례 결과이며 유의성/전체 시장 일반화를 주장하지 않는다. 겹친 창을 독립 표본으로 취급하지 않는다.\n고정 가격 재생과 기대 전환 기준이며 이전 BC/IQL의 확률적 전환 점수와 직접 비교하지 않는다.\n초기 승수와 출력/변화 제한, 잔여예산/적분/직전출력은 모든 방법에서 동일하게 처리했다.\n트리 설정은 FROZEN_SELECTION.json, 규칙은 tree_full/time_only_depth*.txt, 원시 결과는 experiment2/3 CSV 참조.\n'''
    (OUT/'REPORT.md').write_text(text)
    jsave(OUT/'DONE.json',dict(status='COMPLETE',baseline=base,tree_enabled=enabled,oracle_improvement_pct=gain,oracle_vs_future_common_gain_pct=extra_gain,tree_te_improvement_pct=te_gain,tree_value_ratio=value_ratio))
    state('COMPLETE','all_done',report=str(OUT/'REPORT.md'))


def lock():
    OUT.mkdir(parents=True,exist_ok=True);handle=(OUT/'run.lock').open('a')
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close();return None
    return handle


def main():
    if '--status' in sys.argv:
        for name in ['STATUS.json','REPORT.md']:
            path=OUT/name
            if path.exists():print(path.read_text(),flush=True)
        return
    handle=lock()
    if handle is None:print('ALREADY RUNNING:',OUT,flush=True);return
    if '--worker' in sys.argv:
        for attempt in range(2):
            try:
                work();return
            except Exception:
                error=traceback.format_exc()
                with (OUT/'worker_errors.log').open('a') as log:log.write(error+'\n')
                print(error,flush=True)
                state('RETRYING' if attempt==0 else 'INCOMPLETE','worker_exception',attempt=attempt+1,error=error[-2500:])
                if attempt==0:time.sleep(3)
        partial='\n'.join(p.name for p in OUT.glob('*.csv'))
        (OUT/'REPORT.md').write_text('# 실행 미완료\n\n오류로 완료하지 못했습니다. 아래 완료된 단계의 CSV와 오류 기록을 확인하세요. 누락 사례를 조용히 제외하거나 성공으로 표시하지 않았습니다.\n\n'+partial+'\n\n'+error)
        raise SystemExit(1)
    if '--status' in sys.argv:
        for name in ['STATUS.json','REPORT.md']:
            path=OUT/name
            if path.exists():print(path.read_text(),flush=True)
        return
    if (OUT/'DONE.json').exists():print('ALREADY COMPLETE:',OUT/'REPORT.md',flush=True);return
    preflight()
    logfile=(OUT/'overnight.log').open('a')
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker']
    if shutil.which('caffeinate'):command=['caffeinate','-i']+command
    # Child acquires lock after the launching process exits this critical section.
    fcntl.flock(handle,fcntl.LOCK_UN);handle.close()
    worker=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
    logfile.close();jsave(OUT/'LAUNCHED.json',dict(pid=worker.pid))
    print('PREFLIGHT PASSED; STARTED PID:',worker.pid,'\nOUTPUT:',OUT,'\nLOG:',OUT/'overnight.log',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        error=traceback.format_exc();(OUT/'PREFLIGHT_FAILED.txt').write_text(error)
        print(error,flush=True);raise
