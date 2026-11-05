#!/usr/bin/env bash
set -euo pipefail
git rev-parse --show-toplevel >/dev/null
if [ -e auctionnet_lab ]; then
  echo 'auctionnet_lab already exists. Stop to preserve existing work.' >&2
  exit 1
fi
python_bin=''
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; assert (3,10)<=sys.version_info[:2]<(3,13)' 2>/dev/null; then
    python_bin="$candidate"
    break
  fi
done
if [ -z "$python_bin" ]; then
  echo 'Python 3.10-3.12 is required. Install Python 3.12 and run again.' >&2
  exit 1
fi
mkdir auctionnet_lab
cat > auctionnet_lab/.gitignore <<'IGNORE'
.venv/
data/
runs/
vendor/
__pycache__/
IGNORE
cat > auctionnet_lab/requirements.txt <<'REQ'
numpy==1.26.4
pandas==2.2.3
scipy==1.14.1
scikit-learn==1.5.2
hdbscan==0.8.40
optuna==4.2.1
pyarrow==19.0.1
REQ
cat > auctionnet_lab/run.py <<'PYTHON'
"""AuctionNet fixed-price offline replay: regime discovery and PI tuning.
No synthetic data are used by fetch/run. See PROTOCOL.txt for scope.
"""
import argparse, hashlib, importlib.util, json, math, os, shutil, time, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import RobustScaler
import hdbscan, optuna, joblib

ROOT = Path(__file__).resolve().parent
DATA, RUNS = ROOT/'data', ROOT/'runs'
COMMIT = 'c20de631a40e24669d1d6a8e2acefd8f492f172d'
BASE = 'https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/'
NAMES = ['volume','volume_change','volume_cv','value_median','value_spread',
         'price_value_median','price_value_q90','price_value_change',
         'time_left','urgency','burn_rate','price_shift','value_shift','volume_shift',
         'reference_win','reference_capacity','spend_sensitivity','actual_win','actual_cpm','log_multiplier','has_impressions']
PRESETS = {'market':8, 'context':14, 'response':17, 'full':21}
EPS = 1e-8

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)

def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): return
    part = path.with_suffix(path.suffix+'.part')
    start = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={'Range':f'bytes={start}-'} if start else {})
    with urllib.request.urlopen(req, timeout=60) as src:
        append = start > 0 and src.status == 206
        total = start if append else 0
        mark = total
        with part.open('ab' if append else 'wb') as out:
            while True:
                block = src.read(8*1024*1024)
                if not block: break
                out.write(block); total += len(block)
                if total-mark >= 256*1024*1024:
                    print(f'{path.name}: {total/2**30:.2f} GiB', flush=True); mark=total
    part.replace(path)

def fetch(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    wanted = set(args.advertisers)
    for pack in args.packs:
        if pack not in ['7-8','9-10','11-12','13','14-15','16-17','18-19','20-21','22-23','24-25','26-27']:
            raise ValueError(f'Unknown official pack: {pack}')
        target = DATA/f'pack-{pack}'
        if (target/'manifest.json').exists():
            old=json.loads((target/'manifest.json').read_text())
            if old['advertisers'] != sorted(wanted): raise ValueError('Existing pack has different advertiser filter')
            print('Already prepared:', pack); continue
        archive=DATA/f'period-{pack}.zip'
        url=BASE+f'autoBidding_general_track_final_data_period_{pack}.zip'
        print('Downloading official data:', url, flush=True)
        download(url,archive)
        stage=DATA/f'.preparing-{pack}'
        if stage.exists(): shutil.rmtree(stage)
        stage.mkdir(parents=True)
        writers={}; rows=0
        try:
            with zipfile.ZipFile(archive) as z:
                members=[n for n in z.namelist() if n.lower().endswith('.csv') and not n.startswith('__MACOSX/')]
                if not members: raise ValueError('Archive contains no CSV')
                for member in members:
                    with z.open(member) as f:
                        header=pd.read_csv(f,nrows=0).columns.tolist()
                    aid='advertiserNumber' if 'advertiserNumber' in header else 'advertiserIndex'
                    cols=['deliveryPeriodIndex',aid,'budget','CPAConstraint','timeStepIndex','pvIndex','pValue','pValueSigma','leastWinningCost']
                    missing=set(cols)-set(header)
                    if missing: raise ValueError(f'{member}: missing {missing}')
                    with z.open(member) as f:
                        for chunk in pd.read_csv(f,usecols=cols,chunksize=200_000):
                            chunk=chunk[chunk[aid].isin(wanted)].rename(columns={aid:'advertiser'})
                            for (period,ad),group in chunk.groupby(['deliveryPeriodIndex','advertiser']):
                                key=(int(period),int(ad))
                                table=pa.Table.from_pandas(group,preserve_index=False)
                                if key not in writers:
                                    writers[key]=pq.ParquetWriter(stage/f'p{key[0]}-a{key[1]}.parquet',table.schema,compression='zstd')
                                writers[key].write_table(table); rows+=len(group)
            if not rows: raise ValueError('No matching advertiser IDs; check raw CSV IDs')
        finally:
            for writer in writers.values(): writer.close()
        digest=hashlib.sha256()
        with archive.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''): digest.update(block)
        save(stage/'manifest.json',{'source':url,'sha256':digest.hexdigest(),'advertisers':sorted(wanted),'rows':rows})
        if target.exists(): raise FileExistsError(target)
        stage.rename(target)
        if not args.keep_archives: archive.unlink()
        print(f'Prepared {pack}: {rows:,} rows for selected advertisers',flush=True)

def official_env():
    path=ROOT/'vendor'/'offline_env.py'
    download(f'https://raw.githubusercontent.com/alimama-tech/AuctionNet/{COMMIT}/strategy_train_env/bidding_train_env/offline_eval/offline_env.py',path)
    spec=importlib.util.spec_from_file_location('auctionnet_offline',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.OfflineEnv()

def expected_clipped(p,sigma):
    # Official evaluator uses pValueSigma as normal scale, despite paper wording.
    s=np.maximum(sigma,1e-12); a=-p/s; b=(1-p)/s
    result=p*(norm.cdf(b)-norm.cdf(a))+s*(norm.pdf(a)-norm.pdf(b))+norm.sf(b)
    return np.where(sigma>0,np.clip(result,0,1),np.clip(p,0,1))

def prepare_episode(df,seed):
    df=df.sort_values(['timeStepIndex','pvIndex'])
    ticks=sorted(df.timeStepIndex.unique())
    if len(ticks)!=48 or not np.array_equal(np.diff(ticks),np.ones(47)):
        raise ValueError(f'Incomplete episode: expected 48 consecutive ticks, got {ticks}')
    if df.duplicated(['timeStepIndex','pvIndex']).any(): raise ValueError('Duplicate opportunity records')
    first=df.iloc[0]; pid=int(first.deliveryPeriodIndex); ad=int(first.advertiser)
    budget=float(first.budget); cpa=float(first.CPAConstraint)
    if df.budget.nunique()!=1 or df.CPAConstraint.nunique()!=1 or min(budget,cpa)<=0:
        raise ValueError('Invalid or changing campaign parameters')
    episode={'period':pid,'ad':ad,'budget':budget,'cpa':cpa,'ticks':[]}
    history=[]
    for t,(_,g) in enumerate(df.groupby('timeStepIndex',sort=True)):
        p=g.pValue.to_numpy(float); sigma=g.pValueSigma.to_numpy(float); c=g.leastWinningCost.to_numpy(float)
        if not np.isfinite(np.r_[p,sigma,c]).all() or np.any(c<0) or np.any(sigma<0) or np.any((p<0)|(p>1)):
            raise ValueError('Nonfinite values, negative prices or invalid probabilities')
        v=expected_clipped(p,sigma)
        base=cpa*p
        def probe(alpha):
            mask=alpha*base>=c
            return float(mask.mean()),float(c[mask].sum())
        wr,spend=probe(1.0); lo=probe(0.9)[1]; hi=probe(1.1)[1]
        ratio=c/np.maximum(base,EPS)  # scaled by campaign CPA; dimensionless
        q25,q50,q75=np.quantile(p,[.25,.5,.75])
        stat={'n':len(g),'pm':p.mean(),'cm':c.mean(),'q':np.median(ratio)}
        prev=history[-1] if history else stat
        short=history[-2:]+[stat]; long=history[:-2]
        ref=long if long else short
        shift=lambda name: np.log((np.mean([x[name] for x in short])+EPS)/(np.mean([x[name] for x in ref])+EPS))
        features=[np.log1p(len(g)),np.log((len(g)+EPS)/(prev['n']+EPS)),
                  np.std([x['n'] for x in short])/(np.mean([x['n'] for x in short])+EPS),
                  q50,(q75-q25)/(q50+EPS),np.median(ratio),np.quantile(ratio,.9),
                  np.log((stat['q']+EPS)/(prev['q']+EPS)),shift('cm'),shift('pm'),shift('n')]
        # Monotone bid/price thresholds accelerate repeated exact fixed-price replay.
        threshold=c/np.maximum(base,1e-300); order=np.argsort(threshold,kind='stable')
        episode['ticks'].append({'threshold':threshold[order], 'cost':c[order], 'value':v[order],
            'cost_cum':np.r_[0,np.cumsum(c[order])],
            'value_cum':np.r_[0,np.cumsum(v[order])], 'n':len(g),'features':features,
            'ref_win':wr,'ref_spend':spend,'sensitivity':(hi-lo)/(.2*budget),
            'seed':int(np.random.SeedSequence([seed,pid,ad,t]).generate_state(1)[0])})
        history.append(stat)
    return episode

def feature(episode,t,remaining,last):
    old=episode['ticks'][t-1] if t else None; left=(48-t)/48; B=episode['budget']
    a=old['features'] if old else [0.0]*11
    return np.array(a[:8]+[left,remaining/B/left,last['spend']/max(last['before'],EPS)]+a[8:]+[
        old['ref_win'] if old else 0,old['ref_spend']/max(remaining/(48-t),EPS) if old else 0,
        old['sensitivity'] if old else 0,last['win'],last['cpm'],last['logalpha'],last['has']],dtype=float)

def transformed(x):
    return np.sign(x)*np.log1p(np.abs(x))

def simulate(ep,params,model=None,trace=False):
    B=ep['budget']; remaining=B; integ=0.; previous=params['bias']; active=-1; pending=-1; streak=0
    last={'spend':0.,'before':B,'win':0.,'cpm':0.,'logalpha':previous,'has':0.}
    actual=[]
    records=[]; value=0.; alphas=[]; errors=[]; switches=0
    for t,tick in enumerate(ep['ticks']):
        x=feature(ep,t,remaining,last); label=-1; strength=0.
        if model is not None and t>=3:
            z=model['scaler'].transform(transformed(x[model['indices']])[None,:])
            lab,conf=hdbscan.approximate_predict(model['cluster'],z[:,model['keep']])
            label=int(lab[0]); strength=float(conf[0])
            if strength<.5: label=-1
            if label==pending: streak+=1
            else: pending=label; streak=1
            # Noise falls back immediately; known-cluster changes require 2 ticks.
            new=label if label<0 or streak>=2 else active
            switches+=int(new!=active); active=new
        gains=params.get('gains',{}).get(str(active),[params['kp'],params['ki']])
        kp,ki=gains; error=t/48-(B-remaining)/B
        candidate=integ+error/48
        raw=params['bias']+kp*error+ki*candidate
        low=max(-6.,previous-.5); high=min(4.,previous+.5)
        if (raw>high and error>0) or (raw<low and error<0):
            raw=params['bias']+kp*error+ki*integ
        else: integ=candidate
        u=float(np.clip(raw,low,high)); previous=u; alpha=math.exp(u)
        count=int(np.searchsorted(tick['threshold'],alpha,side='right')) if remaining>=.1 else 0
        spend=float(tick['cost_cum'][count]); gained=float(tick['value_cum'][count])
        if spend>remaining:
            # Explicit protocol change: common-seed random-priority budget rationing.
            order=np.random.default_rng(tick['seed']).permutation(tick['n'])
            selected=order[order<count]
            cumulative=np.cumsum(tick['cost'][selected])
            count=int(np.searchsorted(cumulative,remaining,side='right'))
            selected=selected[:count]
            spend=float(tick['cost'][selected].sum()); gained=float(tick['value'][selected].sum())
        before=remaining; remaining=max(0.,remaining-spend); value+=gained
        end_error=(t+1)/48-(B-remaining)/B
        errors.append(abs(end_error)); alphas.append(u)
        if trace: records.append({'period':ep['period'],'advertiser':ep['ad'],'tick':t,
            **dict(zip(NAMES,map(float,x))),'label':label,'active_regime':active,'strength':strength,
            'kp':kp,'ki':ki,'alpha':alpha,'spend':spend,'remaining':remaining,'value':gained,'error':end_error})
        actual.append((count,tick['n'],spend,u)); recent=np.asarray(actual[-3:]); wins=recent[:,0].sum()
        last={'spend':spend,'before':before,'win':wins/recent[:,1].sum(),
              'cpm':1000*recent[:,2].sum()/wins if wins else 0.,
              'logalpha':float(recent[:,3].mean()),'has':float(wins>0)}
    te=float(np.mean(errors)); tv=float(np.mean(np.abs(np.diff(alphas)))); residual=remaining/B
    result={'period':ep['period'],'advertiser':ep['ad'],'te':te,'multiplier_tv':tv,
            'unspent':residual,'expected_conversions':value,'switches':switches,
            'objective':te+.01*tv+.2*residual}
    return result,records

def score(episodes,params,model=None):
    frame=pd.DataFrame([simulate(e,params,model)[0] for e in episodes])
    return float(frame.groupby('period').objective.mean().mean())

def fit_regimes(episodes,fixed,preset,out,min_cluster_size=0,min_samples=10):
    xs=[]; metadata=[]
    # Three policies cover on-policy budget states; evaluation episodes are never used.
    for scale in ([1.] if preset=='market' else [.5,1.,2.]):
        p={**fixed,'kp':fixed['kp']*scale,'ki':fixed['ki']*scale}
        for ep in episodes:
            _,rows=simulate(ep,p,trace=True)
            xs.extend([[r[n] for n in NAMES] for r in rows[3:]])
            metadata.extend([{'period':ep['period'],'advertiser':ep['ad'],'tick':r['tick'],'gain_scale':scale} for r in rows[3:]])
    x=np.asarray(xs); indices=np.arange(PRESETS[preset]); z=transformed(x[:,indices])
    unique=np.sort(np.unique(z,axis=0,return_index=True)[1]); x=x[unique]; z=z[unique]
    metadata=[metadata[i] for i in unique]
    scaler=RobustScaler().fit(z); z=scaler.transform(z)
    keep=np.where(np.std(z,axis=0)>1e-6)[0]
    # Prune near-duplicate dimensions, train-only.
    corr=np.corrcoef(z[:,keep],rowvar=False); selected=[]
    for j in range(len(keep)):
        if not any(abs(corr[j,k])>.97 for k in selected): selected.append(j)
    keep=keep[selected]
    if len(keep)<2: raise ValueError('Insufficient nonconstant feature dimensions')
    minsize=min_cluster_size or max(30,int(len(z)*.03))
    cluster=hdbscan.HDBSCAN(min_cluster_size=minsize,min_samples=min_samples,prediction_data=True).fit(z[:,keep])
    labels,counts=np.unique(cluster.labels_,return_counts=True)
    features=pd.concat([pd.DataFrame(metadata),pd.DataFrame(x,columns=NAMES)],axis=1)
    features['regime']=cluster.labels_; features['strength']=cluster.probabilities_
    features.to_csv(out/'train_features_and_regimes.csv',index=False)
    summary={'features_used':[NAMES[indices[k]] for k in keep],
             'cluster_counts':{str(k):int(v) for k,v in zip(labels,counts)},
             'min_cluster_size':minsize,'min_samples':min_samples,'preset':preset}
    save(out/'regimes.json',summary)
    if sum(labels>=0)<2:
        raise RuntimeError('Fewer than 2 regimes found. See regimes.json; do not force a positive result.')
    model={'indices':indices,'scaler':scaler,'keep':keep,'cluster':cluster}
    joblib.dump(model,out/'regimes.joblib'); return model

def tune(episodes,validation,trials,seed,out,fixed=None,model=None,max_loss=.02):
    name='fixed' if model is None else 'regime'
    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=seed),
        storage=f'sqlite:///{(out/(name+".sqlite3")).resolve()}',study_name=name,load_if_exists=True)
    if not study.trials and fixed is not None:
        study.enqueue_trial({f'{field}_{label}':fixed[field] for label in np.unique(model['cluster'].labels_) if label>=0 for field in ['kp','ki']})
    def objective(trial):
        if model is None:
            p={'bias':trial.suggest_float('bias',-5,2),
               'kp':trial.suggest_float('kp',.01,100,log=True),'ki':trial.suggest_float('ki',.01,100,log=True)}
        else:
            p={**fixed,'gains':{}}
            for label in np.unique(model['cluster'].labels_):
                if label<0: continue
                p['gains'][str(label)]=[trial.suggest_float(f'{k}_{label}',.01,100,log=True) for k in ['kp','ki']]
        loss=score(episodes,p,model); validation_score=score(validation,p,model)
        if fixed is not None:
            for group,is_validation in [(episodes,False),(validation,True)]:
                base_value=np.mean([simulate(e,fixed)[0]['expected_conversions'] for e in group])
                new_value=np.mean([simulate(e,p,model)[0]['expected_conversions'] for e in group])
                penalty=10*max(0.,1-max_loss-new_value/max(base_value,EPS))
                if is_validation: validation_score+=penalty
                else: loss+=penalty
        trial.set_user_attr('parameters',p); trial.set_user_attr('validation',validation_score)
        return loss
    finished=sum(t.state.is_finished() for t in study.trials)
    study.optimize(objective,n_trials=max(0,trials-finished))
    done=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    if not done: raise RuntimeError('No completed tuning trials')
    best=min(done,key=lambda t:t.user_attrs['validation'])
    p=best.user_attrs['parameters']; save(out/(name+'.json'),p)
    study.trials_dataframe().to_csv(out/(name+'_trials.csv'),index=False)
    return p

def run(args):
    out=RUNS/args.name; out.mkdir(parents=True,exist_ok=True)
    files=sorted(DATA.glob('pack-*/*.parquet'))
    if not files: raise ValueError('Run fetch first')
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in DATA.glob('pack-*/manifest.json')}
    config={'data':hashes,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'preset':args.preset,'trials':args.trials,'seed':args.seed,
            'max_value_loss':args.max_value_loss,'min_cluster_size':args.min_cluster_size,'min_samples':args.min_samples,
            'protocol':'fixed_price_expected_value_random_rationing_v1','upstream':COMMIT}
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:
        raise ValueError('Configuration changed: choose a new --name')
    save(out/'config.json',config)
    episodes=[prepare_episode(pd.read_parquet(f),args.seed) for f in files]
    periods=sorted(set(e['period'] for e in episodes))
    if len(periods)<6: raise ValueError('Need >=6 distinct periods for train/validation/test; fetch 7-8,9-10,11-12')
    nt=max(2,int(.6*len(periods))); nv=max(1,int(.2*len(periods)))
    train=[e for e in episodes if e['period'] in periods[:nt]]
    val=[e for e in episodes if e['period'] in periods[nt:nt+nv]]
    test=[e for e in episodes if e['period'] in periods[nt+nv:]]
    save(out/'split.json',{'train':periods[:nt],'validation':periods[nt:nt+nv],'test':periods[nt+nv:]})
    print('Split:',periods[:nt],periods[nt:nt+nv],periods[nt+nv:],flush=True)
    fixed=tune(train,val,args.trials,args.seed,out)
    model=fit_regimes(train,fixed,args.preset,out,args.min_cluster_size,args.min_samples)
    regime=tune(train,val,args.trials,args.seed,out,fixed,model,args.max_value_loss)
    results=[]; traces=[]
    for method,p,m in [('fixed',fixed,None),('regime',regime,model)]:
        for ep in test:
            result,rows=simulate(ep,p,m,True); result['method']=method; results.append(result)
            traces.extend([{**r,'method':method} for r in rows])
    frame=pd.DataFrame(results); frame.to_csv(out/'test_metrics.csv',index=False)
    pd.DataFrame(traces).to_csv(out/'test_traces.csv',index=False)
    summary=frame.groupby(['method','period']).mean(numeric_only=True).groupby('method').mean()
    summary.to_csv(out/'summary.csv'); print(summary.to_string())
    # Paired period bootstrap; very few periods => exploratory intervals only.
    paired=frame.pivot(index=['period','advertiser'],columns='method',values='te')
    delta=(paired['regime']-paired['fixed']).groupby('period').mean().to_numpy()
    rng=np.random.default_rng(args.seed)
    samples=rng.choice(delta,(2000,len(delta)),replace=True).mean(axis=1)
    save(out/'paired_te.json',{'regime_minus_fixed':float(delta.mean()),
         'periods':len(delta),'bootstrap_95':list(map(float,np.quantile(samples,[.025,.975]))),
         'warning':'Exploratory only, especially with few held-out periods. No automatic significance claim.'})
    print('Results:',out)

def check():
    env=official_env(); p=np.array([.1,.2,.3]); c=np.array([1.,2.,3.]); bids=np.array([1.,1.,4.])
    _,cost,won,_=env.simulate_ad_bidding(p,np.zeros(3),bids,c)
    assert np.array_equal(won,[True,False,True]) and np.allclose(cost,[1,0,3])
    assert np.allclose(expected_clipped(p,np.zeros(3)),p)
    print('Official fixed-price win/cost contract and expected-value checks passed.')

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='command',required=True)
    f=sub.add_parser('fetch'); f.add_argument('--packs',nargs='+',default=['7-8','9-10','11-12'])
    f.add_argument('--advertisers',nargs='+',type=int,default=[1,13,25,37]); f.add_argument('--keep-archives',action='store_true')
    r=sub.add_parser('run'); r.add_argument('--name',default='market_pilot'); r.add_argument('--preset',choices=PRESETS,default='market')
    r.add_argument('--trials',type=int,default=30); r.add_argument('--seed',type=int,default=42)
    r.add_argument('--max-value-loss',type=float,default=.02)
    r.add_argument('--min-cluster-size',type=int,default=0); r.add_argument('--min-samples',type=int,default=10)
    sub.add_parser('check'); args=ap.parse_args()
    if args.command=='fetch': fetch(args)
    elif args.command=='run': run(args)
    else: check()

if __name__=='__main__': main()

PYTHON
cat > auctionnet_lab/PROTOCOL.txt <<'PROTOCOL'
AuctionNet regime/PI exploratory protocol v1
Source: alimama-tech/AuctionNet commit c20de631a40e24669d1d6a8e2acefd8f492f172d.
Official general_track_final ZIPs; default periods 7-12 and advertiser IDs 1,13,25,37.
All 48 ticks and all opportunities of each selected advertiser are retained.
Fixed-price replay, NOT full multi-agent GSP simulation or official leaderboard reproduction.
Win/cost threshold matches official offline_env.py. pValueSigma is normal scale, as in code.
Use analytic E[clip(N(pValue,pValueSigma),0,1)] as expected conversions; no sampled-score claim.
At budget exhaustion, seeded random-priority truncation replaces upstream random-drop/re-draw loop.
All features use previous ticks only. Current competitors' future prices are not inputs.
Counterfactual reference features use previous observed fixed-price traffic only.
Price/value ratio is normalized by fixed campaign CPA. Reference base bid = CPA*pValue.
Feature pool: 20 candidates + no-impression flag. No-impression CPM=0 plus flag, not a cheap-market assertion.
Market feature fitting avoids duplicated trajectories; context fitting uses 3 train-only policies and deduplicates identical vectors.
Period order: floor(60%) train, floor(20%) validation, remainder test; no row-random split.
HDBSCAN settings are pilot defaults; not selected using held-out test performance.
Fewer than 2 clusters: export diagnosis and stop, rather than fabricate regimes.
PI uses normalized cumulative error and log bid multiplier, anti-windup, common bounds/rate limit.
Hysteresis: 2 consecutive known-cluster predictions; noise/strength<0.5 -> tuned global PI.
30 TPE trials per method are a pilot budget, not proof of optimal coefficients.
Objective: mean absolute tracking error + .01 log-multiplier TV + .2 unspent budget fraction.
Regime tuning penalizes >2% expected-value loss vs fixed PI on train/validation; no test guarantee.
Per-period paired test summary/bootstrap; two test periods are exploratory, not significance evidence.
Validate on more periods/advertisers/seeds; add clustering sweeps and full official simulator before publication.
References: NeurIPS2024 AuctionNet; CIKM2018 DRLB; WWW Companion2025 GAS;
Complexity2017 A Gain-Scheduling PI Control Based on Neural Networks.
PROTOCOL
"$python_bin" -m venv auctionnet_lab/.venv
auctionnet_lab/.venv/bin/python -m pip install --upgrade pip
auctionnet_lab/.venv/bin/python -m pip install -r auctionnet_lab/requirements.txt
auctionnet_lab/.venv/bin/python auctionnet_lab/run.py check
printf '\nInstallation complete. Downloading and preparing 3 official ZIP packs.\n'
auctionnet_lab/.venv/bin/python auctionnet_lab/run.py fetch --packs 7-8 9-10 11-12
failed=''
for preset in market context response full; do
  if auctionnet_lab/.venv/bin/python auctionnet_lab/run.py run --preset "$preset" --name "${preset}_pilot" --trials 30; then
    echo "Completed: $preset"
  else
    echo "Stopped: $preset. Inspect the traceback and runs/${preset}_pilot/" >&2
    failed="$failed $preset"
  fi
done
if [ -n "$failed" ]; then
  echo "Incomplete presets:$failed" >&2
  exit 1
fi
printf '\nCompleted. Results: auctionnet_lab/runs/*_pilot/summary.csv\n'
