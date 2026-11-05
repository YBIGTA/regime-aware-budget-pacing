"""Official AuctionNet BC/IQL models; streaming data adapter and exploratory evaluation."""
import argparse, hashlib, importlib.util, json, math, os, random, shutil, subprocess, sys, time, traceback, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
from iql import IQL
from bc import BC
from offline_env import OfflineEnv
COMMIT='c20de631a40e24669d1d6a8e2acefd8f492f172d'
BASE='https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/'
KEY=['deliveryPeriodIndex','advertiserNumber','advertiserCategoryIndex','budget','CPAConstraint','timeStepIndex']
MEANS=['bid','leastWinningCost','conversionAction','xi','pValue']
RAW=KEY+MEANS+['remainingBudget','isEnd','isExposed','cost','pValueSigma','pvIndex']
EVAL_ADS=[1,13,25,37]


def write_json(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False))
    tmp.replace(path)


def summarize(chunk):
    c=chunk.copy()
    c['n']=1
    c['r']=c.conversionAction*(c.isExposed==1)
    c['rc']=c.pValue*(c.isExposed==1)
    c['paid']=c.cost*c.isExposed
    return c.groupby(KEY,sort=True).agg({**{x:'sum' for x in MEANS},
        'n':'sum','r':'sum','rc':'sum','paid':'sum','remainingBudget':'first','isEnd':'first'})


def combine(parts):
    return pd.concat(parts).groupby(level=list(range(len(KEY))),sort=True).agg({
        **{x:'sum' for x in MEANS},'n':'sum','r':'sum','rc':'sum','paid':'sum',
        'remainingBudget':'first','isEnd':'first'})


def trajectories(agg):
    rows=[]
    for identity,group in agg.reset_index().groupby(KEY[:-1],sort=True):
        group=group.sort_values('timeStepIndex').reset_index(drop=True)
        if not np.array_equal(group.timeStepIndex.to_numpy(),np.arange(48)):
            raise ValueError(f'Incomplete trajectory: {identity}')
        if (group.n<=0).any(): raise ValueError('Empty tick')
        means=group[MEANS].div(group.n,axis=0)
        past=means.expanding().mean().shift(1).fillna(0)
        short=means.rolling(3,min_periods=1).mean().shift(1).fillna(0)
        states=[]
        for i,r in group.iterrows():
            states.append(np.array([
                (48-int(r.timeStepIndex))/48,r.remainingBudget/r.budget,
                past.bid[i],short.bid[i],past.leastWinningCost[i],past.pValue[i],
                past.conversionAction[i],past.xi[i],short.leastWinningCost[i],
                short.pValue[i],short.conversionAction[i],short.xi[i],means.pValue[i],
                r.n,group.n.iloc[max(0,i-3):i].sum(),group.n.iloc[:i].sum()],dtype=float))
        # Retain official terminal rows, including post-depletion terminal records.
        for i,r in group.iterrows():
            done=int(i==47 or r.isEnd==1)
            rows.append(dict(period=int(r.deliveryPeriodIndex),ad=int(r.advertiserNumber),
                tick=i,state=states[i],action=float(r.bid/r.pValue) if r.pValue>0 else 0.,
                reward=float(r.r),reward_continuous=float(r.rc),done=done,
                next_state=np.zeros(16) if done else states[i+1]))
    return rows


def prepare_pack(pack):
    data=ROOT/'data'; data.mkdir(exist_ok=True)
    result=data/f'transitions_{pack}.npz'
    if result.exists():
        print('REUSE',result.name,flush=True); return
    archive=data/f'raw_{pack}.zip'
    if not archive.exists():
        print('DOWNLOAD',pack,flush=True)
        subprocess.run(['curl','--fail','--location','--retry','8','--retry-delay','5',
            '--connect-timeout','30','--speed-time','120','--speed-limit','1024',
            '--continue-at','-','--output',str(archive)+'.part',
            BASE+f'autoBidding_general_track_final_data_period_{pack}.zip'],check=True)
        Path(str(archive)+'.part').replace(archive)
    parts=[]; count=0; writers={}; excluded={}
    try:
        with zipfile.ZipFile(archive) as z:
            members=[n for n in z.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX/')]
            if not members: raise ValueError('No raw CSV')
            for member in members:
                with z.open(member) as f:
                    header=pd.read_csv(f,nrows=0).columns
                if not set(RAW).issubset(header):
                    raise ValueError(f'Raw columns missing: {set(RAW)-set(header)}')
                with z.open(member) as f:
                    for chunk in pd.read_csv(f,usecols=RAW,chunksize=200000):
                        # V7_BID_NAN_QUARANTINE: never invent missing action labels.
                        checked = chunk[RAW].to_numpy(dtype=float)
                        bad_counts = pd.Series((~np.isfinite(checked)).sum(axis=0), index=RAW)
                        forbidden = bad_counts.drop('bid')
                        if forbidden.sum() or np.isinf(chunk.bid.to_numpy(dtype=float)).any():
                            raise ValueError('Unexpected nonfinite values: '+bad_counts[bad_counts>0].to_json())
                        missing_bid = chunk.bid.isna()
                        if missing_bid.any():
                            for (pid, ad), group in chunk.loc[missing_bid].groupby(['deliveryPeriodIndex','advertiserNumber']):
                                key = (int(pid), int(ad))
                                excluded[key] = excluded.get(key, 0) + len(group)
                            write_json(data/f'exclusions_{pack}.json', {
                                'reason': 'NaN bid; whole advertiser-period trajectory excluded from transitions',
                                'episodes': [dict(period=k[0],advertiser=k[1],missing_bid_rows=n)
                                             for k,n in sorted(excluded.items())]})
                            print('QUARANTINE', pack, sorted(excluded), 'missing rows', int(missing_bid.sum()), flush=True)
                        parts.append(summarize(chunk));count+=len(chunk)
                        if pack=='13':
                            view=chunk[chunk.advertiserNumber.isin(EVAL_ADS)]
                            for (pid,ad),sub in view.groupby(['deliveryPeriodIndex','advertiserNumber']):
                                key=(int(pid),int(ad))
                                if key not in writers:
                                    table=pa.Table.from_pandas(sub,preserve_index=False)
                                    writers[key]=pq.ParquetWriter(data/f'eval_p{key[0]}_a{key[1]}.parquet.part',table.schema,compression='zstd')
                                writers[key].write_table(pa.Table.from_pandas(sub,preserve_index=False))
                        if len(parts)>=20: parts=[combine(parts)]
                        if count%2000000==0: print('PARSED',pack,f'{count:,}',flush=True)
        agg=combine(parts)
        expected=set(range(int(pack.split('-')[0]),int(pack.split('-')[-1])+1))
        if set(agg.index.get_level_values(0))!=expected: raise ValueError('Unexpected periods')
        for pid,gp in agg.reset_index().groupby('deliveryPeriodIndex'):
            if gp.advertiserNumber.nunique()!=48: raise ValueError('Expected 48 advertisers')
        if excluded:
            keep = [(int(pid),int(ad)) not in excluded
                    for pid,ad in zip(agg.index.get_level_values(0),agg.index.get_level_values(1))]
            agg = agg.loc[keep]
            if agg.empty: raise ValueError('No usable trajectories remain')
        rows=trajectories(agg)
        if not all(np.isfinite(np.r_[r['state'], r['next_state'], r['action'], r['reward_continuous']]).all() for r in rows):
            raise ValueError('Nonfinite transitions after quarantine')
        with (result.with_suffix('.npz.part')).open('wb') as f:
            np.savez_compressed(f,s=np.stack([r['state'] for r in rows]),
                ns=np.stack([r['next_state'] for r in rows]),
                a=np.array([r['action'] for r in rows])[:,None],
                r=np.array([r['reward_continuous'] for r in rows])[:,None],
                done=np.array([r['done'] for r in rows])[:,None],
                period=np.array([r['period'] for r in rows]),ad=np.array([r['ad'] for r in rows]))
        for w in writers.values():w.close()
        writers={}
        for path in data.glob('eval_*.parquet.part'):path.replace(path.with_suffix(''))
        result.with_suffix('.npz.part').replace(result)
        print('PREPARED',pack,len(rows),'transitions',flush=True)
    finally:
        for w in writers.values():w.close()
    # Remove only our downloaded compressed archive after successful conversion.
    archive.unlink()


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def scaling(s):
    lo=np.zeros(16);den=np.ones(16)
    for i in [13,14,15]:lo[i]=s[:,i].min();den[i]=s[:,i].max()-lo[i]+.01
    return lo,den


def state_live(t,remaining,budget,p,h):
    def mean(field,n=None):
        seq=h if n is None else h[-n:]
        return float(np.mean([r[field] for r in seq])) if seq else 0.
    return np.array([(48-t)/48,remaining/budget,mean('bid'),mean('bid',3),
        mean('price'),mean('p'),mean('conv'),mean('xi'),mean('price',3),mean('p',3),
        mean('conv',3),mean('xi',3),float(p.mean()),len(p),
        sum(r['n'] for r in h[-3:]),sum(r['n'] for r in h)],dtype=float)


def evaluate(model,lo,den,seed,ratios=(1.,)):
    records=[];env=OfflineEnv()
    for path in sorted((ROOT/'data').glob('eval_*.parquet')):
        df=pd.read_parquet(path).sort_values(['timeStepIndex','pvIndex'])
        if sorted(df.timeStepIndex.unique())!=list(range(48)):raise ValueError('Incomplete eval episode')
        first=df.iloc[0];pid=int(first.deliveryPeriodIndex);ad=int(first.advertiserNumber)
        for ratio in ratios:
            np.random.seed(seed+pid*1000+ad)
            budget=float(first.budget)*ratio;remaining=budget;history=[];reward=0;errors=[]
            for t,(_,tick) in enumerate(df.groupby('timeStepIndex',sort=True)):
                p=tick.pValue.to_numpy();sigma=tick.pValueSigma.to_numpy();price=tick.leastWinningCost.to_numpy()
                state=state_live(t,remaining,budget,p,history)
                with torch.no_grad():action=float(model.take_actions(((state-lo)/den)[None,:]).reshape(-1)[0])
                if not np.isfinite(action):raise ValueError('Nonfinite action')
                bid=np.maximum(action,0)*p
                if remaining<.1:
                    bid=np.zeros_like(p);cost=np.zeros_like(p);status=np.zeros_like(p);conv=np.zeros_like(p)
                else:
                    _,cost,status,conv=env.simulate_ad_bidding(p,sigma,bid,price)
                    for _ in range(100):
                        over=max((cost.sum()-remaining)/(cost.sum()+1e-4),0)
                        if over<=0:break
                        winners=np.flatnonzero(status)
                        drops=np.random.choice(winners,int(math.ceil(len(winners)*over)),replace=False)
                        bid[drops]=0
                        _,cost,status,conv=env.simulate_ad_bidding(p,sigma,bid,price)
                    if cost.sum()>remaining+1e-6:raise ValueError('Budget trimming failed')
                remaining=max(0.,remaining-float(cost.sum()));reward+=float(conv.sum())
                history.append(dict(bid=float(bid.mean()),price=float(price.mean()),p=float(p.mean()),
                                    xi=float(status.mean()),conv=float(conv.mean()),n=len(p)))
                errors.append(abs((t+1)/48-(budget-remaining)/budget))
            spend=budget-remaining;cpa=spend/(reward+1e-10)
            score=reward*min(1.,float(first.CPAConstraint)/(cpa+1e-10))**2
            records.append(dict(period=pid,ad=ad,budget_ratio=ratio,conversions=reward,
                spend=spend,cpa=cpa,score=score,te=float(np.mean(errors)),unspent=remaining/budget))
    if len(records)!=4*len(ratios):raise ValueError('Expected four validation advertisers')
    return pd.DataFrame(records)


def train(args):
    train_parts=[dict(np.load(ROOT/'data'/f'transitions_{p}.npz')) for p in ['7-8','9-10','11-12']]
    s=np.concatenate([p['s'] for p in train_parts]); ns=np.concatenate([p['ns'] for p in train_parts])
    a=np.concatenate([p['a'] for p in train_parts]);r=np.concatenate([p['r'] for p in train_parts]);done=np.concatenate([p['done'] for p in train_parts])
    lo,den=scaling(s);rlo=float(r.min());rden=float(r.max()-r.min()+1e-8)
    tensors=[torch.tensor(x,dtype=torch.float32) for x in [(s-lo)/den,a,(r-rlo)/rden,(ns-lo)/den,done]]
    # Terminal next-state contents are masked by (1-done) in the official model.
    if not all(torch.isfinite(t).all() for t in tensors):raise ValueError('Invalid training tensors')
    write_json(ROOT/'outputs'/'normalization.json',dict(lo=lo.tolist(),den=den.tolist(),reward_min=rlo,reward_range=rden))
    print('TRAIN TRANSITIONS',len(s),flush=True)
    all_scores=[]
    for method in ['BC','IQL']:
        for seed in args.seeds:
            folder=ROOT/'outputs'/f'{method}_seed{seed}';folder.mkdir(exist_ok=True)
            final=folder/'DONE'
            if final.exists():print('REUSE MODEL',folder.name,flush=True);continue
            seed_all(seed)
            model=BC(16,network_random_seed=seed) if method=='BC' else IQL(dim_obs=16,network_random_seed=seed)
            start=0;best=-float('inf');log=[]
            resume=folder/'last.pt'
            optimizers=(['actor_optimizer'] if method=='BC' else ['value_optimizer','critic1_optimizer','critic2_optimizer','actor_optimizer'])
            if resume.exists():
                ck=torch.load(resume,map_location='cpu',weights_only=False)
                model.load_state_dict(ck['model'])
                for name in optimizers:getattr(model,name).load_state_dict(ck[name])
                torch.set_rng_state(ck['rng']);start=ck['step'];best=ck['best'];log=ck['log']
                print('RESUME',folder.name,start,flush=True)
            started=time.monotonic()
            for step in range(start+1,args.steps+1):
                idx=torch.randint(len(s),(100,));batch=[t[idx] for t in tensors]
                losses=model.step(batch[0].numpy(),batch[1].numpy()) if method=='BC' else model.step(*batch)
                if not np.isfinite(np.array(losses)).all():raise ValueError(f'Nonfinite loss: {method}/{seed}/{step}')
                if step%1000==0:print('TRAIN',method,seed,step,'/',args.steps,'seconds',round(time.monotonic()-started),flush=True)
                if step%args.checkpoint==0 or step==args.steps:
                    scores=evaluate(model,lo,den,seed)
                    score=float(scores.score.mean());log.append(dict(step=step,validation_score=score))
                    if score>best:
                        best=score;torch.save(model.state_dict(),folder/'best.pt')
                    ck=dict(model=model.state_dict(),step=step,best=best,log=log,rng=torch.get_rng_state())
                    for name in optimizers:ck[name]=getattr(model,name).state_dict()
                    temp=folder/'last.pt.tmp';torch.save(ck,temp);temp.replace(resume)
                    pd.DataFrame(log).to_csv(folder/'training_curve.csv',index=False)
                    print('CHECKPOINT',method,seed,step,'validation score',score,flush=True)
            model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=False))
            scores=evaluate(model,lo,den,seed,ratios=(.5,.75,1.,1.25,1.5))
            scores.to_csv(folder/'validation_budget_sweep.csv',index=False)
            final.write_text('Training complete; validation only, no final test.\n')
    for method in ['BC','IQL']:
        for seed in args.seeds:
            f=pd.read_csv(ROOT/'outputs'/f'{method}_seed{seed}'/'validation_budget_sweep.csv')
            f['method']=method;f['seed']=seed;all_scores.append(f)
    report=pd.concat(all_scores,ignore_index=True)
    report.to_csv(ROOT/'outputs'/'validation_episodes.csv',index=False)
    report.groupby(['method','budget_ratio'])[['score','conversions','te','unspent']].agg(['mean','std']).to_csv(ROOT/'outputs'/'validation_summary.csv')
    (ROOT/'outputs'/'DONE').write_text('All requested seeds and budget sweeps completed.\n')
    print('ALL DONE',ROOT/'outputs',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=20000)
    parser.add_argument('--checkpoint',type=int,default=5000);parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44]);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    # The original implementation is CPU-safe; force CPU consistently on Mac/Linux.
    torch.cuda.is_available=lambda:False
    (ROOT/'outputs').mkdir(exist_ok=True)
    write_json(ROOT/'outputs'/'protocol.json',dict(commit=COMMIT,train_periods=list(range(7,13)),validation_periods=[13],test_periods='Not downloaded or evaluated',
        training_advertisers='All 48 except audited NaN-bid trajectories',validation_advertisers=EVAL_ADS,seeds=args.seeds,steps=args.steps,
        objective='Official continuous reward IQL; CPA-penalized conversion validation score; TE reported separately',
        adaptations=['Streaming exact tick aggregation','Train-only normalization, same transform at evaluation',
            'Correct last-three history at inference (official wrapper uses a faulty slice)',
            'CPU training; checkpoint/resume; explicit zero participation below minimum budget'],
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'vendor').glob('*.py')}))
    if args.smoke:
        seed_all(42)
        s=torch.randn(100,16);a=torch.rand(100,1)*10;r=torch.rand(100,1);done=torch.zeros(100,1)
        bc=BC(16);print('BC SMOKE',bc.step(s.numpy(),a.numpy()))
        model=IQL(dim_obs=16);print('IQL SMOKE',model.step(s,a,r,s,done));return
    for pack in ['7-8','9-10','11-12','13']:
        if shutil.disk_usage(ROOT).free < 12*2**30:
            raise RuntimeError('Please provide at least 12 GiB free disk space before downloading the next raw pack.')
        prepare_pack(pack)
    train(args)

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc()
        (ROOT/'outputs').mkdir(exist_ok=True)
        (ROOT/'outputs'/'FAILED.txt').write_text(traceback.format_exc())
        raise
