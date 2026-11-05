"""Locked full-advertiser benchmark; finite-action approximate policy improvement.
No promise of positive results. Not an exact AggreVaTe implementation.
"""
import os,sys,json,time,hashlib,traceback,subprocess,fcntl,shutil,zipfile
from pathlib import Path
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parent
if hashlib.sha256((ROOT/'gain_policy_v10.py').read_bytes()).hexdigest()!='43ed2f280ce876015674daa45219308015ef34f103b445f2f6742c1e0f66b77f':raise ValueError('Unmodified gain_policy_v10.py is required')
import gain_policy_v10 as c
v=c.v
CODE=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
OUT=v.LAB/'runs'/('pacing_protocol_v11_'+CODE[:8]);OUT.mkdir(parents=True,exist_ok=True)
CACHE=OUT/'cache';CACHE.mkdir(exist_ok=True)
c.OUT=OUT;c.CACHE=CACHE
TRAIN=list(range(16,22));TUNE=[22,23];TEST=[24,25,26,27]
ANCHORS=list(range(1,48,6));MASTER=20261105
ADS=[];EXPECTED_ADS=48;MIN_STATES=100

def save(name,obj):v.jsave(OUT/name,obj)
def csv(name,df):v.csvsave(OUT/name,df)
def status(stage,kind='RUNNING'):save('STATUS.json',dict(status=kind,stage=stage));print(kind,stage,flush=True)
def get(name,fn):return c.cache(name,fn)
def data_lab():
 lab=v.g.d.load_lab();lab.DATA=OUT/'data';lab.DATA.mkdir(exist_ok=True);return lab

def fetch(pack):
 if (OUT/f'data/pack-{pack}/manifest.json').exists():return
 lab=data_lab()
 for attempt in range(3):
  try:lab.fetch(SimpleNamespace(packs=[pack],advertisers=list(range(49)),keep_archives=False));return
  except Exception as e:
   if isinstance(e,zipfile.BadZipFile):(lab.DATA/f'period-{pack}.zip').unlink(missing_ok=True)
   if attempt==2:raise
   print('RETRY DOWNLOAD',pack,attempt+1,flush=True);time.sleep(5)

def paths(period):return sorted((OUT/'data').glob(f'pack-*/p{period}-a*.parquet'))
def discover(period):
 ids=sorted(int(p.stem.split('-a')[1]) for p in paths(period))
 if len(ids)!=EXPECTED_ADS or len(set(ids))!=EXPECTED_ADS:raise ValueError(f'Expected {EXPECTED_ADS} distinct advertisers in {period}, found {ids}')
 return ids

@lru_cache(maxsize=2)
def episode(period,ad):
 def load():
  found=list((OUT/'data').glob(f'pack-*/p{period}-a{ad}.parquet'))
  if len(found)!=1:raise ValueError(f'Expected one data file: {period}/{ad}')
  ep=data_lab().prepare_episode(pd.read_parquet(found[0]),42)
  if (ep['period'],ep['ad'])!=(period,ad):raise ValueError('Episode identity mismatch')
  return ep
 return get(f'episode_{period}_{ad}',load)

def features(e,t,state,history,bias):
 return np.r_[v.feature(e,t,state,history),np.log(e['budget']),np.log(e['cpa']),bias]

def rollout(e,pool,base,bias,policy=None,collect=False,prefix='fixed'):
 parts=[];state=None;start=0;rows=[]
 sample=set(np.random.default_rng(np.random.SeedSequence([MASTER,e['period'],e['ad']])).choice(ANCHORS,3,replace=False).tolist())
 while start<48:
  end=1 if start==0 else min(start+6,48);action=base
  if start and state[0]>=.1:
   x=features(e,start,state,pd.concat(parts,ignore_index=True),bias)
   if collect and start in sample:rows.append(dict(period=e['period'],ad=e['ad'],prefix=prefix,start=start,state=list(map(float,state)),x=x.tolist()))
   if policy is not None:action=policy.act(x)
  f,states=v.g.roll(e,dict(kp=pool[action][0],ki=pool[action][1]),bias,start,end,state)
  f['candidate']=action;parts.append(f);state=states[end];start=end
 f=pd.concat(parts,ignore_index=True);v.check_trace(f,e['budget']);return f,rows

def label(row,pool,base,bias,continuation,tag):
 state=tuple(row['state']);start=row['start'];period=row['period'];ad=row['ad']
 digest=hashlib.sha256(np.asarray([*state,bias,*np.asarray(pool).ravel()],dtype=float).tobytes()).hexdigest()[:16]
 def compute():
  e=episode(period,ad);js=[];vs=[]
  for kp,ki in pool:
   end=min(start+6,48);f,states=v.g.roll(e,dict(kp=kp,ki=ki),bias,start,end,state);parts=[f];s=states[end];t=end
   while t<48:
    action=base
    if continuation is not None and s[0]>=.1:action=continuation.act(features(e,t,s,pd.concat(parts,ignore_index=True),bias))
    kp2,ki2=pool[action];f,states=v.g.roll(e,dict(kp=kp2,ki=ki2),bias,t,min(t+6,48),s)
    parts.append(f);s=states[min(t+6,48)];t=min(t+6,48)
   f=pd.concat(parts,ignore_index=True);v.check_trace(f,state[0])
   js.append(float(f.error.abs().sum()/48+.01*f.du.sum()/47));vs.append(float(f.value.sum()))
  return dict(row,j=js,value=vs)
 return get(f'Q_{tag}_{period}_{ad}_{start}_{digest}',compute)

def evaluate(periods,pool,base,bias,model=None):
 rows=[]
 for p in periods:
  for ad in ADS:
   e=episode(p,ad);f,_=rollout(e,pool,base,bias[ad],model)
   rows.append(dict(period=p,advertiser=ad,**v.metric(f,e['budget']),changed_ticks=int((f.candidate!=base).sum())))
 return pd.DataFrame(rows)
def avg(frame):return frame.groupby('period')[['te','value','j','tv','unspent','changed_ticks']].mean().mean()

def run():
 global ADS
 status('fetch_training_and_selection_periods')
 for pack in ['16-17','18-19','20-21','22-23']:fetch(pack)
 ADS=discover(TRAIN[0])
 for p in TRAIN+TUNE:
  if discover(p)!=ADS:raise ValueError('Advertiser census changes across periods')
 save('advertisers.json',ADS)
 status('train_only_initialization')
 bias,calibration=v.g.d.initial_bias(episode(p,ad) for p in TRAIN for ad in ADS)
 save('calibration_audit.json',calibration)
 pool=sorted(set([(float(kp),float(ki)) for kp in [4,16,64] for ki in [0,.3,3]]+[(19.188179966639105,.06278467600183289)]))
 reference=pool.index((16.,0.));fixed=[]
 status('select_fixed_baseline_on22_23')
 for offset in [-.4,0.,.4]:
  b={a:float(np.clip(bias[a]+offset,-6,4)) for a in ADS}
  for base in range(len(pool)):
   f=get(f'fixed_{offset}_{base}',lambda base=base,b=b:evaluate(TUNE,pool,base,b))
   a=avg(f);fixed.append(dict(base=base,offset=offset,te=float(a.te),j=float(a.j),value=float(a['value'])))
 ref=next(z for z in fixed if z['base']==reference and z['offset']==0)
 passing=[z for z in fixed if z['value']>=.98*ref['value']]
 selected=min(passing,key=lambda z:(z['j'],z['te'],z['base'],z['offset']))
 base=selected['base'];bias={a:float(np.clip(bias[a]+selected['offset'],-6,4)) for a in ADS}
 csv('fixed_baseline_search.csv',pd.DataFrame(fixed));save('fixed_baseline.json',dict(selection=selected,bias=bias,pool=pool))
 status('collect_representative_training_states')
 raw=[]
 for p in TRAIN:
  for ad in ADS:
   _,r=rollout(episode(p,ad),pool,base,bias[ad],collect=True)
   raw.extend(r)
 save('state_sampling.json',dict(seed=MASTER,sampling='3 of 8 prespecified decision times per advertiser-period, without replacement. Exhausted states ineligible; full episodes always evaluated.',states=len(raw),coverage=[dict(period=p,advertiser=a,states=sum(r['period']==p and r['ad']==a for r in raw)) for p in TRAIN for a in ADS]))
 if len(raw)<MIN_STATES:raise ValueError('Insufficient eligible training contexts')
 initial=[label(r,pool,base,bias[r['ad']],None,'baseline') for r in raw]
 candidates=[];audit=[];models={}
 for family in ['policy_tree','forest_cost','time_only']:
  for penalty in [.1,1.]:
   key=f'{family}_{penalty}';status('fit_'+key)
   initial_model=c.fit(family,initial,base,penalty)
   # Roll in under the learned policy, then label ALL states with the SAME
   # learned-policy continuation. Do not mix baseline-Q and policy-Q targets.
   own=[]
   for p in TRAIN:
    for ad in ADS:
     _,r=rollout(episode(p,ad),pool,base,bias[ad],initial_model,True,key);own.extend(r)
   combined=raw+own
   labelled=[label(r,pool,base,bias[r['ad']],initial_model,key) for r in combined]
   model=c.fit(family,labelled,base,penalty);joblib.dump(model,OUT/(key+'.joblib'));models[key]=model
   audit.append(c.audit(model,labelled,base,penalty,'train_policy_Q',key))
   f=evaluate(TUNE,pool,base,bias,model);csv(key+'_selection_episodes.csv',f);a=avg(f)
   candidates.append(dict(key=key,family=family,penalty=penalty,te=float(a.te),j=float(a.j),value=float(a['value'])))
   if isinstance(model,c.CostTree):save(key+'_rules.json',dict(features=v.F+['log_budget','log_cpa','bias'],gains=pool,tree=model.tree))
   csv('learning_audit.csv',pd.DataFrame(audit));csv('policy_selection.csv',pd.DataFrame(candidates))
 def choose(options):return min(options,key=lambda z:(max(0,.98*selected['value']-z['value']),z['j'],z['key']))
 winner=choose([z for z in candidates if z['family']!='time_only']);timewinner=choose([z for z in candidates if z['family']=='time_only'])
 treewinner=choose([z for z in candidates if z['family']=='policy_tree'])
 enabled=bool(winner['value']>=.98*selected['value'] and winner['te']<=.98*selected['te'] and winner['j']<selected['j'])
 save('FROZEN_BEFORE_TEST.json',dict(winner=winner,time_only=timewinner,matched_full_tree=treewinner,enabled=enabled,baseline=selected))
 (OUT/'REPORT.md').write_text('# 설정 선택 완료\n\n최종 24~27 결과를 읽기 전에 설정을 고정했습니다.\n'+json.dumps(dict(winner=winner,enabled=enabled),ensure_ascii=False,indent=2))
 status('fetch_once_locked_final_test24_27')
 for pack in ['24-25','26-27']:fetch(pack)
 for p in TEST:
  if discover(p)!=ADS:raise ValueError('Final advertiser census differs')
 summaries=[];all_results=[]
 for name,model in [('fixed',None),('learned_raw',models[winner['key']]),('selected_policy',models[winner['key']] if enabled else None),('time_only',models[timewinner['key']]),('matched_full_tree',models[treewinner['key']])]:
  frame=evaluate(TEST,pool,base,bias,model);frame['method']=name;all_results.append(frame);a=avg(frame);summaries.append(dict(method=name,**a.to_dict()))
  csv('final_episodes.csv',pd.concat(all_results,ignore_index=True))
 result=pd.concat(all_results,ignore_index=True);summary=pd.DataFrame(summaries);csv('final_summary.csv',summary)
 csv('final_by_period.csv',result.groupby(['method','period'])[['te','value','j','unspent']].mean().reset_index())
 fixed=result[result.method=='fixed'][['period','advertiser','te','value']].rename(columns={'te':'fixed_te','value':'fixed_value'})
 paired=result.merge(fixed,on=['period','advertiser'],validate='many_to_one');paired['te_gain_pct']=100*(paired.fixed_te-paired.te)/paired.fixed_te.clip(lower=1e-12);paired['value_ratio']=paired.value/paired.fixed_value.clip(lower=1e-12)
 csv('paired_final.csv',paired)
 (OUT/'REPORT.md').write_text('# 고정 프로토콜 최종 결과\n\n16~21 학습, 22~23 설정 선택, 24~27 최종 평가. 광고주 48명 전체.\n\n'+summary.to_string(index=False)+'\n\n통계 단위는 기간이다. 광고주들이 시장을 공유하므로 192개 독립 시장으로 취급하지 않는다. 최종 기간 4개의 기간별·광고주별 대응 결과를 보고하며 유의성을 주장하지 않는다.\n검증에서 탈락하면 selected_policy는 고정 기준이며 learned_raw는 별도로 보고한다.\n고정 가격·기대 전환 재생이며 실제 시장의 인과적 효과와 동일하지 않다.\nQ 라벨은 후보 계수를 최대 6틱 적용한 뒤 현재 정책을 계속 실행해 계산한다. 한 번의 근사 정책 개선이며 최적성·단조 개선 보장은 없다.\n종료 후 24~27로 설정을 수정하면서 새로운 테스트라고 주장하면 안 된다.\n')
 save('DONE.json',dict(status='COMPLETE',enabled=enabled,test_consumed=True));status('all_done','COMPLETE')

def main():
 if '--status' in sys.argv:
  for n in ['STATUS.json','REPORT.md']:
   if (OUT/n).exists():print((OUT/n).read_text())
  return
 lock=(OUT/'run.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:print('ALREADY RUNNING',OUT);return
 if '--worker' in sys.argv:
  try:run()
  except Exception:
   error=traceback.format_exc();(OUT/'ERROR.txt').write_text(error)
   with (OUT/'REPORT.md').open('a') as f:f.write('\n\n실행 미완료:\n'+error)
   status('exception','INCOMPLETE');raise
  return
 if (OUT/'DONE.json').exists():print('ALREADY COMPLETE',OUT);return
 if shutil.disk_usage(OUT).free<24*2**30:raise RuntimeError('Full 48-advertiser experiment requires at least 24 GiB free disk space; no downloads started.')
 # Pre-register before any new outcome data are downloaded.
 save('PROTOCOL.json',dict(code=CODE,training=TRAIN,selection=TUNE,locked_test=TEST,advertisers='All48 census, not an arbitrary four-advertiser sample',state_seed=MASTER,
  primary='Period-macro TE improvement while retaining >=98% baseline mean expected conversions; 2% validation TE activation threshold',
  fixed_search='10 gain pairs x3 common offsets, validation only, train-only calibration',learners='Depth2 direct-cost policy tree,64-tree ExtraTrees,maxdepth5; matched time-only tree; two declared cost penalties',
  decision_schedule=ANCHORS,training_sampling='3 without-replacement timepoints per episode. Evaluations use all eligible decisions and all episodes.',
  cost='Remaining-to-day-end tracking+output variation plus shortfall penalty. Round0 fixed continuation, round1 current-policy continuation for every relabelled example.',
  uncertainty='Deterministic simulator; do not inflate N with overlapping windows or repeated identical seeds. Only four final market periods, no population significance claim.',
  sources={'gain_policy_v10.py':hashlib.sha256((ROOT/'gain_policy_v10.py').read_bytes()).hexdigest(),'gain_final_v9.py':hashlib.sha256((ROOT/'gain_final_v9.py').read_bytes()).hexdigest()}))
 v.g.d.self_test(v.g.d.load_lab())
 # Small actual cost-learning check before background launch.
 x=np.array([[i%2,0.] for i in range(40)]);cost=np.array([[0,1] if i%2==0 else [1,0] for i in range(40)],dtype=float)
 for cls in [c.CostTree,c.ForestCost]:
  model=cls().fit(x,cost,np.ones(40)/40)
  if any(model.act(xx)!=int(cc.argmin()) for xx,cc in zip(x,cost)):raise ValueError('Learner smoke test failed')
 log=(OUT/'overnight.log').open('a');cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker']
 if shutil.which('caffeinate'):cmd=['caffeinate','-i']+cmd
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print('PROTOCOL LOCKED; STARTED PID:',proc.pid,'\nOUTPUT:',OUT,flush=True)
if __name__=='__main__':main()
