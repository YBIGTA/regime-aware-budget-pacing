"""Audited full-information decision-cost learning; adapted policy learning, not exact paper reproduction."""
import os,sys,json,time,hashlib,traceback,subprocess,fcntl,shutil,importlib.util,zipfile
from pathlib import Path
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesRegressor
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'gain_final_v9.py'
EXPECTED='0972308be6d82f5527bee995acf1010be1e3d4422835322f4688ec13d613274a'
if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=EXPECTED:raise RuntimeError('Expected unmodified v9 source; preserve original source beside this file')
spec=importlib.util.spec_from_file_location('audited_v9',SOURCE);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
CODE=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
OUT=v.LAB/'runs'/('gain_policy_v10_'+CODE[:8]+('_fresh' if '--fresh-test' in sys.argv else '_dev'));OUT.mkdir(parents=True,exist_ok=True)
CACHE=OUT/'cache';CACHE.mkdir(exist_ok=True)

def save(name,obj):v.jsave(OUT/name,obj)
def csv(name,frame):v.csvsave(OUT/name,frame)
def status(s,stage):save('STATUS.json',dict(status=s,stage=stage));print(s,stage,flush=True)
def cache(name,fn):
 p=CACHE/(name+'.joblib')
 if p.exists():
  try:return joblib.load(p)
  except Exception:p.rename(p.with_suffix('.broken_'+str(time.time_ns())))
 result=fn();tmp=p.with_suffix('.tmp');joblib.dump(result,tmp,compress=3);tmp.replace(p);return result

@lru_cache(maxsize=2)
def ep(period,ad):
 def load():
  if period<=13:
   old=v.OUT/'cache'/f'episode_{period}_{ad}.joblib'
   if old.exists():
    try:return joblib.load(old)
    except Exception:pass
   path=v.file_inventory()[(period,ad)]
  else:path=OUT/f'fresh_data/pack-14-15/p{period}-a{ad}.parquet'
  df=pd.read_parquet(path)
  if 'advertiserNumber' in df:df=df.rename(columns={'advertiserNumber':'advertiser'})
  return v.g.d.load_lab().prepare_episode(df,42)
 return cache(f'ep_{period}_{ad}',load)

def weights(rows):
 groups=[(r['period'],r['ad'],r['prefix']) for r in rows]
 periods=sorted(set(k[0] for k in groups));unique=set(groups)
 sizes={k:groups.count(k) for k in unique}
 cases={p:sum(k[0]==p for k in unique) for p in periods}
 w=np.array([1/(len(periods)*cases[k[0]]*sizes[k]) for k in groups]);return w/w.sum()

def compute_labels(episode,start,state,bias,pool,base):
 js=[];vs=[]
 for kp,ki in pool:
  a,states=v.g.roll(episode,dict(kp=kp,ki=ki),bias,start,start+6,state)
  if start+6<48:
   b,_=v.g.roll(episode,dict(kp=pool[base][0],ki=pool[base][1]),bias,start+6,48,states[start+6]);a=pd.concat([a,b],ignore_index=True)
  v.check_trace(a,state[0])
  js.append(float(a.error.abs().sum()/48+.01*a.du.sum()/47))
  vs.append(float(a.value.sum()))
 return js,vs

def labels(episode,start,state,bias,pool,base):
 signature=hashlib.sha256(np.array([*state,bias,*np.asarray(pool).ravel(),base],dtype=np.float64).tobytes()).hexdigest()[:20]
 return cache(f'counter_{episode["period"]}_{episode["ad"]}_{start}_{signature}',lambda:compute_labels(episode,start,state,bias,pool,base))

def rollout(episode,pool,base,bias,policy=None,collect=False,prefix='fixed'):
 parts=[];state=None;start=0;rows=[]
 while start<48:
  action=base;x=None
  if start>=6 and state[0]>=.1:
   x=v.feature(episode,start,state,pd.concat(parts,ignore_index=True))
   if collect:
    j,val=labels(episode,start,state,bias,pool,base)
    rows.append(dict(period=episode['period'],ad=episode['ad'],prefix=prefix,start=start,x=x.tolist(),j=j,value=val))
   if policy is not None:action=policy.act(x)
  f,s=v.g.roll(episode,dict(kp=pool[action][0],ki=pool[action][1]),bias,start,start+6,state)
  f['candidate']=action;parts.append(f);state=s[start+6];start+=6
 f=pd.concat(parts,ignore_index=True);v.check_trace(f,episode['budget']);return f,rows

def costs(rows,base,penalty):
 j=np.array([r['j'] for r in rows]);value=np.array([r['value'] for r in rows])
 # Absolute expected-value shortfall divided by max(future baseline value,1):
 # stable near zero; no clipping and no action-specific deployment gate.
 c=j+penalty*np.maximum(0,.98*value[:,[base]]-value)/np.maximum(value[:,[base]],1.)
 return c-c[:,[base]]

class CostTree:
 def fit(self,x,c,w):
  self.x=x;self.c=c;self.w=w
  def node(ids,depth):
   sums=(w[ids,None]*c[ids]).sum(0);action=int(sums.argmin());base=getattr(self,'base',0)
   if sums[base]<=sums[action]+1e-10:action=base
   risk=float(sums[action]);best=None
   if depth<2 and len(ids)>=20:
    for f in range(x.shape[1]):
     for cut in np.unique(np.quantile(x[ids,f],[.2,.4,.6,.8])):
      left=ids[x[ids,f]<=cut];right=ids[x[ids,f]>cut]
      if min(len(left),len(right))<10:continue
      loss=float((w[left,None]*c[left]).sum(0).min()+(w[right,None]*c[right]).sum(0).min())
      if loss<risk-1e-10 and (best is None or loss<best[0]):best=(loss,f,float(cut),left,right)
   if best is None:return dict(action=action,n=len(ids))
   return dict(feature=best[1],cut=best[2],left=node(best[3],depth+1),right=node(best[4],depth+1))
  self.tree=node(np.arange(len(x)),0);del self.x,self.c,self.w;return self
 def act(self,x):
  node=self.tree
  while 'action' not in node:node=node['left'] if x[node['feature']]<=node['cut'] else node['right']
  return node['action']

class SoftCost:
 def fit(self,x,c,w):
  self.center=x.mean(0);self.scale=x.std(0);self.scale[self.scale<1e-8]=1
  z=np.c_[np.ones(len(x)),np.clip((x-self.center)/self.scale,-6,6)]
  k=c.shape[1];d=z.shape[1];scale=max(float(np.quantile(np.ptp(c,axis=1),.75)),.001);cn=c/scale
  def objective(theta):
   b=theta.reshape(d,k);pr=softmax(z@b,axis=1);expected=(pr*cn).sum(1)
   penalty=.001;loss=float(w@expected+.5*penalty*(b[1:]**2).sum())
   grad=z.T@(w[:,None]*pr*(cn-expected[:,None]));grad[1:]+=penalty*b[1:]
   return loss,grad.ravel()
  result=minimize(objective,np.zeros(d*k),jac=True,method='L-BFGS-B',options={'maxiter':300,'ftol':1e-10})
  if not np.isfinite(result.fun) or not np.isfinite(result.x).all():raise ValueError('Nonfinite soft policy optimization')
  self.coef=result.x.reshape(d,k);self.optimizer_message=str(result.message);self.optimizer_success=bool(result.success);return self
 def act(self,x):
  z=np.r_[1,np.clip((x-self.center)/self.scale,-6,6)];score=z@self.coef;base=getattr(self,'base',0)
  return base if score[base]>=score.max()-1e-10 else int(score.argmax())

class ForestCost:
 def fit(self,x,c,w):
  self.model=ExtraTreesRegressor(n_estimators=64,max_depth=5,min_samples_leaf=5,random_state=42,n_jobs=2).fit(x,c,sample_weight=w);self.model.n_jobs=1;return self
 def act(self,x):
  score=self.model.predict(x[None,:])[0];base=getattr(self,'base',0)
  return base if score[base]<=score.min()+1e-10 else int(score.argmin())

class TimeCost(CostTree):
 def fit(self,x,c,w):return super().fit(x[:,[0]],c,w)
 def act(self,x):return super().act(x[[0]])

def fit(name,rows,base,penalty):
 model={'policy_tree':CostTree,'soft_cost':SoftCost,'forest_cost':ForestCost,'time_only':TimeCost}[name]()
 model.base=base
 x=np.array([r['x'] for r in rows]);c=costs(rows,base,penalty);w=weights(rows)
 model.fit(x,c,w)
 if isinstance(model,CostTree):
  chosen=np.array([model.act(a) for a in x]);risk=float(w@c[np.arange(len(c)),chosen]);constant=float((w[:,None]*c).sum(0).min())
  if risk>constant+1e-8:raise ValueError('Direct-cost tree training risk exceeds the best constant action')
 return model

def audit(model,rows,base,penalty,split,name):
 c=costs(rows,base,penalty);x=np.array([r['x'] for r in rows]);pred=np.array([model.act(a) for a in x]);w=weights(rows);chosen=c[np.arange(len(c)),pred];oracle=c.min(1)
 return dict(method=name,split=split,states=len(rows),mean_cost=float(w@chosen),oracle_cost=float(w@oracle),
  decision_regret=float(w@(chosen-oracle)),switch_fraction=float(w@(pred!=base)),solver_message=getattr(model,'optimizer_message','not iterative'),solver_success=getattr(model,'optimizer_success',True))

def evaluate(periods,pool,base,bias,model=None):
 records=[]
 for p in periods:
  for ad in v.ADS:
   f,_=rollout(ep(p,ad),pool,base,bias[ad],model)
   records.append(dict(period=p,advertiser=ad,**v.metric(f,ep(p,ad)['budget']),changed_ticks=int((f.candidate!=base).sum())))
 return pd.DataFrame(records)

def aggregate(frame):return frame.groupby('period')[['te','value','j','unspent','changed_ticks']].mean().mean()

def fresh_data():
 if shutil.disk_usage(OUT).free<4*2**30:raise RuntimeError('Fresh evaluation needs at least 4 GiB free disk space')
 lab=v.g.d.load_lab();previous=lab.DATA;lab.DATA=OUT/'fresh_data';lab.DATA.mkdir(exist_ok=True)
 try:
  for attempt in range(3):
   try:lab.fetch(SimpleNamespace(packs=['14-15'],advertisers=v.ADS,keep_archives=False));return
   except Exception as error:
    if isinstance(error,zipfile.BadZipFile):
     (lab.DATA/'period-14-15.zip').unlink(missing_ok=True)
    if attempt==2:raise
    print('RETRY fresh download',attempt+1,flush=True);time.sleep(5)
 finally:lab.DATA=previous

def work(fresh):
 _,_,bias,pool,original=v.context();old=v.OUT
 frozen=json.loads((old/'baseline.json').read_text());base=int(frozen['candidate'])
 if tuple(pool[base])!=(frozen['kp'],frozen['ki']):raise ValueError('Gain dictionary mismatch')
 status('RUNNING','preparing_train_labels')
 rows=[]
 for p in v.TRAIN:
  for ad in v.ADS:
   def build(p=p,ad=ad):
    result=[]
    # Collect states from fixed and half-Kp control without changing initialization.
    f,states=v.g.roll(ep(p,ad),dict(kp=pool[base][0]*.5,ki=pool[base][1]),bias[ad])
    _,r=rollout(ep(p,ad),pool,base,bias[ad],collect=True);result.extend(r)
    for start in v.ANCHORS:
     if states[start][0]<.1:continue
     j,val=labels(ep(p,ad),start,states[start],bias[ad],pool,base)
     result.append(dict(period=p,ad=ad,prefix='mild',start=start,x=v.feature(ep(p,ad),start,states[start],f.iloc[:start]).tolist(),j=j,value=val))
    return result
   rows.extend(cache(f'labels_{p}_{ad}',build));print('LABELLED',p,ad,flush=True)
 if len(rows)<20:raise ValueError('Too few eligible states')
 csv('baseline_tuning.csv',evaluate(v.TUNE,pool,base,bias))
 baseline=aggregate(evaluate(v.TUNE,pool,base,bias));candidates=[];audits=[]
 for name in ['policy_tree','soft_cost','forest_cost','time_only']:
  for penalty in [.1,1.]:
   status('RUNNING',f'fit_{name}_{penalty}')
   key=f'{name}_{penalty}'
   model=fit(name,rows,base,penalty)
   # One DAgger-inspired round on TRAIN ONLY. Counterfactual labels use
   # candidate for 6 ticks then baseline to end: not optimal Q, no DAgger guarantee.
   def aggregate_states():
    new=[]
    for p in v.TRAIN:
     for ad in v.ADS:
      _,r=rollout(ep(p,ad),pool,base,bias[ad],model,collect=True,prefix=key);new.extend(r)
    return new
   own=cache('own_'+key,aggregate_states);combined=rows+own
   model=fit(name,combined,base,penalty);joblib.dump(model,OUT/(key+'.joblib'))
   if isinstance(model,CostTree):save(key+'_rules.json',dict(features=['time_left'] if name=='time_only' else v.F,gains=pool,tree=model.tree))
   audits.append(audit(model,combined,base,penalty,'train',key))
   val=evaluate(v.TUNE,pool,base,bias,model);csv(key+'_tuning_episodes.csv',val);a=aggregate(val)
   candidates.append(dict(method=name,penalty=penalty,key=key,te=float(a.te),j=float(a.j),value=float(a['value'])))
   # Evaluation labels for fit/generalization diagnosis; NEVER appended to training.
   validation=[]
   for p in v.TUNE:
    for ad in v.ADS:
     _,r=rollout(ep(p,ad),pool,base,bias[ad],model,collect=True,prefix='diagnostic');validation.extend(r)
   audits.append(audit(model,validation,base,penalty,'tuning',key))
   csv('learning_audit.csv',pd.DataFrame(audits));csv('tuning_settings.csv',pd.DataFrame(candidates))
 def select(options):return min(options,key=lambda z:(max(0,.98*baseline['value']-z['value']),z['j'],z['key']))
 winner=select([c for c in candidates if c['method']!='time_only']);timewinner=select([c for c in candidates if c['method']=='time_only'])
 enabled=bool(winner['value']>=.98*baseline['value'] and winner['te']<=.98*baseline.te and winner['j']<baseline.j)
 save('FROZEN.json',dict(winner=winner,time_only=timewinner,enabled=enabled,baseline=base))
 selected=joblib.load(OUT/(winner['key']+'.joblib'));time_policy=joblib.load(OUT/(timewinner['key']+'.joblib'))
 def report(periods,label):
  pieces=[]
  for name,model in [('fixed',None),('learned_raw',selected),('selected_policy',selected if enabled else None),('time_only',time_policy)]:
   f=evaluate(periods,pool,base,bias,model);f['method']=name;pieces.append(f)
  result=pd.concat(pieces,ignore_index=True);csv(label+'_episodes.csv',result)
  summary=result.groupby(['method','period'])[['te','value','unspent','j','changed_ticks']].mean().groupby('method').mean().reset_index();csv(label+'_summary.csv',summary)
  return summary
 diagnostic=report([13],'development13')
 (OUT/'REPORT.md').write_text('# 학습기 재설계 결과\n\n기간 13은 개발 진단이며 최종 테스트가 아닙니다.\n\n'+diagnostic.to_string(index=False)+'\n\n학습/일반화 진단: learning_audit.csv. 결정 손실과 oracle regret, 실제 변경 횟수를 확인하세요.\n최종 모델 선택은 기간 11~12에서만 수행했습니다. 신선한 테스트의 지표는 선택에 사용하지 않습니다.\n')
 if fresh:
  status('RUNNING','fetching_frozen_test14_15');fresh_data();summary=report([14,15],'fresh_test14_15')
  with (OUT/'REPORT.md').open('a') as f:f.write('\n## 설정 고정 후 새 기간 14~15\n\n'+summary.to_string(index=False)+'\n\n두 기간·광고주 4명의 서술적 비교입니다. 유의성이나 전체 시장 일반화를 주장하지 않습니다.\n')
 save('DONE.json',dict(status='COMPLETE',fresh_test=fresh,enabled=enabled));status('COMPLETE','all_done')

def main():
 if '--status' in sys.argv:
  for n in ['STATUS.json','REPORT.md']:
   if (OUT/n).exists():print((OUT/n).read_text())
  return
 lock=(OUT/'run.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:print('ALREADY RUNNING',OUT);return
 if '--worker' in sys.argv:
  try:work('--fresh-test' in sys.argv)
  except Exception:
   error=traceback.format_exc();(OUT/'ERROR.txt').write_text(error)
   with (OUT/'REPORT.md').open('a') as f:f.write('\n## 실행 미완료\n\n'+error)
   status('INCOMPLETE','exception');raise
  return
 if (OUT/'DONE.json').exists():print('ALREADY COMPLETE',OUT);return
 # No mutation of the v9 experiment. Validate every existing episode first.
 status('PREFLIGHT','existing_data')
 manifest=json.loads((v.OUT/'manifest.json').read_text())
 for relative,digest in manifest['files'].items():
  if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=digest:raise ValueError('v9 source data changed: '+relative)
 for name,digest in manifest['dependencies'].items():
  if hashlib.sha256((v.LAB/name).read_bytes()).hexdigest()!=digest:raise ValueError('Replay dependency changed: '+name)
 v.g.d.self_test(v.g.d.load_lab())
 for p in v.TRAIN+v.TUNE+[13]:
  for ad in v.ADS:
   e=ep(p,ad)
   if (e['period'],e['ad'])!=(p,ad):raise ValueError('Episode mismatch')
   print('CHECKED',p,ad,flush=True)
 save('protocol.json',dict(code=CODE,source_v9=EXPECTED,train=v.TRAIN,tune=v.TUNE,development=[13],fresh_test=[14,15] if '--fresh-test' in sys.argv else [],
  source_manifest_sha256=hashlib.sha256((v.OUT/'manifest.json').read_bytes()).hexdigest(),
  cost='tail-to-day-end TE/48 + .01 log-output variation/47 + lambda*max(0,.98*future_baseline_value-future_value)/max(future_baseline_value,1)',
  methods=['greedy direct-cost tree depth2, not exact SPOT','linear softmax expected-cost objective, deterministic argmax deployment','ExtraTrees cost regression,64 trees'],
  aggregation='One policy-induced TRAIN-only state collection/relabel round; DAgger-inspired, no theory guarantee',
  limits='Simulator full-information adaptation; no doubly robust causal claim; fixed prices; 4 advertisers; one-step deviation followed by fixed baseline labels, not optimal Q; existing period13 is development only'))
 # Actual model-path smoke test, including cost-sensitive split and optimizer.
 x=np.array([[i%2,0.] for i in range(40)]);c=np.array([[0.,1.] if i%2==0 else [1.,0.] for i in range(40)]);w=np.ones(40)/40
 for cls in [CostTree,SoftCost,ForestCost]:
  m=cls().fit(x,c,w)
  if not all(m.act(x[i])==int(c[i].argmin()) for i in range(40)):raise ValueError('Separable learner test failed')
 log=(OUT/'overnight.log').open('a');cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker']
 if '--fresh-test' in sys.argv:cmd.append('--fresh-test')
 if shutil.which('caffeinate'):cmd=['caffeinate','-i']+cmd
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print('PREFLIGHT PASSED; STARTED PID:',proc.pid,'\nOUTPUT:',OUT,flush=True)
if __name__=='__main__':
 sys.modules['gain_policy_v10']=sys.modules[__name__]
 for cls in [CostTree,SoftCost,ForestCost,TimeCost]:cls.__module__='gain_policy_v10'
 main()
