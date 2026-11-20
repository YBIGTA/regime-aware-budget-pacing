"""Causal Ki-only refinements. The state-conditioned model forecasts sensitivity;
its training labels come from development auction tapes, never live observations.
"""
import math
from env import rt,np,PRIOR

def usable(obs):
    return [f for f in obs.history if f.tick>=0 and not f.budget_censored and f.action>1e-4]

def slope_features(obs,base):
    h=usable(obs)
    if not h:return np.zeros(18)
    a=np.array([f.action for f in h]);y=np.array([f.spend/obs.budget for f in h])
    count=np.array([f.opportunities for f in h])
    am=float(a.mean());wr=float(np.mean([f.wins/max(1,f.opportunities) for f in h]))
    lc=float(np.log1p(count).mean());bv=float(np.mean([f.base_mean for f in h]))
    ym=float(y.mean());lr=math.log(max(float(y.sum()/(1e-6+a.sum())),1e-8))
    return np.array([base,base*base,math.log(obs.budget),math.sin(2*math.pi*obs.tick/obs.ticks),
       math.cos(2*math.pi*obs.tick/obs.ticks),am,wr,lc,bv,ym,lr,float(a.std()),
       float(count.std()/max(1.,count.mean())),base*wr,base*lc,base*lr,base*bv,obs.error])

def predict_slope(features,model):
    x=np.clip((features-np.asarray(model['mean']))/np.asarray(model['scale']),-8,8)
    return float(np.exp(np.clip(model['intercept']+x@np.asarray(model['coef']),math.log(1e-5),math.log(.2))))

class RefinedPI(rt.StatePI):
    def __init__(self,sp):
        super().__init__(sp['anchor'],sp['weights'],True)
        self.sp=sp
    def reset(self):
        super().reset();self.smoothed=PRIOR
    def act(self,obs):
        sp=self.sp;base,slope=rt.ns['base_and_slope'](obs)
        x=np.append(rt.lib.state_features(obs,base,slope),1. if obs.error>=.005 else -1.)
        kp,ki0=self.net(x);h=usable(obs);valid=len(h)>=6
        gain=PRIOR;gate=1.;factor=1.;clipped=False
        if valid:
            a=np.array([f.action for f in h]);y=np.array([f.spend/obs.budget for f in h])
            ratio=float(y.sum()/(1e-6+a.sum()))
            if sp['method']==8:raw_gain=predict_slope(slope_features(obs,base),sp['slope_model'])
            else:raw_gain=ratio
            self.smoothed+=(raw_gain-self.smoothed)*sp['ema_alpha'];gain=self.smoothed
            if sp['method']==7:
                stderr=float(np.sqrt(len(h)/(len(h)-1)*np.sum((y-ratio*a)**2))/a.sum())
                gate=1/(1+(stderr/max(ratio,1e-8)/.25)**2)
        fallback=not valid or not np.isfinite(gain) or gain<0
        if not fallback:
            if sp['method']==4:raw=(PRIOR/max(gain,.1*PRIOR))**sp['power']
            else:raw=sp['target']*math.exp(gate*(sp['center']-sp['power']*math.log(max(gain,1e-8))))
            factor=float(np.clip(raw,.5,2));clipped=raw<.5 or raw>2
        if sp['method']==0:factor=1.;fallback=clipped=False
        if sp['method']==5:factor=sp['constant'];fallback=clipped=False
        ki=float(np.clip(ki0*factor,0,6000));candidate=self.integral+ki*obs.error/obs.ticks
        raw=base+kp*obs.error+candidate
        blocked=(raw>1 and obs.error>0) or (raw<0 and obs.error<0)
        if blocked:raw=base+kp*obs.error+self.integral
        else:self.integral=candidate
        return float(np.clip(raw,0,1)),dict(kp=float(kp),ki=ki,integral=self.integral,base=base,
          slope=slope,ki_original=float(ki0),g_est=gain,factor=factor,gate=gate,
          fallback=fallback,factor_clipped=clipped,integral_blocked=blocked,saturated=raw<0 or raw>1)
