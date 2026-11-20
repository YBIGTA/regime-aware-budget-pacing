"""Causal remaining-time feedforward and risk-conditioned Ki restoration."""
import math
from env import rt,np,PRIOR

def usable(obs):
    return [f for f in obs.history if f.tick>=0 and not f.budget_censored and f.action>1e-4]

def spend_targets(obs,blend):
    need=max(0.,1-obs.cumulative_spend/obs.budget)/(obs.ticks-obs.tick)
    return need,(1-blend)/obs.ticks+blend*need

def base_and_slope(obs,target):
    if obs.tick==0:return rt.cfg.initial_multiplier,0.,rt.cfg.initial_multiplier,0.,False
    batch=np.repeat(rt.lib.features(obs,0.)[None,:],len(rt.lib.grid),axis=0)
    batch[:,0]=rt.lib.grid
    normalized=rt.scaler.transform(batch).astype(np.float32)
    pred=np.maximum(rt.linear.predict(normalized)*rt.y_std+rt.y_mean,0)
    chosen=int(np.argmin(np.abs(pred.astype(float)-target)))
    nominal=int(np.argmin(np.abs(pred.astype(float)-1/obs.ticks)))
    left,right=max(0,chosen-5),min(len(rt.lib.grid)-1,chosen+5)
    slope=(float(pred[right])-float(pred[left]))/(rt.lib.grid[right]-rt.lib.grid[left])
    return float(rt.lib.grid[chosen]),slope,float(rt.lib.grid[nominal]),float(pred[chosen]),bool(target>float(pred.max()))

def restoration_weight(error,need,forecast,n,ramp):
    if n<6 or error<=0 or need<=0:return 0.
    return float(np.clip((need-forecast)/(ramp*need),0,1))

def restore_factor(factor,weight):return factor+weight*max(0.,1-factor)

class DeadlinePI(rt.StatePI):
    def __init__(self,sp):
        super().__init__(sp['anchor'],sp['weights'],True);self.sp=sp
    def reset(self):
        super().reset();self.smoothed=PRIOR
    def act(self,obs):
        sp=self.sp;need,target=spend_targets(obs,sp['target_blend'])
        base,slope,nominal,pred,above=base_and_slope(obs,target)
        x=np.append(rt.lib.state_features(obs,base,slope),1. if obs.error>=.005 else -1.)
        kp,ki0=self.net(x);h=usable(obs);valid=len(h)>=6
        gain=PRIOR;gate=1.;factor=1.;clipped=False;forecast=0.
        if valid:
            a=np.array([f.action for f in h]);y=np.array([f.spend/obs.budget for f in h])
            ratio=float(y.sum()/(1e-6+a.sum()));forecast=float(y.mean())
            self.smoothed+=(ratio-self.smoothed)*sp['ema_alpha'];gain=self.smoothed
            if sp['method']==7:
                stderr=float(np.sqrt(len(h)/(len(h)-1)*np.sum((y-ratio*a)**2))/a.sum())
                gate=1/(1+(stderr/max(ratio,1e-8)/.25)**2)
        fallback=not valid or not np.isfinite(gain) or gain<0
        if not fallback:
            raw=sp['target']*math.exp(gate*(sp['center']-sp['power']*math.log(max(gain,1e-8))))
            factor=float(np.clip(raw,.5,2));clipped=raw<.5 or raw>2
        if sp['method']==0:factor=1.;fallback=clipped=False
        before=factor
        weight=restoration_weight(obs.error,need,forecast,len(h),sp['risk_ramp']) if sp['risk_restore'] else 0.
        factor=restore_factor(factor,weight)
        ki=float(np.clip(ki0*factor,0,6000));candidate=self.integral+ki*obs.error/obs.ticks
        raw=base+kp*obs.error+candidate
        blocked=(raw>1 and obs.error>0) or (raw<0 and obs.error<0)
        if blocked:raw=base+kp*obs.error+self.integral
        else:self.integral=candidate
        return float(np.clip(raw,0,1)),dict(kp=float(kp),ki=ki,integral=self.integral,base=base,slope=slope,
          ki_original=float(ki0),g_est=gain,factor=factor,gate=gate,fallback=fallback,
          factor_clipped=clipped,integral_blocked=blocked,saturated=raw<0 or raw>1,
          required_rate=need,target_rate=target,forecast_rate=forecast,risk_weight=weight,
          factor_before=before,target_above_prediction=above,base_nominal=nominal,
          predicted_at_base=pred,restoration_applied=factor>before+1e-12)
