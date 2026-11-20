"""Project adaptation of inverse plant-gain scaling, NOT the paper's full controller.

The reference implementation uses only Observation (completed feedback), retains
the weighted integral state, and changes only the current Ki contribution.
Methods: 0 identity; 1 B finite slope; 2 first-day cumulative ratio;
3 trailing-hour ratio; 4 trailing-hour ratio plus 30-minute EMA; 5 constant Ki.
The ratio estimator is biased and is not an unbiased local derivative estimate.
"""
from bridge import rt, np

def correction(g, g_ref, power, valid):
    if not valid or not np.isfinite(g) or g < 0:
        return 1., True, False
    raw=(g_ref / max(g, .1*g_ref))**power
    return float(np.clip(raw,.5,2.)), False, bool(raw<.5 or raw>2.)

class SensitivityPI(rt.StatePI):
    def __init__(self, spec):
        super().__init__(spec['anchor'],spec['weights'],True)
        self.spec=spec

    def reset(self):
        super().reset()
        self.sum_spend=self.sum_action=0.
        self.smoothed=self.spec['g_ref']

    def act(self, obs):
        sp=self.spec
        method=sp['method']; ref=sp['g_ref']
        base,slope=rt.ns['base_and_slope'](obs)
        x=rt.lib.state_features(obs,base,slope)
        x=np.append(x,1. if obs.error>=.005 else -1.)
        kp,ki0=self.net(x)
        if obs.tick:
            last=obs.history[-1]
            if not last.budget_censored and last.action>1e-4:
                self.sum_spend+=last.spend/obs.budget
                self.sum_action+=last.action
        gain=ref; valid=True
        if method==1:
            gain=slope; valid=obs.tick>0
        elif method==2:
            weight=1-obs.tick/obs.ticks
            gain=weight*ref+(1-weight)*self.sum_spend/(1e-6+self.sum_action)
            valid=self.sum_action>1e-4
        elif method in (3,4):
            usable=[f for f in obs.history if f.tick>=0 and not f.budget_censored and f.action>1e-4]
            numerator=sum(f.spend/obs.budget for f in usable)
            denominator=sum(f.action for f in usable)
            valid=len(usable)>=6 and denominator>1e-4
            if valid:
                gain=numerator/(1e-6+denominator)
                if method==4:
                    self.smoothed+=(gain-self.smoothed)*sp['ema_alpha']
                    gain=self.smoothed
        factor,fallback,clipped=correction(gain,ref,sp['strength'],valid)
        if method==0: factor,fallback,clipped=1.,False,False
        if method==5: factor,fallback,clipped=sp['constant'],False,False
        ki=float(np.clip(ki0*factor,0.,2*rt.gain_scale[1]))
        candidate=self.integral+ki*obs.error/obs.ticks
        raw=base+kp*obs.error+candidate
        blocked=(raw>1 and obs.error>0) or (raw<0 and obs.error<0)
        if blocked: raw=base+kp*obs.error+self.integral
        else: self.integral=candidate
        return float(np.clip(raw,0.,1.)), dict(kp=float(kp),ki=ki,
            saturated=bool(raw<0 or raw>1),base=base,slope=slope,ki_original=float(ki0),
            integral=self.integral,g_est=gain,g_ref=ref,factor=factor,
            fallback=fallback,factor_clipped=clipped,integral_blocked=blocked)
