import unittest,math
from dataclasses import replace
from env import *
from controller import RefinedPI,slope_features
from fast_refined import run,make_features,TRACE
from experiment import calibrate_center,choose

def dummy_model():
    return dict(mean=[0.]*18,scale=[1.]*18,coef=np.linspace(-.005,.005,18).tolist(),intercept=math.log(.015))

class RefinementChecks(unittest.TestCase):
    def test_identity_is_frozen_CMA(self):
        from fast_eval import run as original
        for day in PROBE[:2]:
            a,tr0=original(day,rt.baseline_specs()['team_cma_sum']);b,tr1=run(day,spec())
            np.testing.assert_allclose(tr1[:,:6],tr0,rtol=0,atol=1e-12)
            np.testing.assert_allclose([a[m] for m in METRICS],[b[m] for m in METRICS],rtol=0,atol=1e-14)

    def test_legacy_is_previous_sensitivity_policy(self):
        ns={'rt':rt,'np':np}
        rt.definitions(FIRST/'code/controller.py',['correction','SensitivityPI'],ns)
        sp=spec(4);old={**sp,'strength':.25}
        ep=rt.episode(PROBE[0]);a,ra=rt.core.rollout(ep,rt.cfg,ns['SensitivityPI'](old))
        b,rb=rt.core.rollout(ep,rt.cfg,RefinedPI(sp))
        np.testing.assert_allclose([r['action'] for r in ra],[r['action'] for r in rb],rtol=0,atol=1e-12)

    def test_fast_vs_independent_all_methods_strengths_targets(self):
        ep=rt.episode(PROBE[0])
        for method in [6,7,8]:
            for power in [.25,.5,1.]:
                for target in [.75,1.,1.25]:
                    sp=spec(method,power,target,power*math.log(.012),model=dummy_model())
                    fast,tr=run(ep.day,sp);slow,rows=rt.core.rollout(ep,rt.cfg,RefinedPI(sp))
                    with self.subTest(method=method,power=power,target=target):
                        np.testing.assert_allclose([fast[m] for m in METRICS],[slow[m] for m in METRICS],rtol=0,atol=1e-10)
                        for col in ['action','ki','g_est','factor','gate','integral']:
                            np.testing.assert_allclose(tr[:,TRACE.index(col)],[r[col] for r in rows],rtol=1e-9,atol=1e-10)
                        self.assertTrue(np.all((tr[:,0]>=0)&(tr[:,0]<=1)))
                        self.assertTrue(np.all((tr[:,10]>=.5)&(tr[:,10]<=2)))
                        self.assertLessEqual(tr[-1,2],ep.budget+1e-9)

    def test_causal_feature_vectors_match_compiled(self):
        h=tuple(rt.core.Feedback(t,.2+.04*t,2+t,10+t,2+t,1+.1*t,t==4) for t in range(12))
        obs=rt.core.Observation(12,288,1000,40,h);arr=np.zeros((288,7))
        for f in h:arr[f.tick]=[1,f.action,f.spend/1000,math.log1p(f.opportunities),f.wins/f.opportunities,f.base_mean,float(f.budget_censored)]
        np.testing.assert_allclose(make_features(arr,12,1000,.4,obs.error),slope_features(obs,.4),atol=1e-12)

    def test_calibration_matches_mean_with_fallback_and_gating(self):
        gain=np.linspace(.003,.03,100);gate=np.linspace(.2,1,100);valid=np.arange(100)>6
        for p in [.25,.5,1.]:
            for target in [.75,1.,1.25]:
                center,value=calibrate_center(gain,gate,valid,p,target)
                self.assertAlmostEqual(value,target,places=9)

    def test_future_auctions_do_not_change_past_actions(self):
        ep=rt.episode(PROBE[0]);cut=40
        other=replace(ep,tape=ep.tape[:cut]+tuple(replace(t,competitor=t.competitor*4) for t in ep.tape[cut:]))
        for method in [7,8]:
            sp=spec(method,center=-1.2,model=dummy_model())
            _,a=rt.core.rollout(ep,rt.cfg,RefinedPI(sp));_,b=rt.core.rollout(other,rt.cfg,RefinedPI(sp))
            np.testing.assert_array_equal([r['action'] for r in a[:cut]],[r['action'] for r in b[:cut]])

    def test_budget_censored_feedback_cannot_drive_estimator(self):
        h=tuple(rt.core.Feedback(t,.5,1e7,10,10,1,True) for t in range(12))
        obs=rt.core.Observation(12,288,1000,12/288*1000,h)
        for method in [6,7,8]:
            pi=RefinedPI(spec(method));pi.reset();_,d=pi.act(obs)
            self.assertTrue(d['fallback']);self.assertEqual(d['factor'],1.)

    def test_integral_state_not_rescaled(self):
        zero=rt.core.Feedback(-1,0,0,0,0,0,False);obs=rt.core.Observation(0,288,1000,0,(zero,)*12)
        a=RefinedPI(spec());b=RefinedPI(spec(5,constant=2));a.reset();b.reset();a.integral=b.integral=.2
        x,dx=a.act(obs);y,dy=b.act(obs)
        self.assertEqual(x,y);self.assertEqual(dx['integral'],dy['integral'])

    def test_selection_and_split_guards(self):
        scores={'S0_CMA':dict(tracking_error=1,unspent=1,objective=2,multiplier_tv=1),
          'bad':dict(tracking_error=1.1,unspent=.5,objective=1.6,multiplier_tv=.5)}
        self.assertEqual(choose(['bad'],scores)['primary'],'S0_CMA')
        for day in [200000,520000,530400]:
            with self.assertRaises(ValueError):run(day,spec())

if __name__=='__main__':unittest.main()
