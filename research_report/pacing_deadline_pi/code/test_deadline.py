import unittest,math
from dataclasses import replace
from unittest.mock import patch
from env import *
from controller import DeadlinePI,usable,spend_targets,restoration_weight,restore_factor
from fast_deadline import run,TRACE
from experiment import diagnose,MAIN

def observation(t=12,spent=None,history=None):
    zero=rt.core.Feedback(-1,0,0,0,0,0,False)
    return rt.core.Observation(t,288,1000,t/288*1000 if spent is None else spent,
      (zero,)*12 if history is None else history)

class DeadlineChecks(unittest.TestCase):
    def test_frozen_CMA_and_G(self):
        ns=dict(rt=rt,np=np,PRIOR=PRIOR,math=math,usable=usable)
        rt.definitions(PARENT/'code/controller.py',['HypothesisPI'],ns)
        ep=rt.episode(PROBE[0])
        for k in ['CMA','G']:
            sp=policies()[k];old=rt.policy(sp) if k=='CMA' else ns['HypothesisPI'](sp)
            a,ra=rt.core.rollout(ep,rt.cfg,old);b,rb=rt.core.rollout(ep,rt.cfg,DeadlinePI(sp))
            np.testing.assert_allclose([r['action'] for r in ra],[r['action'] for r in rb],rtol=0,atol=1e-12)
            np.testing.assert_allclose([a[m] for m in METRICS],[b[m] for m in METRICS],rtol=0,atol=1e-12)

    def test_all_policies_fast_vs_reference(self):
        for day in PROBE[:2]:
            ep=rt.episode(day)
            for k,sp in policies().items():
                fast,tr=run(day,sp);slow,rows=rt.core.rollout(ep,rt.cfg,DeadlinePI(sp))
                with self.subTest(day=day,model=k):
                    np.testing.assert_allclose([fast[m] for m in METRICS],[slow[m] for m in METRICS],rtol=0,atol=1e-10)
                    for col in ['action','ki','integral','factor','required_rate','target_rate','forecast_rate','risk_weight','base','base_nominal','predicted_at_base']:
                        np.testing.assert_allclose(tr[:,TRACE.index(col)],[r[col] for r in rows],rtol=1e-9,atol=1e-10)
                    self.assertTrue(np.all((tr[:,0]>=0)&(tr[:,0]<=1)))
                    self.assertTrue(np.all((tr[:,4]>=0)&(tr[:,4]<=6000)))
                    self.assertLessEqual(tr[-1,2],ep.budget+1e-8)

    def test_target_boundary_and_catchup(self):
        for t in [0,12,240,287]:
            for blend in [0.,.5,1.]:
                q,target=spend_targets(observation(t),blend)
                self.assertAlmostEqual(q,1/288);self.assertAlmostEqual(target,1/288)
        obs=observation(240,spent=800)
        q,target=spend_targets(obs,.5)
        self.assertAlmostEqual(q,.2/48);self.assertAlmostEqual(target,(1/288+.2/48)/2)
        obs=observation(287,spent=990);self.assertAlmostEqual(spend_targets(obs,1.)[1],.01)
        self.assertEqual(spend_targets(observation(287,spent=1000),1.)[1],0.)
        self.assertLess(spend_targets(observation(240,spent=900),1.)[1],1/288)

    def test_risk_gates_and_monotonicity(self):
        for err,n,need in [(0.,12,.01),(-.1,12,.01),(.1,5,.01),(.1,12,0.)]:
            self.assertEqual(restoration_weight(err,need,0.,n,.25),0.)
        self.assertEqual(restoration_weight(.01,.01,.012,12,.25),0.)
        weights=[restoration_weight(.01,.01,f,12,.25) for f in [.01,.009,.008,.0075,.005,0.]]
        self.assertEqual(weights,sorted(weights));self.assertEqual(weights[-1],1.)
        self.assertAlmostEqual(restoration_weight(.01,.01,.00875,12,.25),.5)
        for factor in [.5,.75,1.,1.5,2.]:
            for w in [0.,.5,1.]:
                value=restore_factor(factor,w);self.assertGreaterEqual(value,factor)
                self.assertLessEqual(value,max(1.,factor))
                if factor>=1:self.assertEqual(value,factor)

    def test_censored_observations_cannot_trigger_restore(self):
        h=tuple(rt.core.Feedback(t,.5,0,10,0,1,True) for t in range(12))
        obs=observation(12,spent=1,history=h)
        p=DeadlinePI(policies()['G_R']);p.reset();_,d=p.act(obs)
        self.assertEqual(d['risk_weight'],0);self.assertEqual(d['factor'],1);self.assertTrue(d['fallback'])

    def test_weighted_integral_is_preserved(self):
        h=tuple(rt.core.Feedback(t,.5,0,10,0,1,False) for t in range(12))
        obs=observation(12,spent=(12/288-.001)*1000,history=h)
        p=DeadlinePI(policies()['G_R']);p.reset();p.integral=.1;p.net=lambda x:(0.,100.)
        with patch('controller.base_and_slope',return_value=(.2,0.,.2,1/288,False)):
            _,d=p.act(obs)
        self.assertEqual(d['factor'],1.)
        self.assertAlmostEqual(d['integral'],.1+d['ki']*obs.error/288)

    def test_future_data_does_not_change_actions(self):
        ep=rt.episode(PROBE[0]);cut=50
        future=replace(ep,tape=ep.tape[:cut]+tuple(replace(t,competitor=t.competitor*5) for t in ep.tape[cut:]))
        for k in ['CMA_T05','G_R','G_T05_R','G_T1_R']:
            sp=policies()[k];_,a=rt.core.rollout(ep,rt.cfg,DeadlinePI(sp));_,b=rt.core.rollout(future,rt.cfg,DeadlinePI(sp))
            np.testing.assert_array_equal([r['action'] for r in a[:cut]],[r['action'] for r in b[:cut]])

    def test_factors_and_original_target_path(self):
        for k,sp in policies().items():
            score,tr=run(PROBE[0],sp)
            self.assertTrue(np.all(tr[:,10]>=tr[:,22]-1e-12))
            self.assertAlmostEqual(np.abs(tr[:,1]).mean(),score['tracking_error'],places=14)
            d=diagnose(tr);self.assertAlmostEqual(d['remaining_pre_final']-d['final_spend'],score['unspent'],places=13)
        specs=policies()
        for a,b in [('CMA','G'),('CMA_T05','G_T05'),('CMA_T1','G_T1')]:
            _,x=run(PROBE[0],specs[a]);_,y=run(PROBE[0],specs[b]);np.testing.assert_allclose(x[:6,:6],y[:6,:6],rtol=0,atol=1e-12)

    def test_only_requested_switches_differ(self):
        s=policies()
        for a,b,changes in [('G_T05','G',{'target_blend'}),('G_R','G',{'risk_restore'}),
          ('G_T05_R','G',{'target_blend','risk_restore'})]:
            self.assertEqual({k for k in s[a] if s[a][k]!=s[b][k]},changes)
        self.assertEqual(len(s),9);self.assertEqual(len(MAIN),4)

    def test_sealed_dates(self):
        for day in [200000,520000,530000,540000,551000,450000,460000]:
            with self.assertRaises(ValueError):run(day,policies()['CMA'])

if __name__=='__main__':unittest.main()
