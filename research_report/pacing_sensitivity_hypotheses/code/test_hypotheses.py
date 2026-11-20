import unittest,math
from dataclasses import replace
from env import *
from controller import HypothesisPI,usable,slope_features,predict_slope
from fast_hypotheses import run,TRACE
from experiment import diagnose,MAIN

class HypothesisChecks(unittest.TestCase):
    def test_frozen_parent_paths_unchanged(self):
        ns=dict(rt=rt,np=np,PRIOR=PRIOR,math=math,usable=usable,slope_features=slope_features,predict_slope=predict_slope)
        rt.definitions(PARENT/'code/controller.py',['RefinedPI'],ns)
        ep=rt.episode(PROBE[0])
        for k in ['CMA','G','C_all','R','L']:
            sp=policies()[k]
            _,a=rt.core.rollout(ep,rt.cfg,ns['RefinedPI'](sp))
            _,b=rt.core.rollout(ep,rt.cfg,HypothesisPI(sp))
            np.testing.assert_allclose([r['action'] for r in a],[r['action'] for r in b],rtol=0,atol=1e-12)

    def test_all_policies_fast_reference(self):
        for day in PROBE[:2]:
            ep=rt.episode(day)
            for k,sp in policies().items():
                fast,tr=run(day,sp);slow,rows=rt.core.rollout(ep,rt.cfg,HypothesisPI(sp))
                with self.subTest(day=day,model=k):
                    np.testing.assert_allclose([fast[m] for m in METRICS],[slow[m] for m in METRICS],rtol=0,atol=1e-10)
                    for col in ['action','ki','integral','g_est','factor','fallback']:
                        np.testing.assert_allclose(tr[:,TRACE.index(col)],[r[col] for r in rows],rtol=1e-9,atol=1e-10)
                    self.assertLessEqual(tr[-1,2],ep.budget+1e-8)

    def test_warm_controls_share_initial_trajectory(self):
        specs=policies();_,g=run(PROBE[0],specs['G'])
        for k in ['CMA','C_warm','C_rule','C_matched','G_neutral','G_first']:
            _,t=run(PROBE[0],specs[k]);np.testing.assert_allclose(g[:6,:6],t[:6,:6],rtol=0,atol=1e-12)
        _,t=run(PROBE[0],specs['C_all']);self.assertGreater(np.max(np.abs(t[:6,0]-g[:6,0])),1e-6)

    def test_late_fallback_excludes_censored_feedback(self):
        h=tuple(rt.core.Feedback(t,.5,100,10,5,1,True) for t in range(12))
        obs=rt.core.Observation(20,288,1000,20/288*1000,h)
        specs=policies()
        for k in ['G','G_first','R_neutral','L_noema','C_rule','C_matched']:
            p=HypothesisPI(specs[k]);p.reset();_,d=p.act(obs)
            self.assertTrue(d['fallback']);self.assertEqual(d['factor'],1.)
        p=HypothesisPI(specs['C_warm']);p.reset();_,d=p.act(obs)
        self.assertFalse(d['fallback']);self.assertEqual(d['factor'],specs['C_warm']['constant'])

    def test_initialization_and_first_observation(self):
        zero=rt.core.Feedback(-1,0,0,0,0,0,False)
        h=(zero,)*6+tuple(rt.core.Feedback(t,.5,2,10,2,1,False) for t in range(6))
        obs=rt.core.Observation(6,288,1000,12,h);raw=12/1000/(3+1e-6)
        for key in ['G','R']:
            sp=policies()[key+'_neutral'];p=HypothesisPI(sp);p.reset()
            expected=math.exp(sp['center']/sp['power']);self.assertAlmostEqual(p.smoothed,expected)
            _,d=p.act(obs);self.assertAlmostEqual(d['g_est'],expected+sp['ema_alpha']*(raw-expected))
            sp=policies()[key+'_first'];p=HypothesisPI(sp);p.reset();_,d=p.act(obs)
            self.assertAlmostEqual(d['g_est'],raw)

    def test_single_factor_ablations(self):
        s=policies()
        for a,b,changed in [('L_noema','L',{'ema_alpha'}),('L_wide','L',{'factor_low','factor_high'}),
          ('R_neutral','R',{'init_mode'}),('C_warm','C_all',{'constant_mode'})]:
            self.assertEqual({k for k in s[a] if s[a][k]!=s[b][k]},changed)
        for k in ['R_neutral','G_neutral','R_first','G_first']:
            zero=rt.core.Feedback(-1,0,0,0,0,0,False)
            obs=rt.core.Observation(0,288,1000,0,(zero,)*12)
            p=HypothesisPI(s[k]);p.reset();p.integral=.2;_,d=p.act(obs)
            self.assertEqual(d['integral'],.2)

    def test_future_data_does_not_change_past(self):
        ep=rt.episode(PROBE[0]);cut=40
        future=replace(ep,tape=ep.tape[:cut]+tuple(replace(t,competitor=t.competitor*5) for t in ep.tape[cut:]))
        for key in ['G_first','R_neutral','C_matched','L_noema','L_wide']:
            sp=policies()[key];_,a=rt.core.rollout(ep,rt.cfg,HypothesisPI(sp));_,b=rt.core.rollout(future,rt.cfg,HypothesisPI(sp))
            np.testing.assert_array_equal([r['action'] for r in a[:cut]],[r['action'] for r in b[:cut]])

    def test_diagnostic_segments_sum_to_TE(self):
        for k,sp in policies().items():
            score,tr=run(PROBE[0],sp);d=diagnose(tr,sp)
            self.assertAlmostEqual(sum(d[m] for m in d if m.startswith('TE_')),score['tracking_error'],places=14)

    def test_sealed_dates_and_unique_main_comparisons(self):
        for day in [200000,520000,530000,540800,430000]:
            with self.assertRaises(ValueError):run(day,policies()['CMA'])
        self.assertEqual(len(MAIN),4);self.assertEqual(len(policies()),14)
        self.assertFalse(set(CAL)&set(FRESH));self.assertFalse(set(FRESH)&set(PROBE))

if __name__=='__main__':unittest.main()
