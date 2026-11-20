import unittest
from dataclasses import replace
from bridge import *
from controller import SensitivityPI, correction
from fast_sensitivity import run, TRACE
from experiment import choose

class SensitivityChecks(unittest.TestCase):
    def test_identity_matches_frozen_controller_and_kernel(self):
        from fast_eval import run as old_run
        for day in PROBE[:2]:
            old,tr0=old_run(day,rt.baseline_specs()['team_cma_sum'])
            new,tr1=run(day,spec())
            np.testing.assert_allclose(tr1[:,:6],tr0,rtol=0,atol=1e-12)
            np.testing.assert_allclose([new[m] for m in METRICS],[old[m] for m in METRICS],rtol=0,atol=1e-14)

    def test_independent_python_replay_all_candidates(self):
        ep=rt.episode(PROBE[0])
        for key,sp in candidates()[0].items():
            fast,tr=run(ep.day,sp)
            slow,rows=rt.core.rollout(ep,rt.cfg,SensitivityPI(sp))
            with self.subTest(key=key):
                np.testing.assert_allclose([fast[m] for m in METRICS],[slow[m] for m in METRICS],rtol=0,atol=1e-10)
                for column in ['action','ki','g_est','factor','integral']:
                    np.testing.assert_allclose(tr[:,TRACE.index(column)],[r[column] for r in rows],rtol=1e-9,atol=1e-10)

    def test_correction_bounds_and_invalid_fallback(self):
        for g in [-1,float('nan'),float('inf')]:
            self.assertEqual(correction(g,1,1,True),(1.,True,False))
        self.assertEqual(correction(0,1,1,False),(1.,True,False))
        self.assertEqual(correction(0,1,1,True),(2.,False,True))
        self.assertEqual(correction(100,1,1,True),(.5,False,True))
        self.assertEqual(correction(1,1,1,True),(1.,False,False))

    def test_integral_not_rescaled_when_Ki_changes(self):
        ep=rt.episode(PROBE[0]);zero=rt.core.Feedback(-1,0,0,0,0,0,False)
        obs=rt.core.Observation(0,288,ep.budget,0,(zero,)*12)
        p1=SensitivityPI(spec());p2=SensitivityPI(spec(5,constant=2))
        p1.reset();p2.reset();p1.integral=p2.integral=.2
        a,x=p1.act(obs);b,y=p2.act(obs)
        self.assertEqual(a,b)
        self.assertEqual(x['integral'],y['integral'])
        self.assertAlmostEqual(y['ki'],2*x['ki'])

    def test_future_auctions_do_not_change_past_actions(self):
        ep=rt.episode(PROBE[0]);cut=30
        changed=replace(ep,tape=ep.tape[:cut]+tuple(replace(t,competitor=t.competitor*4) for t in ep.tape[cut:]))
        _,a=rt.core.rollout(ep,rt.cfg,SensitivityPI(spec(4)))
        _,b=rt.core.rollout(changed,rt.cfg,SensitivityPI(spec(4)))
        np.testing.assert_array_equal([r['action'] for r in a[:cut]],[r['action'] for r in b[:cut]])

    def test_budget_censored_feedback_is_excluded(self):
        hist=tuple(rt.core.Feedback(t,.5,1e7,10,10,1,True) for t in range(12))
        obs=rt.core.Observation(12,288,1000,12/288*1000,hist)
        policy=SensitivityPI(spec(4));policy.reset()
        _,d=policy.act(obs)
        self.assertTrue(d['fallback']);self.assertEqual(d['factor'],1.)

    def test_output_and_budget_bounds(self):
        for method in [1,2,3,4,5]:
            ep=rt.episode(PROBE[1]);score,rows=rt.core.rollout(ep,rt.cfg,SensitivityPI(spec(method)))
            self.assertTrue(all(0<=r['action']<=1 and 0<=r['ki']<=6000 for r in rows))
            self.assertLessEqual(rows[-1]['cumulative_spend'],ep.budget+1e-9)

    def test_selection_does_not_adopt_accuracy_tradeoff(self):
        scores={'base':dict(tracking_error=1,unspent=1,objective=2,multiplier_tv=1),
                'bad':dict(tracking_error=1.1,unspent=.5,objective=1.6,multiplier_tv=.5)}
        self.assertEqual(choose(['bad'],scores,'base')['primary'],'base')
        self.assertEqual(choose(['bad'],scores,'base')['diagnostic'],'bad')

    def test_splits_and_final_test_guard(self):
        self.assertEqual(len(candidates()[0]),24)
        with self.assertRaises(ValueError):rt.episode(200000)
        with self.assertRaises(ValueError):rt.episode(520300)
        with self.assertRaises(ValueError):run(200000,spec())

if __name__=='__main__':unittest.main()
