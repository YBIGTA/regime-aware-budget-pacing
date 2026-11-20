import unittest,math
from dataclasses import replace
from unittest.mock import patch
from env import *
from controller import AsymmetricPI,usable,spend_targets,base_and_slope,restoration_weight,restore_factor
from fast_asymmetric import run,TRACE
from experiment import diagnose,MAIN

def observation(t=12,spent=None,history=None):
    zero=rt.core.Feedback(-1,0,0,0,0,0,False)
    return rt.core.Observation(t,288,1000,t/288*1000 if spent is None else spent,
      (zero,)*12 if history is None else history)

class AsymmetricChecks(unittest.TestCase):
    def test_all_six_legacy_conditions_match_parent(self):
        ns=dict(rt=rt,np=np,PRIOR=PRIOR,math=math,usable=usable,base_and_slope=base_and_slope,
          restoration_weight=restoration_weight,restore_factor=restore_factor)
        rt.definitions(DEADLINE/'code/controller.py',['spend_targets','DeadlinePI'],ns)
        ep=rt.episode(PROBE[0])
        for k,sp in policies().items():
            if '_A' in k:continue
            a,ra=rt.core.rollout(ep,rt.cfg,ns['DeadlinePI'](sp))
            b,rb=rt.core.rollout(ep,rt.cfg,AsymmetricPI(sp))
            np.testing.assert_allclose([r['action'] for r in ra],[r['action'] for r in rb],rtol=0,atol=1e-12)
            np.testing.assert_allclose([a[m] for m in METRICS],[b[m] for m in METRICS],rtol=0,atol=1e-12)

    def test_ten_policies_fast_vs_reference(self):
        for day in PROBE[:2]:
            ep=rt.episode(day)
            for k,sp in policies().items():
                fast,tr=run(day,sp);slow,rows=rt.core.rollout(ep,rt.cfg,AsymmetricPI(sp))
                with self.subTest(day=day,model=k):
                    np.testing.assert_allclose([fast[m] for m in METRICS],[slow[m] for m in METRICS],rtol=0,atol=1e-10)
                    for col in ['action','ki','integral','factor','required_rate','target_rate','base','base_nominal','predicted_at_base']:
                        np.testing.assert_allclose(tr[:,TRACE.index(col)],[r[col] for r in rows],rtol=1e-9,atol=1e-10)
                    self.assertTrue(np.all((tr[:,0]>=0)&(tr[:,0]<=1)))
                    self.assertTrue(np.all((tr[:,4]>=0)&(tr[:,4]<=6000)))
                    self.assertLessEqual(tr[-1,2],ep.budget+1e-8)

    def test_target_same_when_behind_and_fixed_when_ahead(self):
        for t in [0,12,240,287]:
            for spent in [0.,max(0,t/288*1000-10),t/288*1000,min(1000,t/288*1000+10),1000.]:
                obs=observation(t,spent)
                for blend in [0.,.5,1.]:
                    _,s=spend_targets(obs,blend,False);_,a=spend_targets(obs,blend,True)
                    self.assertGreaterEqual(a,1/288)
                    if obs.error>0:self.assertEqual(a,s)
                    else:self.assertAlmostEqual(a,1/288)
                    self.assertAlmostEqual(a,1/288+blend*max(0,obs.error)/(288-t))
        self.assertAlmostEqual(spend_targets(observation(287,990),1,True)[1],.01)
        self.assertAlmostEqual(spend_targets(observation(287,1000),1,True)[1],1/288)

    def test_same_state_base_floor(self):
        ep=rt.episode(PROBE[0]);_,rows=rt.core.rollout(ep,rt.cfg,AsymmetricPI(policies()['G_A05']))
        for r in rows:
            self.assertGreaterEqual(r['target_rate'],1/288)
            self.assertGreaterEqual(r['base']+1e-12,r['base_nominal'])
        self.assertTrue(any(r['base']>r['base_nominal'] for r in rows))

    def test_integral_preserved_and_ki_not_restored(self):
        obs=observation(240,(240/288-.001)*1000)
        p=AsymmetricPI(policies()['G_A05']);p.reset();p.integral=.1;p.net=lambda x:(0.,100.)
        with patch('controller.base_and_slope',return_value=(.2,0.,.2,1/288,False)):
            _,d=p.act(obs)
        self.assertEqual(d['risk_weight'],0)
        self.assertEqual(d['factor'],d['factor_before'])
        self.assertAlmostEqual(d['integral'],.1+d['ki']*obs.error/288)

    def test_future_auctions_do_not_change_past(self):
        ep=rt.episode(PROBE[0]);cut=50
        future=replace(ep,tape=ep.tape[:cut]+tuple(replace(t,competitor=t.competitor*5) for t in ep.tape[cut:]))
        for k in ['CMA_A05','CMA_A1','G_A05','G_A1']:
            sp=policies()[k];_,a=rt.core.rollout(ep,rt.cfg,AsymmetricPI(sp));_,b=rt.core.rollout(future,rt.cfg,AsymmetricPI(sp))
            np.testing.assert_array_equal([r['action'] for r in a[:cut]],[r['action'] for r in b[:cut]])

    def test_original_target_and_terminal_accounting(self):
        for k,sp in policies().items():
            score,tr=run(PROBE[0],sp)
            expected=np.arange(1,289)/288-tr[:,2]/rt.episode(PROBE[0]).budget
            np.testing.assert_allclose(tr[:,1],expected,rtol=0,atol=1e-14)
            self.assertAlmostEqual(np.abs(tr[:,1]).mean(),score['tracking_error'],places=14)
            d=diagnose(tr);self.assertAlmostEqual(d['remaining_pre_final']-d['final_spend'],score['unspent'],places=13)
            if '_A' in k:self.assertEqual(d['target_lower_rate'],0.)
            self.assertEqual(d['restoration_rate'],0.)

    def test_only_target_switches_differ(self):
        s=policies()
        for parent in ['CMA','G']:
            for suffix in ['05','1']:
                a=s[parent+'_A'+suffix];b=s[parent+'_T'+suffix]
                self.assertEqual({k for k in a if a[k]!=b[k]},{'asymmetric'})
        self.assertEqual(len(s),10);self.assertEqual(len(MAIN),4)

    def test_sealed_and_prior_dates(self):
        for day in [200000,200499,520000,530000,540000,550000,561000,450000,460000]:
            with self.assertRaises(ValueError):run(day,policies()['CMA'])

if __name__=='__main__':unittest.main()
