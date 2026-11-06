import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from core import Config, Gains, PI, Tick, Episode, Observation, Feedback, PredictiveBase
from core import generate, execute, rollout, features
import run

class Tests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(minutes=60, interval=5, history=3, train_days=3,
                          validation_days=2, test_days=3, trials=4)

    def test_seed_and_independent_days(self):
        a = generate(self.cfg,1000)
        self.assertEqual(a.digest(),generate(self.cfg,1000).digest())
        self.assertNotEqual(a.digest(),generate(self.cfg,1001).digest())
        self.assertFalse(a.tape[0].base.flags.writeable)

    def test_first_price_and_chronological_budget(self):
        tick = Tick(np.array([1.,2.,1.]),np.array([.3,.6,.2]))
        self.assertEqual(execute(tick,.5,10),(2.,3,False))
        # Pay bid, not competitor price; skip unaffordable middle auction.
        self.assertEqual(execute(tick,.5,1.1),(1.,2,True))
        self.assertEqual(execute(tick,0,10),(0.,0,False))
        self.assertEqual(execute(Tick(np.array([1.]),np.array([.5])),.5,10),(0.,0,False))

    def test_antiwindup_and_recovery_direction(self):
        p = PI(self.cfg,Gains(20,10))
        p.integral = .1
        obs = Observation(6,12,100,0,())
        action,debug = p.act(obs)
        self.assertEqual(action,1.)
        self.assertTrue(debug['integral_blocked'])
        self.assertEqual(p.integral,.1)
        # While at a high output, negative error must be allowed to unwind.
        p.integral = 2.
        p.act(Observation(1,12,100,50,()))
        self.assertLess(p.integral,2.)
        p.reset()
        self.assertEqual(p.integral,0.)

    def test_budget_bounds_and_repeatable_rollout(self):
        ep = generate(self.cfg,1000)
        first,rows = rollout(ep,self.cfg,PI(self.cfg,Gains(20,10)))
        again,_ = rollout(ep,self.cfg,PI(self.cfg,Gains(20,10)))
        self.assertEqual(first,again)
        self.assertTrue(all(0 <= r['action'] <= 1 for r in rows))
        self.assertLessEqual(rows[-1]['cumulative_spend'],ep.budget+1e-9)
        self.assertEqual(rows[0]['action'],self.cfg.initial_multiplier)
        self.assertEqual(len(rows),self.cfg.ticks)

    def test_future_tape_cannot_change_prior_actions(self):
        ep = generate(self.cfg,1000)
        captured = []
        def predictor(x):
            captured.append(x.copy())
            return x[:,0]/self.cfg.ticks
        def evaluate(episode):
            return rollout(episode,self.cfg,PI(self.cfg,Gains(4,1),
                PredictiveBase(self.cfg,predictor)))[1]
        a = evaluate(ep)
        empty = Tick(np.array([]),np.array([]))
        changed = replace(ep,tape=ep.tape[:6]+(empty,)*6)
        b = evaluate(changed)
        # t=6 sees history through t=5; action must be identical too.
        self.assertEqual([r['action'] for r in a[:7]], [r['action'] for r in b[:7]])
        self.assertEqual(captured[0].shape,(101,5+7*self.cfg.history))
        self.assertEqual(captured[0][0,5],0.)  # padded-history validity mask

    def test_common_controller_identical_when_base_identical(self):
        ep = generate(self.cfg,1000)
        a = rollout(ep,self.cfg,PI(self.cfg,Gains(4,1)))
        b = rollout(ep,self.cfg,PI(self.cfg,Gains(4,1),lambda o:self.cfg.initial_multiplier))
        self.assertEqual(a,b)

    def test_empty_supply_and_complete_exhaustion(self):
        empty = Tick(np.array([]),np.array([]))
        ep = Episode(1,100.,(empty,)*self.cfg.ticks)
        score,rows = rollout(ep,self.cfg,PI(self.cfg,Gains(100,100)))
        self.assertEqual(score['unspent'],1.)
        self.assertGreater(score['capacity_shortfall'],.99)
        cheap = Tick(np.array([1.]*20),np.array([.01]*20))
        ep = Episode(2,.1,(cheap,)*self.cfg.ticks)
        score,rows = rollout(ep,self.cfg,PI(self.cfg,Gains(0,0)))
        # Affordable bids cannot be invented by clipping to remaining budget.
        self.assertTrue(all(r['spend']==0 for r in rows))

    def test_invalid_settings(self):
        with self.assertRaises(ValueError):
            replace(self.cfg,test_start=self.cfg.train_start)
        with self.assertRaises(ValueError):
            Gains(float('nan'),1)
        with self.assertRaises(ValueError):
            PI(self.cfg,Gains(1,1),lambda o:2).act(Observation(0,12,100,0,()))

    def test_end_to_end_freeze_and_test_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            run.init(out,self.cfg)
            run.collect(out)
            run.tune(out)
            self.assertFalse((out/'test_A').exists())
            with self.assertRaises(ValueError):
                run.tune(out)
            run.evaluate(out)
            summary = (out/'test_A/summary.json').read_bytes()
            run.evaluate(out)
            self.assertEqual(summary,(out/'test_A/summary.json').read_bytes())
            frozen = run.read_json(out/'frozen_A.json')
            frozen['gains']['kp'] += 1
            run.write_json(out/'frozen_A.json',frozen)
            with self.assertRaises(ValueError):
                run.evaluate(out)

if __name__ == '__main__':
    unittest.main()
