import unittest
from run import BOA, Mixture, run_day, rt, np, PARENT


class Tests(unittest.TestCase):
    def test_bernstein_formula(self):
        b = BOA()
        loss = np.array([0.2, 0.7])
        excess = loss - loss.mean()
        expected = np.exp(-0.5*excess-0.25*excess**2)
        b.update(loss)
        np.testing.assert_allclose(b.weights, expected/expected.sum())
        self.assertGreater(b.weights[0], 0.5)

    def test_equal_feedback_stable(self):
        b = BOA()
        for _ in range(100):
            b.update([0.4, 0.4])
        np.testing.assert_array_equal(b.weights, [0.5, 0.5])

    def test_invalid_feedback(self):
        for value in ([np.nan, 0.1], [0.2, 1.1], [0.2]):
            with self.assertRaises(ValueError):
                BOA().update(value)

    def test_rollout_causality_budget_and_mixture(self):
        specs = rt.read_json(PARENT/'results/protocol.json')['design']['policies']
        b = BOA()
        score, rows = run_day(560000, specs, b)
        for row in rows:
            self.assertAlmostEqual(row['action'], row['weight_G']*row['action_G']+
                                   row['weight_A05']*row['action_A05'])
            self.assertGreaterEqual(row['remaining'], 0.)
        self.assertEqual(rows[0]['weight_G'], 0.5)
        self.assertEqual(len(rows), rt.cfg.ticks)
        self.assertAlmostEqual(score['tracking_error'], np.mean([abs(r['error']) for r in rows]))
        # Resetting PI state does not erase BOA learning from completed windows.
        m = Mixture(specs, b)
        w = b.weights.copy()
        m.reset()
        np.testing.assert_array_equal(w, b.weights)
        ep = rt.episode(560000)
        zero = rt.core.Feedback(-1, 0., 0., 0, 0, 0., False)
        obs = rt.core.Observation(0, rt.cfg.ticks, ep.budget, 0., (zero,)*rt.cfg.history)
        before = b.weights.copy()
        m.act(obs)
        np.testing.assert_array_equal(before, b.weights)
        with self.assertRaises(RuntimeError):
            m.act(obs)
        m.feedback(obs, ep.tape[0])
        with self.assertRaises(RuntimeError):
            m.feedback(obs, ep.tape[0])
        for day in (200000, 200499):
            with self.assertRaises(ValueError):
                rt.episode(day)


if __name__ == '__main__':
    unittest.main()

