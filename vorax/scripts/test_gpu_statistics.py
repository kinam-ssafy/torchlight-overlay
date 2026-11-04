import unittest
from gpu_statistics import analyze, paired_rates, planning_games, release_quality, wilson_interval, paired_exact_power


class SimulationStatisticsTests(unittest.TestCase):
    def test_exact_power_null_size_and_sample_growth(self):
        self.assertLessEqual(paired_exact_power(100,.4,0),.05+1e-12)
        self.assertEqual(paired_exact_power(5,1,1),0.)
        self.assertAlmostEqual(paired_exact_power(6,1,1),1.)
        self.assertGreater(paired_exact_power(8192,.4,.02),.80)
        self.assertLess(paired_exact_power(4096,.4,.02),.60)

    def test_wilson_boundary_and_known_half_rate(self):
        self.assertAlmostEqual(wilson_interval(0, 100)[1], .03699349820698568)
        self.assertAlmostEqual(wilson_interval(100, 100)[0], .9630065017930143)
        low, high = wilson_interval(50, 100)
        self.assertAlmostEqual(low + high, 1)
        self.assertAlmostEqual(low, .4038315303659956)

    def test_matched_pair_test_uses_discordant_games(self):
        a = {i: i < 50 for i in range(100)}
        b = {i: i < 70 for i in range(100)}
        r = paired_rates(a, b)
        self.assertEqual((r['gained'], r['lost'], r['discordant']), (20, 0, 20))
        self.assertAlmostEqual(r['paired_exact_p'], 2 / 2**20)
        self.assertEqual(paired_rates(a, a)['paired_exact_p'], 1)

    def test_half_difference_requires_about_four_times_games(self):
        # Halving the detectable difference quadruples the required sample.
        self.assertLessEqual(abs(planning_games(.35, .02) - 4 * planning_games(.35, .04)), 4)
        self.assertGreater(planning_games(.35, .03), 3000)

    def test_low_score_direction_and_disjoint_input(self):
        a = [{'seed': i, 'score': 200000 if i == 0 else 900000} for i in range(10)]
        b = [{'seed': i, 'score': 900000} for i in range(10)]
        r = analyze(a, b)
        self.assertEqual(r['below300k']['difference_pp'], -10)
        for bad in [b[:-1], [{**row, 'seed': 100} for row in b]]:
            with self.assertRaises(ValueError):
                analyze(a, bad)

    def test_release_requires_fresh_stage_enough_games_and_search_depth(self):
        self.assertTrue(release_quality(True, 4096, 256))
        for args in [(False, 4096, 256), (True, 512, 256), (True, 4095, 256), (True, 8192, 64)]:
            self.assertFalse(release_quality(*args))

    def test_invalid_inputs_rejected(self):
        for counts in [(0, 0), (-1, 100), (101, 100)]:
            with self.assertRaises(ValueError):
                wilson_interval(*counts)
        with self.assertRaises(ValueError):
            planning_games(.1, .2)


if __name__ == '__main__':
    unittest.main()
