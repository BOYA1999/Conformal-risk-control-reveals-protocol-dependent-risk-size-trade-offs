import unittest

import numpy as np

from molxai_crc import calibrate_crc, loss_table, top_fraction_set


class MolXAICRCTest(unittest.TestCase):
    def test_tie_rule_and_fraction(self):
        self.assertEqual(top_fraction_set([1.0, 1.0, 0.0], 1 / 3), {0})
        self.assertEqual(top_fraction_set([1.0, 1.0, 0.0], 0), set())

    def test_perfect_ranking_selects_small_set(self):
        scores = [np.arange(10, 0, -1, dtype=float) for _ in range(100)]
        rationales = [{0, 1} for _ in range(100)]
        fractions = np.linspace(0, 1, 11)
        table = loss_table(scores, rationales, fractions)
        result = calibrate_crc(table, fractions, alpha=0.1)
        self.assertEqual(result["fraction"], 0.2)
        self.assertEqual(result["empirical_risk"], 0.0)

    def test_too_small_calibration_is_rejected(self):
        with self.assertRaises(ValueError):
            calibrate_crc(np.zeros((5, 2)), [0.5, 1.0], alpha=0.1)

    def test_optimized_table_matches_direct_sets(self):
        rng = np.random.default_rng(7)
        scores = [rng.normal(size=17) for _ in range(8)]
        rationales = [set(rng.choice(17, 4, replace=False)) for _ in scores]
        fractions = np.linspace(0, 1, 21)
        optimized = loss_table(scores, rationales, fractions)
        direct = np.array([
            [1 - len(top_fraction_set(score, fraction) & truth) / len(truth) for fraction in fractions]
            for score, truth in zip(scores, rationales)
        ])
        np.testing.assert_allclose(optimized, direct)

    def test_random_rankings_simulation(self):
        rng = np.random.default_rng(42)
        n_cal, n_test, n_atoms = 1000, 5000, 12
        fractions = np.linspace(0, 1, 101)
        scores = [rng.normal(size=n_atoms) for _ in range(n_cal + n_test)]
        rationales = [set(rng.choice(n_atoms, 2, replace=False)) for _ in scores]
        cal = loss_table(scores[:n_cal], rationales[:n_cal], fractions)
        test = loss_table(scores[n_cal:], rationales[n_cal:], fractions)
        result = calibrate_crc(cal, fractions, alpha=0.2)
        self.assertLessEqual(test[:, result["index"]].mean(), 0.23)


if __name__ == "__main__":
    unittest.main()
