import io
import pickle
import sys
import unittest
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'full_grid'))
sys.path.insert(0, str(ROOT / 'independent_reference'))
import run_full_grid as runner
import validate_and_compare as independent
from audit_liver_inputs import ArrayReader


class CurrentMetricsTest(unittest.TestCase):
    def test_both_policies_against_direct_metrics(self):
        rng = np.random.default_rng(20260912)
        scores = [rng.integers(-3, 4, n).astype(float) for n in [1, 2, 7, 20, 31, 87]]
        masks = [np.arange(len(s)) % 3 == 0 for s in scores]
        for policy in runner.POLICIES:
            curves = runner.curves(scores, masks, policy)
            self.assertTrue(np.array_equal(curves['risk'], independent.loss_matrix(scores, masks, policy.endswith('v2'))))
            for j, fraction in enumerate(runner.FRACTIONS):
                direct = independent.direct_metrics(scores, masks, fraction, policy.endswith('v2'))
                for metric in curves:
                    self.assertAlmostEqual(curves[metric][:, j].mean(), direct[metric], places=12)

    def test_frozen_selector_and_float_schedule(self):
        self.assertEqual(runner.policy_check()['status'], 'PASS')
        fraction = runner.FRACTIONS[35]
        self.assertEqual(len(runner.top_fraction_set(np.arange(20), fraction)), int(np.ceil(fraction * 20)))
        self.assertEqual(int(np.ceil(fraction * 20)), 8)

    def test_restricted_reader_rejects_other_globals(self):
        with self.assertRaises(pickle.UnpicklingError):
            ArrayReader(io.BytesIO(pickle.dumps(print))).load()


if __name__ == '__main__':
    unittest.main()
