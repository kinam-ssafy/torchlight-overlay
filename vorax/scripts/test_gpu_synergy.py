"""Synergy residual fitting and fixed, resumable validation boundaries."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from gpu_synergy import correction
from gpu_gate import checked_rows, freeze


class SynergyTests(unittest.TestCase):
    def test_zero_residual_keeps_champion(self):
        x = np.array([[1., .2, 0.], [1., .7, 0.]])
        prior = np.array([.2, .8])
        np.testing.assert_array_equal(correction(x, prior, prior, np), np.zeros(3))

    def test_extra_input_learns_distinct_outcomes(self):
        z = np.array([[1., 0., 0.], [0., 1., 0.]] * 64)
        base, y = np.full(128, .4), np.array([.6, .2] * 64)
        w = correction(z, y, base, np)
        self.assertGreater(w[0], 0)
        self.assertLess(w[1], 0)
        self.assertEqual(w[2], 0)
        self.assertLess(np.mean((base + z @ w - y)**2), np.mean((base - y)**2))

    @unittest.skipUnless(os.environ.get('VORAX_TEST_CUDA') == '1', 'explicit CUDA test')
    def test_cuda_matches_cpu_reference(self):
        import cupy as cp
        rng = np.random.default_rng(62)
        z = np.c_[rng.normal(size=(256, 9)), np.zeros(256)]
        y, raw = rng.uniform(0, .84, 256), rng.normal(.5, .3, 256)
        gpu = cp.asnumpy(correction(cp.asarray(z), cp.asarray(y), cp.asarray(raw), cp))
        np.testing.assert_allclose(gpu, correction(z, y, raw, np), rtol=1e-10, atol=1e-10)

    def test_holdout_cannot_change_after_start(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plan.json'
            a = {'games': 4096, 'candidate': 'a', 'seed0': 101_000_000}
            freeze(path, a)
            freeze(path, a)
            for change in [{'games': 8192}, {'candidate': 'b'}, {'seed0': 102_000_000}]:
                with self.assertRaises(ValueError):
                    freeze(path, {**a, **change})

    def test_chunks_reject_missing_duplicate_and_wrong_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rows.json'
            row = {'seed': 10, 'score': 900000, 'monsters': 3, 'groups': 1}
            for rows in [[], [row, row], [{**row, 'seed': 11}], [{**row, 'score': -1}]]:
                path.write_text(json.dumps(rows))
                with self.assertRaises(ValueError):
                    checked_rows(path, 10, 1)
            path.write_text(json.dumps([row]))
            self.assertEqual(checked_rows(path, 10, 1)[10], row)


if __name__ == '__main__':
    unittest.main()
