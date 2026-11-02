"""CPU-only tests for export boundaries and data/evaluation isolation."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import xgboost as xgb
from gpu_model import export_model, predict_export, split_boundary
from gpu_search import assert_separate, eligible_for_review, load_rows, paired


class ExportTests(unittest.TestCase):
    def test_float64_boundary_matches_float32_branch(self):
        rng = np.random.default_rng(17)
        thresholds = np.r_[np.float32(0), np.float32(1), np.float32(-1), rng.normal(size=2048).astype('f4')]
        for t in thresholds:
            prev = np.nextafter(t, np.float32(-np.inf))
            mid = (float(prev) + float(t)) / 2
            x = np.array([float(prev), np.nextafter(mid, -np.inf), mid, np.nextafter(mid, np.inf), float(t)])
            np.testing.assert_array_equal(x < split_boundary(t), x.astype('f4') < t)

    def test_saved_tree_cpu_prediction(self):
        rng = np.random.default_rng(22)
        x = rng.normal(size=(1024, 5)).astype('f4')
        y = .4 + .05 * x[:, 0] - .03 * x[:, 1] ** 2
        model = xgb.train({'tree_method': 'hist', 'device': 'cpu', 'base_score': .4,
                           'objective': 'reg:squarederror', 'nthread': 2, 'max_depth': 3},
                          xgb.DMatrix(x, y, nthread=2), num_boost_round=16)
        exported = json.loads(json.dumps(export_model(model), allow_nan=False))
        np.testing.assert_allclose(predict_export(exported, x), model.predict(xgb.DMatrix(x, nthread=2)), atol=2e-7)

    def test_classification_export_rejected(self):
        x = np.arange(40, dtype='f4').reshape(20, 2)
        b = xgb.train({'objective': 'binary:logistic', 'nthread': 2},
                      xgb.DMatrix(x, np.arange(20) % 2, nthread=2), num_boost_round=1)
        with self.assertRaises(ValueError):
            export_model(b)

    def test_nan_features_rejected(self):
        with self.assertRaises(ValueError):
            predict_export({'base': 0, 'trees': []}, [[np.nan]])


class EvaluationTests(unittest.TestCase):
    def test_game_seeds_not_random_row_split(self):
        assert_separate({'seed0': 1, 'games': 100}, {'seed0': 101, 'games': 10})
        with self.assertRaises(ValueError):
            assert_separate({'seed0': 1, 'games': 100}, {'seed0': 100, 'games': 10})

    def test_seed_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            paired({1: {'score': 1}}, {2: {'score': 1}})

    def test_paired_outcomes(self):
        a = {i: {'score': 0} for i in range(8)}
        b = {i: {'score': 840_000} for i in range(8)}
        result = paired(a, b)
        self.assertEqual((result['gained'], result['lost']), (8, 0))
        self.assertLess(result['paired_p'], .01)
        self.assertEqual(paired(a, a)['paired_p'], 1)

    def test_truncated_rows_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'rows.bin'
            p.write_bytes(b'\0' * 4)
            Path(str(p) + '.json').write_text(json.dumps({'dtype': 'float32-le', 'target': 840_000, 'rows': 2, 'columns': 3}))
            with self.assertRaises(ValueError):
                load_rows(p)

    def test_improvement_cannot_hide_bad_floor_latency_or_small_probe(self):
        baseline = {'p10': 500_000, 'below300k': 2, 'seconds': 10}
        good = {'engine_gate_pass': True, 'paired_p': .01, 'p10': 510_000, 'below300k': 1, 'seconds': 11}
        self.assertTrue(eligible_for_review(baseline, good, True))
        self.assertFalse(eligible_for_review(baseline, good, False))
        for key, value in [('engine_gate_pass', False), ('paired_p', .1), ('p10', 490_000), ('below300k', 3), ('seconds', 13)]:
            with self.subTest(key=key):
                self.assertFalse(eligible_for_review(baseline, {**good, key: value}, True))


if __name__ == '__main__':
    unittest.main()
