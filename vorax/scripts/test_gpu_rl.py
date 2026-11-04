"""Guard policy feedback and seed isolation independently of the GPU experiments."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from gpu_rl import iterate, seed_plan, validate_rows
from gpu_search import assert_separate, write_json


class PolicyIterationTests(unittest.TestCase):
    def test_all_generations_and_validation_games_are_disjoint(self):
        plan = seed_plan(14_000_000, 6, 50_000, 5_000)
        ranges = [spec for iteration in plan for spec in iteration.values()]
        for i, a in enumerate(ranges):
            for b in ranges[i + 1:]:
                assert_separate(a, b)
        self.assertEqual(sum(p['train']['games'] for p in plan), 300_000)
        self.assertEqual(plan[-1]['valid']['seed0'] + 5_000, 14_330_000)

    def test_bad_counts_and_seed_overflow_rejected(self):
        for args in [(-1, 6, 50_000, 5_000), (0, 0, 1, 1), (0, 1, 0, 1), (2**32 - 6, 2, 2, 1)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                seed_plan(*args)

    def test_row_schema_and_provenance_checked(self):
        expected = {'seed0': 10, 'games': 5}
        meta = {**expected, 'features': 'linear-rounds', 'columns': 658}
        validate_rows(meta, expected, 656)
        for key, value in [('features', 'tree'), ('columns', 660), ('seed0', 11), ('games', 4)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_rows({**meta, key: value}, expected, 656)

    def test_next_generation_acts_with_new_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            write_json(run / 'champion.json', {'generation': 0})
            args = SimpleNamespace(run=run, seed0=100, iterations=3, games=2, valid_games=1)
            observed = []

            def fake_fit(args, index, behavior, seeds, cp):
                observed.append(json.loads(behavior.read_text())['generation'])
                name = f'iteration-{index:02d}'
                write_json(run / (name + '.json'), {'generation': index})
                return {'name': name, 'data': {'train': {'rows': 40}}, 'gpu_seconds': 0}

            manifest = {'candidates': []}
            with patch('gpu_rl.fit_iteration', side_effect=fake_fit), patch('gpu_rl.plot_learning'):
                iterate(args, manifest, None)
            self.assertEqual(observed, [0, 1, 2])
            self.assertEqual(len(json.loads((run / 'fit.json').read_text())['candidates']), 3)


if __name__ == '__main__':
    unittest.main()
