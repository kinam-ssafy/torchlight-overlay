"""Paired, uncapped score comparison for `vorax sim --scores` outputs (stdlib only)."""
import argparse
import json
import random
import statistics
from pathlib import Path


def read(path):
    rows = [tuple(map(int, line.split())) for line in path.read_text().splitlines() if line.strip()]
    assert rows and all(len(row) == 2 for row in rows)
    scores = dict(rows)
    assert len(scores) == len(rows), 'duplicate seeds'
    return scores


def compare(before, after):
    assert before.keys() == after.keys(), 'both runs must use exactly the same seeds'
    seeds = sorted(before)
    delta = [after[s] - before[s] for s in seeds]
    rng = random.Random(0x5EED)
    bootstrap = sorted(statistics.mean(rng.choices(delta, k=len(delta))) for _ in range(10_000))
    def summary(scores):
        values = list(scores.values())
        return {'mean': statistics.mean(values), 'median': statistics.median(values),
                'reached840k': sum(x >= 840_000 for x in values) / len(values),
                'reached1120k': sum(x >= 1_120_000 for x in values) / len(values)}
    old, new = summary(before), summary(after)
    return {'games': len(seeds), 'seedMin': min(seeds), 'seedMax': max(seeds),
            'before': old, 'after': new, 'meanChangePercent': (new['mean'] / old['mean'] - 1) * 100,
            'pairedMeanDifference': statistics.mean(delta),
            'pairedBootstrap95PercentInterval': [bootstrap[249], bootstrap[9749]],
            'bootstrapResamples': 10_000, 'bootstrapSeed': 0x5EED}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(read(args.before), read(args.after)), indent=2))
