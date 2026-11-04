"""Paired simulation uncertainty and prospective sample-size planning.

Wilson intervals describe individual rates; exact McNemar tests compare policies
on the same seeds. Power estimates are conditional on pilot discordance, not a
guarantee for a different policy. Never extends a run until a p-value passes.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

MIN_RELEASE_GAMES = 4096


def release_quality(holdout, games, futures):
    """Legacy minimum effort gate, NOT a power calculation or proof of accuracy."""
    return holdout and games >= MIN_RELEASE_GAMES and futures >= 256


def paired_exact_power(games, discordance, difference, alpha=.05):
    """Prospective power of the exact two-sided McNemar/binomial test.

    Sum over Binomial(N, discordance) discordant pairs, then the conditional
    Binomial(M, .5 + difference/(2*discordance)) improvements. Tails totaling
    at most 2e-12 are omitted. Requires independent games and specified q/delta;
    it is not post-hoc 'observed power' or a guarantee for unknown live odds.
    """
    import numpy as np
    from scipy.stats import binom
    if games <= 0 or not 0 <= difference <= discordance <= 1 or discordance == 0 or not 0 < alpha < 1:
        raise ValueError('invalid prospective power assumptions')
    lo, hi = binom.ppf([1e-12, 1-1e-12], games, discordance).astype(int)
    m = np.arange(lo,hi+1)
    k = binom.ppf(alpha/2,m,.5)
    k -= binom.cdf(k,m,.5) > alpha/2
    p = .5 + difference/(2*discordance)
    conditional = binom.cdf(k,m,p) + binom.sf(m-k-1,m,p)
    return float(np.sum(binom.pmf(m,games,discordance)*conditional))


def wilson_interval(successes, games, confidence=.95):
    if not 0 <= successes <= games or games <= 0 or not 0 < confidence < 1:
        raise ValueError('invalid binomial count or confidence')
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / games
    denominator = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denominator
    radius = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def planning_games(discordance, difference, power=.8, alpha=.05):
    """Large-sample paired-binary approximation, two-sided alpha.

    Var(candidate - champion) = discordance - difference**2. Using discordance
    alone is slightly conservative at the specified alternative.
    """
    if not 0 < difference <= discordance <= 1 or not .5 < power < 1 or not 0 < alpha < 1:
        raise ValueError('require 0 < difference <= discordance <= 1 and valid power/alpha')
    normal = NormalDist()
    z = normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)
    return math.ceil(discordance * z * z / (difference * difference))


def paired_rates(a, b):
    from scipy.stats import binomtest
    if not a or a.keys() != b.keys():
        raise ValueError('nonempty identical game seeds required')
    gained = sum(b[k] and not a[k] for k in a)
    lost = sum(a[k] and not b[k] for k in a)
    discordant = gained + lost
    n = len(a)
    return {'games': n, 'champion_count': sum(a.values()), 'candidate_count': sum(b.values()),
            'champion_interval95': wilson_interval(sum(a.values()), n),
            'candidate_interval95': wilson_interval(sum(b.values()), n),
            'difference_pp': (sum(b.values()) - sum(a.values())) / n * 100,
            'gained': gained, 'lost': lost, 'discordant': discordant,
            'paired_exact_p': float(binomtest(gained, discordant, .5).pvalue) if discordant else 1.}


def analyze(champion, candidate):
    if not champion or len(champion) != len(candidate):
        raise ValueError('nonempty matched outcomes required')
    a = {r['seed']: r['score'] for r in champion}
    b = {r['seed']: r['score'] for r in candidate}
    if len(a) != len(champion) or len(b) != len(candidate) or a.keys() != b.keys():
        raise ValueError('duplicate or mismatched game seeds')
    success = paired_rates({k: v >= 840000 for k, v in a.items()}, {k: v >= 840000 for k, v in b.items()})
    low = paired_rates({k: v < 300000 for k, v in a.items()}, {k: v < 300000 for k, v in b.items()})
    q = success['discordant'] / len(a)
    q_high = wilson_interval(success['discordant'], len(a))[1]
    plan = [{'difference_pp': d * 100, 'games_using_pilot': planning_games(q, d) if q >= d else None,
             'games_using_pilot_upper95': planning_games(q_high, d) if q_high >= d else None}
            for d in [.01, .02, .03, .05]]
    return {'success840k': success, 'below300k': low,
            'sample_plan': {'alpha': .05, 'power': .8, 'discordance': q,
                            'discordance_upper95': q_high, 'estimates': plan,
                            'assumption': 'new policy has similar discordance; predeclare fresh fixed seeds'},
            'release_minimum_games': MIN_RELEASE_GAMES,
            'note': 'individual rate intervals are descriptive; paired exact p compares policies; low-score gained means a worse outcome'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--champion', required=True, type=Path)
    p.add_argument('--candidate', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    result = analyze(json.loads(args.champion.read_text()), json.loads(args.candidate.read_text()))
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
