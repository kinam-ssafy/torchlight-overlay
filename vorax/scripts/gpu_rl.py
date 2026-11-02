"""Bounded Monte Carlo policy iteration: fresh games -> CUDA value fit -> next policy.

Uses the existing exploratory training player and CPU linear model format. This is
approximate policy iteration, not PPO, and does not change the shipped model.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
from gpu_finetune import correction, verify
from gpu_search import ROOT, TARGET, assert_separate, command, load_rows, plotting, settings, sha, write_json


def seed_plan(seed0, iterations, games, valid_games):
    if min(iterations, games, valid_games) <= 0 or not 0 <= seed0 < seed0 + iterations * (games + valid_games) < 2**32:
        raise ValueError('positive counts and a valid u32 seed range are required')
    valid_start = seed0 + iterations * games
    return [{'train': {'seed0': seed0 + i * games, 'games': games},
             'valid': {'seed0': valid_start + i * valid_games, 'games': valid_games}}
            for i in range(iterations)]


def validate_rows(meta, expected, columns):
    if (meta.get('features') != 'linear-rounds' or meta['columns'] != columns + 2
            or meta['seed0'] != expected['seed0'] or meta['games'] != expected['games']):
        raise ValueError('training feature schema or game seed range mismatch')


def fit_iteration(args, index, behavior, seeds, cp):
    name = f'iteration-{index:02d}'
    prior = np.array(json.loads(behavior.read_text())['rounds'], dtype=np.float64)
    if prior.ndim != 2 or prior.shape[0] != 13 or not np.isfinite(prior).all():
        raise ValueError('policy iteration requires 13 finite linear weight vectors')
    eps = args.eps_start + (args.eps_end - args.eps_start) * (index - 1) / max(1, args.iterations - 1)
    result = {'name': name, 'iteration': index, 'behavior_model': behavior.name,
              'behavior_sha256': sha(behavior), 'eps': eps, 'data': {}, 'round_metrics': []}
    paths = {label: args.run / f'{name}-{label}.bin' for label in seeds}
    for label, spec in seeds.items():
        command(args, f'{name}-{label}-rows', 'train', '--rows', paths[label], '--linear',
                '--games', spec['games'], '--seed0', spec['seed0'], '--eps', eps,
                '--model', behavior, *settings(args))
    train, tm = load_rows(paths['train'])
    valid, vm = load_rows(paths['valid'])
    for label, meta in [('train', tm), ('valid', vm)]:
        validate_rows(meta, seeds[label], prior.shape[1])
        result['data'][label] = {**meta, 'sha256': sha(paths[label])}
    assert_separate(tm, vm)
    train_ids = np.asarray(train[:, 0], dtype=np.int32)
    valid_ids = np.asarray(valid[:, 0], dtype=np.int32)
    if not all(np.isin(ids, np.arange(1, 14)).all() for ids in [train_ids, valid_ids]):
        raise ValueError('unexpected game round')
    weights = prior.copy()
    gpu_seconds = 0.0
    for r in range(1, 14):
        tr = np.asarray(train[train_ids == r])
        vr = np.asarray(valid[valid_ids == r])
        if not len(tr) or not len(vr):
            raise ValueError(f'missing round {r}')
        cp.cuda.Stream.null.synchronize()
        started = time.perf_counter()
        x, y = cp.asarray(tr[:, 1:-1], dtype=cp.float64), cp.asarray(tr[:, -1], dtype=cp.float64)
        vx, vy = cp.asarray(vr[:, 1:-1], dtype=cp.float64), cp.asarray(vr[:, -1], dtype=cp.float64)
        original = cp.asarray(prior[r - 1])
        # Undistorted terminal capped reward; no outcome-dependent sampling weight.
        updated = original + args.strength * correction(x, y, original, 1.0, cp)
        weights[r - 1] = cp.asnumpy(updated)
        before = cp.clip(vx @ original, 0, TARGET / 1e6) - vy
        after = cp.clip(vx @ updated, 0, TARGET / 1e6) - vy
        result['round_metrics'].append({'round': r, 'train_rows': len(tr), 'valid_rows': len(vr),
                                       'before_rmse': float(cp.sqrt(cp.mean(before ** 2)).get()) * 1e6,
                                       'after_rmse': float(cp.sqrt(cp.mean(after ** 2)).get()) * 1e6})
        cp.cuda.Stream.null.synchronize()
        gpu_seconds += time.perf_counter() - started
        del x, y, vx, vy, tr, vr, original, updated, before, after
        cp.get_default_memory_pool().free_all_blocks()
    if not np.isfinite(weights).all():
        raise ValueError('non-finite fitted coefficients')
    path = args.run / (name + '.json')
    write_json(path, {'rounds': weights.tolist()})
    result.update({'sha256': sha(path), 'gpu_seconds': gpu_seconds,
                   'parity': verify(args, weights, path, name, cp)})
    # Keep metadata/checkpoints and hashes; raw dense rows are reproducible and large.
    del train, valid
    for path in paths.values():
        path.unlink()
    return result


def iterate(args, manifest, cp):
    behavior = args.run / 'champion.json'
    for index, seeds in enumerate(seed_plan(args.seed0, args.iterations, args.games, args.valid_games), 1):
        result = fit_iteration(args, index, behavior, seeds, cp)
        manifest['candidates'].append(result)
        # The next generation acts with the newly fitted checkpoint, not the champion.
        behavior = args.run / (result['name'] + '.json')
        write_json(args.run / 'fit.json', manifest)
        print(f'{index}/{args.iterations}: {index * args.games:,} training games; '
              f'{result["data"]["train"]["rows"]:,} states; GPU {result["gpu_seconds"]:.1f}s', flush=True)
        plot_learning(args.run, manifest)


def plot_learning(run, manifest):
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    candidates = manifest['candidates']
    x = [c['iteration'] * manifest['games_per_iteration'] / 10_000 for c in candidates]
    for key, label in [('before_rmse', '업데이트 전'), ('after_rmse', 'GPU 학습 후')]:
        values = [np.sqrt(sum(r[key] ** 2 * r['valid_rows'] for r in c['round_metrics']) /
                          sum(r['valid_rows'] for r in c['round_metrics'])) / 10_000 for c in candidates]
        axes[0].plot(x, values, 'o-', label=label)
    axes[0].set(xlabel='누적 학습 게임 (만 판)', ylabel='검증 예측 오차 (만 점)', title='각 세대의 별도 검증 게임')
    axes[0].legend()
    axes[1].plot(x, np.cumsum([c['data']['train']['rows'] for c in candidates]) / 1e6, 'o-')
    axes[1].set(xlabel='누적 학습 게임 (만 판)', ylabel='누적 학습 상태 (백만 개)', title='정책 갱신 후 새 게임 생성')
    fig.savefig(run / 'learning.png', dpi=140)
    plt.close(fig)


def fit(args):
    os.environ.setdefault('CUPY_CACHE_DIR', str(args.run.parent / 'cupy-cache'))
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError('a CUDA GPU is required; no CPU fitting fallback')
    cp.linalg.solve(cp.eye(2), cp.ones(2))
    cp.cuda.Stream.null.synchronize()
    source = ROOT / 'data/boardValue.json'
    shutil.copyfile(source, args.run / 'champion.json')
    manifest = {'kind': 'approximate-monte-carlo-policy-iteration', 'device': 'cuda:0',
                'cupy': cp.__version__, 'engine_sha256': sha(args.engine), 'champion_sha256': sha(source),
                'threads': args.threads, 'potion_refreshes': args.potion_refreshes,
                'instrument_refreshes': args.instrument_refreshes, 'target': TARGET,
                'candidate_count': args.iterations, 'candidates': [], 'promoted': False,
                'games_per_iteration': args.games, 'valid_games_per_iteration': args.valid_games,
                'train_seed': args.seed0, 'train_games': args.iterations * args.games,
                'valid_seed': args.seed0 + args.iterations * args.games,
                'valid_games': args.iterations * args.valid_games,
                'eps_start': args.eps_start, 'eps_end': args.eps_end, 'strength': args.strength,
                'regularization': {'diagonal_scale': .1, 'minimum_per_row': 1e-4},
                'training_player': 'train::play, two-sample value policy with exploration',
                'reward': 'min(final_score, 840000) / 1000000'}
    write_json(args.run / 'fit.json', manifest)
    iterate(args, manifest, cp)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--engine', type=Path, default=ROOT / 'target/release/vorax.exe')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--iterations', type=int, default=6)
    p.add_argument('--games', type=int, default=50_000)
    p.add_argument('--valid-games', type=int, default=5_000)
    p.add_argument('--seed0', type=int, default=14_000_000)
    p.add_argument('--eps-start', type=float, default=.08)
    p.add_argument('--eps-end', type=float, default=.02)
    p.add_argument('--strength', type=float, default=.25)
    p.add_argument('--potion-refreshes', type=int, default=3)
    p.add_argument('--instrument-refreshes', type=int, default=1)
    args = p.parse_args()
    try:
        seed_plan(args.seed0, args.iterations, args.games, args.valid_games)
    except ValueError as error:
        p.error(str(error))
    if args.threads <= 0 or not all(0 <= n <= 9 for n in [args.potion_refreshes, args.instrument_refreshes]):
        p.error('positive threads and rerolls between 0 and 9 are required')
    if not (0 <= args.eps_end <= args.eps_start <= 1 and 0 < args.strength <= 1):
        p.error('require 0 <= eps-end <= eps-start <= 1 and 0 < strength <= 1')
    args.run, args.engine = args.run.resolve(), args.engine.resolve()
    args.run.mkdir(parents=True, exist_ok=True)
    if (args.run / 'fit.json').exists():
        p.error('output already exists; use a fresh run directory')
    fit(args)


if __name__ == '__main__':
    main()
