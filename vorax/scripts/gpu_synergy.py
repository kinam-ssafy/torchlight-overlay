"""Fit additional synergy features on CUDA; freeze all champion coefficients.

Training rollouts use the champion. Selection and final solver games are separate.
The exported model requires the synergy-v1 CPU reader; it is never installed here.
"""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import numpy as np
from gpu_search import ROOT, TARGET, assert_separate, command, load_rows, settings, sha, write_json

VARIANTS = [('synergy-10', .10), ('synergy-25', .25), ('synergy-50', .50)]


def correction(z, target, legacy_raw, xp):
    matrix = z.T @ z
    matrix += xp.diag(.1 * xp.diag(matrix) + len(target) * 1e-4)
    return xp.linalg.solve(matrix, z.T @ (target - legacy_raw))


def verify(args, legacy, extra, path, cp):
    name = path.stem + '-probe'
    out = args.run / (name + '.jsonl')
    command(args, name, 'train', '--probe', out, '--synergy', '--games', 48,
            '--seed0', 96_000_000, '--model', path, *settings(args))
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    rounds = np.array([r['round'] for r in rows]) - 1
    x = np.array([r['features'] for r in rows], dtype=np.float64)
    cpu = np.array([r['prediction'] for r in rows])
    weights = np.concatenate([legacy, extra], axis=1)
    prediction = np.clip(np.sum(x * weights[rounds], axis=1) * 1e6, 0, TARGET)
    gpu = cp.asnumpy(cp.clip(cp.sum(cp.asarray(x) * cp.asarray(weights)[cp.asarray(rounds)], axis=1) * 1e6, 0, TARGET))
    error, gpu_error = float(np.max(abs(prediction - cpu))), float(np.max(abs(gpu - cpu)))
    if max(error, gpu_error) > 1e-6:
        raise ValueError(f'CPU synergy export mismatch: {error}, CUDA={gpu_error}')
    return {'boards': len(rows), 'cpu_export_max_error': error, 'cpu_gpu_max_error_points': gpu_error}


def fit(args):
    os.environ.setdefault('CUPY_CACHE_DIR', str(args.run.parent / 'cupy-cache'))
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError('CUDA required; no CPU training fallback')
    cp.linalg.solve(cp.eye(2), cp.ones(2))
    cp.cuda.Stream.null.synchronize()
    source = ROOT / 'data/boardValue.json'
    original = json.loads(source.read_text())
    legacy = np.asarray(original['rounds'], dtype=np.float64)
    if legacy.shape != (13, 656) or not np.isfinite(legacy).all():
        raise ValueError('expected frozen 13-round, 656-feature champion')
    write_json(args.run / 'champion.json', original)
    report = {'engine_sha256': sha(args.engine), 'champion_sha256': sha(source),
              'device': 'cuda:0', 'cupy': cp.__version__, 'threads': args.threads,
              'potion_refreshes': args.potion_refreshes, 'instrument_refreshes': args.instrument_refreshes,
              'target': TARGET, 'kind': 'frozen-legacy-synergy-v1', 'legacy_features': 656, 'extra_features': 564,
              'candidate_count': len(VARIANTS), 'candidates': [], 'eps': .03,
              'train_seed': args.seed0, 'train_games': args.games,
              'valid_seed': args.seed0 + args.games, 'valid_games': args.valid_games,
              'regularization': {'diagonal_scale': .1, 'minimum_per_row': 1e-4},
              'variants_predeclared': VARIANTS, 'gpu_seconds': 0., 'round_metrics': {}}
    write_json(args.run / 'fit.json', report)
    for label, start, games in [('train', args.seed0, args.games), ('valid', args.seed0 + args.games, args.valid_games)]:
        command(args, label + '-rows', 'train', '--rows', args.run / (label + '.bin'), '--synergy',
                '--games', games, '--seed0', start, '--eps', .03, '--model', args.run / 'champion.json', *settings(args))
    train, tm = load_rows(args.run / 'train.bin')
    valid, vm = load_rows(args.run / 'valid.bin')
    assert_separate(tm, vm)
    for meta, seed, games in [(tm, args.seed0, args.games), (vm, args.seed0 + args.games, args.valid_games)]:
        if meta['features'] != 'linear-synergy-v1' or meta['columns'] != 1222 or meta['seed0'] != seed or meta['games'] != games:
            raise ValueError('unexpected synergy row schema/provenance')
    report['data'] = {'train': tm, 'valid': vm}
    extras = {name: np.zeros((13, 564)) for name, _ in VARIANTS}
    metrics = {name: [] for name in ['champion', *extras]}
    train_ids, valid_ids = np.asarray(train[:, 0], dtype=np.int32), np.asarray(valid[:, 0], dtype=np.int32)
    for r in range(1, 14):
        tr, vr = np.asarray(train[train_ids == r]), np.asarray(valid[valid_ids == r])
        if not len(tr) or not len(vr):
            raise ValueError(f'missing round {r}')
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        x, y = cp.asarray(tr[:, 1:-1], dtype=cp.float64), cp.asarray(tr[:, -1], dtype=cp.float64)
        vx, vy = cp.asarray(vr[:, 1:-1], dtype=cp.float64), cp.asarray(vr[:, -1], dtype=cp.float64)
        w = cp.asarray(legacy[r - 1])
        delta = correction(x[:, 656:], y, x[:, :656] @ w, cp)
        baseline = vx[:, :656] @ w
        for name, strength in [('champion', 0.), *VARIANTS]:
            if strength:
                extras[name][r - 1] = cp.asnumpy(strength * delta)
            residual = cp.clip(baseline + strength * (vx[:, 656:] @ delta), 0, TARGET / 1e6) - vy
            low = vy < .3
            count = int(cp.count_nonzero(low).get())
            metrics[name].append({'round': r, 'rmse': float(cp.sqrt(cp.mean(residual ** 2)).get()) * 1e6,
                                  'low_cases': count, 'low_rmse': float(cp.sqrt(cp.mean(residual[low] ** 2)).get()) * 1e6 if count else None})
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        report['gpu_seconds'] += elapsed
        report['round_metrics'] = metrics
        write_json(args.run / 'fit.json', report)
        print(f'CUDA {r}/13: {len(tr)} train / {len(vr)} valid boards, {elapsed:.2f}s', flush=True)
        del x, y, vx, vy, w, delta, baseline, residual, low, tr, vr
        cp.get_default_memory_pool().free_all_blocks()
    for name, strength in VARIANTS:
        extra = extras[name]
        if not np.isfinite(extra).all() or not np.any(extra):
            raise ValueError('non-finite or inactive synergy weights')
        path = args.run / (name + '.json')
        write_json(path, {'synergy_version': 1, 'legacy_rounds': legacy.tolist(), 'synergy_rounds': extra.tolist()})
        parity = verify(args, legacy, extra, path, cp)
        report['candidates'].append({'name': name, 'strength': strength, 'sha256': sha(path), 'parity': parity,
                                     'active_extra_coefficients': int(np.count_nonzero(extra))})
        write_json(args.run / 'fit.json', report)
    print(f'Finished {len(VARIANTS)} candidates; CUDA fit/validation {report["gpu_seconds"]:.2f}s', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--engine', type=Path, default=ROOT / 'target/release/vorax.exe')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--games', type=int, default=50_000)
    p.add_argument('--valid-games', type=int, default=10_000)
    p.add_argument('--seed0', type=int, default=16_000_000)
    p.add_argument('--potion-refreshes', type=int, default=3)
    p.add_argument('--instrument-refreshes', type=int, default=1)
    args = p.parse_args()
    if min(args.threads, args.games, args.valid_games) <= 0 or not 0 <= args.seed0 < args.seed0 + args.games + args.valid_games < 2**32:
        p.error('positive counts and non-overflowing u32 seeds required')
    if not all(0 <= n <= 9 for n in [args.potion_refreshes, args.instrument_refreshes]):
        p.error('rerolls must be between 0 and 9')
    args.run, args.engine = args.run.resolve(), args.engine.resolve()
    args.run.mkdir(parents=True, exist_ok=True)
    if (args.run / 'fit.json').exists():
        p.error('use a fresh run directory')
    fit(args)


if __name__ == '__main__':
    main()
