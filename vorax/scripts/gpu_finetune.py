"""GPU residual fine-tuning of the shipped per-round linear model (CPU format unchanged)."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
from gpu_search import ROOT, TARGET, assert_separate, command, load_rows, plotting, settings, sha, write_json

VARIANTS = [('anchor-10', 1.0, .10), ('anchor-25', 1.0, .25),
            ('floor-10', 2.0, .10), ('floor-25', 2.0, .25)]


def correction(x, y, prior, low_weight, xp):
    """Regularized residual fit, anchored to the existing clipped CPU prediction.

    GPU arrays use float64. Unobserved features keep their original weights.
    xp also permits a small CPU reference calculation in the unit tests.
    """
    weight = xp.where(y < .3, low_weight, 1.0)
    residual = y - xp.clip(x @ prior, 0, TARGET / 1e6)
    xw = x * xp.sqrt(weight)[:, None]
    matrix = xw.T @ xw
    penalty = .1 * xp.diag(matrix) + len(y) * 1e-4
    matrix += xp.diag(penalty)
    rhs = x.T @ (weight * residual)
    return xp.linalg.solve(matrix, rhs)


def predict_linear(weights, rounds, x):
    return np.clip(np.sum(x * weights[rounds - 1], axis=1) * 1e6, 0, TARGET)


def verify(args, weights, path, name, cp):
    out = args.run / (name + '-probe.jsonl')
    command(args, name + '-probe', 'train', '--probe', out, '--linear', '--games', 48,
            '--seed0', 95_000_000, '--model', path, *settings(args))
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    rounds = np.array([r['round'] for r in rows], dtype=np.int64)
    x = np.array([r['features'] for r in rows], dtype=np.float64)
    cpu = np.array([r['prediction'] for r in rows])
    numpy_pred = predict_linear(weights, rounds, x)
    gpu_pred = cp.asnumpy(cp.clip(cp.sum(cp.asarray(x) * cp.asarray(weights)[cp.asarray(rounds - 1)], axis=1) * 1e6, 0, TARGET))
    error = float(np.max(np.abs(cpu - numpy_pred)))
    gpu_error = float(np.max(np.abs(cpu - gpu_pred)))
    if max(error, gpu_error) > 1e-6:
        raise ValueError(f'linear CPU export mismatch: {error}, GPU={gpu_error}')
    return {'boards': len(rows), 'cpu_export_max_error': error, 'cpu_gpu_max_error_points': gpu_error}


def plot_rounds(run, metrics):
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for name, rounds in metrics.items():
        if name == 'champion':
            continue
        axes[0].plot(range(1, 14), [r['rmse'] - b['rmse'] for r, b in zip(rounds, metrics['champion'])], label=name)
        axes[1].plot(range(1, 14), [None if r['low_rmse'] is None else r['low_rmse'] - b['low_rmse']
                                  for r, b in zip(rounds, metrics['champion'])], label=name)
    for ax in axes:
        ax.axhline(0, color='gray', linestyle='--')
    axes[0].set(title='별도 검증 게임: 전체 예측 오차 변화', xlabel='라운드', ylabel='기존 대비 RMSE 차이 (점, 음수=개선)')
    axes[1].set(title='최종 30만 미만 사례의 예측 오차 변화', xlabel='라운드', ylabel='기존 대비 RMSE 차이 (점, 음수=개선)')
    axes[0].legend(fontsize=8)
    fig.savefig(run / 'finetune.png', dpi=140)
    plt.close(fig)


def fit(args):
    os.environ.setdefault('CUPY_CACHE_DIR', str(args.run.parent / 'cupy-cache'))
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError('a CUDA GPU is required; no CPU fitting fallback')
    # Initialize cuSOLVER before generating the larger linear feature files.
    cp.linalg.solve(cp.eye(2), cp.ones(2))
    cp.cuda.Stream.null.synchronize()
    source = ROOT / 'data/boardValue.json'
    prior = np.array(json.loads(source.read_text())['rounds'], dtype=np.float64)
    if prior.ndim != 2 or prior.shape[0] != 13 or not np.isfinite(prior).all():
        raise ValueError('fine-tuning requires 13 finite linear weight vectors')
    write_json(args.run / 'champion.json', {'rounds': prior.tolist()})
    manifest = {'engine_sha256': sha(args.engine), 'champion_sha256': sha(source),
                'device': 'cuda:0', 'cupy': cp.__version__, 'threads': args.threads,
                'potion_refreshes': args.potion_refreshes, 'instrument_refreshes': args.instrument_refreshes,
                'target': TARGET, 'kind': 'linear-residual', 'candidate_count': len(VARIANTS), 'candidates': [],
                'train_seed': args.seed0, 'train_games': args.games,
                'valid_seed': args.seed0 + args.games, 'valid_games': args.valid_games, 'eps': .03}
    write_json(args.run / 'fit.json', manifest)
    for label, start, games in [('train', args.seed0, args.games), ('valid', args.seed0 + args.games, args.valid_games)]:
        command(args, label + '-rows', 'train', '--rows', args.run / (label + '.bin'), '--linear',
                '--games', games, '--seed0', start, '--eps', .03, '--model', args.run / 'champion.json', *settings(args))
    train, tm = load_rows(args.run / 'train.bin')
    valid, vm = load_rows(args.run / 'valid.bin')
    assert_separate(tm, vm)
    if any(m.get('features') != 'linear-rounds' or m['columns'] != prior.shape[1] + 2 for m in [tm, vm]):
        raise ValueError('linear feature schema/weight count mismatch')
    manifest['data'] = {'train': tm, 'valid': vm}
    manifest['regularization'] = {'diagonal_scale': .1, 'minimum_per_row': 1e-4}
    weights = {name: prior.copy() for name, _, _ in VARIANTS}
    metrics = {name: [] for name in ['champion', *weights]}
    gpu_seconds = 0.0
    round_ids = np.asarray(train[:, 0], dtype=np.int32)
    valid_ids = np.asarray(valid[:, 0], dtype=np.int32)
    for round_id in range(1, 14):
        tr = np.asarray(train[round_ids == round_id])
        vr = np.asarray(valid[valid_ids == round_id])
        if not len(tr) or not len(vr):
            raise ValueError(f'missing round {round_id}')
        cp.cuda.Stream.null.synchronize()
        started = time.perf_counter()
        x, y = cp.asarray(tr[:, 1:-1], dtype=cp.float64), cp.asarray(tr[:, -1], dtype=cp.float64)
        vx, vy = cp.asarray(vr[:, 1:-1], dtype=cp.float64), cp.asarray(vr[:, -1], dtype=cp.float64)
        original = cp.asarray(prior[round_id - 1])
        for low_weight in [1.0, 2.0]:
            delta = correction(x, y, original, low_weight, cp)
            for name, weight, strength in VARIANTS:
                if weight == low_weight:
                    weights[name][round_id - 1] = cp.asnumpy(original + strength * delta)
        for name in metrics:
            w = original if name == 'champion' else cp.asarray(weights[name][round_id - 1])
            residual = cp.clip(vx @ w, 0, TARGET / 1e6) - vy
            low = vy < .3
            low_count = int(cp.count_nonzero(low).get())
            metrics[name].append({'round': round_id, 'rmse': float(cp.sqrt(cp.mean(residual ** 2)).get()) * 1e6,
                                  'low_cases': low_count,
                                  'low_rmse': float(cp.sqrt(cp.mean(residual[low] ** 2)).get()) * 1e6 if low_count else None})
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - started
        gpu_seconds += elapsed
        print(f'GPU round {round_id}/13: {len(tr)} train / {len(vr)} validation boards, {elapsed:.2f}s', flush=True)
        manifest['round_metrics'] = metrics
        manifest['gpu_seconds'] = gpu_seconds
        write_json(args.run / 'fit.json', manifest)
        del x, y, vx, vy, tr, vr, original, delta, w, residual, low
        cp.get_default_memory_pool().free_all_blocks()
    for name, low_weight, strength in VARIANTS:
        path = args.run / (name + '.json')
        if not np.isfinite(weights[name]).all():
            raise ValueError('non-finite fitted coefficients')
        write_json(path, {'rounds': weights[name].tolist()})
        parity = verify(args, weights[name], path, name, cp)
        manifest['candidates'].append({'name': name, 'low_score_weight': low_weight, 'strength': strength,
                                      'sha256': sha(path), 'parity': parity})
        write_json(args.run / 'fit.json', manifest)
    plot_rounds(args.run, metrics)
    print(f'GPU fine-tune complete: {len(VARIANTS)} candidates, {gpu_seconds:.2f}s GPU transfers/fit/validation', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--engine', type=Path, default=ROOT / 'target/release/vorax.exe')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--games', type=int, default=50_000)
    p.add_argument('--valid-games', type=int, default=10_000)
    p.add_argument('--seed0', type=int, default=12_000_000)
    p.add_argument('--potion-refreshes', type=int, default=3)
    p.add_argument('--instrument-refreshes', type=int, default=1)
    args = p.parse_args()
    if min(args.threads, args.games, args.valid_games) <= 0 or not 0 <= args.seed0 < args.seed0 + args.games + args.valid_games < 2**32:
        p.error('positive counts and a valid u32 seed range are required')
    if not all(0 <= n <= 9 for n in [args.potion_refreshes, args.instrument_refreshes]):
        p.error('rerolls must be between 0 and 9')
    args.run, args.engine = args.run.resolve(), args.engine.resolve()
    args.run.mkdir(parents=True, exist_ok=True)
    if (args.run / 'fit.json').exists():
        p.error('output already exists; use a fresh run directory')
    fit(args)


if __name__ == '__main__':
    main()
