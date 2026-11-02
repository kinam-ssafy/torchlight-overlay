"""GPU candidate fitting, CPU simulation gates; never installs or promotes a model.

Stages: fit -> screen -> holdout. See docs/GPU-MODEL-SEARCH.md.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

# Bound helper-library CPU pools too. CUDA owns model fitting, not game rules.
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
from gpu_model import export_model, predict_export
from gpu_statistics import release_quality, wilson_interval

ROOT = Path(__file__).resolve().parents[1]
TARGET = 840_000
CANDIDATES = {'compact': [(2, 32), (3, 32), (4, 32)], 'full': [(3, 64), (5, 96), (7, 128)]}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def command(args, name, *tail):
    logfile = args.run / (name + '.log')
    cmd = [str(args.engine), *map(str, tail), '--threads', str(args.threads)]
    print(f'{name}: CPU threads={args.threads}', flush=True)
    started = time.perf_counter()
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == 'nt' else 0
    with logfile.open('w', encoding='utf-8') as log:
        child = subprocess.Popen(cmd, stdout=log, stderr=log, creationflags=flags)
        try:
            code = child.wait()
        except BaseException:
            child.terminate()
            child.wait()
            raise
    seconds = time.perf_counter() - started
    with (args.run / 'commands.jsonl').open('a', encoding='utf-8') as out:
        out.write(json.dumps({'name': name, 'seconds': seconds, 'exit': code, 'argv': cmd}) + '\n')
    if code:
        raise RuntimeError(f'{name} failed ({code}); see {logfile}')
    print(f'{name}: {seconds:.1f}s', flush=True)
    return seconds


def settings(args):
    return ['--potion-refreshes', args.potion_refreshes, '--instrument-refreshes', args.instrument_refreshes]


def load_rows(path):
    meta = json.loads(Path(str(path) + '.json').read_text())
    if meta.get('dtype') != 'float32-le' or meta.get('target') != TARGET:
        raise ValueError('rows require the current schema and 840k target')
    if path.stat().st_size != meta['rows'] * meta['columns'] * 4:
        raise ValueError('truncated or mismatched row file')
    rows = np.memmap(path, mode='r', dtype='<f4', shape=(meta['rows'], meta['columns']))
    if not np.isfinite(rows).all():
        raise ValueError('non-finite training rows')
    return rows, meta


def assert_separate(a, b):
    if max(a['seed0'], b['seed0']) < min(a['seed0'] + a['games'], b['seed0'] + b['games']):
        raise ValueError('game seed ranges overlap')


def verify_cpu(args, booster, path, name):
    probe = args.run / (name + '-probe.jsonl')
    command(args, name + '-probe', 'train', '--probe', probe, '--games', 48,
            '--seed0', 90_000_000, '--model', path, *settings(args))
    rows = [json.loads(line) for line in probe.read_text().splitlines()]
    x = np.array([r['features'] for r in rows], dtype=np.float64)
    cpu = np.array([r['prediction'] for r in rows])
    model = json.loads(path.read_text())
    exported = np.clip(predict_export(model, x) * 1e6, 0, TARGET)
    import xgboost as xgb
    gpu = np.clip(booster.predict(xgb.DMatrix(x.astype('f4'), nthread=args.threads)) * 1e6, 0, TARGET)
    export_error = float(np.max(np.abs(cpu - exported)))
    gpu_error = float(np.max(np.abs(cpu - gpu)))
    # Float32 tree accumulation differs slightly from the CPU's ordered float64 sum.
    if export_error > 1e-6 or gpu_error > 2.0:
        raise ValueError(f'export mismatch: CPU/interpreter={export_error}, CPU/GPU={gpu_error}')
    return {'boards': len(rows), 'cpu_export_max_error': export_error, 'cpu_gpu_max_error_points': gpu_error}


def fit(args):
    import xgboost as xgb
    if not xgb.build_info().get('USE_CUDA'):
        raise RuntimeError('XGBoost was built without CUDA; no silent CPU fallback')
    # Fail fast before generating data if CUDA cannot actually train a model.
    x = np.arange(64, dtype='f4').reshape(32, 2)
    probe = xgb.train({'device': 'cuda', 'tree_method': 'hist', 'max_depth': 1, 'nthread': args.threads},
                      xgb.QuantileDMatrix(x, x[:, 0], nthread=args.threads), num_boost_round=1)
    device = json.loads(probe.save_config())['learner']['generic_param']['device']
    if not device.startswith('cuda'):
        raise RuntimeError(f'GPU requested but XGBoost used {device}')
    manifest = {'engine_sha256': sha(args.engine), 'champion_sha256': sha(ROOT / 'data/boardValue.json'),
                'xgboost': xgb.__version__, 'device': device, 'threads': args.threads,
                'potion_refreshes': args.potion_refreshes, 'instrument_refreshes': args.instrument_refreshes,
                'target': TARGET, 'candidates': [], 'train_seed': args.seed0,
                'train_games': args.games, 'valid_seed': args.seed0 + args.games, 'valid_games': args.valid_games,
                'profile': args.profile}
    write_json(args.run / 'fit.json', manifest)
    for label, start, games in [('train', args.seed0, args.games), ('valid', args.seed0 + args.games, args.valid_games)]:
        path = args.run / (label + '.bin')
        command(args, label + '-rows', 'train', '--rows', path, '--games', games,
                '--seed0', start, '--eps', .15, *settings(args))
    train, tm = load_rows(args.run / 'train.bin')
    valid, vm = load_rows(args.run / 'valid.bin')
    assert_separate(tm, vm)
    manifest['data'] = {'train': tm, 'valid': vm}
    write_json(args.run / 'fit.json', manifest)
    plot_data(args.run, train, valid)
    print(f'GPU fitting: {len(train)} training boards, {len(valid)} validation boards on {device}', flush=True)
    dt = xgb.QuantileDMatrix(train[:, :-1], train[:, -1], max_bin=128, nthread=args.threads)
    dv = xgb.QuantileDMatrix(valid[:, :-1], valid[:, -1], ref=dt, max_bin=128, nthread=args.threads)
    for depth, rounds in CANDIDATES[args.profile]:
        name = f'd{depth}-n{rounds}'
        history = {}
        started = time.perf_counter()
        params = {'device': 'cuda', 'tree_method': 'hist', 'objective': 'reg:squarederror',
                  'base_score': .4, 'max_depth': depth, 'eta': .08, 'max_bin': 128,
                  'min_child_weight': 40, 'lambda': 8, 'subsample': .85,
                  'seed': 713, 'nthread': args.threads, 'eval_metric': 'rmse'}
        booster = xgb.train(params, dt, num_boost_round=rounds, evals=[(dt, 'train'), (dv, 'valid')],
                            evals_result=history, early_stopping_rounds=16, verbose_eval=32)
        config = json.loads(booster.save_config())
        if not config['learner']['generic_param']['device'].startswith('cuda'):
            raise RuntimeError('candidate silently fell back to CPU')
        booster = booster[:booster.best_iteration + 1]
        elapsed = time.perf_counter() - started
        booster.save_model(args.run / (name + '.xgb.json'))
        path = args.run / (name + '.json')
        write_json(path, export_model(booster))
        parity = verify_cpu(args, booster, path, name)
        result = {'name': name, 'depth': depth, 'trees': booster.num_boosted_rounds(),
                  'fit_seconds': elapsed, 'validation_rmse_points': min(history['valid']['rmse']) * 1e6,
                  'history': history, 'parity': parity, 'sha256': sha(path)}
        manifest['candidates'].append(result)
        write_json(args.run / 'fit.json', manifest)
        plot_fit(args.run, manifest)
        print(f'{name}: GPU {elapsed:.1f}s; CPU/GPU max delta {parity["cpu_gpu_max_error_points"]:.3f} points', flush=True)


def scores(path):
    rows = json.loads(path.read_text())
    result = {r['seed']: r for r in rows}
    if not result or len(result) != len(rows):
        raise ValueError('empty or duplicate evaluation seeds')
    return result


def summarize(rows):
    s = np.array([r['score'] for r in rows.values()])
    return {'games': len(s), 'rate840k': float(np.mean(s >= TARGET)),
            'rate840k_interval95': wilson_interval(int(np.sum(s >= TARGET)), len(s)),
            'below300k_interval95': wilson_interval(int(np.sum(s < 300_000)), len(s)),
            'capped_mean': float(np.minimum(s, TARGET).mean()), 'p10': float(np.quantile(s, .1)),
            'below300k': int(np.sum(s < 300_000)),
            'monsters': float(np.mean([r['monsters'] for r in rows.values()])),
            'groups': float(np.mean([r['groups'] for r in rows.values()]))}


def paired(champion, candidate):
    if champion.keys() != candidate.keys():
        raise ValueError('evaluation seeds must match exactly')
    gained = sum(candidate[k]['score'] >= TARGET > champion[k]['score'] for k in champion)
    lost = sum(champion[k]['score'] >= TARGET > candidate[k]['score'] for k in champion)
    from scipy.stats import binomtest
    p = float(binomtest(gained, gained + lost, p=.5).pvalue) if gained + lost else 1.0
    return {'gained': gained, 'lost': lost, 'paired_p': p}


def eligible_for_review(baseline, candidate, release_quality):
    return (release_quality and candidate['engine_gate_pass'] and candidate.get('adjusted_p', candidate['paired_p']) < .05
            and candidate['p10'] >= baseline['p10'] and candidate['below300k'] <= baseline['below300k']
            and candidate['seconds'] <= baseline['seconds'] * 1.25)


def select_candidate(results):
    candidates = results[1:]
    timely = [r for r in candidates if r['seconds'] <= results[0]['seconds'] * 1.25]
    qualified = [r for r in candidates if r.get('eligible')]
    winner = max(qualified or timely or candidates,
                 key=lambda r: (r['rate840k'], r['capped_mean'], r['p10'], -r['monsters']))
    return winner, bool(timely)


def evaluate(args, holdout=False):
    manifest = json.loads((args.run / 'fit.json').read_text())
    if sha(args.engine) != manifest['engine_sha256'] or sha(ROOT / 'data/boardValue.json') != manifest['champion_sha256']:
        raise ValueError('engine/champion changed since fit; use a fresh run')
    if (args.potion_refreshes, args.instrument_refreshes) != (manifest['potion_refreshes'], manifest['instrument_refreshes']):
        raise ValueError('evaluation rerolls must match training')
    expected = manifest['candidate_count'] if 'candidate_count' in manifest else len(CANDIDATES[manifest.get('profile', 'full')])
    if len(manifest['candidates']) != expected:
        raise ValueError('fit did not finish all candidates')
    for candidate in manifest['candidates']:
        if sha(args.run / (candidate['name'] + '.json')) != candidate['sha256']:
            raise ValueError('exported candidate changed since fit')
    phase = 'holdout' if holdout else 'screen'
    seed = args.eval_seed + (1_000_000 if holdout else 0)
    for prefix in ('train', 'valid'):
        assert_separate({'seed0': seed, 'games': args.eval_games},
                        {'seed0': manifest[prefix + '_seed'], 'games': manifest[prefix + '_games']})
    if holdout:
        screening = json.loads((args.run / 'screen.json').read_text())
        assert_separate({'seed0': seed, 'games': args.eval_games}, screening)
        names = args.finalists.split(',') if args.finalists else [screening['winner']]
        known = {c['name'] for c in manifest['candidates']}
        if len(names) != len(set(names)) or not set(names) <= known:
            raise ValueError('finalists must be distinct fitted candidates')
    else:
        names = [c['name'] for c in manifest['candidates']]
    report = {'phase': phase, 'policy': 'solver', 'futures': args.futures,
              'seed0': seed, 'games': args.eval_games, 'results': [], 'promoted': False,
              'release_quality': release_quality(holdout, args.eval_games, args.futures),
              'finalists': names, 'comparison_count': len(names)}
    base = None
    for name in ['champion', *names]:
        model = ['--model', ROOT / 'data/boardValue.json' if name == 'champion' else args.run / (name + '.json')]
        path = args.run / f'{phase}-{name}.json'
        scorefile = args.run / f'{phase}-{name}.txt'
        elapsed = command(args, f'{phase}-{name}', 'sim', '--policy', 'solver', '--goal', 'target',
                          '--games', args.eval_games, '--seed0', seed, '--futures', args.futures,
                          '--outcomes', path, '--scores', scorefile, *model, *settings(args))
        rows = scores(path)
        if set(rows) != set(range(seed, seed + args.eval_games)):
            raise ValueError('simulation did not return the requested seeds')
        result = {'name': name, 'seconds': elapsed, **summarize(rows)}
        if base is None:
            base = rows
        else:
            result.update(paired(base, rows))
            result['adjusted_p'] = min(1.0, result['paired_p'] * len(names))
            command(args, f'{phase}-{name}-gate', 'gate', args.run / f'{phase}-champion.txt', scorefile)
            gate = (args.run / f'{phase}-{name}-gate.log').read_text(encoding='utf-8')
            result['engine_gate_pass'] = 'pass=yes' in gate
            first = report['results'][0]
            # No distribution or latency regression accepted for a shipping recommendation.
            result['eligible'] = eligible_for_review(first, result, report['release_quality'])
        report['results'].append(result)
        write_json(args.run / (phase + '.json'), report)
        plot_scores(args.run, report)
    winner, timely = select_candidate(report['results'])
    report['winner'] = winner['name']
    report['within_cpu_budget'] = timely
    report['decision'] = ('eligible_for_review' if winner['eligible'] else 'keep_champion') if holdout else 'screening_only'
    write_json(args.run / (phase + '.json'), report)
    print(json.dumps({'phase': phase, 'winner': winner['name'], 'decision': report['decision']}), flush=True)


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Malgun Gothic' if os.name == 'nt' else 'DejaVu Sans',
                         'axes.unicode_minus': False, 'font.size': 11})
    return plt


def plot_data(run, train, valid):
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.7), layout='constrained')
    for rows, label in [(train, '학습'), (valid, '검증')]:
        rounds, counts = np.unique(rows[:, 0], return_counts=True)
        axes[0].plot(rounds, counts, 'o-', label=f'{label}: {len(rows):,}개 상태')
        axes[1].hist(rows[:, -1] * 100, bins=np.linspace(0, 84, 22), density=True, histtype='step', linewidth=2, label=label)
    axes[0].set(xlabel='라운드', ylabel='상태 수', title='서로 다른 게임으로 학습/검증 분리')
    axes[1].set(xlabel='최종 점수 (만, 84만 상한)', ylabel='상대 밀도', title='GPU 학습에 사용하는 정답 분포')
    for ax in axes:
        ax.legend(fontsize=9)
    fig.savefig(run / 'data.png', dpi=140)
    plt.close(fig)


def plot_fit(run, report):
    plt = plotting()
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    for c in report['candidates']:
        ax.plot(np.arange(1, len(c['history']['valid']['rmse']) + 1), np.array(c['history']['valid']['rmse']) * 100,
                label=f'{c["name"]}: {c["fit_seconds"]:.1f}초')
    ax.set(xlabel='추가한 트리 수', ylabel='검증 RMSE (만 점)', title='GPU 후보 학습: 낮을수록 예측 오차가 작음')
    ax.legend(title='후보 / GPU 학습 시간')
    fig.savefig(run / 'learning.png', dpi=140)
    plt.close(fig)


def plot_scores(run, report):
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for r in report['results']:
        rows = scores(run / f'{report["phase"]}-{r["name"]}.json')
        values = np.sort([v['score'] for v in rows.values()]) / 10_000
        label = f'{r["name"]}: {r["rate840k"]:.1%}'
        axes[0].plot(values, np.arange(1, len(values) + 1) / len(values) * 100, label=label)
    axes[0].axvline(84, color='gray', linestyle='--', label='목표 84만')
    label = '후보 선별' if report['phase'] == 'screen' else ('독립 최종 검증' if report.get('release_quality') else '독립 추가 검증')
    axes[0].set(xlabel='최종 점수 (만, 0~140만 구간)', ylabel='누적 비율 (%)', xlim=(0, 140), title=f'{label}: CPU 시뮬레이션 점수 분포')
    axes[0].legend(fontsize=9)
    names = [r['name'] for r in report['results']]
    bars = axes[1].barh(names, [r['seconds'] / r['games'] for r in report['results']])
    axes[1].bar_label(bars, fmt='%.3f', padding=4)
    axes[1].margins(x=.15)
    axes[1].set(xlabel='게임당 경과 시간 (초)', title=f'동일 조건: {report["games"]}게임, futures={report["futures"]}')
    fig.savefig(run / (report['phase'] + '.png'), dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['fit', 'screen', 'holdout'])
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--engine', type=Path, default=ROOT / 'target/release/vorax.exe')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--games', type=int, default=50_000)
    p.add_argument('--valid-games', type=int, default=10_000)
    p.add_argument('--seed0', type=int, default=9_200_000)
    p.add_argument('--eval-seed', type=int, default=92_000_000)
    p.add_argument('--eval-games', type=int, default=128)
    p.add_argument('--futures', type=int, default=64)
    p.add_argument('--profile', choices=CANDIDATES, default='compact')
    p.add_argument('--finalists', help='preselected comma-separated candidate names for holdout; Bonferroni adjusted')
    p.add_argument('--potion-refreshes', type=int, default=3)
    p.add_argument('--instrument-refreshes', type=int, default=1)
    args = p.parse_args()
    if args.finalists and args.stage != 'holdout':
        p.error('--finalists is only supported for holdout')
    if min(args.threads, args.games, args.valid_games, args.eval_games, args.futures) <= 0:
        p.error('counts must be positive')
    if not (0 <= args.potion_refreshes <= 9 and 0 <= args.instrument_refreshes <= 9):
        p.error('rerolls must be between 0 and 9')
    if not (0 <= args.seed0 < args.seed0 + args.games + args.valid_games < 2**32
            and 0 <= args.eval_seed < args.eval_seed + 1_000_000 + args.eval_games < 2**32):
        p.error('seed ranges must fit u32')
    args.run = args.run.resolve()
    args.engine = args.engine.resolve()
    args.run.mkdir(parents=True, exist_ok=True)
    if args.stage == 'fit':
        if (args.run / 'fit.json').exists():
            p.error('fit output already exists; choose a new run directory')
        fit(args)
    else:
        evaluate(args, holdout=args.stage == 'holdout')


if __name__ == '__main__':
    main()
