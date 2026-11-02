"""Render measured synergy training and fixed paired holdout results."""
import argparse
import json
from pathlib import Path
import numpy as np
from gpu_search import plotting
from gpu_statistics import wilson_interval


def progress(run, out):
    path = run / 'holdout-plan.json'
    if not path.exists():
        return
    plan = json.loads(path.read_text())
    names = ['champion', plan['candidate']]
    counts, totals, points = [], [0, 0], [[], []]
    games = 0
    for chunk in range((plan['games'] + plan['chunk'] - 1) // plan['chunk']):
        labels = [f'chunk-{chunk:03d}-{name}' for name in names]
        if not all((run / (label + '-complete.json')).exists() for label in labels):
            break
        rows = [json.loads((run / (label + '.json')).read_text()) for label in labels]
        if [r['seed'] for r in rows[0]] != [r['seed'] for r in rows[1]]:
            raise ValueError('progress seeds differ')
        games += len(rows[0])
        counts.append(games)
        for i in range(2):
            totals[i] += sum(r['score'] >= 840000 for r in rows[i])
            points[i].append([totals[i] / games, *wilson_interval(totals[i], games)])
    if not games:
        return
    plt = plotting()
    fig, ax = plt.subplots(figsize=(9, 4.4), layout='constrained')
    for i, label in enumerate(['기존 1.0.9', '시너지 후보']):
        values = np.asarray(points[i]) * 100
        line, = ax.plot(counts, values[:, 0], marker='.', label=label)
        ax.fill_between(counts, values[:, 1], values[:, 2], alpha=.12, color=line.get_color())
    ax.set(xlabel='완료한 동일 시드 쌍 (판)', ylabel='누적 84만 달성률 (%)',
           title=f'{games:,} / {plan["games"]:,}쌍: 누적 달성률과 개별 95% Wilson 구간')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=10)
    fig.savefig(out / 'gpu-synergy-progress.png', dpi=145)
    plt.close(fig)


def learning(run, out):
    fit = json.loads((run / 'fit.json').read_text())
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout='constrained')
    baseline = fit['round_metrics']['champion']
    for c in fit['candidates']:
        name = c['name']
        metrics = fit['round_metrics'][name]
        label = f"추가 가중치 {c['strength']:.0%}"
        axes[0].plot(range(1, 14), [(r['rmse'] - b['rmse']) / 1000 for r, b in zip(metrics, baseline)], marker='.', label=label)
        axes[1].plot(range(1, 14), [(r['low_rmse'] - b['low_rmse']) / 1000 for r, b in zip(metrics, baseline)], marker='.', label=label)
    for ax in axes:
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set(xlabel='진행 라운드', ylabel='기존 대비 예측 오차 변화 (천 점)', xticks=[1, 3, 5, 7, 9, 11, 13])
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_title(f'별도 검증 {fit["valid_games"]:,}판: 전체 상태')
    axes[1].set_title('최종 30만 미만 게임의 상태')
    axes[0].legend(fontsize=9)
    fig.suptitle('시너지 입력 학습: 0 아래면 예측 오차 감소 (추천 성적과 별개)', fontsize=13)
    fig.savefig(out / 'gpu-synergy-learning.png', dpi=145)
    plt.close(fig)


def holdout(run, out):
    report = json.loads((run / 'holdout.json').read_text())
    a, b = report['results']
    plt = plotting()
    fig, grid = plt.subplots(2, 2, figsize=(10.5, 7.6), layout='constrained')
    axes = grid.ravel()
    names = ['기존 1.0.9', '시너지 후보']
    for i, r in enumerate([a, b]):
        rate = r['rate840k'] * 100
        lo, hi = np.asarray(r['rate840k_interval95']) * 100
        axes[0].errorbar(i, rate, yerr=[[rate - lo], [hi - rate]], fmt='o', capsize=6, color=['#326da8', '#be641b'][i])
        axes[0].annotate(f'{rate:.2f}%', (i, hi), xytext=(0, 9), textcoords='offset points', ha='center')
        rows = json.loads((run / f'holdout-{r["name"]}.json').read_text())
        points = np.sort([row['score'] for row in rows]) / 10000
        axes[1].plot(points, np.arange(1, len(points) + 1) / len(points) * 100, label=names[i])
    axes[0].set(xticks=[0, 1], xticklabels=names, xlim=(-.5, 1.5), ylabel='84만 달성률 (%)', title='달성률과 95% Wilson 구간')
    axes[0].margins(y=.3)
    axes[1].axvline(84, color='gray', linestyle='--', linewidth=1)
    axes[1].set(xlabel='최종 점수 (만 점)', ylabel='누적 비율 (%)', xlim=(0, 140), title='최종 점수 분포 (140만까지)')
    axes[1].legend(fontsize=9)
    bars = axes[2].bar(names, [r['below300k'] for r in [a, b]], color=['#326da8', '#be641b'], width=.5)
    axes[2].bar_label(bars, padding=4)
    axes[2].set(ylabel=f'30만 미만 횟수 / {report["games"]}판', title='낮은 점수 발생 빈도')
    axes[2].margins(y=.2)
    bars = axes[3].bar(names, [r['seconds'] / r['games'] for r in [a, b]], color=['#326da8', '#be641b'], width=.5)
    axes[3].bar_label(bars, fmt='%.3f', padding=4)
    axes[3].set(ylabel='게임당 경과 시간 (초)', title='CPU 완주 시뮬레이션 비용')
    axes[3].margins(y=.2)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    plan = json.loads((run / 'holdout-plan.json').read_text())
    fig.suptitle(f'독립 {report["games"]:,}개 시드 / {report["futures"]} futures / CPU {plan["threads"]}스레드\n84만 달성률 차이의 paired p={b["paired_p"]:.4f}', fontsize=13)
    fig.savefig(out / 'gpu-synergy-independent.png', dpi=145)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    learning(args.run, args.out)
    progress(args.run, args.out)
    if (args.run / 'holdout.json').exists():
        holdout(args.run, args.out)


if __name__ == '__main__':
    main()
