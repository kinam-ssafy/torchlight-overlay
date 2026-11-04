"""Auditable, gradient-free edits of the shipped linear value model.

Development only. The existing screen/paired-holdout tools evaluate these files;
this script never replaces the shipped model or installs an executable.
"""
import argparse
import json
from pathlib import Path

from gpu_finetune import verify
from gpu_search import ROOT, sha, write_json, plotting
import numpy as np


VARIANTS = [('resource-contract', 0.0), ('grounded-10', .10), ('grounded-25', .25)]


def edited(prior, strength):
    if prior.shape != (13, 656) or not np.isfinite(prior).all() or not 0 <= strength <= 1:
        raise ValueError('expected finite 13x656 legacy weights and strength in [0,1]')
    weights = prior.copy() * (1 - strength)
    # Feature 3 is min(current score, 950k)/1e6. The existing CPU output clamp
    # remains 840k. This is a conservative heuristic candidate, not a theorem
    # that a high current score cannot be lost through later tool effects.
    weights[:, 3] += strength
    # Extra optional rerolls cannot reduce the value of the available choices.
    weights[:, 23:25] = np.maximum(weights[:, 23:25], 0)
    # game.rs stops dealing potions after round 10. Unused potion rerolls then
    # confer no remaining choices; retaining their bonus can hit the value cap.
    weights[10:, 23] = 0
    return weights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--potion-refreshes', type=int, default=3)
    p.add_argument('--instrument-refreshes', type=int, default=1)
    args = p.parse_args()
    if args.threads < 1 or not all(0 <= n <= 9 for n in [args.potion_refreshes,args.instrument_refreshes]):
        p.error('invalid thread count or rerolls')
    args.run, args.engine = args.run.resolve(), args.engine.resolve()
    if (args.run / 'fit.json').exists():
        p.error('use a fresh output directory')
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError('GPU required for CPU/CUDA export verification')
    args.run.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'data/boardValue.json'
    prior = np.asarray(json.loads(source.read_text())['rounds'],dtype=np.float64)
    (args.run / 'champion.json').write_bytes(source.read_bytes())
    manifest = {'method':'direct semantic coefficient edits; no training or gradients',
                'engine_sha256':sha(args.engine),'champion_sha256':sha(source),
                'train_seed':0,'train_games':0,'valid_seed':0,'valid_games':0,
                'potion_refreshes':args.potion_refreshes,'instrument_refreshes':args.instrument_refreshes,
                'candidate_count':len(VARIANTS),'candidates':[],
                'screen_plan':{'seed0':102_000_000,'games':512,'futures':128},
                'holdout_plan':{'seed0':103_000_000,'games':4096,'futures':256},
                'promotion':'paired holdout gate only; never automatic installation'}
    write_json(args.run / 'fit.json',manifest)
    for name,strength in VARIANTS:
        weights = edited(prior,strength)
        path = args.run / (name + '.json')
        write_json(path,{'rounds':weights.tolist()})
        changes = [{'round':int(r+1),'feature':int(c),'before':float(prior[r,c]),'after':float(weights[r,c])}
                   for r,c in np.argwhere(weights != prior)]
        write_json(args.run / (name + '-changes.json'),changes)
        parity = verify(args,weights,path,name,cp)
        manifest['candidates'].append({'name':name,'strength':strength,'sha256':sha(path),
                                       'changed_coefficients':len(changes),'parity':parity})
        write_json(args.run / 'fit.json',manifest)
    plt = plotting()
    fig,axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    rounds = np.arange(1,14)
    for name,w in [('champion',prior),('resource-contract',edited(prior,0))]:
        axes[0].plot(rounds,w[:,23]*1e6/3,marker='o',label=name)
        axes[1].plot(rounds,w[:,24]*1e6/2,marker='o',label=name)
    for ax,title in zip(axes,['물약 리롤 1회가 더하는 예측 점수','도구 리롤 1회가 더하는 예측 점수']):
        ax.set(title=title,xlabel='라운드',ylabel='점수 (최종 출력 상한 적용 전)')
        ax.axhline(0,color='gray',linewidth=.7)
        ax.legend(fontsize=8)
    fig.savefig(args.run / 'weight-audit.png',dpi=140)
    plt.close(fig)
    print(json.dumps({'candidates':manifest['candidates'],'shipped_model_unchanged':sha(source)==manifest['champion_sha256']}),flush=True)


if __name__ == '__main__':
    main()
