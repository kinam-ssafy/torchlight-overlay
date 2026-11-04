"""Bounded Monte Carlo policy iteration on CUDA, with CPU-only exported values.

Two fresh 50k-game generations, separate validation, then a frozen screen/holdout.
Special entries are deliberately stratified experiments, NOT live drop odds.
No installation, promotion, sequential p-value stopping, or automatic retry loop.
"""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import numpy as np
from gpu_search import ROOT, command, settings, sha, write_json
from gpu_statistics import paired_rates, planning_games, wilson_interval

TARGET = 1_120_000
FEATURES = 1316
TRAIN = [30_000, 5_000, 5_000, 5_000, 5_000]
VALID = [3_000, 500, 500, 500, 500]


def experiment(scenario, objective):
    return ['--goal', objective, '--target-score', TARGET, '--scenario', scenario]


def load(path, success):
    meta = json.loads(Path(str(path) + '.json').read_text())
    if (meta['features'], meta['columns'], meta['target'], meta['success_objective']) != ('linear-chain-v1', FEATURES + 2, TARGET, success):
        raise ValueError('wrong training schema or objective')
    if path.stat().st_size != meta['rows'] * meta['columns'] * 4:
        raise ValueError('truncated data')
    rows = np.memmap(path, mode='r', dtype='<f4', shape=(meta['rows'], meta['columns']))
    for start in range(0, len(rows), 8192):
        if not np.isfinite(rows[start:start+8192]).all():
            raise ValueError('non-finite data')
    return rows, meta


def weights(path):
    model = json.loads(path.read_text())
    if 'chain_version' in model:
        return np.concatenate([model['legacy_rounds'], model['chain_rounds']], axis=1)
    return np.concatenate([model['rounds'], np.zeros((13, 660))], axis=1)


def fit(args):
    os.environ.setdefault('CUPY_CACHE_DIR', str(args.run.parent / 'cupy-cache'))
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError('CUDA is required; no silent CPU fit fallback')
    cp.linalg.solve(cp.eye(2), cp.ones(2))
    incumbent = args.run / 'champion.json'
    write_json(incumbent, json.loads((ROOT / 'data/boardValue.json').read_text()))
    plan = {
        'algorithm': 'two-generation regularized Monte Carlo fitted policy iteration',
        'engine_sha256': sha(args.engine), 'champion_sha256': sha(incumbent),
        'target': TARGET, 'target_status': 'community preparation-score threshold, not verified current loot probability',
        'objective': args.objective, 'seed_offset': args.seed_offset,
        'train_games_per_generation': TRAIN, 'valid_games_per_generation': VALID,
        'train_seed_base': args.seed_offset+40_000_000, 'valid_seed_base': args.seed_offset+50_000_000,
        'screen_seed': args.seed_offset+60_000_000, 'screen_games': 512, 'screen_futures': 64,
        'holdout_seed': args.seed_offset+70_000_000, 'holdout_games': 8192, 'holdout_futures': 256,
        'strata_seed': args.seed_offset+80_000_000, 'strata_games_per_family': 256,
        'alpha': .05, 'planned_power': .8, 'detectable_difference_pp': 2.,
        'assumed_discordance': .4, 'required_games_approx': planning_games(.4, .02),
        'selection': 'highest screen >=1120k count among retargeted incumbent and two fitted generations satisfying floor guard; no holdout selection',
        'screen_floor_guard': 'below300k no more than incumbent; p10 at least incumbent; CPU at most 1.35x incumbent',
        'release_guard': 'positive primary difference with paired p<.05; also report 840k, lower tail and latency; no automatic promotion',
        'step_size': .5, 'ridge_diagonal': .1, 'ridge_minimum_per_row': 1e-4,
        'exploration': .05, 'threads': args.threads, 'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        'settings': settings(args), 'scenario_weights': '60% ordinary, 10% each of four forced round-2 entries; not live odds',
        'baseline': 'same audited engine, incumbent weights and legacy target840k objective',
        'limitations': ['unknown real entry frequencies', 'unverified rarity/event ordering for some chains',
                        'round-2 entry strata do not establish late-entry performance',
                        'independent games, not correlated state rows, are statistical units',
                        'same initial seeds do not imply identical subsequent offers after divergent actions'],
    }
    if args.resume:
        saved = json.loads((args.run/'plan.json').read_text())
        expected = json.loads((args.run/'plan.sha.json').read_text())['sha256']
        if saved != plan or sha(args.run/'plan.json') != expected:
            raise ValueError('resume cannot change the preregistered experiment')
        report = json.loads((args.run/'fit.json').read_text())
        for g in report['generations']:
            if sha(args.run/f'g{g["generation"]}.json') != g['candidate_sha256']:
                raise ValueError('completed model changed')
            for label in ('train','valid'):
                for scenario in range(5):
                    path=args.run/f'g{g["generation"]}-{label}-s{scenario}.bin'
                    if path.exists(): path.unlink()
    else:
        write_json(args.run / 'plan.json', plan)
        write_json(args.run / 'plan.sha.json', {'sha256': sha(args.run / 'plan.json')})
        report = {'generations': [], 'total_training_games': 0, 'cuda_seconds': 0.}
    completed=len(report['generations'])
    behavior = incumbent if completed==0 else args.run/f'g{completed}.json'
    for generation in range(completed+1,3):
        paths, metadata = {}, {}
        for label, counts, seedbase in [('train',TRAIN,40_000_000),('valid',VALID,50_000_000)]:
            paths[label] = []
            metadata[label] = []
            for scenario, count in enumerate(counts):
                seed = args.seed_offset + seedbase + generation * 1_000_000 + scenario * 100_000
                name = f'g{generation}-{label}-s{scenario}'
                path = args.run / (name + '.bin')
                command(args, name, 'train', '--rows', path, '--chain', '--games', count,
                        '--seed0', seed, '--eps', .05, '--model', behavior, *experiment(scenario,args.objective), *settings(args))
                rows, meta = load(path,args.objective=='success')
                del rows
                if meta['games'] != count or meta['seed0'] != seed or meta['scenario'] != scenario:
                    raise ValueError('data provenance mismatch')
                metadata[label].append({**meta, 'sha256': sha(path)})
                paths[label].append(path)
        old = weights(behavior)
        new = old.copy()
        round_metrics = []
        for r in range(1,14):
            started = time.perf_counter()
            matrix = cp.zeros((FEATURES, FEATURES), dtype=cp.float64)
            rhs = cp.zeros(FEATURES, dtype=cp.float64)
            w = cp.asarray(old[r-1])
            count = 0
            for path in paths['train']:
                rows, _ = load(path,args.objective=='success')
                for at in range(0,len(rows),8192):
                    chunk = np.asarray(rows[at:at+8192])
                    part = chunk[chunk[:,0] == r]
                    if not len(part): continue
                    x = cp.asarray(part[:,1:-1], dtype=cp.float64)
                    y = cp.asarray(part[:,-1], dtype=cp.float64)
                    matrix += x.T @ x
                    rhs += x.T @ (y - x @ w)
                    count += len(part)
                del rows, chunk, part
            matrix += cp.diag(.1 * cp.diag(matrix) + count * 1e-4)
            delta = cp.linalg.solve(matrix, rhs)
            new[r-1] += .5 * cp.asnumpy(delta)
            squared = np.zeros(2)
            vn = 0
            nw = cp.asarray(new[r-1])
            for path in paths['valid']:
                rows, _ = load(path,args.objective=='success')
                for at in range(0,len(rows),8192):
                    chunk = np.asarray(rows[at:at+8192])
                    part = chunk[chunk[:,0] == r]
                    if not len(part): continue
                    x = cp.asarray(part[:,1:-1], dtype=cp.float64)
                    y = cp.asarray(part[:,-1], dtype=cp.float64)
                    for i, model in enumerate([w,nw]):
                        squared[i] += float(cp.sum((cp.clip(x @ model,0,TARGET/1e6)-y)**2).get())
                    vn += len(part)
                del rows, chunk, part
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter()-started
            report['cuda_seconds'] += elapsed
            round_metrics.append({'round':r,'train_rows':count,'valid_rows':vn,'rmse_probability':(np.sqrt(squared/vn)/(TARGET/1e6)).tolist()})
            print(f'g{generation} CUDA round {r}/13: {count} rows, {elapsed:.1f}s',flush=True)
            del matrix,rhs,w,nw,delta,x,y
            cp.get_default_memory_pool().free_all_blocks()
        if not np.isfinite(new).all(): raise ValueError('non-finite model')
        candidate = args.run / f'g{generation}.json'
        write_json(candidate, {'chain_version':1,'legacy_rounds':new[:,:656].tolist(),'chain_rounds':new[:,656:].tolist()})
        parity = []
        for scenario in range(5):
            name = f'g{generation}-probe-s{scenario}'
            path = args.run / (name+'.jsonl')
            command(args,name,'train','--probe',path,'--chain','--games',32,'--seed0',args.seed_offset+96_000_000+scenario*100,
                    '--model',candidate,*experiment(scenario,args.objective),*settings(args))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            x = np.array([r['features'] for r in rows])
            indices = np.array([r['round']-1 for r in rows])
            cpu = np.array([r['prediction'] for r in rows])
            gpu = cp.asnumpy(cp.clip(cp.sum(cp.asarray(x)*cp.asarray(new)[cp.asarray(indices)],axis=1)*1e6,0,TARGET))
            error = float(np.max(abs(gpu-cpu)))
            if error > 1e-6: raise ValueError(f'CPU/CUDA mismatch {error}')
            parity.append({'scenario':scenario,'boards':len(rows),'max_error_points':error,'chain_rows':int(np.count_nonzero(np.any(x[:,1220:]!=0,axis=1)))})
        report['total_training_games'] += sum(TRAIN)
        report['generations'].append({'generation':generation,'behavior_sha256':sha(behavior),'candidate_sha256':sha(candidate),
                                      'data':metadata,'metrics':round_metrics,'parity':parity})
        write_json(args.run/'fit.json',report)
        behavior = candidate
        # Only our exact validated row files, never a recursive directory delete.
        for group in paths.values():
            for path in group: path.unlink()


def simulate(args, name, model, games, seed, futures, scenario=0, legacy=False):
    path = args.run/(name+'.json')
    if path.exists(): raise ValueError('refusing to overwrite evaluation outcomes')
    opts = ['--goal','target','--target-score',840_000,'--scenario',scenario] if legacy else experiment(scenario,args.objective)
    seconds = command(args,name,'sim','--policy','solver','--futures',futures,'--games',games,'--seed0',seed,
                      '--outcomes',path,'--model',model,*opts,*settings(args))
    rows = json.loads(path.read_text())
    if {r['seed'] for r in rows} != set(range(seed,seed+games)) or len(rows)!=games:
        raise ValueError('missing, duplicate or mismatched evaluation games')
    return rows, seconds


def summary(rows):
    scores = np.array([r['score'] for r in rows])
    return {'games':len(rows),'success1120k':int(np.sum(scores>=TARGET)),
            'success840k':int(np.sum(scores>=840_000)),'below300k':int(np.sum(scores<300_000)),
            'p10':float(np.quantile(scores,.1)),'mean':float(scores.mean()),
            'monsters':float(np.mean([r['monsters'] for r in rows])),
            'groups':float(np.mean([r['groups'] for r in rows]))}


def compare(a,b):
    if not a or len(a)!=len(b) or len({r['seed'] for r in a})!=len(a) or len({r['seed'] for r in b})!=len(b):
        raise ValueError('nonempty, matched unique game seeds required')
    result = {}
    for name, predicate in [('success1120k',lambda s:s>=TARGET),('success840k',lambda s:s>=840_000),('below300k',lambda s:s<300_000)]:
        aa = {r['seed']:predicate(r['score']) for r in a}
        bb = {r['seed']:predicate(r['score']) for r in b}
        stats = paired_rates(aa,bb)
        # Paired multinomial bootstrap preserves within-game dependence.
        counts = [stats['gained'],stats['lost'],len(a)-stats['discordant']]
        draws = np.random.default_rng(911).multinomial(len(a),np.array(counts)/len(a),size=20000)
        stats['difference_pp_bootstrap95'] = np.quantile((draws[:,0]-draws[:,1])*100/len(a),[.025,.975]).tolist()
        result[name] = stats
    return result


def screen_eligible(baseline, candidate):
    return (candidate['below300k']<=baseline['below300k'] and candidate['p10']>=baseline['p10']
            and candidate['seconds']<=1.35*baseline['seconds'])


def evaluate(args):
    plan = json.loads((args.run/'plan.json').read_text())
    if sha(args.engine)!=plan['engine_sha256']: raise ValueError('engine changed after preregistration')
    if (args.objective,args.seed_offset,settings(args),args.threads)!=(plan['objective'],plan['seed_offset'],plan['settings'],plan['threads']):
        raise ValueError('evaluation differs from preregistration')
    if sha(args.run/'plan.json')!=json.loads((args.run/'plan.sha.json').read_text())['sha256']:
        raise ValueError('preregistered plan changed')
    write_json(args.run/'retargeted.json',json.loads((args.run/'champion.json').read_text()))
    screen = []
    for name in ['champion','retargeted','g1','g2']:
        rows, seconds = simulate(args,'screen-'+name,args.run/(name+'.json'),512,plan['screen_seed'],64,legacy=name=='champion')
        screen.append({'model':name,**summary(rows),'seconds':seconds})
        write_json(args.run/'screen.json',screen)
    eligible=[s for s in screen[1:] if screen_eligible(screen[0],s)]
    if not eligible:
        write_json(args.run/'rejected.json',{'status':'rejected_on_screen','reason':'no candidate met the predeclared floor/CPU guard','holdout_run':False,'promoted':False})
        return
    selected = max(eligible,key=lambda s:(s['success1120k'],-s['below300k'],s['p10']))['model']
    write_json(args.run/'selected.json',{'model':selected,'sha256':sha(args.run/(selected+'.json')),'selected_before_holdout':True})
    report = {'selected':selected,'plan_sha256':sha(args.run/'plan.json'),'comparisons':{}}
    for scenario in range(5):
        games, seed = (8192,plan['holdout_seed']) if scenario==0 else (256,plan['strata_seed']+scenario*100_000)
        a, ta = simulate(args,f'holdout-s{scenario}-champion',args.run/'champion.json',games,seed,256,scenario,True)
        b, tb = simulate(args,f'holdout-s{scenario}-{selected}',args.run/(selected+'.json'),games,seed,256,scenario)
        report['comparisons'][str(scenario)] = {'champion':summary(a),'candidate':summary(b),'paired':compare(a,b),'seconds':[ta,tb]}
        write_json(args.run/'evaluation.json',report)
        plot(args.run, report)
    primary = report['comparisons']['0']['paired']['success1120k']
    report['primary_pass'] = primary['difference_pp']>0 and primary['paired_exact_p']<.05
    report['promoted'] = False
    report['promotion_note'] = 'Research only; uncertain game rules/occurrence rates and deployment latency require separate review.'
    write_json(args.run/'evaluation.json',report)


def plot(run, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(15,4.5))
    entry = report['comparisons']['0']
    for ax,key,title in zip(axes,['success1120k','below300k','monsters'],['Preparation score >= 1.12M','Preparation score < 300k','Final monster quantity']):
        values = [entry[m][key] for m in ['champion','candidate']]
        if key!='monsters': values = [100*v/8192 for v in values]
        ax.bar(['Incumbent','Candidate'],values,color=['#8899aa','#2685c6'])
        for i,v in enumerate(values): ax.text(i,v,f'{v:.2f}',ha='center',va='bottom')
        ax.set_title(title)
        ax.set_ylabel('% of simulated games' if key!='monsters' else 'Mean count')
        ax.set_ylim(0,max(values)*1.2+1)
    fig.suptitle('8192 paired games - simulation only, special entry odds uncalibrated')
    fig.tight_layout()
    fig.savefig(run/'evaluation.png',dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['fit','evaluate'])
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--engine',type=Path,default=ROOT/'target/release/vorax.exe')
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--potion-refreshes',type=int,default=3)
    p.add_argument('--instrument-refreshes',type=int,default=1)
    p.add_argument('--objective',choices=['target','success'],default='target')
    p.add_argument('--seed-offset',type=int,default=1_000_000_000)
    p.add_argument('--resume',action='store_true',help='continue after a completed generation, with the identical plan')
    args = p.parse_args()
    args.run,args.engine=args.run.resolve(),args.engine.resolve()
    args.run.mkdir(parents=True,exist_ok=True)
    if args.stage=='fit' and (args.run/'plan.json').exists() and not args.resume: p.error('use a fresh run directory or resume completed generations')
    if not 1<=args.threads<=8 or not all(0<=v<=9 for v in [args.potion_refreshes,args.instrument_refreshes]): p.error('invalid CPU/reroll limits')
    if not 0<=args.seed_offset<4_000_000_000: p.error('seed offset outside u32 range')
    (fit if args.stage=='fit' else evaluate)(args)


if __name__=='__main__':
    main()
