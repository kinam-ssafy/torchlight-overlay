"""Fixed 4096-game paired holdout, checkpointed without adaptive stopping.

The screen chooses one candidate before this run. Alternate chunk order to
reduce timing drift. Resume accepts only the exact frozen experiment.
"""
import argparse
import json
from pathlib import Path

from gpu_search import (ROOT, assert_separate, command, eligible_for_review, paired,
                        scores, settings, sha, summarize, write_json)
from gpu_statistics import analyze, release_quality


def checked_rows(path, seed, games):
    rows = scores(path)
    if set(rows) != set(range(seed, seed + games)):
        raise ValueError('chunk seeds are incomplete or mismatched')
    for row in rows.values():
        if any(type(row[k]) is not int or row[k] < 0 for k in ['score', 'monsters', 'groups']):
            raise ValueError('invalid simulation outcome')
    return rows


def freeze(path, plan):
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError('frozen holdout plan changed; cannot reuse or extend this holdout')
    else:
        write_json(path, plan)


def gate(args):
    fit = json.loads((args.run / 'fit.json').read_text())
    screen = json.loads((args.run / 'screen.json').read_text())
    name = screen['winner']
    candidate = next(c for c in fit['candidates'] if c['name'] == name)
    if len(fit['candidates']) != fit['candidate_count']:
        raise ValueError('incomplete fit')
    models = {'champion': args.run / 'champion.json', name: args.run / (name + '.json')}
    if sha(args.engine) != fit['engine_sha256'] or sha(ROOT / 'data/boardValue.json') != fit['champion_sha256']:
        raise ValueError('engine or shipped champion changed since fit')
    if json.loads(models['champion'].read_text()) != json.loads((ROOT / 'data/boardValue.json').read_text()):
        raise ValueError('champion copy differs')
    if sha(models[name]) != candidate['sha256']:
        raise ValueError('candidate changed since fit')
    if (args.potion_refreshes, args.instrument_refreshes) != (fit['potion_refreshes'], fit['instrument_refreshes']):
        raise ValueError('reroll settings changed since fit')
    seeds = {'seed0': args.seed0, 'games': args.games}
    assert_separate(seeds, screen)
    for prefix in ['train', 'valid']:
        assert_separate(seeds, {'seed0': fit[prefix + '_seed'], 'games': fit[prefix + '_games']})
    plan = {**seeds, 'futures': args.futures, 'threads': args.threads, 'chunk': args.chunk,
            'potion_refreshes': args.potion_refreshes, 'instrument_refreshes': args.instrument_refreshes,
            'engine_sha256': sha(args.engine), 'screen_sha256': sha(args.run / 'screen.json'),
            'models': {key: sha(path) for key, path in models.items()}, 'candidate': name,
            'selection': 'screen winner fixed before holdout; no adaptive stopping',
            'criteria': 'native gate, paired p<.05, p10 and below300k noninferior, CPU<=1.25x'}
    freeze(args.run / 'holdout-plan.json', plan)
    all_rows = {key: {} for key in models}
    times = {key: 0. for key in models}
    for chunk, start in enumerate(range(args.seed0, args.seed0 + args.games, args.chunk)):
        n = min(args.chunk, args.seed0 + args.games - start)
        order = list(models) if chunk % 2 == 0 else list(reversed(models))
        for key in order:
            label = f'chunk-{chunk:03d}-{key}'
            path, completed = args.run / (label + '.json'), args.run / (label + '-complete.json')
            if completed.exists():
                record = json.loads(completed.read_text())
                if record['sha256'] != sha(path):
                    raise ValueError('completed chunk changed')
                elapsed = record['seconds']
            else:
                elapsed = command(args, label, 'sim', '--policy', 'solver', '--goal', 'target',
                                  '--games', n, '--seed0', start, '--futures', args.futures,
                                  '--outcomes', path, '--model', models[key], *settings(args))
                checked_rows(path, start, n)
                write_json(completed, {'seconds': elapsed, 'sha256': sha(path)})
            all_rows[key].update(checked_rows(path, start, n))
            times[key] += elapsed
        count = len(all_rows['champion'])
        progress = {'completed_pairs': count, 'planned_pairs': args.games, 'candidate': name,
                    'seconds': times, 'partial_not_a_test': True,
                    'results': [{'name': key, **summarize(rows)} for key, rows in all_rows.items()]}
        write_json(args.run / 'holdout-progress.json', progress)
        print(f'Paired holdout {count}/{args.games}; fixed candidate={name}', flush=True)
    report = {'phase': 'holdout', 'policy': 'solver', 'seed0': args.seed0, 'games': args.games,
              'futures': args.futures, 'release_quality': release_quality(True, args.games, args.futures),
              'results': [], 'finalists': [name], 'comparison_count': 1, 'promoted': False, 'winner': name}
    for key, rows in all_rows.items():
        ordered = [rows[k] for k in sorted(rows)]
        write_json(args.run / f'holdout-{key}.json', ordered)
        (args.run / f'holdout-{key}.txt').write_text(''.join(f'{r["seed"]} {r["score"]}\n' for r in ordered))
        report['results'].append({'name': key, 'seconds': times[key], **summarize(rows)})
    first, result = report['results']
    result.update(paired(all_rows['champion'], all_rows[name]))
    result['adjusted_p'] = result['paired_p']
    command(args, 'holdout-native-gate', 'gate', args.run / 'holdout-champion.txt', args.run / f'holdout-{name}.txt')
    result['engine_gate_pass'] = 'pass=yes' in (args.run / 'holdout-native-gate.log').read_text(encoding='utf-8')
    result['eligible'] = eligible_for_review(first, result, report['release_quality'])
    report['decision'] = 'eligible_for_review' if result['eligible'] else 'keep_champion'
    report['paired_statistics'] = analyze(list(all_rows['champion'].values()), list(all_rows[name].values()))
    write_json(args.run / 'holdout.json', report)
    print(json.dumps({'decision': report['decision'], 'results': report['results']}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--games', type=int, default=4096)
    p.add_argument('--seed0', type=int, default=101_000_000)
    p.add_argument('--futures', type=int, default=256)
    p.add_argument('--chunk', type=int, default=128)
    p.add_argument('--potion-refreshes', type=int, default=3)
    p.add_argument('--instrument-refreshes', type=int, default=1)
    args = p.parse_args()
    if min(args.threads, args.games, args.futures, args.chunk) <= 0 or not 0 <= args.seed0 < args.seed0 + args.games < 2**32:
        p.error('positive counts and valid u32 seeds required')
    args.run, args.engine = args.run.resolve(), args.engine.resolve()
    gate(args)


if __name__ == '__main__':
    main()
