"""Run declared secondary checks after the frozen main comparison completes."""
import argparse
import json
from pathlib import Path
from gpu_chain_rl import simulate,summary,compare
from gpu_search import sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--engine',required=True,type=Path)
    args=p.parse_args()
    args.run,args.engine=args.run.resolve(),args.engine.resolve()
    plan=json.loads((args.run/'plan.json').read_text())
    final=json.loads((args.run/'evaluation.json').read_text())
    if len(final['comparisons'])!=5 or 'primary_pass' not in final:
        raise ValueError('main evaluation is incomplete')
    if sha(args.engine)!=plan['engine_sha256']: raise ValueError('engine changed')
    selected=json.loads((args.run/'selected.json').read_text())
    candidate=args.run/(selected['model']+'.json')
    if sha(candidate)!=selected['sha256']: raise ValueError('candidate changed')
    args.threads,args.potion_refreshes,args.instrument_refreshes=8,3,1
    args.objective='target'
    secondary={}
    isolated=json.loads((args.run/'isolated-model-plan.json').read_text())
    seed,games=isolated['seed0'],isolated['games']
    a,seconds=simulate(args,'isolated-retargeted',args.run/'champion.json',games,seed,256)
    b=[r for r in json.loads((args.run/f'holdout-s0-{selected["model"]}.json').read_text()) if seed<=r['seed']<seed+games]
    secondary['same_target_weights']={'incumbent':summary(a),'candidate':summary(b),'paired':compare(a,b),'incumbent_seconds':seconds}
    write_json(args.run/'secondary.json',secondary)
    pets=json.loads((args.run/'pet-sensitivity-plan.json').read_text())
    for rerolls in pets['instrument_refreshes']:
        args.instrument_refreshes=rerolls
        seed=pets['seed0']+rerolls*100_000
        a,ta=simulate(args,f'pet-{rerolls}-champion',args.run/'champion.json',256,seed,256,legacy=True)
        b,tb=simulate(args,f'pet-{rerolls}-candidate',candidate,256,seed,256)
        secondary[f'instrument_rerolls_{rerolls}']={'incumbent':summary(a),'candidate':summary(b),'paired':compare(a,b),'seconds':[ta,tb]}
        write_json(args.run/'secondary.json',secondary)


if __name__=='__main__': main()
