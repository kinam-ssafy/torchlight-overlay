"""Fresh fixed-size confirmation of weights, with the target held constant.

The earlier 1024-game isolation was exploratory. Never extend that sample after
seeing its p-value; use untouched seeds, frozen weights, and a new fixed plan.
"""
import argparse
import json
from pathlib import Path
import shutil
from gpu_chain_rl import simulate, summary, compare
from gpu_search import sha, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--engine',type=Path,required=True)
    args=p.parse_args()
    args.source,args.run,args.engine=args.source.resolve(),args.run.resolve(),args.engine.resolve()
    if args.run.exists(): raise ValueError('a fresh confirmation directory is required')
    previous=json.loads((args.source/'plan.json').read_text())
    selected=json.loads((args.source/'selected.json').read_text())
    model=args.source/(selected['model']+'.json')
    if sha(args.engine)!=previous['engine_sha256'] or sha(model)!=selected['sha256']:
        raise ValueError('frozen engine/model changed')
    if sha(args.source/'champion.json')!=previous['champion_sha256']:
        raise ValueError('incumbent coefficients changed')
    args.run.mkdir(parents=True)
    shutil.copy2(model,args.run/'candidate.json')
    shutil.copy2(args.source/'champion.json',args.run/'champion.json')
    args.threads,args.potion_refreshes,args.instrument_refreshes=8,3,1
    args.objective='target'
    seed,games=1_270_000_000,8192
    plan={'purpose':'fresh confirmation of learned weights at an identical target',
          'declared_after':'run02 main comparison and exploratory 1024-game isolation; no new model selection',
          'engine_sha256':sha(args.engine),'candidate_sha256':sha(args.run/'candidate.json'),
          'champion_sha256':sha(args.run/'champion.json'),
          'seed0':seed,'games_per_policy':games,'futures':256,'threads':8,
          'goal':'target','target_score_both':1120000,'potion_refreshes':3,'instrument_refreshes':1,
          'primary':'paired difference in preparation score >=1120000',
          'alpha':.05,'planned_power':.8,'detectable_difference_pp':2.,'assumed_discordance':.4,
          'decision':'positive primary difference and exact two-sided McNemar p<.05; no automatic promotion',
          'diagnostics':['840k rate','below300k rate','p10','mean','monster quantity','occupied tanks','CPU wall time'],
          'limitations':['same uncertain simulator rules as run02','no live reward/combat model',
                         'ordinary-entry distribution; special and pet diagnostics remain separate',
                         'fixed size; no interim outcome inspection, extension, or reselection']}
    write_json(args.run/'plan.json',plan)
    write_json(args.run/'plan.sha.json',{'sha256':sha(args.run/'plan.json')})
    a,ta=simulate(args,'holdout-champion',args.run/'champion.json',games,seed,256)
    b,tb=simulate(args,'holdout-candidate',args.run/'candidate.json',games,seed,256)
    result={'plan':plan,'plan_sha256':sha(args.run/'plan.json'),
            'champion':summary(a),'candidate':summary(b),'paired':compare(a,b),'seconds':[ta,tb]}
    primary=result['paired']['success1120k']
    result['primary_pass']=primary['difference_pp']>0 and primary['paired_exact_p']<.05
    result['promoted']=False
    write_json(args.run/'evaluation.json',result)
    print('Fixed matched-target confirmation complete',flush=True)


if __name__=='__main__': main()
