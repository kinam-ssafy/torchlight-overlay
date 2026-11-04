"""Fixed screen then untouched holdout for one additional root target.

Weights/target/rerolls stay fixed. The default path must reproduce old games.
Extra candidates retain full futures; measured CPU cost is an explicit guard.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from gpu_search import sha,write_json
from gpu_chain_rl import summary,compare


def run(args,name,engine,extra,games,seed,futures):
    path=args.run/(name+'.json')
    if path.exists(): raise ValueError('refusing to overwrite outcomes')
    cmd=[str(engine),'sim','--policy','solver','--games',str(games),'--seed0',str(seed),
         '--futures',str(futures),'--threads','8','--model',str(args.model),
         '--goal','target','--target-score','840000','--potion-refreshes','3','--instrument-refreshes','1',
         '--outcomes',str(path)]
    if extra:cmd.append('--value-candidates')
    started=time.perf_counter()
    print(name,flush=True)
    with (args.run/(name+'.log')).open('w',encoding='utf-8') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    elapsed=time.perf_counter()-started
    rows=json.loads(path.read_text())
    if len(rows)!=games or {r['seed'] for r in rows}!=set(range(seed,seed+games)):
        raise ValueError('missing/duplicate/mismatched seeds')
    return rows,elapsed


def stats(a,b,times):
    sa,sb=summary(a),summary(b)
    for rows,s in [(a,sa),(b,sb)]:s['capped_mean840k']=float(np.mean([min(r['score'],840000) for r in rows]))
    return {'baseline':sa,'candidate':sb,'paired':compare(a,b),'seconds':times}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','baseline','candidate','model']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    for name in ['run','baseline','candidate','model']:setattr(args,name,getattr(args,name).resolve())
    if args.run.exists():raise ValueError('fresh experiment directory required')
    args.run.mkdir(parents=True)
    plan={'target':840000,'goal':'target','rerolls':[3,1],'threads':8,
          'model_sha256':sha(args.model),'baseline_sha256':sha(args.baseline),'candidate_sha256':sha(args.candidate),
          'screen':{'seed0':1310000000,'games':512,'futures':64},
          'holdout':{'seed0':1320000000,'games':8192,'futures':256},
          'screen_guard':'840k count +5 or more; capped mean and p10 not lower; below300k count not higher; CPU <=1.15x',
          'primary':'paired 840k success difference; exact two-sided McNemar alpha .05',
          'holdout_guards':'positive primary with p<.05; capped mean and p10 not lower; below300k upper bootstrap95 <=0.5pp; CPU <=1.15x',
          'sampling':'fresh whole-game seeds, one frozen candidate, no extension/reselection',
          'limitations':['uncertain simulator rules/entry odds','ordinary games only','CPU guard uses wall time','no automatic promotion']}
    write_json(args.run/'plan.json',plan)
    plan_hash=sha(args.run/'plan.json')
    a,ta=run(args,'screen-baseline',args.baseline,False,512,1310000000,64)
    same,ts=run(args,'screen-default-parity',args.candidate,False,512,1310000000,64)
    if a!=same:raise ValueError('default policy changed')
    b,tb=run(args,'screen-candidate',args.candidate,True,512,1310000000,64)
    result={'plan':plan,'plan_sha256':plan_hash,'default_bit_exact_games':512,'default_seconds':ts,
            'screen':stats(a,b,[ta,tb]),'promoted':False}
    x,y=result['screen']['baseline'],result['screen']['candidate']
    passed=y['success840k']>=x['success840k']+5 and y['capped_mean840k']>=x['capped_mean840k'] and y['p10']>=x['p10'] and y['below300k']<=x['below300k'] and tb<=ta*1.15
    result['screen_pass']=passed
    write_json(args.run/'evaluation.json',result)
    if not passed:
        print('Screen rejected; holdout not run',flush=True)
        return
    a,ta=run(args,'holdout-baseline',args.baseline,False,8192,1320000000,256)
    b,tb=run(args,'holdout-candidate',args.candidate,True,8192,1320000000,256)
    result['holdout']=stats(a,b,[ta,tb])
    x,y=result['holdout']['baseline'],result['holdout']['candidate']
    primary=result['holdout']['paired']['success840k']
    result['primary_pass']=primary['difference_pp']>0 and primary['paired_exact_p']<.05
    result['quality_pass']=result['primary_pass'] and y['capped_mean840k']>=x['capped_mean840k'] and y['p10']>=x['p10'] and result['holdout']['paired']['below300k']['difference_pp_bootstrap95'][1]<=.5 and tb<=ta*1.15
    write_json(args.run/'evaluation.json',result)
    print('Fixed holdout complete; no automatic promotion',flush=True)


if __name__=='__main__':main()
