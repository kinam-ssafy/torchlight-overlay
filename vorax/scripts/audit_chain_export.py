"""Check full-precision CPU values on every round/chain stage against CUDA."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import cupy as cp
from gpu_chain_rl import weights, TARGET
from gpu_search import sha,write_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True)
args=p.parse_args()
report={}
for name in ['champion','g1','g2']:
    rows=[json.loads(line) for line in (args.run/f'audit-{name}.jsonl').read_text().splitlines()]
    if {r['stage'] for r in rows}!=set(range(13)) or {r['round'] for r in rows}!=set(range(1,14)):
        raise ValueError('incomplete stage/round coverage')
    x=np.array([r['features'] for r in rows])
    rounds=np.array([r['round']-1 for r in rows])
    cpu=np.array([r['prediction'] for r in rows])
    w=weights(args.run/f'{name}.json')
    gpu=cp.asnumpy(cp.clip(cp.sum(cp.asarray(x)*cp.asarray(w)[cp.asarray(rounds)],axis=1)*1e6,0,TARGET))
    error=float(np.max(abs(cpu-gpu)))
    if error>1e-6: raise ValueError(f'{name}: CPU/CUDA error {error}')
    report[name]={'boards':len(rows),'max_error_points':error,'model_sha256':sha(args.run/f'{name}.json'),
                  'cpu_timing':json.loads((args.run/f'audit-{name}-timing.json').read_text())}
write_json(args.run/'export-audit.json',report)
print(json.dumps(report,indent=2))
