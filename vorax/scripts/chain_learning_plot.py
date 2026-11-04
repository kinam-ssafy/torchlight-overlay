"""Plot the saved validation losses; they are not estimates of policy improvement."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',required=True,type=Path)
args=p.parse_args()
fit=json.loads((args.run/'fit.json').read_text())
plan=json.loads((args.run/'plan.json').read_text())
reward='capped score' if plan.get('objective')=='target' else 'binary success'
fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharey=True)
for ax,g in zip(axes,fit['generations']):
    rows=g['metrics']
    for column,label in [(0,'Previous value'),(1,'CUDA fitted value')]:
        ax.plot([r['round'] for r in rows],[r['rmse_probability'][column] for r in rows],marker='.',label=label)
    ax.set(title=f'Generation {g["generation"]}: separate 5,000-game validation',xlabel='Round',ylabel=f'RMSE of normalized {reward} reward')
    ax.legend()
fig.suptitle('100,000 training games - lower fit error does not prove better decisions')
fig.tight_layout()
fig.savefig(args.run/'learning.png',dpi=160)
print('Saved',args.run/'learning.png')
