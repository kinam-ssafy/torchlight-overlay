"""Reproducible prospective precision/power audit; no game outcomes are used."""
import json
from pathlib import Path
from gpu_statistics import wilson_interval, planning_games, paired_exact_power


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    rows=[]
    for n in [4096,8192]:
        for q in [.25,.4]:
            for delta in [.01,.02,.03]:
                rows.append({'games':n,'discordance':q,'difference_pp':delta*100,
                             'exact_power':paired_exact_power(n,q,delta)})
    report={'independent_unit':'complete game, not state row or search future',
            'individual_rate_intervals95':[
                {'games':n,'successes':k,'interval':wilson_interval(k,n)}
                for n in [4096,8192] for k in [0,round(.01*n),round(.5*n),round(.735*n)]],
            'prospective_power':rows,
            'approximate_required_games':[{'discordance':q,'difference_pp':d*100,'games':planning_games(q,d)}
                                          for q in [.25,.4] for d in [.01,.02,.03]],
            'assumptions':['independent game seeds under the same frozen simulator distribution',
                           'fixed holdout size and one candidate chosen on separate screen data',
                           'specified paired discordance and effect size, not estimated after seeing the final result'],
            'does_not_validate':['live game rules','special-card frequency','reward threshold','global optimality','unseen pet settings']}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.7))
    for n in [4096,8192]:
        r=[x for x in rows if x['games']==n and x['discordance']==.4]
        ax.plot([x['difference_pp'] for x in r],[100*x['exact_power'] for x in r],marker='o',label=f'{n} paired games')
    ax.axhline(80,color='gray',linestyle='--',label='80% power')
    ax.set(xlabel='True improvement (percentage points)',ylabel='Probability of detecting improvement (%)',
           title='Exact paired test power, assuming 40% discordant outcomes',ylim=(0,105))
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out.with_suffix('.png'),dpi=160)
    print(json.dumps(rows,indent=2))


if __name__=='__main__': main()
