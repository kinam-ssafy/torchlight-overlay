"""Publish reproducible research artifacts only after every declared check completes."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from gpu_search import ROOT,sha,write_json
from gpu_statistics import wilson_interval


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def p_text(value):
    return '<1e-300 (floating-point underflow)' if value == 0 else f'{value:.6g}'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--binary-run',type=Path,required=True)
    p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--confirm',type=Path,required=True)
    args=p.parse_args()
    run,failed=args.run.resolve(),args.binary_run.resolve()
    evaluation=read(run/'evaluation.json')
    secondary=read(run/'secondary.json')
    plan=read(run/'plan.json')
    selected=read(run/'selected.json')
    confirmation=read(args.confirm/'evaluation.json')
    ca=read(args.confirm/'holdout-champion.json')
    cb=read(args.confirm/'holdout-candidate.json')
    if (len(ca),len(cb),confirmation['plan']['target_score_both'])!=(8192,8192,1120000):
        raise ValueError('incomplete matched-target confirmation')
    if confirmation['plan']['candidate_sha256']!=selected['sha256'] or confirmation['plan']['engine_sha256']!=plan['engine_sha256']:
        raise ValueError('confirmation differs from frozen candidate/engine')
    if sha(args.confirm/'plan.json')!=confirmation['plan_sha256'] or confirmation['plan']['champion_sha256']!=plan['champion_sha256']:
        raise ValueError('confirmation plan/incumbent changed')
    for role in ['champion','candidate']:
        if sha(args.confirm/(role+'.json'))!=confirmation['plan'][role+'_sha256']:
            raise ValueError('confirmation weights changed: '+role)
    if len(evaluation['comparisons'])!=5 or len(secondary)!=3 or 'primary_pass' not in evaluation:
        raise ValueError('incomplete evaluation')
    if sha(args.engine)!=plan['engine_sha256'] or sha(run/(selected['model']+'.json'))!=selected['sha256']:
        raise ValueError('frozen engine/model changed')
    a=read(run/'holdout-s0-champion.json')
    b=read(run/f'holdout-s0-{selected["model"]}.json')
    if len(a)!=8192 or len(b)!=8192: raise ValueError('incomplete primary outcomes')
    for name,rows in [('champion',a),('candidate',b)]:
        (run/f'legacy-gate-{name}.txt').write_text(''.join(f'{r["seed"]} {r["score"]}\n' for r in rows),encoding='utf-8')
    gate=subprocess.check_output([str(args.engine.resolve()),'gate',str(run/'legacy-gate-champion.txt'),str(run/'legacy-gate-candidate.txt')],text=True,encoding='utf-8')
    gate=gate.replace(str(run),'<run>')
    (run/'legacy-gate.log').write_text(gate,encoding='utf-8')
    for name,rows in [('champion',ca),('candidate',cb)]:
        (args.confirm/f'legacy-gate-{name}.txt').write_text(''.join(f'{r["seed"]} {r["score"]}\n' for r in rows),encoding='utf-8')
    matched_gate=subprocess.check_output([str(args.engine.resolve()),'gate',str(args.confirm/'legacy-gate-champion.txt'),str(args.confirm/'legacy-gate-candidate.txt')],text=True,encoding='utf-8')
    matched_gate=matched_gate.replace(str(args.confirm),'<confirmation>')
    (args.confirm/'legacy-gate.log').write_text(matched_gate,encoding='utf-8')
    base=ROOT.parent
    output=base/'docs/benchmarks/chain-audit-20261104.json'
    packet={'scope':'simulation research, not live loot or combat-time validation',
            'binary_ablation':{'plan':read(failed/'plan.json'),'fit':read(failed/'fit.json'),'screen':read(failed/'screen.json'),'rejected':read(failed/'rejected.json')},
            'target_experiment':{'plan':plan,'plan_sha256':sha(run/'plan.json'),'fit':read(run/'fit.json'),'screen':read(run/'screen.json'),
                                 'selected':selected,'evaluation':evaluation,'secondary':secondary,'export_audit':read(run/'export-audit.json'),
                                 'secondary_plans':{name:{'plan':read(run/name),'sha256':sha(run/name)} for name in ['isolated-model-plan.json','pet-sensitivity-plan.json']}},
            'statistics_audit':read(base/'qa/research/sample-size.json'),'seed_audit':read(base/'qa/research/seed-audit.json'),
            'static_game_inspection':read(base/'qa/research/key-scan.json'),'legacy_840k_gate':gate,
            'raw_outcomes':{path.stem:read(path) for path in sorted(run.glob('holdout-s*-*.json'))},
            'raw_secondary':{path.stem:read(path) for pattern in ['isolated-retargeted.json','pet-*-champion.json','pet-*-candidate.json'] for path in run.glob(pattern)}}
    packet['matched_target_confirmation']={'evaluation':confirmation,'champion_outcomes':ca,'candidate_outcomes':cb,'legacy_840k_gate':matched_gate}
    model_out=base/'docs/benchmarks/chain-target-candidate-20261104.json'
    # Public Git normalizes text to LF; record that artifact's hash separately
    # from the immutable Windows experiment file's hash.
    model_out.write_bytes((run/(selected['model']+'.json')).read_text(encoding='utf-8').encode('utf-8'))
    if read(model_out)!=read(run/(selected['model']+'.json')):
        raise ValueError('exported model coefficients changed')
    packet['exported_model']={'path':model_out.name,'sha256':sha(model_out),
                              'experiment_sha256':selected['sha256'],'line_endings':'LF'}
    write_json(output,packet)
    images=base/'docs/images'
    for source,name in [(run/'learning.png','chain-learning.png'),(failed/'learning.png','chain-binary-learning.png'),
                        (base/'qa/research/sample-size.png','chain-sample-size.png')]:
        shutil.copy2(source,images/name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    colors=['#8899aa','#2685c6']
    for rows,label,color in [(a,'Incumbent policy',colors[0]),(b,'Candidate policy',colors[1])]:
        scores=np.sort([r['score'] for r in rows])
        axes[0,0].plot(scores/1e6,np.arange(1,len(scores)+1)/len(scores),label=label,color=color)
    axes[0,0].set(xlim=(0,2),ylim=(0,1),xlabel='Preparation score (millions)',ylabel='Fraction of games at or below score',title='Independent paired score distribution')
    axes[0,0].legend()
    for j,threshold in enumerate([840000,1120000]):
        for i,rows in enumerate([a,b]):
            count=sum(r['score']>=threshold for r in rows)
            value=100*count/len(rows)
            ci=np.array(wilson_interval(count,len(rows)))*100
            axes[0,1].bar(j+(i-.5)*.32,value,width=.3,color=colors[i],yerr=[[value-ci[0]],[ci[1]-value]],capsize=3)
    axes[0,1].set(xticks=[0,1],xticklabels=['>=840k','>=1.12M'],ylabel='% of simulated games',title='95% Wilson intervals (individual rates)')
    for i,rows in enumerate([a,b]):
        count=sum(r['score']<300000 for r in rows)
        value=100*count/len(rows)
        ci=np.array(wilson_interval(count,len(rows)))*100
        axes[1,0].bar(i,value,color=colors[i],yerr=[[value-ci[0]],[ci[1]-value]],capsize=3)
        axes[1,0].text(i,ci[1],f'{count}/8192',ha='center',va='bottom')
        quantity=np.mean([r['monsters'] for r in rows])
        axes[1,1].bar(i,quantity,color=colors[i])
        axes[1,1].text(i,quantity,f'{quantity:.0f}',ha='center',va='bottom')
    axes[1,0].set(xticks=[0,1],xticklabels=['Incumbent','Candidate'],ylabel='% of simulated games',title='Lower-tail failures: score <300k')
    axes[1,1].set(xticks=[0,1],xticklabels=['Incumbent','Candidate'],ylabel='Mean final quantity',title='Combat population proxy, not measured time')
    fig.suptitle('8192 paired games, 256 futures, CPU 8 threads; special entry odds uncalibrated')
    fig.tight_layout()
    fig.savefig(images/'chain-holdout.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.7))
    for ax,test,title in [(axes[0],lambda score:score>=1120000,'Preparation score >=1.12M'),
                          (axes[1],lambda score:score<300000,'Lower-tail failures <300k')]:
        for i,rows in enumerate([ca,cb]):
            count=sum(test(r['score']) for r in rows)
            value=100*count/len(rows)
            ci=np.array(wilson_interval(count,len(rows)))*100
            ax.bar(i,value,color=colors[i],yerr=[[value-ci[0]],[ci[1]-value]],capsize=3)
            ax.text(i,ci[1],f'{count}/8192',ha='center',va='bottom')
        ax.set(xticks=[0,1],xticklabels=['Existing weights','Learned weights'],ylabel='% of simulated games',title=title)
        ax.margins(y=.15)
    fig.suptitle('Fresh 8192 paired games: identical 1.12M target, only weights differ')
    fig.tight_layout()
    fig.savefig(images/'chain-weight-confirmation.png',dpi=160)
    plt.close(fig)
    e=evaluation['comparisons']['0']
    x,y=e['champion'],e['candidate']
    r=e['paired']['success1120k']
    ci=r['difference_pp_bootstrap95']
    low=e['paired']['below300k']
    low_ci=low['difference_pp_bootstrap95']
    s=secondary['same_target_weights']['paired']['success1120k']
    holy=evaluation['comparisons']['2']
    cpu_change=100*(e['seconds'][1]/e['seconds'][0]-1)
    quantity_change=100*(y['monsters']/x['monsters']-1)
    cw=confirmation['paired']['success1120k']
    cw_ci=cw['difference_pp_bootstrap95']
    cx,cy=confirmation['champion'],confirmation['candidate']
    cl=confirmation['paired']['below300k']
    cl_ci=cl['difference_pp_bootstrap95']
    confirm_conclusion='같은 목표에서도 후보의 달성률 개선을 확인했다.' if confirmation['primary_pass'] else '같은 목표에서 후보가 우월하다는 근거는 확보하지 못했다.'
    special='\n'.join(f'| {name} | {evaluation["comparisons"][str(i)]["champion"]["success1120k"]}/256 | {evaluation["comparisons"][str(i)]["candidate"]["success1120k"]}/256 | {evaluation["comparisons"][str(i)]["champion"]["below300k"]} / {evaluation["comparisons"][str(i)]["candidate"]["below300k"]} | {evaluation["comparisons"][str(i)]["champion"]["p10"]:,.0f} / {evaluation["comparisons"][str(i)]["candidate"]["p10"]:,.0f} |' for i,name in [(1,'부화'),(2,'성수'),(3,'해부'),(4,'만능약')])
    text=f'''# 블러드 러스트 GPU 학습 및 독립 평가 결과

이번 작업에서 독립 학습 200,000판, 검증 20,000판으로 두 목적함수와 각 두 세대를 시험했다.
GPU는 RTX 4060 Laptop이다. 배포 기본 모델은 교체하지 않았다. 아래 수치는 동일한 감사용
시뮬레이터의 정책 비교이며 실제 전리품 확률이 아니다.

동일 목표의 새 8,192쌍 확증: {confirm_conclusion}
기존 {cw['champion_count']}/8192, 후보 {cw['candidate_count']}/8192,
차이 {cw['difference_pp']:+.2f}%포인트, 95% 대응 구간 [{cw_ci[0]:+.2f}, {cw_ci[1]:+.2f}].

**준비 점수 개선 후보를 찾았지만 배포 기준을 모두 해결한 것은 아니다.**
아래 전체 정책 비교에서 30만 미만 감소는 통계적으로 확실하지 않았고,
성수 계열에서는 저점 악화가 관측됐다. 이 전체 정책 비교의 CPU 완주 경과시간은 {cpu_change:+.1f}%,
최종 몬스터 수량은 {quantity_change:+.1f}% 변했다. 같은 목표의 가중치 비교 결과는 별도 표에 구분한다.
따라서 저점 안정성이나 전투시간까지 개선한 최적 모델로 부르지 않는다.

## 목적함수와 최종 결과

달성/미달만 보상한 첫 실험은 512판 선별에서 30만 미만이 7회에서 132/139회로 늘어 기각했다.
첫 실험의 예정된 8,192판 검증은 완료하지 않았다. 실패한 결과와 원인을 보존했다.

두 번째 실험은 목표까지의 점수를 연속적으로 보상한다. 512판 선별의 저점/CPU 조건을
통과한 `{selected['model']}`를 고정하고, 학습이나 선택에 쓰지 않은 동일 시드 8,192개를
각 정책으로 완주했다. 256 futures, CPU 8스레드, 포션/도구 리롤 3/1 조건이다.
기존 가중치의 목표는 840k, 후보의 목표는 1.12M이므로 전체 정책을 비교하는 주 평가다.

| 지표 | 기존 정책 | 새 후보 |
|---|---:|---:|
| 준비 점수 1.12M 이상 | {x['success1120k']}/8192 ({x['success1120k']/8192:.2%}) | {y['success1120k']}/8192 ({y['success1120k']/8192:.2%}) |
| 준비 점수 840k 이상 | {x['success840k']}/8192 | {y['success840k']}/8192 |
| 300k 미만 | {x['below300k']}/8192 | {y['below300k']}/8192 |
| 하위 10% 점수 | {x['p10']:,.0f} | {y['p10']:,.0f} |
| 평균 점수 | {x['mean']:,.0f} | {y['mean']:,.0f} |
| 평균 최종 몬스터 수량 | {x['monsters']:,.1f} | {y['monsters']:,.1f} |
| 평균 사용 탱크 수 | {x['groups']:.2f} | {y['groups']:.2f} |
| 전체 완주 경과시간(초) | {e['seconds'][0]:.1f} | {e['seconds'][1]:.1f} |

주 지표의 대응 차이는 {r['difference_pp']:+.2f}%포인트, paired bootstrap 95% 구간
[{ci[0]:+.2f}, {ci[1]:+.2f}]%포인트, exact McNemar p={p_text(r['paired_exact_p'])}이다.
새로 목표를 넘긴 판 {r['gained']}개, 기존에는 넘겼으나 후보에서 못 넘긴 판 {r['lost']}개다.
30만 미만 비율의 차이는 {low['difference_pp']:+.2f}%포인트,
대응 bootstrap 95% 구간 [{low_ci[0]:+.2f}, {low_ci[1]:+.2f}],
exact p={p_text(low['paired_exact_p'])}다. 이 지표는 음수일수록 좋다.
주 지표 이외의 검정은 탐색적 진단이며 다중 검정을 보정한 확증 결과가 아니다.
하위 10% 점수의 상승만으로 저점 전체의 비열등성을 주장하지 않는다.
기존 840k 전용 native gate: `{'pass' if 'pass=yes' in gate else 'fail'}`.
CPU 완주 시간은 OCR/화면 지연 측정이 아니다. 일부 빌드/검증 작업이 병행되었으므로
경과시간 비율을 정밀한 단독 처리량 측정으로 보지 않는다.

![독립 평가](images/chain-holdout.png)

목표 변경과 가중치 변경을 분리한 추가 1,024판도 같은 1.12M 목표에서 비교했다.
학습에 쓰지 않은 주 평가 시드의 앞 1,024개를 사용했고, 후보를 다시 고르지 않았다.
기존 가중치 {s['champion_count']}/1024, 후보 {s['candidate_count']}/1024,
차이 {s['difference_pp']:+.2f}%포인트, paired exact p={p_text(s['paired_exact_p'])}.
이는 별도로 선언한 부가 진단이며 주 검정으로 바꿔 부르지 않는다.

## 새 시드에서 같은 목표로 검증한 가중치 효과

위 1,024판은 신뢰구간이 0을 포함했다. 그 결과를 보고 같은 표본을 늘리지 않고,
새 시드 1,270,000,000부터 8,192개를 고정해 두 정책 모두 `--goal target --target-score 1120000`으로
다시 비교했다. 후보 가중치와 엔진은 그대로 고정했고 추가 학습/재선별을 하지 않았다.
가정 q=.4, 차이 2%포인트, 양측 유의수준 .05에서 계획 검정력은 약 81.2%다.

| 동일 목표 지표 | 기존 가중치 | 학습 가중치 |
|---|---:|---:|
| 1.12M 이상 | {cx['success1120k']}/8192 ({cx['success1120k']/8192:.2%}) | {cy['success1120k']}/8192 ({cy['success1120k']/8192:.2%}) |
| 840k 이상 | {cx['success840k']}/8192 | {cy['success840k']}/8192 |
| 300k 미만 | {cx['below300k']}/8192 | {cy['below300k']}/8192 |
| 하위 10% 점수 | {cx['p10']:,.0f} | {cy['p10']:,.0f} |
| 평균 몬스터 수량 | {cx['monsters']:,.1f} | {cy['monsters']:,.1f} |
| CPU 완주 경과시간(초) | {confirmation['seconds'][0]:.1f} | {confirmation['seconds'][1]:.1f} |

가중치 효과의 대응 차이 {cw['difference_pp']:+.2f}%포인트, bootstrap 95% 구간
[{cw_ci[0]:+.2f}, {cw_ci[1]:+.2f}], exact McNemar p={p_text(cw['paired_exact_p'])}.
{confirm_conclusion} 이는 목표 변경을 포함한 앞선 +31.70%포인트와 다른 비교다.
개선 폭의 구간이 넓으므로 p값의 통과 여부만으로 실용적 우월성을 선언하지 않는다.
같은 목표에서 300k 미만의 차이는 {cl['difference_pp']:+.2f}%포인트,
95% 구간 [{cl_ci[0]:+.2f}, {cl_ci[1]:+.2f}], exact p={p_text(cl['paired_exact_p'])}로
저점 감소를 확정할 수 없다. 하위 10% 점수도 {cx['p10']:,.0f}에서 {cy['p10']:,.0f}로 변했다.
평균 몬스터 수량은 {100*(cy['monsters']/cx['monsters']-1):+.2f}%, CPU 완주 경과시간은
{100*(confirmation['seconds'][1]/confirmation['seconds'][0]-1):+.2f}% 변했다.
이 동일 목표 결과의 840k native gate는 `{'pass' if 'pass=yes' in matched_gate else 'fail'}`이다.
판수/시드를 고정한 별도 확증 실험이며, 앞선 1,024판을 합산하거나 좋은 결과만 선별하지 않았다.
규칙/보상/전투시간의 불확실성과 특수 조건의 저점 문제는 이 검정만으로 해결되지 않는다.

![동일 목표의 가중치 비교](images/chain-weight-confirmation.png)

## 특수 포션과 펫 설정

각 계열을 2라운드에 강제로 제공한 별도 256판의 1.12M 달성 횟수다.
실제 출현 확률을 모르는 상태에서 아래 결과를 일반 판과 합쳐 실전 성공률로 표시하지 않는다.

| 강제 진입 계열 | 기존 1.12M 이상 | 후보 1.12M 이상 | 300k 미만 기존/후보 | 하위 10% 기존/후보 |
|---|---:|---:|---:|---:|
{special}

성수 계열의 300k 미만은 {holy['champion']['below300k']}회에서 {holy['candidate']['below300k']}회로 늘었다.
보정 전 exact p={p_text(holy['paired']['below300k']['paired_exact_p'])}이지만,
여러 계열/지표를 함께 본 탐색 검사이고 표본도 작으므로 확증적 악화 검정으로 주장하지 않는다.
관측된 위험 신호와 하위 10% 점수 하락을 공개하며, 전체 점수 개선만으로 이 문제를 덮지 않는다.

펫의 도구 리롤 0/2회도 각 256판씩 비교했으며 원시 결과와 대응 구간을 JSON에 담았다.
이 작은 각 조건의 표본으로 2%포인트 개선이나 전 설정의 비열등성을 보증하지 않는다.

## 검증과 한계

- 가치 함수 입력 1,316개: 기존 656, 시너지/시간 564, 연계 상태 96.
- 모든 13라운드/13연계 상태를 포함한 1,352개 합성 보드에서 각 모델의 CPU/CUDA 대조를 통과했다.
  실제 모의게임 경로를 쓰는 별도 대조도 수행했다. CUDA를 끈 Rust 실행도 성공했다.
- 모델 적합은 CUDA, 게임 전이와 최종 평가는 CPU다. 학습용 CUDA/Python은 배포 의존성이 아니다.
- Rust 전체 회귀 102개와 추가 CLI 설정 검증 1개를 작업본과 공개용 체크아웃에서 모두 통과했다.
  Python 39개도 CUDA 대조와 동일 목표/가중치 파일 보존 검사를 포함해 통과했다.
  기존 dead-code 경고 6개는 남아 있다.
- 배포 기본 모델 SHA256은 `31acf8fc57fd7af8000ca0a8ab57191ecb1730d6431ce532d235fdc74c3582af`로 유지했다.
- 새 상태 전이 19개는 공개 설명을 반영한 실험 구현이다. 실제 출현율, 일부 희귀도/반응 순서,
  10라운드 이후 연계는 미검증이다. 준비 점수와 전투 후 보상, 몬스터 수량과 실제 전투시간도 구분한다.
- 마지막 3회의 위험도 선택과 처치/사망 보정은 모델에 없다. 840k 목표가 이 추가 점수를
  기대하는 기존 전략이라는 근거도 있어, 1.12M 준비 점수 증가를 그대로 보상 증가로 환산하지 않는다.
- REA로 PE 분석은 했지만 암호화된 게임 규칙 테이블 추출에는 실패했다. 판수를 늘려 해결할 수 없는 한계다.
- 추가 확증 시험의 첫 시작은 가중치와 출력 파일명의 충돌을 덮어쓰기 방지가 차단했다.
  0판 실행 상태에서 파일명을 분리해 새 폴더로 시작했으며 실패 로그와 원 가중치를 보존했다.

![학습 오차](images/chain-learning.png)

[통계 설계와 규칙 출처](RULES-AND-EVALUATION-AUDIT.md),
[설정/해시/모든 최종 시드별 결과](benchmarks/chain-audit-20261104.json),
[실험용 후보 가중치](benchmarks/chain-target-candidate-20261104.json).
공개 가중치 파일 SHA256: `{sha(model_out)}`. 실험 파일과 계수는 같고,
Git의 LF 줄바꿈 정규화 때문에 원 실행 파일의 해시와 구분해 기록한다.
후보 가중치는 기본 모델 파일로 자동 복사하지 않는다. 코드/실험 결과는 검토용이며
새 실행 파일이나 자동 업데이트 릴리스를 배포한 것이 아니다.
이 후보의 재현 설정은 `--goal target --target-score 1120000`이다. 기존 앱의 840k 설정에서
가중치 파일만 바꿨을 때의 성능은 이 보고서로 검증된 것이 아니다.
'''
    (base/'docs/CHAIN-MODEL-RESULTS.md').write_text(text,encoding='utf-8')
    print('Wrote',output)
    print('Primary difference pp:',r['difference_pp'],'p:',r['paired_exact_p'])


if __name__=='__main__': main()
