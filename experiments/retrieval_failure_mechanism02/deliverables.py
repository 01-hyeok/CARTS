"""Materialize a cell's required small tables and inspectable partial report.

No study-wide GO decision is fabricated by this mechanical aggregation step.
"""
import csv,shutil
from pathlib import Path
import numpy as np
from io_utils import read,dump,sha,now
from protocol import ROOT,CELLS

def rows(p):return list(csv.DictReader(Path(p).open()))
def table(values,columns):
    text=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for r in values:
        vals=[]
        for c in columns:
            v=r.get(c,'')
            try:v=f'{float(v):.6f}'
            except (TypeError,ValueError):pass
            vals.append(str(v))
        text.append('| '+' | '.join(vals)+' |')
    return '\n'.join(text)

def finish(cell,smoke=False):
    root=ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02';dest=root/('summary_smoke' if smoke else 'cells')/cell
    g=read(dest/'AGGREGATION_GATE.json');s=read(dest/'CPU_SUPPLEMENT_GATE.json')
    assert g['checkpoint_count']==6 and s['v1_included_both_training_supports']
    for name,hashval in g['files'].items():assert sha(dest/name)==hashval
    aliases={'RETRIEVAL_BASELINE_COMPARISON':'CORE_COMPARISON','V0_V1_V5_COMPARISON':'CORE_COMPARISON',
      'RETRIEVAL_REGRET_DECOMPOSITION':'REGRET_DECOMPOSITION','GLOBAL_RANKING_QUALITY':'RANKING_QUALITY',
      'MISSED_ORACLE_RANKS':'MISSED_ORACLE','TOPK_EXCHANGE_REGRET':'CROSS_SUPPORT',
      'CROSS_SUPPORT_EVALUATION':'CROSS_SUPPORT','SOFTMAX_SUPPORT_SENSITIVITY':'NORMALIZER_CONTROL',
      'HEAD_LEVEL_RETRIEVAL_QUALITY':'HEAD_UTILITY','BEST_SINGLE_VS_ALL5':'HEAD_UTILITY',
      'ORACLE_POSITIVE_MULTIMODALITY':'GEOMETRY_TEST','PAST_FUTURE_DISAGREEMENT':'FEATURE_PAIRED',
      'PAST_FUTURE_LEARNABILITY':'PROBE_COMPARISON'}
    for name,source in aliases.items():shutil.copy2(dest/(source+'.csv'),dest/(name+'.csv'))
    main=rows(dest/'CORE_COMPARISON.csv');reg=rows(dest/'REGRET_DECOMPOSITION.csv');norm=rows(dest/'NORMALIZER_CONTROL.csv')
    report=[f'# {cell} — Mechanism02 조건별 결과',
      'SMOKE ONLY: 공식 결과가 아닙니다.' if smoke else '이 문서는 한 Dataset/Horizon 조건의 결과입니다. 전체 8조건 완료를 의미하지 않습니다.',
      '모든 학습 모델은 seed0의 기존 validation 선택 checkpoint를 고정했습니다. Stage1 검색 지표이며 Stage2 결과가 아닙니다. 서로 다른 학습 support 비교와 동일 checkpoint의 support 개입을 별도 표에 기록했습니다.',
      '## Native support 주 비교',table(main,['support','method','retmse','agg_mse','agg_mae','recall10']),
      'Raw Cosine은 정규화된 과거 입력의 직접 cosine입니다. Delta-last와 Pearson은 별도 비교군입니다. Last-repeat는 후보 선택을 하지 않아 RetMSE/Recall이 없습니다. Oracle은 개별 후보 MSE 최적이며 평균 예측 MSE/MAE의 이론적 최적을 뜻하지 않습니다.',
      '## 후보군과 순위 손실',table(reg,['model','training_support','variant','pool','ranking','total','pool_share','ranking_share','full_oracle_coverage100']),
      '비중은 평균 regret의 비율입니다. Full 모델이 Full에서 고른 결과에 P100 Oracle 하한을 강제하지 않았습니다. 각 query×channel에서 regret 및 exchange 항등식을 검증했습니다.',
      '## 동일 logits의 정규화 분모 개입',table(norm,['model','training_support','changed_order_fraction','same_set_fraction','restricted_minus_local_retmse']),
      '이 대조는 같은 logits·mask·temperature·후보 ID에 FP64 logsumexp를 적용합니다. Native FP32와의 비교를 분모 효과로 해석하지 않습니다.',
      '## 검증과 해석 범위',
      '- 원래 Full checkpoint 생성 런타임은 UNKNOWN입니다. 모델별 전체 validation/test의 ordered-ID와 저장 예측 재현 PASS는 별도 증거입니다.',
      '- Full V1 일부 역사적 실행은 early stopping을 썼습니다. 독립 V1과 V5 비교는 모델 비교이며 head 수만의 인과효과로 주장하지 않습니다.',
      '- 비싼 global correlation은 고정 최대256 시점·전 채널에서 정확 계산했습니다. Core 지표는 모든 query×channel입니다.',
      '- Block bootstrap은 H/2H/4H를 사전 고정하고 시점의 모든 채널을 같이 재표집합니다. 4블록 미만은 CI 불충분이며 모든 비교는 탐색적입니다. seed 변동성은 추정하지 않습니다.',
      '- Probe는 고정256 train시점×전채널×100 future-free 후보로 학습한 기존 Ridge를 재사용했습니다. 전체 train으로 학습했다고 주장하지 않습니다.',
      '- Geometry는 validation 규칙이 통과하지 않으면 k=1/null입니다. 별도 disjoint future-MSE-matched control은 생성하지 않았습니다. Random/Past-nearest 대조와 후보 overlap을 함께 보고하며 geometry만으로 다중모드의 유용성을 단정하지 않습니다.',
      '- Weather sentinel 채널 제외 표는 진단용입니다. 공식 전체 채널 성능을 대체하거나 scaler/data를 수정하지 않았습니다.',
      '## 현재 결론',
      '연구 전체의 최종 GO 판정은 INCONCLUSIVE입니다. 모든 조건의 실제 수치 및 기존 실패 연구와의 중복 검토를 완료한 뒤 학습 목표 우선순위를 확정해야 합니다. 조건별 관측값은 CSV와 SOURCES.json을 근거로 검토할 수 있습니다.']
    (dest/'PARTIAL_REPORT_KO.md').write_text('\n\n'.join(report)+'\n')
    (dest/'WEATHER_RECALL_ANOMALY_AUDIT.md').write_text('# Recall / 미래 후보 품질 진단\n\n정확한 Oracle ID Recall과 near-optimal 후보의 양·선택 비율은 서로 다른 지표입니다. '+
      'NEAR_ORACLE_CANDIDATE_ANALYSIS.csv에서 두 값을 함께 확인합니다. Weather sentinel 영향은 WEATHER_CHANNEL_SENSITIVITY.csv의 공식 all_channels와 진단 채널군을 구분합니다. 낮은 Recall의 원인을 선험적으로 단정하지 않습니다.\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not smoke:
        fig,axes=plt.subplots(1,2,figsize=(15,5))
        for ax,support in zip(axes,['P100','FULL']):
            group=[r for r in main if r.get('support')==support and r.get('retmse') not in ('',None)]
            ax.bar(range(len(group)),[float(r['retmse']) for r in group]);ax.set_xticks(range(len(group)),[r['method'] for r in group],rotation=60,ha='right')
            ax.set_title(cell+' '+support);ax.set_ylabel('Stage1 RetMSE@10 (lower is better)')
        fig.tight_layout();fig.savefig(dest/'baseline_retrieval_comparison.png',dpi=160);plt.close(fig)
    dump(dest/'CELL_ARTIFACT_GATE.json',dict(status='SMOKE_PASS' if smoke else 'CORE_AND_CPU_ANALYSES_PASS_STUDY_REVIEW_PENDING',at=now(),cell=cell,
      v1_included=True,originals_unchanged=True,study_final_go='INCONCLUSIVE_PENDING_ALL_CONDITIONS',
      limitations=['fixed 256 timestamps for expensive global correlation','seed0 only','no disjoint future-MSE-matched geometry control'],
      files={p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='CELL_ARTIFACT_GATE.json'}))
    coverage={c:('CORE_AND_CPU_ANALYSES_SAVED' if (root/'cells'/c/'CELL_ARTIFACT_GATE.json').exists() else 'PENDING') for c in CELLS}
    if not smoke:dump(root/'00_Audit/STUDY_COVERAGE.json',dict(at=now(),cells=coverage,all_conditions_finished=all(v!='PENDING' for v in coverage.values()),final_review='PENDING'))
    print('CELL ARTIFACTS',cell,'SMOKE_PASS' if smoke else 'CORE_AND_CPU_PASS; STUDY_REVIEW_PENDING',flush=True)
