"""Separate execution status from completeness; evidence-based descriptive report."""
import argparse
from supplement_common import *

def cell_gate(support,cell):
    src=SOURCE/support/cell;dest=OUT/support/cell
    old=read(src/'COMPLETE.json');cpu=read(dest/'CPU_GATE.json');inf=read(dest/'INFERENCE_GATE.json')
    assert cpu['status']==inf['status']=='PASS'
    assert cpu['source_complete_sha256']==inf['source_complete_sha256']==sha(src/'COMPLETE.json')
    for s in inf['splits']:
        assert s['status']=='PASS' and s['queries']==old['query_counts'][s['split']] and s['channels']==old['channels'] and s['all_methods_ordered_top10_and_oracle_ids_exact']
    rows=csvread(dest/'MISSED_FALSE_SCORE_AND_POOLED_RANK.csv')
    for split in ['val','test']:
        schema=read(src/'aggregate'/f'{split}_schema.json')
        assert {r['method'] for r in rows if r['split']==split}==set(schema['methods'])
    assert len(csvread(dest/'INPUT_STRATA_THRESHOLDS.csv'))==6*old['channels']
    assert read(dest/'PROBE_SAMPLING_DISCLOSURE.json')['all_channels']
    if support=='FULL' and old['models']:
        assert (dest/'FULL_MODEL_FIXED_P100_RANKING.csv').exists()
    result=dict(status='PASS',execution_status=old['status'],deliverable_status='COMPLETE_EXCEPT_DECLARED_FULL_DEPENDENCIES' if support=='P100' else 'COMPLETE_WITH_DECLARED_MISSING_MODELS' if old['missing_models'] else 'COMPLETE',cell=cell,support=support,source_complete_sha256=sha(src/'COMPLETE.json'),missing_models=old['missing_models'],deferred=['Full Oracle','pool regret','Full Oracle P100 coverage'] if support=='P100' else [],inference_gate_sha256=sha(dest/'INFERENCE_GATE.json'),cpu_gate_sha256=sha(dest/'CPU_GATE.json'),inadequate_CI='explicitly excluded from significance; not fabricated')
    dump(dest/'DELIVERABLE_GATE.json',result);return result

def phase_gate(support):
    rows=[cell_gate(support,c) for c in CELLS]
    dump(OUT/f'{support}_DELIVERABLE_GATE.json',dict(status='PASS',cells=rows,non_full_dependent_gaps_remaining=0 if support=='P100' else None))
    return rows

def summary(final=False):
    rows=[];comparison=[];notes=[]
    for support in ['P100','FULL']:
        for cell in CELLS:
            src=SOURCE/support/cell;dest=OUT/support/cell
            executed=(src/'COMPLETE.json').exists();cpu=(dest/'CPU_GATE.json').exists();inf=(dest/'INFERENCE_GATE.json').exists();delivered=(dest/'DELIVERABLE_GATE.json').exists()
            rows.append(dict(cell=cell,support=support,execution_status=read(src/'COMPLETE.json')['status'] if executed else 'PENDING',cpu_supplement='PASS' if cpu else 'PENDING',detail_inference='PASS' if inf else 'PENDING',deliverable_status=read(dest/'DELIVERABLE_GATE.json')['deliverable_status'] if delivered else 'INCOMPLETE'))
            if not executed:continue
            values={r['method']:r for r in csvread(src/'RETRIEVAL_BASELINE_COMPARISON.csv') if r['split']=='test'}
            simple=min([values[n] for n in ['Raw Cosine','Delta-last Cosine','Pearson','PAA-16 Cosine']],key=lambda r:float(r['retmse']))
            v0=values.get('V0_'+('P100Trained' if support=='P100' else 'FullTrained'));v5=values.get('V5_'+('P100Trained' if support=='P100' else 'FullTrained'))
            row=dict(cell=cell,support=support,best_simple_test_descriptive=simple['method'],simple_retmse=float(simple['retmse']),v0_retmse=float(v0['retmse']) if v0 else None,v5_retmse=float(v5['retmse']) if v5 else None,ridge_retmse=float(values['Ridge_val_selected']['retmse']),oracle_retmse=float(values['IndividualMSE_Oracle']['retmse']),best_simple_is_test_descriptive_not_deployable_selection=True)
            if v5:
                regret=float(v5['ranking_regret']);row.update(v5_recall=float(v5['recall']),v5_spearman=float(v5['spearman']),v5_rank11_20_exchange_share=float(v5['boundary_11_20_exchange'])/regret if regret>1e-12 else None)
                row['observed_v5_gain_vs_best_simple_pct']=100*(row['simple_retmse']-row['v5_retmse'])/row['simple_retmse'] if row['simple_retmse']>1e-12 else None
            dp=SOURCE/'FULL'/cell/'RETRIEVAL_REGRET_DECOMPOSITION.csv'
            if support=='P100' and dp.exists():
                ds=[r for r in csvread(dp) if r['split']=='test' and r['method']=='V5_P100Trained'];assert len(ds)==1
                for k in ['pool_regret','ranking_regret','total_regret','pool_share','ranking_share']:row[k]=float(ds[0][k]) if ds[0][k] else None
            comparison.append(row)
    csvwrite(OUT/'EXECUTION_VS_DELIVERABLE_STATUS.csv',rows);csvwrite(OUT/'EVIDENCE_COMPARISON.csv',comparison)
    text='# Retrieval 진단 산출물 보완 보고\n\n'+('선행 진단의 전체 산출물 gate를 검증했다. Solar Full-trained 누락 8개는 후속 학습 대상이며 현재 결과로 채우지 않았다.\n' if final else 'PARTIAL. 셀 계산 완료와 산출물 완료는 별도이며, 미완료 행은 실행 큐에 남아 있다.\n')
    text+='\nP100 12셀의 비-Full 의존 산출물을 먼저 완료하고 Full을 실행한다. 원본 체크포인트, 결과, 코드와 기존 48개 학습은 보존했다.\n\n'
    text+='|Support|Cell|계산|CPU 보완|추론 보완|산출물|\n|---|---|---|---|---|---|\n'
    text+=''.join(f"|{r['support']}|{r['cell']}|{r['execution_status']}|{r['cpu_supplement']}|{r['detail_inference']}|{r['deliverable_status']}|\n" for r in rows)
    text+='\n## 실제 Test RetMSE 비교\n\nTest 최선 단순 baseline은 기술적 비교일 뿐, 배포 모델 선택이 아니다. V5 대 V0는 projection·학습 파라미터 수 차이도 포함하며 head 수의 고유 효과를 입증하지 않는다.\n\n|Support|Cell|단순 최선|V0|V5|Ridge|Individual-MSE Oracle|\n|---|---|---:|---:|---:|---:|---:|\n'
    fmt=lambda x:'MISSING' if x is None else f'{x:.6f}'
    for r in comparison:text+=f"|{r['support']}|{r['cell']}|{fmt(r['simple_retmse'])}|{fmt(r['v0_retmse'])}|{fmt(r['v5_retmse'])}|{fmt(r['ridge_retmse'])}|{fmt(r['oracle_retmse'])}|\n"
    text+='\n## 해석과 검증 한계\n\n'
    for r in comparison:
        if r['v5_retmse'] is None:continue
        text+=f"- {r['support']} {r['cell']}: V5 대 단순 최선 RetMSE 개선율 {r['observed_v5_gain_vs_best_simple_pct']:.3f}% (분모=동일 셀 단순 최선 RetMSE), Spearman {r['v5_spearman']:.4f}. "
        if r['v5_rank11_20_exchange_share'] is not None:text+=f"Rank11–20의 정해진 교환 손실 기여는 ranking regret의 {100*r['v5_rank11_20_exchange_share']:.2f}%. 나머지 순위 오류와 함께 판단한다. "
        if 'pool_share' in r and r['pool_share'] is not None:text+=f"P100 V5 총 regret에서 pool {100*r['pool_share']:.2f}%, ranking {100*r['ranking_share']:.2f}%. "
        text+='관찰된 분해이며 원인에 대한 개입 실험은 아니다.\n'
    text+='\n입력 strata는 query history의 변동성·추세·급변·스펙트럼·주기 특징으로 정의했다. validation에서 채널별 tercile을 고정하고 test에 적용했다. 채널별 표와 동일 가중 channel macro를 함께 제공한다. 상수 입력의 중복 경계는 억지로 세 그룹에 나누지 않으며 정의 불가 주기는 별도 행이다.\n'
    text+='\n기존 circular block bootstrap의 block=4H 규칙을 유지한다. 전체 query 길이로 clipping되어 한 블록뿐인 H720 CI 또는 유효 블록 수가 2 미만인 CI는 충분한 불확실성 추정이 아니므로 표시와 유의성 판단에서 제외했다. 결과에 유리한 block 길이를 선택하지 않았다. 다른 CI도 다중비교 무보정·seed0의 탐색적 결과다.\n'
    text+='\nP100 Kendall은 4,950쌍 exact, Full Kendall은 미래를 보지 않고 고정한 최대 4,096쌍의 APPROXIMATE tau-b다. Probe는 train-only 고정 균등간격 최대 256 query × 모든 채널 × 100 후보의 Ridge 학습이며 전체 train 학습이 아니다. feature scaling은 해당 train 표본, alpha 선택은 validation만 사용했다. 전체 validation/test의 모든 query/channel은 평가했다.\n'
    text+='\nMissed/false-selected score는 실제 검색 score를 재계산하고 모든 query/channel/method의 저장 Top10 ID와 Oracle ID 순서가 같음을 검증한 값이다. 서로 다른 score 정의의 크기를 직접 비교하지 않는다. Pooled missed-rank median/P90은 전체 누락 후보의 정확한 정수 rank histogram에서 구하며 query별 분위수 평균과 구분한다.\n'
    text+='\nFull-trained 모델의 고정 P100 내부 순위는 동일 Full 전역 score를 P100에 제한한 보조 분석이다. Head softmax를 P100에서 다시 계산한 모델 또는 P100-trained 모델로 해석하지 않는다. Full 모델에 P100 Oracle ≤ Full 모델 RetMSE를 강제하지 않는다.\n'
    text+='\nWeather -9999 sentinel과 기존 scaler를 그대로 유지한 정규화 공간 진단이다. L=H 동시 변경, 인접 후보 window, 겹치는 query 및 채널 상관, seed0 한계가 있다. Probe 성공/실패와 past–future ambiguity만으로 정보론적 불가능성이나 encoder의 인과적 실패를 확정하지 않는다.\n'
    recommendations=[]
    for r in comparison:
        if r['v5_retmse'] is None:continue
        evidence=f"{r['cell']} {r['support']}: V5 RetMSE={r['v5_retmse']:.6f}, Oracle={r['oracle_retmse']:.6f}, Spearman={r['v5_spearman']:.6f}, rank11-20 exchange share={r['v5_rank11_20_exchange_share']}; pool share={r.get('pool_share','DEFERRED_OR_NOT_APPLICABLE')}."
        candidates=[
            ('Original KL','기존 분포 학습 기준','현재 teacher 전체 분포와 실제 Top10 품질을 연결하는 기준점','동일 init/pool/예산의 기존 기준 유지'),
            ('Weighted Cross-Entropy','teacher 동일 시 KL과 gradient 동등','고정 teacher entropy 상수만 제거하면 새 학습 신호가 아님','실제 teacher와 reduction 동일성 검증; 이름만 바꾼 효과 주장 금지'),
            ('RankNet / Weighted RankNet','좋은 후보가 순위 하단에 있는 오류','후보 쌍 순서·미래 MSE 차이 가중으로 전체 분포 fitting과 다름','동일 후보에서 고정 pair sample의 작은 후속 ablation; teacher noise 위험'),
            ('Margin Ranking','정답 후보와 잘못 선택된 후보의 분리','고정/거리 기반 margin으로 비교하며 확률 분포는 직접 맞추지 않음','미래 MSE 차이가 작은 tie에 큰 margin을 강제하지 않기'),
            ('Choice Loss','선택 집합 오류','정확한 choice 정의 없이는 KL과 차별성 판정 불가','원 수식·정규화 정의 후 동일 support 대조'),
            ('Top-K aligned auxiliary','경계 근처의 실제 교환 손실','KL에 경계·regret 신호를 추가','위 rank11-20 exchange share를 근거로 검토; auxiliary scale confound'),
            ('Differentiable Top-K','hard Top10 선택과 학습 신호 불일치','연속 선택 surrogate를 통한 gradient','hard/soft 선택 정합성을 먼저 검증; temperature 불안정'),
            ('Hard Negative Mining','높은 점수의 false-selected 후보','loss보다 학습 pair/candidate sampling을 바꿈','동일 P100 유지하며 sampling만 통제; support 변경 confound 금지')]
        for loss,target,difference,minimal in candidates:recommendations.append(dict(cell=r['cell'],support=r['support'],candidate=loss,target=target,evidence=evidence,difference=difference,minimal_validation=minimal,status='EVIDENCE_LINKED_PROPOSAL_ONLY_NOT_TRAINED',causal_effect='NOT_ESTABLISHED'))
    if recommendations:csvwrite(OUT/'EVIDENCE_LINKED_LOSS_RECOMMENDATIONS.csv',recommendations)
    text+='\n## 후속 학습 신호 검토\n\nEVIDENCE_LINKED_LOSS_RECOMMENDATIONS.csv는 각 셀의 실제 RetMSE·Oracle·Spearman·rank11–20 교환 손실·가능한 pool share를 여덟 loss 후보에 연결한다. 현 실험에서 새 loss를 학습하거나 우수성을 입증하지 않았다. Pool와 ranking 비중 및 오류 rank 범위를 함께 보고 최소 통제 실험을 정해야 하며, 단일 원인을 강제하지 않는다. 동일 고정 teacher에 대한 WCE는 KL과 teacher entropy 상수 차이뿐이면 새로운 gradient 신호가 아니다.\n'
    (OUT/'FINAL_REPORT_KO.md').write_text(text,encoding='utf8')
    if final:
        assert read(OUT/'P100_DELIVERABLE_GATE.json')['status']==read(OUT/'FULL_DELIVERABLE_GATE.json')['status']=='PASS'
        assert all(r['deliverable_status']!='INCOMPLETE' for r in rows)
        assert read(SOURCE/'07_Summary/SUMMARY_GATE.json')['status']=='PASS'
        dump(OUT/'FINAL_DELIVERABLE_GATE.json',dict(status='PASS',cells=24,all_nonmissing_requested_details_complete=True,missing_full_solar_checkpoints=8,uncertainty_limitations_disclosed=True,no_significance_claims=True,original_summary_sha256=sha(SOURCE/'07_Summary/SUMMARY_GATE.json'),p100_gate_sha256=sha(OUT/'P100_DELIVERABLE_GATE.json'),full_gate_sha256=sha(OUT/'FULL_DELIVERABLE_GATE.json')))
    print('SUPPLEMENT SUMMARY', 'FINAL' if final else 'PARTIAL',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',nargs=2);p.add_argument('--phase',choices=['P100','FULL']);p.add_argument('--final',action='store_true');a=p.parse_args()
    if a.cell:cell_gate(*a.cell)
    if a.phase:phase_gate(a.phase)
    summary(a.final)
