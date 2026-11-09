"""Evidence-only rolling summary. Every previous snapshot remains immutable."""
import csv,json,shutil
from pathlib import Path
from d56_aggregate import OUT,sha,dump,csvout,now

FAMILIES=['D5_ORACLE_LAG_DIVERSITY','D5_HEAD_LAG_SPECIALIZATION','D5_LAG_AMBIGUITY','D5_LAG_DISTRIBUTION_STABILITY',
 'D6_TOPR_ORACLE_COVERAGE','D6_TOPR_RESTRICTED_ORACLE','D6_SHIFT_RERANKING','D6_CANDIDATE_VS_RERANKING_REGRET',
 'D6_ORACLE_INITIAL_RANK','D6_PAIRED_BLOCK_CI']
def main():
    stamp=now().replace(':','').replace('.','');snap=OUT/'summary_snapshots'/stamp;snap.mkdir(parents=True)
    values={k:[] for k in FAMILIES};gates=[];inventory=[];resources=[]
    for p in sorted((OUT/'complete_analysis').glob('*/*/ANALYSIS_GATE.json')):
        g=json.loads(p.read_text());assert g['status']=='PASS_D5_D6_CASE'
        ident=g['identity'];gates.append(dict(path=str(p.relative_to(OUT)),sha256=sha(p),identity=ident))
        for name,h in g['files'].items():
            q=p.parent/name;assert sha(q)==h
            row=dict(path=str(q.relative_to(OUT)),bytes=q.stat().st_size,sha256=h,**ident)
            if q.suffix=='.csv':
                with q.open() as stream:rows=list(csv.DictReader(stream))
                row['rows']=len(rows);row['columns']=json.dumps(list(rows[0]) if rows else [])
                if q.stem in values:values[q.stem].extend(rows)
            inventory.append(row)
        source=OUT/'complete_features'/ident['cell']/(ident['model']+'_'+ident['training_support']+'_to_'+ident['evaluation_support'])/'FEATURE_GATE.json'
        assert sha(source)==g['source_feature_gate_sha256'];f=json.loads(source.read_text())
        resources.append(dict(**ident,feature_wall_seconds=f['seconds'],cpu_aggregation_seconds=g['seconds'],
            peak_allocated_bytes=f['peak_allocated_bytes'],peak_reserved_bytes=f['peak_reserved_bytes'],
            physical_gpu=1,query_timestamps_per_split=32,channels=f['channels'],full_bank_count=f['full_bank_count'],
            shared_distance_and_lag_cache='first consumer pays construction; later consumers reuse; not independent benchmark'))
    for family,rows in values.items():
        if rows:csvout(snap/(family+'.csv'),rows)
    if inventory:csvout(snap/'D5_D6_ARTIFACT_INVENTORY.csv',inventory)
    if resources:csvout(snap/'D5_D6_RESOURCE_SUMMARY.csv',resources)
    native=sum(g['identity']['variant']=='NATIVE' for g in gates);cross=len(gates)-native
    lines=['# D5·D6 확장 분석 보고서','',
        '검증 완료: native %d/48, 동일 checkpoint cross-support %d/48. **%s**.'%(native,cross,'전체 지정 비교 완료' if len(gates)==96 else '부분 결과 — 나머지는 미완료'),
        '', '각 조건은 D4와 같은 validation/test 각각 32개 고정 timestamp와 전체 채널이다. 후보 Bank와 mask를 유지하고 Full Top-R는 전체 후보 score에서 선택했다. 전체 query population 결과로 해석하지 않는다.',
        '', 'V0/V1/V5 재학습은 없다. 기존 checkpoint·D4·SoftLag 결과는 불변이다. Full-trained와 P100-trained 및 native/cross-support를 분리한다. Solar는 재개하지 않았다.',
        '', 'D5 lag는 query[t] 대 candidate[t+δ], 반경 min(16,L//8), patch16/stride8, 공통 유효 query 구간, patch별 평균 제거와 L2 정규화이다. 미래값은 Oracle/그룹 정의에만 쓰고 shift score 입력에 쓰지 않는다.',
        '', 'Near-Oracle은 같은 masked support의 d10+max(1e-8,0.05*d10) 이내이다. 전체 Near-Oracle 후보를 포함했고 동일 개수 Random 대조군을 사용했다. Head Future-MSE 맞춤 대조군은 같은 support의 캐시 후보 합집합에서 선택한 진단용 근접 매칭이며, 전체 support 최적 매칭을 주장하지 않는다.',
        '', 'lag ambiguity: top1−top2 cosine≤0.01인 후보 비율과, 최대 cosine에서 0.01 이내인 lag 대안 비율을 구분한다. Effective lag 수는 exp(attention entropy), attention temperature는0.1. std≤1e-8인 후보는 평탄으로 분류했다.',
        '', 'D6은 Original / validation-selected ShiftOnly / ShiftRecent / Oracle@R를 비교한다. R=10/20/50/100, Full은200/500도 포함한다. 계수는 고정 grid에서 validation RetMSE로 선택했고 test는 선택에 사용하지 않았다. Oracle@R는 실제 배포 가능한 방법이 아닌 진단 하한이다.',
        '', '모든 Query×Channel에서 total regret=candidate limitation+reranking regret, Oracle 부등식, R 증가에 따른 coverage/Oracle@R 단조성, R10의 원래 선택 집합·RetMSE 일치를 검증했다. 음수 regret를 clamp하지 않았다.',
        '', '아래 표는 native V5의 test 평균이며, 전체 상세 모델·support 행은 CSV에 있다. Worst1/5%는 channel 평균 timestamp의 양의 악화량 기여율이다. Query-channel win과 timestamp win을 구분한다.',
        '', '| 조건 | 학습/eval support | R | 방식 | Original RetMSE | Rerank RetMSE | Oracle@R | Support Oracle | O10 Coverage | Near Coverage |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    rows=values['D6_SHIFT_RERANKING']
    for r in rows:
        if r['model']=='V5' and r['variant']=='NATIVE' and r['channel']=='ALL' and r['method']!='Original' and int(r['R']) in [100,500]:
            lines.append('| %s | %s | %s | %s | %.9f | %.9f | %.9f | %.9f | %.6f | %.6f |'%(r['cell'],r['evaluation_support'],r['R'],r['method'],*[float(r[k]) for k in ['original_retmse','reranked_retmse','oracle_at_r','support_oracle_retmse','oracle10_coverage','near_oracle_coverage']]))
    lines.extend(['','D5의 유용 후보 lag 다양성(동일 크기 Random과 함께 판단):','','| 조건 | support | 그룹 | Unique best lag 평균 | lag entropy 평균 | shift gain 평균 | near-tied 후보 비율 |','|---|---|---|---:|---:|---:|---:|'])
    for r in values['D5_ORACLE_LAG_DIVERSITY']:
        if r['model']=='V5' and r['variant']=='NATIVE' and r['split']=='test' and r['group'] in ['Oracle10','Random10','Oracle30','Random30'] and r.get('count'):
            lines.append('| %s | %s | %s | %.6f | %.6f | %.6f | %.6f |'%(r['cell'],r['evaluation_support'],r['group'],*[float(r[k]) for k in ['unique_best_lags','useful_candidate_lag_entropy','shift_gain_mean','near_tied_candidate_fraction']]))
    lines.extend(['','판단은 현재 **INCONCLUSIVE**이다. 다양한 argmax lag 또는 head별 분포 차이만으로 multi-head의 필요성이나 인과효과를 주장하지 않는다. 실제 RetMSE와 고유 유용 후보 기여, ambiguity, validation/test 분포 안정성을 함께 확인해야 한다.',
      '', 'D6_PAIRED_BLOCK_CI.csv는 H/2H/4H paired 시간 블록, 같은 시점 모든 채널 공동 resampling, seed0·2000회·percentile95%이다. 4개 미만 블록은 불충분으로 표시한다. 여러 학습 seed 불확실성을 나타내지 않으며 다중 비교의 탐색적 결과다.',
      '', '완료 gate는 검증된 raw 파일 hash·query/channel 수·checkpoint와 연결된다. 코드/원래 생성 runtime UNKNOWN은 수치 재현 PASS와 별개로 유지한다. 실제 미측정 시간·GPU 사용률은 추정해 채우지 않는다. 원시 모든 행을 이 본문에 넣었다고 주장하지 않으며 inventory에서 파일/행수/hash를 제공한다.'])
    report='\n'.join(lines)
    for name in ['D5_REPORT.md','D6_REPORT.md','D5_D6_FINAL_REPORT.md']:(snap/name).write_text(report,encoding='utf8')
    decision=['# 다음 연구 판단','', '현 단계 결과와 완료 범위는 D5_D6_FINAL_REPORT.md 및 CSV와 함께 읽는다. 새 loss/encoder/head 학습은 실행하지 않았다.',
       '', 'Q1·Q2: D1 고유 유용 후보와 D2 fusion/discovery regret의 검증 결과를 결합해야 한다. 해당 원시 분석의 완료 여부는 기존 D2_SAVED_ARRAYS/CPU_SET_GATE 및 보완 gate로 확인하며 D5/D6 완료만으로 대체하지 않는다.',
       '', 'Q3: Native와 동일 checkpoint cross-support normalizer 결과는 기존 Mechanism02 NORMALIZER_CONTROL/HEAD_PROBABILITY_MASS를 정확한 checkpoint·pool hash로 연결해야 한다. 서로 다른 모델 파라미터 효과를 support 효과로 해석하지 않는다.',
       '', 'Q4·Q7: 위 표의 ShiftOnly/ShiftRecent 실제 RetMSE와 Oracle@R, coverage가 직접 근거다. 제한 후보군 Oracle 개선 가능성은 학습으로 반드시 회복 가능한 양이 아니다.',
       '', 'Q5·Q6: D5 Oracle10/30 vs 동일 크기 Random, head unique-useful lag, uncertainty 및 val/test 안정성을 함께 판단한다. 관찰 자료만으로 Multi-head 우월성은 INCONCLUSIVE다.',
       '', 'Q8·Q9: 남은 조건과 D1~D4 보완 완료 전 하나의 병목이나 신규 학습을 확정하지 않는다. Score Calibration에는 fusion/normalizer 효과, Single Soft-Lag에는 실제 rerank 이득, Multi-head/Role Loss에는 고유 유용 후보와 안정된 유용 lag 분담이 필요하다.',
       '', 'D1–D6 통합 최종 결론은 각 트랙의 산출물 gate가 모두 충족된 후 별도 문서로 작성한다. 이 파일은 누락을 숨기지 않는 중간 판단이다.']
    (snap/'NEXT_RESEARCH_DECISION.md').write_text('\n'.join(decision),encoding='utf8')
    dump(snap/'D5_D6_VALIDATION_GATE.json',dict(status='PASS_D5_D6_ALL_CASES' if len(gates)==96 else 'PARTIAL_COMPLETE_CASES_VERIFIED',at=now(),native_complete=native,cross_support_complete=cross,
        expected_native=48,expected_cross_support=48,case_gates=gates,files={p.name:sha(p) for p in snap.iterdir() if p.is_file()},
        D1_D4_completion_not_implied=True,sampled_timestamps_per_split=32))
    # Every aggregate update has an immutable snapshot; older results remain.
    for p in snap.iterdir():
        t=OUT/(p.name+'.new');shutil.copyfile(p,t);t.replace(OUT/p.name)
    dump(OUT/'LATEST_SUMMARY.json',dict(path=str(snap),at=now(),native_complete=native,cross_support_complete=cross))
    print('D56 SUMMARY',native,cross,flush=True)
if __name__=='__main__':main()
