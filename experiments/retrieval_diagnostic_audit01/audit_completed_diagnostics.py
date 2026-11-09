"""CPU-only additive audit; never edits active diagnostic source/results."""
from pathlib import Path
import csv,json,datetime,itertools
import numpy as np
ROOT=Path('/data/pjh_workspace/CARTS')
SOURCE=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01'
OUT=SOURCE/'00_Audit/COMPLETENESS_AUDIT_20261030'
OUT.mkdir(parents=True,exist_ok=True)

def read(p):return json.loads(p.read_text())
def rows(p):return list(csv.DictReader(p.open()))
def write_csv(p,rs):
    if not rs:return
    fields=list(dict.fromkeys(k for r in rs for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rs)

audit=[];overlap=[];intervals=[];bins=[];comparison=[]
for h in [96,192,336,720]:
    cell=f'ETTh1_{h}';p=SOURCE/'P100'/cell;done=read(p/'COMPLETE.json');splits={}
    for split in ['val','test']:
        gate=read(p/split/'SPLIT_GATE.json');count=0;prev=None;pair_sums={};names=None
        for f in sorted((p/split).glob('batch*.npz')):
            with np.load(f) as a:
                q=a['query_start'];ids=a['selected_ids'];names=a['method_names'].tolist()
                assert ids.shape==(len(q),7,len(names),10)
                assert np.all(np.diff(q)==1) and (prev is None or q[0]==prev+1)
                assert np.all(np.diff(np.sort(ids,axis=-1),axis=-1)>0)
                prev=int(q[-1]);count+=len(q)
                for i,j in itertools.combinations(range(len(names)),2):
                    hit=(ids[:,:,i,:,None]==ids[:,:,j,None,:]).any(-1).mean(-1)
                    pair_sums[i,j]=pair_sums.get((i,j),0.)+hit.sum(dtype=np.float64)
        assert count==done['query_counts'][split]==gate['query_count'] and count*7==gate['units']
        splits[split]=dict(queries=count,channels=7,units=count*7,unique_query_ids=True,selected_top10_ids_unique=True,canonical_gate=gate['status'],max_canonical_difference=max(x['max_canonical_difference'] for x in gate['model_gates']))
        for (i,j),v in pair_sums.items():overlap.append(dict(cell=cell,support='P100',split=split,method_a=names[i],method_b=names[j],top10_overlap=v/(count*7),queries=count,channels=7,definition='mean intersection size divided by10; exact saved selected IDs'))
    for filename in ['BASELINE_WIN_RATE.csv','FEATURE_GROUP_COMPARISON.csv']:
        for row in rows(p/filename):
            rawlo=row.get('ci_low');rawhi=row.get('ci_high');block=float(row['block']);times=float(row['query_times']);eff=times/block
            status='DEGENERATE_ONE_TIME_BLOCK' if eff<=1 else ('VERY_FEW_TIME_BLOCKS' if eff<2 else 'EXPLORATORY_UNADJUSTED_MULTIPLE_COMPARISONS')
            row.update(source_file=str(p/filename),ci_status=status,original_ci_low=rawlo,original_ci_high=rawhi,usable_for_significance=False)
            if eff<=1:row['ci_low']=row['ci_high']=''
            intervals.append(row)
    for row in rows(p/'MISSED_ORACLE_RANK_DISTRIBUTION.csv'):
        total=float(row['missed_count']);new=dict(cell=cell,split=row['split'],method=row['method'],mean_missed_count=total,mean_query_median_rank=row['missed_rank_median'],pooled_candidate_median_rank='NOT_STORED_REQUIRES_RANKS')
        for key in ['11_20','21_50','51_100','101_500','501_1000','1001_plus']:
            new['missed_candidate_fraction_'+key]=float(row['missed_'+key])/total if total else ''
        bins.append(new)
    metric=[x for x in rows(p/'RETRIEVAL_BASELINE_COMPARISON.csv') if x['split']=='test'];lookup={x['method']:x for x in metric}
    simple=min([lookup[n] for n in ['Raw Cosine','Delta-last Cosine','Pearson','PAA-16 Cosine']],key=lambda x:float(x['retmse']))
    v0=lookup['V0_P100Trained'];v5=lookup['V5_P100Trained'];reg=float(v5['ranking_regret'])
    comparison.append(dict(cell=cell,best_simple_test_descriptive=simple['method'],simple_retmse=float(simple['retmse']),v0_retmse=float(v0['retmse']),v5_retmse=float(v5['retmse']),ridge_retmse=float(lookup['Ridge_val_selected']['retmse']),oracle_retmse=float(lookup['IndividualMSE_Oracle']['retmse']),v5_spearman=float(v5['spearman']),v5_recall=float(v5['recall']),v5_boundary_rank11_20_exchange_share=float(v5['boundary_11_20_exchange'])/reg if reg else None,improvement_vs_best_simple_pct=100*(float(simple['retmse'])-float(v5['retmse']))/float(simple['retmse']),test_used_to_change_configuration=False))
    probe=read(p/'probe_train.json');selection=read(p/'probe_selection.json')
    audit.append(dict(cell=cell,splits=splits,original_complete_status=done['status'],probe_train_query_count=probe['train_query_count'],probe_pair_count=probe['pairs'],probe='Ridge only; fixed three alphas validation-selected',test_used_for_selection=selection['test_used_for_selection'],figures=len(list((p/'figures').glob('*.png'))),original_results_unchanged=True))

write_csv(OUT/'TOP10_METHOD_OVERLAP.csv',overlap)
write_csv(OUT/'CI_VALIDITY_REVIEW.csv',intervals)
write_csv(OUT/'MISSED_RANK_BIN_FRACTIONS.csv',bins)
write_csv(OUT/'ETTH1_DESCRIPTIVE_COMPARISON.csv',comparison)
remaining=[
    dict(item='Full Oracle / pool-regret / Full coverage',status='DEFERRED_FULL_DEPENDENCY',reason='P100 all12 must finish before any Full computation'),
    dict(item='Mean Student score of Missed Oracle / False Selected',status='NOT_COMPUTED',reason='Scores are not present in saved per-query diagnostics; cannot reconstruct from Top10 IDs alone'),
    dict(item='Pooled missed-rank median/P90',status='NOT_STORED',reason='Current CSV averages per-query rank summaries; not a pooled candidate distribution'),
    dict(item='Trend/volatility/periodicity input strata',status='NOT_COMPUTED',reason='Current strata use validation regret terciles per channel; input strata can be added from saved feature means'),
    dict(item='Full ranking within frozen P100 membership',status='NOT_YET_AUDITED',reason='Full worker computes global and Student Top50/100; explicit same-Full-model P100 restricted ranking is not currently emitted'),
    dict(item='Dataset-level failure attribution and loss recommendations',status='PENDING_FINAL_EVIDENCE_SYNTHESIS',reason='Current final report generator is a conservative generic template; quantitative recommendations still need actual all-cell evidence'),
]
payload=dict(status='PARTIAL_REQUIRED_DETAIL_GAPS',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),cells=audit,remaining=remaining,statistics=dict(bootstrap_reps=2000,seed=0,resampling='circular time blocks; all simultaneous channels jointly',nominal_block='4H clipped to query count',ETTh1_H720_CI='UNINFORMATIVE_DEGENERATE; do not claim statistical significance',ETTh1_H336_CI='fewer than two effective nonoverlap blocks; exploratory only',multiple_comparisons='not corrected; no significance claims'),added_cpu_only_summaries=True,original_outputs_preserved=True,all_requested_details_complete=False)
(OUT/'AUDIT.json').write_text(json.dumps(payload,indent=2)+'\n')
text='''# ETTh1 P100 진단 완결성 감사

현재 셀 COMPLETE는 구현된 계산과 저장을 마쳤다는 상태다. 요청된 모든 세부 산출물의 완결성을 뜻하지 않는다. 기존 결과와 실행 코드는 수정하지 않았다.

4셀 val/test 전체 query와 7채널, 연속 query ID 및 중복 없는 Top10, 기존 모델 재현 gate를 확인했다. 각 셀에 17개 주요 CSV와 8개 그림, query별 Parquet가 있다. P100 Kendall은 4,950쌍 exact, Full Kendall은 예정된 고정4,096쌍 근사다. 주 지표 평가는 전체 query이며 Ridge 학습만 256개 train query/channel의 작은 진단 예산을 쓴다. MLP·pairwise probe는 실행하지 않았다.

## 통계 한계

H720은 실제 block2161 = 전체 query2161이라 circular bootstrap 평균이 불변이다. 원래 폭0 CI는 유효한 불확실성 추정으로 취급하지 않는다. CI_VALIDITY_REVIEW.csv에서 원래 수치와 상태를 보존하고 수정 표시에서는 CI를 비워 두었다. H336은 유효 시간 블록1.89개로 근거가 약하다. 전체 CI는 다중비교 보정 없이 탐색적으로만 해석한다.

## CPU로 보완한 표

TOP10_METHOD_OVERLAP.csv는 저장된 실제 후보 ID의 방법 간 overlap이다. MISSED_RANK_BIN_FRACTIONS.csv는 평균 누락 개수로 가중한 후보별 rank-bin 비율이다. 기존 per-query median 평균을 pooled median으로 표기하지 않는다.

## 아직 남은 항목

'''
for x in remaining:text+=f"- {x['item']}: {x['status']}. {x['reason']}\n"
text+='\nFull 단계 및 후순위 Solar Full8은 아직 완료되지 않았다. 모든 세부 산출물까지 완료됐다고 보고하지 않는다.\n'
(OUT/'AUDIT_KO.md').write_text(text,encoding='utf8')
print(json.dumps(dict(audit_path=str(OUT),status=payload['status'],cells=[x['cell'] for x in audit],overlap_rows=len(overlap),ci_rows=len(intervals),rank_bin_rows=len(bins),remaining=remaining),ensure_ascii=False))
