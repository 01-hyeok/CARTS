"""CPU-only, source-verified interpretation of all 16 sampled SoftLag pilots.

Independent of the delayed four legacy pair exports; never claims those exist.
Original values and every completed checkpoint/artifact remain unchanged.
"""
import csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from contracts import *

def read_csv(p):
    with p.open() as f:return list(csv.DictReader(f))
def write_csv(p,rows):
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
def main():
    dest=OUT/'sampled_results_review02';dest.mkdir(parents=True,exist_ok=True)
    existing=checked_gate(dest/'REVIEW_GATE.json','PASS_ALL16_SAMPLED_METRICS_WITH_EXPLICIT_EXPORT_GAPS')
    if existing:print(json.dumps(existing));return
    tests=[];comparisons=[];cis=[];heads=[];availability=[];sources=[]
    for cell,support in JOBS:
        reused=(cell,support) in REUSE;folder=(OLD if reused else OUT)/(cell+'_'+support)
        g=checked_gate(folder/'FINAL_GATE.json','PASS_PILOT_CELL_COMPLETE' if reused else 'PASS_EXPANSION_CELL_COMPLETE')
        assert g and g['samples']=={'train':64,'val':32,'test':32} and g['epochs_saved']==list(range(6))
        cp=json.loads((folder/'PROVENANCE.json').read_text())['identity']['checkpoint'];assert sha(cp['checkpoint'])==cp['checkpoint_sha256']
        C=7 if cell.startswith('ETTh1') else 21
        tags=dict(cell=cell,support=support,queries=32,channels=C,seed=0,reused=reused,coverage='fixed 32 test timestamps; all channels; NOT full query population',checkpoint_sha256=cp['checkpoint_sha256'])
        p=folder/'TEST_RESULTS.csv';rows=read_csv(p);indexed={x['method']:x for x in rows};assert set(indexed)=={'B0','B1','B2','random','oracle','LastRepeat'}
        for row in rows:
            assert int(row['queries'])==32 and int(row['channels'])==C
            tests.append(dict(row,**{k:v for k,v in tags.items() if k not in ['queries','channels']},source=str(p)))
        raw={};starts=None
        for arm in ['B0','B1','B2']:
            p=folder/(arm+'_test_queries.npz')
            with np.load(p) as z:
                raw[arm]=z['metrics'].copy();assert raw[arm].shape==(32,C,6) and np.isfinite(raw[arm]).all()
                assert z['selected_ids'].shape==(32,C,10)
                assert bool(np.all(np.diff(np.sort(z['selected_ids'],axis=-1),axis=-1)>0))
                if starts is None:starts=z['query_starts'].copy()
                else:assert np.array_equal(starts,z['query_starts'])
            for j,key in enumerate(['retmse','agg_mse','agg_mae','recall']):assert float(raw[arm][...,j].mean())==float(indexed[arm][key])
            sources.append(dict(path=str(p),sha256=sha(p)))
        for a,b in [('B1','B0'),('B2','B0'),('B2','B1')]:
            va=raw[a][...,0];vb=raw[b][...,0];ra=float(va.mean());rb=float(vb.mean())
            comparisons.append(dict(**tags,comparison=a+'-'+b,retmse_a=ra,retmse_b=rb,retmse_delta=ra-rb,
                relative_improvement_percent=100*(1-ra/rb),outcome='WIN' if ra<rb else 'LOSS' if ra>rb else 'TIE',
                timestamp_win_fraction=float((va.mean(1)<vb.mean(1)).mean()),query_channel_win_fraction=float((va<vb).mean()),
                query_channel_tie_fraction=float((va==vb).mean()),agg_mse_delta=float(raw[a][...,1].mean()-raw[b][...,1].mean()),
                agg_mae_delta=float(raw[a][...,2].mean()-raw[b][...,2].mean()),recall_delta=float(raw[a][...,3].mean()-raw[b][...,3].mean()),source=str(folder)))
        p=folder/'PAIRED_TIME_BLOCK_CI.csv'
        for row in read_csv(p):
            assert int(row['repeats'])==2000 and float(row['confidence'])==.95
            row.update(cell=cell,support=support,channels=C,source=str(p),block_rule=str(int(row['block_length'])//int(cell.rsplit('_',1)[1]))+'H')
            row['interpretation']='INSUFFICIENT_BLOCKS' if row['sufficient'].lower()!='true' else 'EXPLORATORY_UNADJUSTED_PERCENTILE_95_CI'
            cis.append(row)
        p=folder/'HEAD_USEFULNESS.csv'
        for row in read_csv(p):row.update(cell=cell,support=support,source=str(p));heads.append(row)
        pair=folder/('PAIR_EXPORT_GATE.json' if reused else 'FINAL_GATE.json')
        availability.append(dict(**tags,training_selection_test='PASS',head_usefulness='PASS',pair_overlap='PASS' if pair.exists() else 'PENDING_LEGACY_EXPORT',source=str(folder)))
        for name in ['FINAL_GATE.json','TEST_RESULTS.csv','PROVENANCE.json','PAIRED_TIME_BLOCK_CI.csv','HEAD_USEFULNESS.csv']:
            p=folder/name;sources.append(dict(path=str(p),sha256=sha(p)))
    assert len(tests)==96 and len(comparisons)==48 and len(cis)==144 and len(heads)==240
    summary=[]
    for comparison in ['B1-B0','B2-B0','B2-B1']:
        for support in ['ALL','FULL','P100']:
            rows=[r for r in comparisons if r['comparison']==comparison and (support=='ALL' or r['support']==support)]
            summary.append(dict(comparison=comparison,support=support,conditions=len(rows),wins=sum(r['outcome']=='WIN' for r in rows),losses=sum(r['outcome']=='LOSS' for r in rows),ties=sum(r['outcome']=='TIE' for r in rows),
                equal_condition_mean_relative_improvement_percent=float(np.mean([r['relative_improvement_percent'] for r in rows])),
                median_relative_improvement_percent=float(np.median([r['relative_improvement_percent'] for r in rows])),
                interpretation='Descriptive condition counts and equal-weight relative changes; not pooled MSE, independent trials, or significance'))
    for name,rows in [('TEST_RESULTS.csv',tests),('PAIRED_COMPARISONS.csv',comparisons),('DESCRIPTIVE_SUMMARY.csv',summary),('PAIRED_TIME_BLOCK_CI.csv',cis),('HEAD_USEFULNESS.csv',heads),('COVERAGE_AND_GAPS.csv',availability),('SOURCE_INVENTORY.csv',sources)]:write_csv(dest/name,rows)
    gaps=[r for r in availability if r['pair_overlap']!='PASS']
    lines=['# SoftLag 16조건 고정 표본 평가 검토','',
        '**평가 범위: 고정 train 64 / validation 32 / test 32 timestamps × 전체 채널. 전체 query 평가가 아니다.** ETTh1은 7채널, Weather는 21채널이다. 각 horizon 96/192/336/720과 Full/P100 총16조건을 포함한다. seed0만 사용했다.',
        '', '16조건의 학습·validation 선택·selected test·조건별 최종 gate가 완료됐다. 기존 4조건의 head-pair 상세 export는 %d조건 미완료이며, 이 파일은 전체 워크플로 최종 완료 보고서가 아니다. 신규12조건은 원본과 다른 경로에 저장했고 기존4조건은 검증 후 재사용했다.'%len(gaps),
        '', 'B0는 해당 support의 기존 frozen V5이다. B1은 모든 head가 공유하는 lag bias와 alpha, B2는 head별 lag bias와 공통 alpha를 학습한다. 세 방법 모두 기존 5개 query head를 유지한다. B1은 독립 V1 retriever가 아니다. Encoder/query-head 파라미터는 변경하지 않았다.',
        '', 'B1/B2 각5epoch, batch8,40optimizer step, Adam lr0.01, 동일 배치 순서를 사용했다. epoch0~5 모두 보존하고 validation RetMSE 최소 epoch(동률이면 이른 epoch)를 선택한 뒤 test를 평가했다. H96의 학습 파라미터는 B1 26/B2 126개, 나머지 horizon은 34/166개이므로 용량 차이가 있다.',
        '', 'Full은 전체 masked candidate bank이며 P100은 기존 ordered 후보100개이다. 모든 채널이 하나의 GPU batch 축에 들어간다. Lag는 query[t]와 candidate 과거[t+delta]를 비교하며 원형 이동·미래 입력을 사용하지 않는다. Teacher는 기존 negative z-score future-MSE 분포다. 평가지표는 train split scaler로 표준화한 값 공간에서 delta-last 복원 후보 미래와 query 미래를 비교한다. Teacher의 z-score를 평가 MSE에 적용하지 않는다.',
        '', 'RetMSE는 선택 후보10개의 개별 future MSE 평균, AggMSE/MAE는 후보10개 평균 예측의 오차, Recall은 같은 support의 Oracle10과 겹치는 비율이다. TEST_RESULTS.csv에 B0/B1/B2·Random·Oracle·LastRepeat 모두 포함했다. LastRepeat의 검색 RetMSE/Recall은 적용 불가로 비워 두었다.',
        '', '| 조건 | Support | B0 RetMSE | B1 RetMSE | B2 RetMSE | B1/B2 선택 epoch |','|---|---|---:|---:|---:|---|']
    for c,s in JOBS:
        r={x['method']:x for x in tests if x['cell']==c and x['support']==s}
        lines.append('| %s | %s | %.9f | %.9f | %.9f | %s/%s |'%(c,s,float(r['B0']['retmse']),float(r['B1']['retmse']),float(r['B2']['retmse']),r['B1']['selected_epoch'],r['B2']['selected_epoch']))
    lines+=['','## 관찰과 해석','', '| 비교 | 개선 조건 | 악화 조건 | 동일 조건 |','|---|---:|---:|---:|']
    for r in summary:
        if r['support']=='ALL':lines.append('| %s | %d | %d | %d |'%(r['comparison'],r['wins'],r['losses'],r['ties']))
    lines+=['','판정: **INCONCLUSIVE**. 조건별 이득과 손해가 함께 존재한다. 특히 Weather H336 P100에서는 B1이 B2보다 RetMSE가 낮아, head별 lag bias가 공유 lag보다 일관되게 낫다는 근거가 없다. 더 많은 파라미터, validation에서 선택된 epoch 차이, 작은 고정 표본과 단일 seed가 해석을 제한한다.',
        '', 'Full/P100은 서로 다른 학습 support의 frozen checkpoint를 사용한다. 두 support 사이 수치 차이를 동일 파라미터에서 후보 수만 바꾼 효과로 해석하면 안 된다. Dataset/horizon마다 오차 척도도 달라 전체 절대MSE를 단순 통합하지 않았다. DESCRIPTIVE_SUMMARY는 조건별 상대 변화에 동일 가중치를 준 기술통계다.',
        '', 'CI는 같은 timestamp의 모든 채널을 공동 resampling하는 H/2H/4H 시간 block, seed0 2000회 percentile95%다. 4개 미만 block은 불충분이며 CSV에 표시했다. 다중 비교 보정을 하지 않았고, 이는 여러 training seed의 불확실성을 측정한 것이 아니다. CI가 좁거나 0을 제외하더라도 head specialization의 인과적 입증으로 해석하지 않는다.',
        '', 'Head별 고유 유용 후보는 Oracle10 경계 MSE + max(1e-8, 경계MSE×1/5/10%)의 진단 정의를 사용한다. 미래는 이 유용성 평가에만 사용하고 검색 lag feature 입력에는 사용하지 않았다. HEAD_USEFULNESS.csv의 관찰은 후속 설명 근거이며 causality를 확정하지 않는다.',
        '', '시간은 다른 진단과 공존한 wall time이며 단독 GPU 성능으로 해석하지 않는다. 별도 학습 GPU kernel time/validation wall time은 미측정이다. 기존 runtime provenance UNKNOWN을 유지하며 현재 재현 PASS와 구분한다.',
        '', '남은 작업: 기존 ETTh1 H96/H336 Full/P100의 head-pair export 및 최종 통합 gate. D1–D6의 복구·보완·보고서 작업은 별도 실행 중이다. 이 검토는 새 학습/추론을 실행하지 않았고 Solar를 재개하지 않았다.',
        '', '재현 파일: TEST_RESULTS.csv(96행), PAIRED_COMPARISONS.csv(48행), DESCRIPTIVE_SUMMARY.csv(9행), PAIRED_TIME_BLOCK_CI.csv(144행), HEAD_USEFULNESS.csv(240행), COVERAGE_AND_GAPS.csv(16행), SOURCE_INVENTORY.csv. 원본 숫자 문자열과 source 경로·SHA256를 보존했다.']
    (dest/'REVIEW_KO.md').write_text('\n'.join(lines),encoding='utf8')
    gate=dict(status='PASS_ALL16_SAMPLED_METRICS_WITH_EXPLICIT_EXPORT_GAPS',at=now(),cells=16,train_timestamps=64,val_timestamps=32,test_timestamps=32,all_channels=True,full_population=False,verdict='INCONCLUSIVE',pending_pair_exports=[dict(cell=x['cell'],support=x['support']) for x in gaps],files={p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='REVIEW_GATE.json'})
    dump(dest/'REVIEW_GATE.json',gate);print(json.dumps(dict(path=str(dest),gate=gate,descriptive_summary=summary)),flush=True)
if __name__=='__main__':main()
