"""CPU-only final deliverable assembly, gated by actual immutable artifacts.

This file never turns partial computation into COMPLETE and never starts GPU
work. It preserves every prior track and separates population vs sample scope.
"""
import csv,json,os,time,traceback
from pathlib import Path
import numpy as np
from d56_aggregate import sha,dump,csvout,now
ROOT=Path('/data/pjh_workspace/CARTS');BASE=ROOT/'results/TRACK-HEAD-SHIFT-DIAGNOSTIC01'
EXT=BASE/'extensions_d5_d6';POP=BASE/'full_population_completion';OUT=BASE/'final_bundle01'
CELLS=[d+'_'+str(h) for d in ['ETTh1','Weather'] for h in [96,192,336,720]]

def verified_gate(path,expected):
    if not path.exists():return False
    g=json.loads(path.read_text());assert g['status']==expected
    for name,h in g.get('files',{}).items():assert sha(path.parent/name)==h
    return g

def requirements():
    gaps=[];gates=[]
    for c in CELLS:
        for training in ['FULL','P100']:
            p=POP/'analysis'/c/('V5_'+training)/'DELIVERABLE_GATE.json'
            if not p.exists():gaps.append(str(p));continue
            gates.append((p,'PASS_FULL_QUERY_D1_D2_D3'))
        for m in ['V0','V1','V5']:
            for t in ['FULL','P100']:
                for e in ['FULL','P100']:
                    for folder,name,status in [('complete_analysis','ANALYSIS_GATE.json','PASS_D5_D6_CASE'),('d14_statistics','SAMPLE_STATISTICS_GATE.json','PASS_D4_SAMPLE_STATISTICS_AND_D12_SAMPLE_ADDITIONS')]:
                        p=EXT/folder/c/(m+'_'+t+'_to_'+e)/name
                        if not p.exists():gaps.append(str(p))
                        else:gates.append((p,status))
    soft=ROOT/'results/TRACK-SOFTLAG-HEAD-PILOT01/DELIVERABLE_COMPLETENESS_GATE.json'
    if not soft.exists():gaps.append(str(soft))
    return gaps,gates,soft

def combine(name,paths):
    rows=[]
    for p in paths:
        with p.open() as f:part=list(csv.DictReader(f))
        rows.extend([dict(source_file=str(p.relative_to(BASE)),**r) for r in part])
    if rows:csvout(OUT/name,rows)
    return rows

def macro(rows,keys,metric):
    groups={}
    for r in rows:
        if r.get('split')!='test' or r.get(metric,'') in ['',None]:continue
        groups.setdefault(tuple(r[k] for k in keys),[]).append(float(r[metric]))
    return {k:float(np.mean(v)) for k,v in groups.items()}

def parquet_features():
    import pyarrow as pa
    import pyarrow.parquet as pq
    folder=OUT/'D4_TEMPORAL_SHIFT_FEATURES.parquet';folder.mkdir(exist_ok=True)
    manifest=[]
    for gate in sorted((EXT/'complete_features').glob('*/*/FEATURE_GATE.json')):
        g=json.loads(gate.read_text());cell=gate.parent.parent.name;case=gate.parent.name
        for split in ['val','test']:
            target=folder/cell/case/(split+'.parquet');target.parent.mkdir(parents=True,exist_ok=True);side=target.with_suffix('.json')
            if target.exists():
                v=json.loads(side.read_text());assert v['source_gate_sha256']==sha(gate) and v['sha256']==sha(target);manifest.append(dict(path=str(target.relative_to(OUT)),**v));continue
            temp=target.with_suffix('.tmp');writer=None;n=0
            try:
                for p in sorted((gate.parent/split).glob('query*.npz')):
                    assert sha(p)==g['files'][str(p.relative_to(gate.parent))]
                    with np.load(p) as f:z={k:f[k].copy() for k in f.files}
                    c,j=np.nonzero(z['valid']);q=int(z['query_index'][0]);count=len(c)
                    columns=dict(query_index=np.full(count,q,np.int64),query_start=np.full(count,int(z['query_start'][0]),np.int64),channel=c.astype(np.int16),
                        candidate_id=z['global_ids'][c,j],eligible=z['eligible'][c,j],near_oracle=z['near_mask'][c,j],query_std=z['query_std'][c])
                    for key in ['mse','zero','best','gain','best_lag','margin','entropy','recent_difference','candidate_std','near_tie_lag_fraction']:columns[key]=z[key][c,j]
                    # Per-candidate group memberships preserve eligible/out-of-support
                    # distinction. Oracle groups are diagnostics, not score inputs.
                    ids=z['global_ids'];sel=z['ordered_ids'][:,:10];oracle=z['oracle_ids'][:,:10]
                    for key,ref in [('selected',sel),('oracle10',oracle),('oracle30',z['oracle_ids']),('random10',z['random_ids'][:,:10])]:
                        columns[key]=np.array([int(ids[cc,jj]) in set(ref[cc]) for cc,jj in zip(c,j)])
                    table=pa.table(columns)
                    if writer is None:writer=pq.ParquetWriter(temp,table.schema,compression='zstd')
                    writer.write_table(table);n+=count
            finally:
                if writer:writer.close()
            assert n>0 and pq.ParquetFile(temp).metadata.num_rows==n;temp.replace(target)
            v=dict(rows=n,sha256=sha(target),source_gate_sha256=sha(gate),cell=cell,case=case,split=split,
                coverage='32 fixed timestamps; all channels; diagnostic candidate union with explicit eligibility; NOT full-query/Full-bank feature tensor')
            dump(side,v);manifest.append(dict(path=str(target.relative_to(OUT)),**v))
    csvout(OUT/'D4_PARQUET_INVENTORY.csv',manifest)
    return manifest

def figures(d1,d2,d3,d4,groups,d5,d6):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8,'figure.dpi':130})
    def bars(name,title,values,ylabel):
        labels=[' / '.join(map(str,k)) for k in values];fig,ax=plt.subplots(figsize=(12,max(3,len(labels)*.16)))
        y=np.arange(len(labels));ax.barh(y,list(values.values()));ax.set_yticks(y);ax.set_yticklabels(labels);ax.invert_yaxis();ax.set_xlabel(ylabel);ax.set_title(title);fig.tight_layout();fig.savefig(OUT/name);plt.close(fig)
    bars('head_unique_useful_candidates.png','V5 exclusive Near-Oracle candidates (5%); full query population',macro(d1,['cell','training_support','variant'],'near_count_0.05'),'Mean count per head/query/channel; descriptive')
    bars('fusion_vs_discovery_regret.png','V5 fusion regret; full query population',macro(d2,['cell','training_support','variant'],'fusion_regret'),'RetMSE gap (lower is better)')
    bars('normalizer_only_effect.png','Same-checkpoint P100 normalization-only delta RB - RA',macro(d3,['cell','training_support'],'normalization_only_delta'),'RetMSE change; negative improves')
    chosen=[r for r in groups if r['model']=='V5' and r['training_support']==r['evaluation_support'] and r['group'] in ['MissedOracle','FalseSelected']]
    bars('shift_gain_distribution.png','Mean shift gain by group; 32 timestamps/all channels',macro(chosen,['cell','training_support','group'],'gain_mean_over_nonempty_qc'),'Mean past-only cosine gain; not a quality score')
    selected=[r for r in d4 if r['model']=='V5' and r['training_support']==r['evaluation_support'] and r['method']=='ShiftRecent' and r['R']=='100']
    bars('shift_reranking_retmse.png','Validation-selected Shift+Recent R100; 32 timestamps/all channels',{(r['cell'],r['training_support']):float(r['retmse']) for r in selected},'Test RetMSE@10 (lower is better)')
    rel=[]
    for p in (EXT/'d14_statistics').glob('*/*/D4_SHIFT_RELEVANCE_ANALYSIS.csv'):
        with p.open() as f:
            rel.extend(r for r in csv.DictReader(f) if r['model']=='V5' and r['training_support']==r['evaluation_support'] and r['split']=='test' and r['group']=='TopR' and r['gain_vs_future_mse_pearson'])
    bars('shift_gain_vs_future_mse.png','Past shift gain vs future MSE Pearson; descriptive',macro(rel,['cell','training_support'],'gain_vs_future_mse_pearson'),'Mean defined query/channel Pearson; no causal claim')

def main():
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True);lock=(BASE/'00_Audit/final_d1_d6.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        gaps,gates,soft=requirements()
        if not gaps:break
        dump(OUT/'CURRENT.json',dict(status='WAITING_ACTUAL_DELIVERABLE_GATES',at=now(),pid=os.getpid(),missing_count=len(gaps),missing=gaps,gpu_used=False));time.sleep(30)
    gate_inventory=[]
    for p,status in gates:
        g=verified_gate(p,status);assert g
        gate_inventory.append(dict(path=str(p.relative_to(BASE)),sha256=sha(p),status=g['status'],queries=json.dumps(g['queries']),channels=g['channels']))
    softgate=json.loads(soft.read_text());assert softgate['status'].startswith('PASS'),softgate
    for n,h in softgate.get('files',{}).items():assert sha(soft.parent/n)==h
    dump(OUT/'CURRENT.json',dict(status='ASSEMBLING_VERIFIED_CPU_DELIVERABLES',at=now(),pid=os.getpid(),gates=len(gates),gpu_used=False))
    src=POP/'analysis';sample=EXT/'d14_statistics';ana=EXT/'complete_analysis'
    families={}
    for name in ['D1_HEAD_UNIQUE_USEFUL_CANDIDATES','D1_HEAD_RELEVANCE_CONTRIBUTION','D1_HEAD_DIVERSITY','D2_FUSION_DISCOVERY_REGRET','D2_HEAD_LEAVE_ONE_OUT','D3_NORMALIZER_ONLY_RESULTS','D3_HEAD_PROBABILITY_MASS']:
        families[name]=combine(name+'.csv',sorted(src.glob('*/*/'+name+'.csv')))
    for name in ['D4_SHIFT_GROUP_COMPARISON','D4_SHIFT_RELEVANCE_ANALYSIS','D4_SHIFT_RERANKING_PROBE','D4_GROUP_PAIRED_CI','D4_VALIDATION_FIXED_STRATA','D5_SAME_SIZE_BLOCK_CI']:
        families[name]=combine(name+'.csv',sorted(sample.glob('*/*/'+name+'.csv')))
    for name in ['D5_ORACLE_LAG_DIVERSITY','D5_HEAD_LAG_SPECIALIZATION','D5_LAG_AMBIGUITY','D5_LAG_DISTRIBUTION_STABILITY','D6_SHIFT_RERANKING','D6_TOPR_ORACLE_COVERAGE','D6_CANDIDATE_VS_RERANKING_REGRET']:
        families[name]=combine(name+'.csv',sorted(ana.glob('*/*/'+name+'.csv')))
    lost=[]
    for p in src.glob('*/*/D2_LOST_GOOD_CANDIDATES.jsonl.gz'):lost.append(dict(path=str(p.relative_to(BASE)),sha256=sha(p),bytes=p.stat().st_size,scope='all-query lost-candidate records; compressed JSONL'))
    csvout(OUT/'D2_LOST_GOOD_CANDIDATES.csv',lost)
    csvout(OUT/'PHASE0_ARTIFACT_INVENTORY.csv',gate_inventory)
    manifest=parquet_features()
    resources=[]
    for p in (EXT/'complete_features').glob('*/*/FEATURE_GATE.json'):
        g=json.loads(p.read_text());resources.append(dict(track='D4_D5_D6_sample_features',cell=p.parent.parent.name,case=p.parent.name,seconds=g['seconds'],peak_allocated_bytes=g['peak_allocated_bytes'],peak_reserved_bytes=g['peak_reserved_bytes'],queries_per_split=32,channels=g['channels'],source=str(p.relative_to(BASE)),source_sha256=sha(p)))
    for p in (POP/'raw').glob('*/*/GATE.json'):
        g=json.loads(p.read_text());resources.append(dict(track='D1_D2_D3_full_query_missing_traces',cell=p.parent.parent.name,case=p.parent.name,seconds=g['seconds'],peak_allocated_bytes=g['peak_allocated_bytes'],peak_reserved_bytes=g['peak_reserved_bytes'],queries=json.dumps(g['queries']),channels=g['channels'],source=str(p.relative_to(BASE)),source_sha256=sha(p)))
    csvout(OUT/'RESOURCE_SUMMARY.csv',resources)
    d1=families['D1_HEAD_UNIQUE_USEFUL_CANDIDATES'];d2=families['D2_FUSION_DISCOVERY_REGRET'];d3=families['D3_NORMALIZER_ONLY_RESULTS'];d4=families['D4_SHIFT_RERANKING_PROBE'];gr=families['D4_SHIFT_GROUP_COMPARISON'];d5=families['D5_ORACLE_LAG_DIVERSITY'];d6=families['D6_SHIFT_RERANKING']
    figures(d1,d2,d3,d4,gr,d5,d6)
    lines=['# D1–D6 검증 완료 보고서','',
      '실행 범위: ETTh1·Weather × H96/192/336/720, seed0. 기존 checkpoint 불변, Stage2·새 Retriever 학습 없음. Solar를 재개하지 않았다.',
      '', '**Coverage를 구분한다.** D1/D2/D3 V5 본 분석은 전체 validation/test query와 모든 채널이다. D4/D5/D6는 사전 고정한 split별 32시점·모든 채널의 진단 표본이며, V0·독립 V1·V5 × 학습/평가 Full/P100의 96조합이다. Full 후보 수를 줄이지 않았으나 전체-query lag 분석이라고 주장하지 않는다.',
      '', 'P100_NATIVE는 원래 P100 연산, P100_CONTROLLED는 동일 Full 인코딩/점수에서 공통 P100을 제한한 대조다. 두 결과를 합치지 않는다. D3 A/B는 동일 FP32 logit/tau에서 FP64로 정규화하고 C는 기존 canonical Full mixture이다. 전체 경로 차이는 인과적 분해가 아니다.',
      '', 'D1: head 고유 후보는 다른 4개 head Top10과 겹치지 않는 후보다. Useful은 동일 support Oracle10 또는 d10+max(1e-8, εd10), ε=1/5/10%이다. D2 C=모든 head Top10 합집합 ∪ mixture Top10, fusion=Rmix−RoracleC, discovery=RoracleC−RoracleSupport. 후보별 탈락 rank/logit/probability는 압축 JSONL에 모두 보존했다.',
      '', 'D2 제거 delta는 All5−without-head이므로 양수일수록 해당 head 제거가 개선한다. 재학습 ablation이 아니다. D3 m_h=Zp100/Zfull, 암묵적 가중치=m_h/Σm_h, 동일 P100 A/B top10 overlap·tie-aware Spearman·질량 분산을 보존했다. 음수 delta를 clamp하지 않았다.',
      '', 'D4 lag는 query[t]와 candidate[t+δ], 반경 min(16,L//8), 공통 유효 query 구간, patch16/stride8, patch 평균 제거·L2 정규화다. 미래는 평가·Oracle 그룹에만 사용하고 lag 점수와 후보 preselection에는 사용하지 않았다. α/β 및 feature strata는 validation만 사용했다. Reranking은 원래 future-blind TopR 안에서 수행한다.',
      '', 'NDCG는 g(d)=1/(1+d) direct gain과 log2 discount, 같은 support의 Oracle10 IDCG를 사용했다. Missed recovery는 기존에 놓친 Oracle 중 새로 선택한 비율이다. False promotion은 신규 진입 후보 중 비Oracle 비율이며 신규 진입이 없으면 undefined다. K 대비 비율도 별도 저장했다.',
      '', 'D5는 Oracle10/30, 전체 Near-Oracle, 동일 크기 Random 및 head별 useful 후보의 lag 분포·entropy·ambiguity를 비교한다. MSE matched control은 저장된 동일 support 후보 합집합 내 greedy matching이며 전체 후보의 최적 매칭은 아니다. D6 Full R=10/20/50/100/200/500, P100 R=10/20/50/100; Oracle@R과 support Oracle를 구분한다.',
      '', '통계: H/2H/4H, seed0, 2000회, timestamp block 및 모든 채널 공동 resampling, percentile95%. 4블록 미만은 CI 불충분이다. 이는 학습 seed 불확실성이 아니다. 상관관계와 head lag 차이를 인과적 역할 또는 필연적 multi-head 필요성으로 해석하지 않는다.',
      '', '| 조건 | 학습 support | 평가 variant | RetMSE | Fusion regret | Discovery regret | 고유 Near5% 후보/head |', '|---|---|---|---:|---:|---:|---:|']
    keys=['cell','training_support','variant'];ret=macro(d2,keys,'r_mix');fus=macro(d2,keys,'fusion_regret');disc=macro(d2,keys,'discovery_regret');uniq=macro(d1,keys,'near_count_0.05')
    for k in sorted(ret):lines.append('| %s | %s | %s | %.9f | %.9f | %.9f | %.6f |'%(*k,ret[k],fus[k],disc[k],uniq[k]))
    lines+=['','| 조건 | 학습 support | 정규화 Δ RB−RA | support Δ RC−RB | A/B overlap | A/B Spearman |','|---|---|---:|---:|---:|---:|']
    for r in d3:
        if r['split']=='test':lines.append('| %s | %s | %.9f | %.9f | %.6f | %s |'%(r['cell'],r['training_support'],float(r['normalization_only_delta']),float(r['support_after_normalization_delta']),float(r['A_B_top10_overlap']),r['A_B_common_P100_spearman']))
    lines+=['','아래는 native V5의 표본 진단이다. 다른 모델 및 cross-support 수치는 동일 이름 CSV에 포함된다.','','| 조건 | support | R | 방법 | Original RetMSE | Rerank RetMSE | Oracle@R | Support Oracle | Coverage O10 |','|---|---|---:|---|---:|---:|---:|---:|---:|']
    for r in d6:
        if r['model']=='V5' and r['training_support']==r['evaluation_support'] and r['channel']=='ALL' and r['method']!='Original' and r['R'] in ['100','500']:
            lines.append('| %s | %s | %s | %s | %.9f | %.9f | %.9f | %.9f | %.6f |'%(r['cell'],r['training_support'],r['R'],r['method'],*[float(r[k]) for k in ['original_retmse','reranked_retmse','oracle_at_r','support_oracle_retmse','oracle10_coverage']]))
    lines+=['','최종 연구 방향 판정은 **INCONCLUSIVE**다. 조건별 fusion/discovery와 정규화 delta, validation-selected shift 이득은 위의 수치로 판단하되, seed0 및 표본 기반의 무학습 분석만으로 특정 새 loss의 효과나 head 역할 분화의 인과성을 입증하지 않는다. 기존 SoftLag pilot의 B0/B1/B2 결과는 별도 트랙이며 D4의 무학습 reranking과 혼용하지 않는다.',
      '', '후속 최소 비교: fusion 및 normalizer 영향이 큰 조건에서는 score calibration 대조, shift reranking 이득이 안정적인 조건에서는 single SoftLag 대조, 고유 useful 후보와 안정적 lag 분화가 동시에 나타난 조건에서는 multi-head specialization 대조를 검토한다. 본 분석에서 새 loss나 추가 모델을 학습하지 않았다.',
      '', '측정 wall time에는 코드에 기록된 inference/CPU 단계만 포함하며 GPU 연산 시간과 동일하지 않다. Shared cache 최초 생성 비용과 재사용 비용이 달라 독립 모델 속도 비교에 쓰지 않는다. 과거 원본 런타임 UNKNOWN과 현재 재현 PASS를 구분한다.',
      '', '모든 원시 CSV 행을 이 본문에 펼쳐 싣지 않았다. ARTIFACT_INVENTORY.csv와 PHASE0_ARTIFACT_INVENTORY.csv에 경로·크기·SHA256·행수/컬럼을 제공한다. candidate raw features는 partitioned Parquet와 원본 NPZ에 있으며 원래 결과를 덮어쓰지 않았다.']
    report='\n'.join(lines)
    for name in ['FINAL_REPORT.md','D1_HEAD_DIVERSITY_SUMMARY.md','D2_FUSION_FAILURE_REPORT.md','D3_SUPPORT_SENSITIVITY_SUMMARY.md','D4_SHIFT_DIAGNOSTIC_REPORT.md','NEXT_RESEARCH_DECISION.md']:(OUT/name).write_text(report,encoding='utf8')
    (OUT/'PHASE0_REPRODUCTION_AUDIT.md').write_text('각 source gate와 checkpoint/pool/data identity를 검증했다. 전체-query D1–D3는 조건별 first/tail smoke와 native ordered Top10 검증 뒤 실행했다. D4–D6는 96개 조합 각각 feature·analysis·통계 gate를 검증했다. 본 보고서는 artifact 존재만으로 PASS를 부여하지 않는다.\n',encoding='utf8')
    inventory=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or p.name in ['CURRENT.json','VALIDATION_GATE.json','ARTIFACT_INVENTORY.csv']:continue
        r=dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p))
        if p.suffix=='.csv':
            with p.open() as f:reader=csv.reader(f);r['columns']=json.dumps(next(reader,[]));r['rows']=sum(1 for _ in reader)
        inventory.append(r)
    csvout(OUT/'ARTIFACT_INVENTORY.csv',inventory)
    dump(OUT/'VALIDATION_GATE.json',dict(status='PASS_D1_D6_ALL_SPECIFIED_CELLS_WITH_DECLARED_SAMPLING',at=now(),full_query_V5_cases=16,D456_model_support_cases=96,
        full_query_sampling_not_implied_for_D456=True,source_gates=gate_inventory,softlag_deliverable_gate_sha256=sha(soft),parquet_files=len(manifest),
        files={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['CURRENT.json','VALIDATION_GATE.json']}))
    dump(OUT/'CURRENT.json',dict(status='COMPLETE',at=now(),pid=os.getpid(),gate_sha256=sha(OUT/'VALIDATION_GATE.json'),gpu_used=False))
    print('FINAL D1-D6 DELIVERABLE GATE PASS',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        OUT.mkdir(parents=True,exist_ok=True);dump(OUT/'CURRENT.json',dict(status='BLOCKED_FINAL_ASSEMBLY',at=now(),error=repr(e),traceback=traceback.format_exc()));raise
