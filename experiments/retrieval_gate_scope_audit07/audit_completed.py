"""Read existing CPU artifacts; never import torch or modify source results."""
import csv,hashlib,json,datetime
from pathlib import Path
import numpy as np

ROOT=Path('/data/pjh_workspace/CARTS')
SOURCE=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01'
SUPPLEMENT=ROOT/'results/TRACK-RETRIEVAL-DIAGNOSTIC-SUPPLEMENT01'
CELLS=['ETTh1_96','ETTh1_192','ETTh1_336','ETTh1_720','Weather_96','Weather_192']
MODELS=['V0_FullTrained','V5_FullTrained']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def manifest_sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()


def raw_id_evidence(directory, queries, channels, first_query, candidates):
    h={m:hashlib.sha256() for m in MODELS}
    qh={m:hashlib.sha256() for m in MODELS}
    files=sorted(directory.glob('batch*.npz'))
    assert len(files)==(queries+31)//32,('batch_count',directory,len(files),queries)
    count=0
    for batch,p in enumerate(files):
        assert p.name=='batch%05d.npz'%batch
        with np.load(p,allow_pickle=False) as z:
            names=z['method_names'].tolist();ids=z['selected_ids'];starts=z['query_start']
        b=min(32,queries-count)
        assert ids.shape==(b,channels,len(names),10)
        assert np.array_equal(starts,np.arange(first_query+count,first_query+count+b))
        assert len(set(names))==len(names) and all(m in names for m in MODELS)
        for m in MODELS:
            x=ids[:,:,names.index(m)]
            assert x.dtype==np.int64,('Unexpected candidate dtype',x.dtype)
            assert (x>=0).all() and (x<candidates).all()
            assert (np.diff(np.sort(x,axis=-1),axis=-1)!=0).all()
            h[m].update(x.transpose(1,0,2).copy().tobytes())
            qh[m].update(x.copy().tobytes())
        count+=b
    assert count==queries
    return {m:dict(raw_to_reproduction_id_sha256=h[m].hexdigest(),
                   raw_query_major_id_sha256=qh[m].hexdigest(),
                   queries=count,units=count*channels,batches=len(files)) for m in MODELS}


def main():
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=SOURCE/'00_Audit/GATE_SCOPE07'/stamp;out.mkdir(parents=True,exist_ok=False)
    rows=[];cell_records=[];all_hash_details={}
    for cell in CELLS:
        p=SOURCE/'FULL'/cell;prov=read(p/'PROVENANCE.json');gate=read(p/'CANONICAL_REPRODUCTION.json')
        models=read(p/'MODEL_STATUS.json')['models'];done=read(p/'COMPLETE.json') if (p/'COMPLETE.json').exists() else None
        assert prov['seed']==0 and prov['support']=='FULL' and prov['all_channels'] and prov['all_queries']
        assert gate['status']=='PASS'
        expected={(m,s) for m in MODELS for s in ['val','test']}
        assert len(gate['rows'])==4 and {(r['model'],r['split']) for r in gate['rows']}==expected
        hash_details=[]
        for rel,digest in prov['source_hashes'].items():
            f=ROOT/rel;actual=sha(f);assert actual==digest,('changed source',cell,rel)
            hash_details.append(dict(path=rel,recorded=digest,actual=actual,status='PASS'))
        all_hash_details[cell]=hash_details
        if done:
            assert done['status']=='COMPLETE' and done['source_hashes']==prov['source_hashes']
            delivered=read(SUPPLEMENT/'FULL'/cell/'DELIVERABLE_GATE.json')
            assert delivered['status']=='PASS' and delivered['source_complete_sha256']==sha(p/'COMPLETE.json')
            assert done['missing_models']==[]
        for split in ['val','test']:
            nq=prov['data']['counts'][split];c=prov['data']['channels'];n=prov['data']['candidate_count']
            raw=raw_id_evidence(p/split,nq,c,prov['data']['starts'][split][0],n) if done else None
            split_gate=read(p/split/'SPLIT_GATE.json') if done else None
            if done:
                assert split_gate['status']=='PASS' and split_gate['query_count']==nq
                assert split_gate['channels']==c and split_gate['units']==nq*c
            for name in MODELS:
                arch=name.split('_')[0];model=next(m for m in models if m['architecture']==arch)
                cp=sha(model['checkpoint']);assert cp==model['checkpoint_sha256']
                g=next(r for r in gate['rows'] if r['model']==name and r['split']==split)
                assert g['status']=='PASS' and g['units']==nq*c and g['max_stored_prediction_difference']==0
                pair='NOT_RECORDED_INDEPENDENT_ALLCHANNEL_PAIR'
                refsource='native canonical implementation; no original saved ID reference'
                rt=prov['torch_version'];paired_path=''
                if 'canonical_ordered_id_sha256' in g:
                    assert g['canonical_ordered_id_sha256']==g['ordered_candidate_id_sha256'] and g['ordered_mismatches']==0
                    pair='PASS_FULL_RECOMPUTED_CANONICAL_VS_BATCHED';refsource='canonical runtime recomputed every query'
                    paired_path=str(p/'CANONICAL_REPRODUCTION.json')
                elif cell=='Weather_96' and arch=='V5':
                    f=SOURCE/'00_Audit/ALL_CHANNEL03/Weather_96_V5_TRUE_BATCH_V2_FULL_GATE.json';pg=read(f)
                    v=next(r for r in pg['rows'] if r['split']==split)
                    assert v['queries']==nq and v['channels']==c and v['units']==nq*c
                    assert v['ordered_mismatch_units']==0 and v['canonical_id_sha256']==v['concurrent_id_sha256']
                    assert raw[name]['raw_query_major_id_sha256']==v['canonical_id_sha256']
                    assert pg['checkpoint_sha256']==cp
                    assert pg['source_hashes']['packed_batch.py']==prov['source_hashes']['experiments/retrieval_all_channel03/packed_batch.py']
                    pair='PASS_FULL_RECOMPUTED_CANONICAL_VS_BATCHED';refsource='independent V5-only whole-split gate + raw IDs matched'
                    paired_path=str(f)
                cache=SOURCE/'full_v0_canonical_scores'/cell/'COMPLETE.json'
                cache_state='NOT_APPLICABLE';cache_sha=''
                if arch=='V0' and cache.exists():
                    cg=read(cache);v=next(x for x in cg['splits'] if x['split']==split)
                    assert cg['status']=='PASS' and cg['provenance']['checkpoint_sha256']==cp
                    assert (v['queries'],v['channels'],v['candidates'],v['units'])==(nq,c,n,nq*c)
                    assert v['ordered_id_sha256']==g['ordered_candidate_id_sha256']
                    saved=next(m for m in done['models'] if m['model']==name)
                    assert saved['canonical_score_gate_sha256']==sha(cache)
                    cache_state='PASS_CANONICAL_SCORE_MANIFEST_AND_IDS';cache_sha=sha(cache);rt=cg['provenance']['torch']
                    refsource='immutable canonical Torch2 score cache; full cache-byte rehash not repeated in this audit'
                if done:
                    assert raw[name]['raw_to_reproduction_id_sha256']==g['ordered_candidate_id_sha256']
                    sg=next(x for x in split_gate['model_gates'] if x['checkpoint_sha256']==cp)
                    assert sg['queries']==nq and sg['channels']==c and sg['units']==nq*c and sg['status']=='PASS'
                    if split=='test':assert sg['test_retmse_recall_agg_reproduced'] is True
                rows.append(dict(dataset=cell.rsplit('_',1)[0],horizon=int(cell.rsplit('_',1)[1]),cell=cell,
                                 model=arch,split=split,seed=0,queries=nq,channels=c,query_channel_units=nq*c,candidates=n,
                                 execution_status='COMPLETE' if done else 'RUNNING',
                                 deliverable_status='PASS' if done else 'NOT_COMPLETE',
                                 prediction_reproduction='PASS_ALL_QUERIES_MAXDIFF_0',
                                 aggregate_metric_reference='PASS_TEST_RETMSE_AGGMSE_RECALL_LEGACY_NDCG' if split=='test' else 'VAL_PREDICTIONS_PASS_METRICS_RECORDED_NOT_INDEPENDENTLY_COMPARED',
                                 independent_allchannel_ordered_id_pair=pair,
                                 canonical_score_cache=cache_state,
                                 raw_ids_vs_reproduction='PASS_FULL_SPLIT' if done else 'PENDING_MAIN_SPLIT_COMPLETION',
                                 checkpoint_sha256=cp,source_hash_count=len(hash_details),
                                 source_manifest_sha256=manifest_sha(prov['source_hashes']),source_hash_verification='ALL_MATCH_CURRENT_FILES',
                                 verification_runtime=rt,original_generation_runtime='NOT_INFERRED_FROM_VERIFICATION_RUNTIME',
                                 reference_source=refsource,gate_path=str(p/'CANONICAL_REPRODUCTION.json'),
                                 gate_sha256=sha(p/'CANONICAL_REPRODUCTION.json'),paired_gate_path=paired_path,
                                 score_cache_gate_sha256=cache_sha))
        cell_records.append(dict(cell=cell,execution='COMPLETE' if done else 'RUNNING',
                                 model_split_rows=4,source_hash_count=len(hash_details),
                                 checkpoint_hashes_verified=2,scope_missing_model=False,
                                 note='Inventory MODEL_STATUS metric_reproduction=NOT_RUN and CURRENT RUNNING are initial/progress snapshots; COMPLETE plus linked gates are terminal authority when present'))
        print('AUDITED',cell,'COMPLETE' if done else 'RUNNING',flush=True)
    with (out/'MODEL_SPLIT_GATE_SCOPE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(status='CPU_ARTIFACT_SCOPE_AUDIT_PASS_WITH_EXPLICIT_LIMITS',at=stamp,
                 cells=cell_records,model_split_rows=len(rows),completed_model_split_rows=sum(r['execution_status']=='COMPLETE' for r in rows),
                 independent_allchannel_pair_rows=sum(r['independent_allchannel_ordered_id_pair'].startswith('PASS') for r in rows),
                 raw_full_split_id_rows=sum(r['raw_ids_vs_reproduction'].startswith('PASS') for r in rows),
                 missing_model_split_rows=0,checkpoint_hashes_all_pass=True,source_hashes_all_pass=True,
                 source_hash_details=all_hash_details,
                 limits=['Original saved prediction equality is not a proof of identical candidate IDs.',
                         'Native canonical/cached canonical paths and an independent new all-channel-vs-canonical comparison are separate scopes.',
                         'No new GPU test and no rehash of all large original score-cache bytes.',
                         'Weather192 reproduction PASS does not mean its main diagnosis or supplements are complete.',
                         'No historical generation runtime is inferred from numerical verification runtime.'])
    (out/'SCOPE_AUDIT.json').write_text(json.dumps(summary,indent=2))
    report='''# 완료 셀 gate 범위 감사\n\n완료 5개 셀과 진행 중 Weather192 모두 V0/V5 × val/test 네 행이 존재한다.\n모델 누락을 셀 전체 PASS로 일반화한 추가 사례는 발견되지 않았다.\n24행 모두 전체 query/channel의 저장 예측 재현 최대 차이 0이며, test gate는 RetMSE/AggMSE/Recall/기존 NDCG를 원본과 비교한다.\nValidation은 전체 저장 예측을 비교하고 지표를 기록하지만, 별도 저장된 validation RetMSE 기준치와의 독립 대조로 표현하지 않는다.\n\n완료 5셀의 20행은 원시 selected_ids와 query_start를 CPU로 전수 읽어 표본 수·채널 수·범위·중복·순서를 확인했고, 재현 gate의 ID hash와 일치했다.\n체크포인트 12개 및 기록된 소스 hash는 모두 현재 파일과 일치했다.\n\n독립적인 새 all-channel 대 canonical ordered ID 전수 대조는 Weather96 V5 2행과 Weather192 V0/V5 4행에 명시적인 증거가 있다.\n나머지를 독립 all-channel gate PASS로 표시하지 않는다. ETTh1의 native canonical 경로와 V0의 canonical Torch2 score cache 경로는 별도 근거다.\nV0 cache manifest hash와 생성 ID hash는 검증했으나 대형 score cache 전체 바이트를 이번에 재해시하지 않았다.\n\nWeather192는 재현만 PASS이고 본 계산·보완은 진행 중이다. 원본 완료 파일과 결과는 변경하지 않았다.\n초기 MODEL_STATUS의 NOT_RUN과 마지막 CURRENT의 RUNNING은 오래된 snapshot이며, 완료 판정에는 COMPLETE와 연결된 산출물 gate를 사용한다.\n\n행별 정확한 모델·split·count·checkpoint·source hash·근거 파일은 MODEL_SPLIT_GATE_SCOPE.csv에 기록했다.\n'''
    (out/'REPORT_KO.md').write_text(report)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['cells','source_hash_details']},indent=2))
    print('OUT',out)


if __name__=='__main__':main()
