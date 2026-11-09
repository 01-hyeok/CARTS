"""Read-only evidence export for five completed Mechanism02 cells. No GPU."""
from pathlib import Path
import csv, datetime, hashlib, io, json, shutil, zipfile, math

ROOT=Path('/data/pjh_workspace/CARTS')
R=ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02'
CELLS=['ETTh1_96','ETTh1_192','ETTh1_336','ETTh1_720','Weather_96']
OUT=ROOT/'reports/mechanism02_completed_20261103_v2'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def savejson(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def savecsv(p,data):
    keys=list(dict.fromkeys(k for row in data for k in row))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
def table(data,keys):
    def fmt(x):
        if x in [None,'']:return '—'
        try:return '%.6f'%float(x) if '.' in str(x) or 'e-' in str(x) else str(x)
        except ValueError:return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']+['| '+' | '.join(fmt(row.get(k)) for k in keys)+' |' for row in data])

def main():
    assert not OUT.exists(), 'Do not overwrite a report snapshot'
    OUT.mkdir(parents=True)
    at=datetime.datetime.now(datetime.timezone.utc).isoformat();checks=[];inventory=[];master=[];sources=[];large=[]
    def copy(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(str(src),str(dst))
        sources.append(dict(source=str(src),snapshot=str(dst.relative_to(OUT)),sha256=sha(src),bytes=src.stat().st_size))
    for cell in CELLS:
        c=R/'cells'/cell;gate=read(c/'CELL_ARTIFACT_GATE.json');proof=read(c/'SOURCES.json')
        assert gate['status']=='CORE_AND_CPU_ANALYSES_PASS_STUDY_REVIEW_PENDING'
        assert read(c/'CPU_SUPPLEMENT_GATE.json')['status']=='PASS' and proof['smoke'] is False
        for name,digest in gate['files'].items():assert sha(c/name)==digest,(cell,name,'cell gate hash mismatch')
        for p in sorted(c.iterdir()):
            if p.is_file() and p.suffix in ['.csv','.json','.md','.png']:
                if p.stat().st_size>10*1024**2:
                    large.append(dict(cell=cell,source=str(p),bytes=p.stat().st_size,sha256=sha(p),reason='per-query ranking details; original remains on server; exact sampled-row aggregate included'))
                else:copy(p,OUT/'cells'/cell/p.name)
        groups={}
        with (c/'RANKING_QUALITY.csv').open(newline='') as f:
            for row in csv.DictReader(f):
                key=tuple(row[k] for k in ['model','training_support','checkpoint_sha256','split','variant','subset'])
                g=groups.setdefault(key,dict(n=0,queries=set(),channels=set(),spearman_sum=0.,spearman_n=0,kendall_sum=0.,kendall_n=0))
                g['n']+=1;g['queries'].add(row['query_id']);g['channels'].add(row['channel'])
                for col,prefix in [('spearman','spearman'),('kendall_tau_b','kendall')]:
                    v=float(row[col]) if row[col] else float('nan')
                    if math.isfinite(v):g[prefix+'_sum']+=v;g[prefix+'_n']+=1
        summary=[]
        for key,g in sorted(groups.items()):
            row=dict(zip(['model','training_support','checkpoint_sha256','split','variant','subset'],key))
            row.update(cell=cell,rows=g['n'],sampled_timestamps=len(g['queries']),channels=len(g['channels']),source_sha256=gate['files']['RANKING_QUALITY.csv'],sampling='fixed linspace max256/allchannels; not whole population')
            for prefix in ['spearman','kendall']:
                row[prefix+'_finite_count']=g[prefix+'_n'];row[prefix+'_mean']=g[prefix+'_sum']/g[prefix+'_n'] if g[prefix+'_n'] else ''
            summary.append(row)
        savecsv(OUT/'cells'/cell/'RANKING_SAMPLED_AGGREGATE_EXPORT.csv',summary)
        expected_c=7 if cell.startswith('ETTh1') else 21
        for name,item in proof['checkpoints'].items():
            d=R/'checkpoint_runs'/cell/name;core=read(d/'CORE_COMPLETE.json');repro=read(d/'REPRODUCTION_GATE.json')
            assert core['status']=='CORE_COMPLETE_NOT_ALL_PHASES' and repro['status']=='PASS'
            assert core['channels']==expected_c and core['query_counts']==proof['queries']
            assert sha(d/'REPRODUCTION_GATE.json')==item['reproduction_sha256']
            assert sha(Path(item['checkpoint']))==item['sha256']==core['identity']['checkpoint_sha256']
            assert {x['split'] for x in repro['rows']}=={'val','test'}
            for row in repro['rows']:
                assert row['status']=='PASS' and row['queries']==proof['queries'][row['split']] and row['channels']==expected_c
                assert row['ordered_mismatch_count']==0
            for f in ['CORE_COMPLETE.json','REPRODUCTION_GATE.json']:
                copy(d/f,OUT/'checkpoint_evidence'/cell/name/f)
            src=repro['source'];inventory.append(dict(cell=cell,model=name.split('_')[0],training_support=name.split('_')[1],
                checkpoint=item['checkpoint'],checkpoint_sha256=item['sha256'],selected_epoch=src.get('selected_epoch'),
                epochs_completed=src.get('epochs_completed'),selection=item['selection'],
                val_queries=proof['queries']['val'],test_queries=proof['queries']['test'],channels=expected_c,
                full_bank_count=core['identity']['data']['candidate_count'],reproduction_status='PASS',
                original_runtime=item['original_runtime'],new_retriever_training=False))
        reuse=read(c/'BASELINE_QUERY_REUSE_GATE.json');assert reuse['status']=='PASS'
        for sup in ['P100','FULL']:
            prior=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01'/sup/cell/'RETRIEVAL_BASELINE_COMPARISON.csv'
            entry=next(x for x in reuse['sources'] if x['support']==sup)
            assert sha(prior)==entry['files']['RETRIEVAL_BASELINE_COMPARISON.csv']
            prior_test=[x for x in rows(prior) if x['split']=='test']
            assert prior_test and all(int(x['queries'])==proof['queries']['test'] and int(x['channels'])==expected_c for x in prior_test)
            copy(prior,OUT/'prior_baselines'/sup/cell/prior.name)
        base=rows(c/'RETRIEVAL_BASELINE_COMPARISON.csv')
        assert len(base)==20
        for sup in ['P100','FULL']:
            names={x['method'] for x in base if x['support']==sup}
            assert {'V0','V1','V5','IndividualMSE_Oracle','Random','Last-repeat','Raw Cosine'}<=names
        for row in base:
            row=dict(row);row['queries']=str(proof['queries']['test']);row['channels']=str(expected_c)
            row['split']='test';row['seed']='0';row['units']=str(proof['queries']['test']*expected_c)
            row['origin']='Mechanism02 native checkpoint inference' if row['model'] else 'Attribution01 baseline reused with ID/mean gate; Last-repeat also independently rechecked'
            row['source_cell_csv_sha256']=sha(c/'RETRIEVAL_BASELINE_COMPARISON.csv');master.append(row)
        checks.append(dict(cell=cell,status=gate['status'],cell_gate_sha256=sha(c/'CELL_ARTIFACT_GATE.json'),
            cell_files_fresh_hash_pass=len(gate['files']),checkpoint_fresh_hash_pass=6,
            validation_query_count=proof['queries']['val'],test_query_count=proof['queries']['test'],channels=expected_c,
            raw_prediction_archives='Referenced by stored core manifests; not freshly re-read for this report'))
    for f in ['PROTOCOL.json','CHECKPOINT_AUDIT.csv','P100_COMPARABILITY.json','FULL_V1_ORIGINAL_SELECTION.json','FROZEN_PROBE_REUSE_GATE.json','STUDY_COVERAGE.json','CONTROLLER_STATUS.json']:
        copy(R/'00_Audit'/f,OUT/'audit'/f)
    savecsv(OUT/'UNIFIED_NATIVE_TEST_COMPARISON.csv',master);savecsv(OUT/'CHECKPOINT_INVENTORY_COMPLETED.csv',inventory)
    savejson(OUT/'EXPORT_VALIDATION.json',dict(at=at,status='PASS_FIVE_CELL_EVIDENCE_EXPORT',cells=checks,
        incomplete_cells=['Weather_192','Weather_336','Weather_720'],whole_study_complete=False))
    savejson(OUT/'SOURCE_MANIFEST.json',sources)
    savejson(OUT/'LARGE_FILES_RETAINED_ON_SERVER.json',large)
    doc=['# Mechanism02 완료 5조건 분석용 자료',f'\n추출 확인: {at}. ETTh1 96/192/336/720 및 Weather96만 포함합니다. Weather192/336/720은 종합 미완료이므로 결과를 채우지 않았습니다.',
    '\n각 셀은 `CORE_AND_CPU_ANALYSES_PASS_STUDY_REVIEW_PENDING`입니다. 전체 연구 완료 또는 최종 GO가 아닙니다. 완료 30개 checkpoint의 validation/test 재현 gate, 실제 checkpoint SHA256 및 셀 산출물 hash를 재확인했습니다. 대용량 query 예측 배열은 이번 보고 추출에서 다시 읽지 않았으며 기존 검증 manifest를 근거로 합니다.',
    '\n## 지표·지원 후보군·출처',
    '\n모든 수치는 Stage1 retrieval입니다. Stage2 forecasting 수치가 아닙니다. `training_support`는 가중치를 학습한 후보군, `support`/`variant`는 평가 후보군입니다. Native 표의 P100/FULL 모델은 서로 다른 checkpoint이므로 후보군만 바꾼 인과 대조가 아닙니다. 동일 checkpoint 개입은 CROSS_SUPPORT 표에서 SHA256을 고정해 봅니다.',
    '\nTrain에서만 fit한 StandardScaler 공간이며 역변환하지 않습니다. 후보 미래는 `v_i(t)=candidate_future_i(t)-candidate_history_last_i+query_history_last`로 정렬합니다. L=H입니다. 먼저 horizon 평균을 낸 뒤 모든 query×channel에 동일 가중치를 줍니다. 누적합/분모는 float64입니다.',
    '\n- `RetMSE@10 = mean_(q,c) mean_(i in Top10) mean_t (v_i(t)-y_qc(t))²`.',
    '- `AggMSE@10 = mean_(q,c,t) (mean_(i in Top10) v_i(t)-y_qc(t))²`; AggMAE는 같은 평균 예측의 절대오차입니다.',
    '- `Recall@10 = mean_(q,c) |S10 ∩ O10_same_support|/10`. 후보군이 다르면 Oracle 목표 집합도 다르므로 Recall 원시값의 단순 우열 해석을 피합니다.',
    '- Oracle은 같은 support 안에서 개별 미래 MSE가 가장 작은 10개를 선택합니다. RetMSE의 최적 기준이며 AggMSE/MAE 최적 집합이라는 뜻은 아닙니다. Last-repeat는 마지막 관측을 반복하므로 RetMSE/Recall이 정의되지 않습니다.',
    '- Pool regret=R_P100_oracle−R_Full_oracle, Ranking regret=R_model_P100−R_P100_oracle, Total=두 항의 합. 비중은 집계 regret의 비율입니다. Full에서 선택한 모델에 R_P100_oracle≤R_model을 강요하지 않습니다.',
    '- Exchange regret=(false selected의 MSE 합−missed Oracle의 MSE 합)/10이며 동일 support Ranking regret과 같습니다.',
    '- Raw Cosine은 정규화된 과거 입력의 직접 cosine입니다. Delta-last Cosine·Pearson은 서로 별도 방법입니다. Random/Oracle/Raw 등 baseline은 기존 Attribution01의 검증된 동일 population 결과를 명시적으로 재사용했습니다. 새 계산으로 가장하지 않습니다.',
    '- 모델 checkpoint는 원래 validation 선택을 고정했으며 test로 재선택하지 않았습니다. 모델별 실제 선택 규칙/epoch는 아래 inventory 원문에 있습니다. Full V1 일부는 early stopping이므로 V1/V5를 순수 head 수 인과효과로 단정하지 않습니다.',
    '\n## 표 읽는 법과 한계',
    '\n표는 소수 6자리, 아래 CSV 원문은 원시 정밀도입니다. 개선율은 `(비교 기준−대상)/비교 기준×100`이며 RetMSE/AggMSE/MAE의 최선이 다를 수 있습니다. 원래 checkpoint 생성 런타임은 UNKNOWN이며 현재 전체 ordered-ID/예측 재현 PASS와 구분합니다.',
    '\nCore는 전체 query×channel입니다. 고비용 global correlation은 사전 고정 최대256시점×전체채널이며, 기존 Ridge probe도 train256시점×전체채널×100 future-free 후보로 학습한 모델을 검증 후 재사용했습니다. seed0만이므로 seed 간 불확실성은 미측정입니다. Bootstrap은 전체채널을 묶은 시간블록이며 H/2H/4H 사전 고정, 주 블록 유효 개수4 미만은 유의성 근거가 아닙니다. 다중비교는 탐색적입니다.',
    '\nGeometry는 null/no-cluster를 허용하고 validation 선택을 고정했습니다. 별도 disjoint future-MSE-matched control은 미구현이므로 multimodality 인과 결론은 INCONCLUSIVE입니다. Weather −9999/scaler 영향은 공식 전체채널 결과에 남아 있고 채널 제외 진단은 별도입니다.',
    '\n대용량 query별 ranking 원본 CSV 두 alias는 서버에 보존하고 hash와 경로를 기록했습니다. 첨부에는 그 원본의 동일 가중 평균·유효 개수·표본 시점/채널 수를 다시 계산한 RANKING_SAMPLED_AGGREGATE_EXPORT.csv를 제공합니다. 이 파일은 새 inference 결과가 아닌 보고용 집계입니다.',
    '\n## 확인된 coverage',table(checks,['cell','validation_query_count','test_query_count','channels','cell_files_fresh_hash_pass','checkpoint_fresh_hash_pass'])]
    for cell in CELLS:
        c=OUT/'cells'/cell;doc+=['\n## '+cell]
        for sup in ['P100','FULL']:
            data=[x for x in master if x['cell']==cell and x['support']==sup]
            order=['Last-repeat','Random','Raw Cosine','Delta-last Cosine','Pearson','PAA-16 Cosine','V0','V1','V5','IndividualMSE_Oracle']
            data.sort(key=lambda x:order.index(x['method']))
            doc+=['\n### Native '+sup+' — 전체 test',table(data,['method','retmse','agg_mse','agg_mae','recall10'])]
        cross=rows(c/'CROSS_SUPPORT.csv');doc+=['\n### 동일 checkpoint의 후보군 개입',
          'P100_CONTROLLED는 같은 Full logits를 P100 ID로 제한한 비교이고 P100_NATIVE는 기존 native 연산 재현입니다. 분모 대조는 같은 logits의 LOCAL_NORMALIZER/FULL_NORMALIZER끼리 비교합니다. 서로 바꾸어 쓰지 않습니다.',
          table([x for x in cross if x['variant'] in ['FULL','P100_CONTROLLED','P100_NATIVE']],['model','training_support','variant','retmse','agg_mse','agg_mae','recall10']),
          '\n### P100 native의 regret 분해',table([x for x in rows(c/'REGRET_DECOMPOSITION.csv') if x['variant']=='P100_NATIVE'],['model','pool','ranking','total','pool_share','ranking_share','full_oracle_coverage100']),
          '\n### V5 내부 head 결합 (validation 선택 single)',table([x for x in rows(c/'HEAD_UTILITY.csv') if x['method'] in ['All5','ValidationBestSingle'] and (x['variant']=='FULL' and x['training_support']=='FULL' or x['variant']=='P100_NATIVE')],['training_support','variant','method','validation_selected_head','retmse','agg_mse','recall10']),
          '\nGeometry·normalizer·missed/exchange·probe·channel/strata 원문은 다음 부록에 보존했습니다. 전체 결론은 남은 Weather 3조건 완료 후 판단합니다.']
    doc+=['\n## 체크포인트 provenance 원문','```csv',(OUT/'CHECKPOINT_INVENTORY_COMPLETED.csv').read_text().rstrip(),'```','\n## 사전 규약 원문','```json',(OUT/'audit/PROTOCOL.json').read_text().rstrip(),'```']
    # Include each distinct CSV once, preserving every original header/value.
    seen={}
    for cell in CELLS:
        doc+=['\n## '+cell+' 원시 CSV 및 gate 부록']
        for p in sorted((OUT/'cells'/cell).glob('*.csv')):
            digest=sha(p)
            doc+=['\n### '+cell+'/'+p.name, 'SHA256: `'+digest+'`']
            if digest in seen:doc+=['같은 내용: '+seen[digest]+' (파일명 alias; 원본 hash 동일).']
            else:
                seen[digest]=cell+'/'+p.name;doc+=['```csv',p.read_text().rstrip(),'```']
        for f in ['CELL_ARTIFACT_GATE.json','SOURCES.json','BASELINE_QUERY_REUSE_GATE.json','LAST_REPEAT_RECHECK.json']:
            p=OUT/'cells'/cell/f;doc+=['\n### '+cell+'/'+f,'```json',p.read_text().rstrip(),'```']
    doc+=['\n## 추출 검증','```json',(OUT/'EXPORT_VALIDATION.json').read_text().rstrip(),'```',
          '\n## 서버에 보존한 대용량 파일','```json',(OUT/'LARGE_FILES_RETAINED_ON_SERVER.json').read_text().rstrip(),'```',
          '\n## 원본 경로·파일 hash manifest','```json',(OUT/'SOURCE_MANIFEST.json').read_text().rstrip(),'```']
    report=OUT/'MECHANISM02_COMPLETED_5CELLS_KO.md';report.write_text('\n\n'.join(doc)+'\n')
    manifest={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
    savejson(OUT/'DELIVERABLE_SHA256.json',manifest)
    z=OUT.with_suffix('.zip')
    with zipfile.ZipFile(z,'x',compression=zipfile.ZIP_DEFLATED) as out:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():out.write(str(p),str(p.relative_to(OUT)))
    print(json.dumps(dict(report=str(report),report_bytes=report.stat().st_size,report_sha256=sha(report),zip=str(z),zip_bytes=z.stat().st_size,zip_sha256=sha(z),cells=checks),ensure_ascii=False))
if __name__=='__main__':main()
