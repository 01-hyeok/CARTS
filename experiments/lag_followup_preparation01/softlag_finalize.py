"""Exact cached-score replay and missing head diagnostics; no extra training."""
import argparse,csv,datetime,fcntl,json,os,time,traceback
from pathlib import Path
import numpy as np
from server_bridge import *
from softlag_run import batch,probability,evaluation,metrics,write_csv
from softlag import torch_parameters
from past_lag import indices

BASE=ROOT/'results/TRACK-SOFTLAG-HEAD-PILOT01'

def load_rows(out,split):
    result=[]
    for p in sorted((out/'cache'/split).glob('*.npz')):
        m=json.loads(p.with_suffix('.json').read_text());assert sha(p)==m['sha256']
        with np.load(p) as z:result.append({k:z[k].copy() for k in z.files})
    assert len(result)==32
    return result

@torch.no_grad()
def head_detail(data,rows,alpha,bias,support):
    values=[];unique=[];counts=[]
    for a in range(0,len(rows),8):
        b=batch(rows,range(a,min(a+8,len(rows))));u,p=probability(b,alpha,bias,support)
        ids=topk(u.masked_fill(~b['mask'][:,:,None],-float('inf')),10)
        oracle=topk(-b['distance'].masked_fill(~b['mask'],float('inf')),10)
        boundary=b['distance'].gather(-1,oracle)[...,-1]
        hm=[];hu=[]
        for h in range(5):
            mm,_=metrics(data,b,u[:,:,h]);hm.append(mm)
            others=torch.cat([ids[:,:,j] for j in range(5) if j!=h],-1)
            exclusive=~(ids[:,:,h,:,None]==others[:,:,None]).any(-1)
            dh=b['distance'].gather(-1,ids[:,:,h])
            exact=(ids[:,:,h,:,None]==oracle[:,:,None]).any(-1)
            fields=[exclusive.sum(-1).double(),(exclusive&exact).sum(-1).double()]
            for eps in [.01,.05,.10]:fields.append((exclusive&(dh<=boundary[...,None]+(eps*boundary).clamp_min(1e-8)[...,None])).sum(-1).double())
            hu.append(torch.stack(fields,-1).cpu().numpy())
        values.append(np.stack(hm,2));unique.append(np.stack(hu,2))
    return np.concatenate(values),np.concatenate(unique)

def bootstrap(values,starts,length):
    # Paired timestamps, all seven channels together; bins fixed in time units.
    v=values.mean(1);labels=(starts-starts.min())//length;groups=[v[labels==k] for k in np.unique(labels)]
    rng=np.random.RandomState(0);est=[]
    for _ in range(2000):
        picked=rng.randint(0,len(groups),len(groups));est.append(np.concatenate([groups[i] for i in picked]).mean())
    return dict(mean=float(v.mean()),ci_low=float(np.quantile(est,.025)),ci_high=float(np.quantile(est,.975)),blocks=len(groups),sufficient=len(groups)>=4,queries=len(v),block_length=length,repeats=2000,confidence=.95)

def main(support_filter):
    lock=(BASE/'00_Audit/finalize.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if support_filter!='REPORT':before=start_gpu()
    allresults=[];allheads=[];allbias=[];cis=[];resources=[];gates=[]
    for cell,support in [('ETTh1_96','FULL'),('ETTh1_336','FULL'),('ETTh1_96','P100'),('ETTh1_336','P100')]:
        out=BASE/(cell+'_'+support);gate=json.loads((out/'COMPONENT_GATE.json').read_text())
        assert gate['status']=='PILOT_TRAIN_TEST_COMPLETE_HEAD_DETAIL_REVIEW_PENDING'
        if support_filter=='REPORT':
            complete=json.loads((out/'FINAL_GATE.json').read_text());assert complete['status']=='PASS_PILOT_CELL_COMPLETE'
            for name,h in complete['files'].items():assert sha(out/name)==h
            for filename,target in [('HEAD_USEFULNESS.csv',allheads),('BIAS_SIMILARITY.csv',allbias),('PAIRED_TIME_BLOCK_CI.csv',cis)]:
                with (out/filename).open() as stream:target.extend(list(csv.DictReader(stream)))
            with (out/'TEST_RESULTS.csv').open() as stream:allresults.extend([dict(cell=cell,support=support,**row) for row in csv.DictReader(stream)])
            resource=json.loads((out/'RESOURCE_SUMMARY.json').read_text());resources.append(dict(cell=cell,support=support,wall_seconds=resource['wall_seconds'],peak_allocated_bytes=resource['peak_allocated_bytes'],peak_reserved_bytes=resource['peak_reserved_bytes'],scope='cache+fixed 5epochs each B1/B2+validation+selected test; may share GPU with D4'))
            gates.append(complete);continue
        if support!=support_filter:continue
        data,f,identity=frozen(cell,support);proof=json.loads((out/'PROVENANCE.json').read_text())
        val=load_rows(out,'val');test=load_rows(out,'test');raw={};_,lags=indices(data.h);caseheads=[];casebias=[]
        for arm in ['B0','B1','B2']:
            alpha,bias=torch_parameters(arm!='B1',len(lags),device=data.device,dtype=torch.float32)
            if arm!='B0':
                hist=json.loads((out/arm/'VALIDATION_HISTORY.json').read_text());assert [x['epoch'] for x in hist]==list(range(6))
                chosen=json.loads((out/arm/'SELECTION.json').read_text());assert not chosen['test_used']
                expected=min(hist,key=lambda z:(z['val_retmse'],z['epoch']));assert expected==chosen['selected']
                assert sha(chosen['checkpoint'])==chosen['sha256']
                ck=torch.load(chosen['checkpoint'],map_location=data.device);assert ck['identity']==proof
                alpha.data.copy_(ck['alpha']);bias.data.copy_(ck['bias'])
                for ep in range(6):assert (out/arm/('epoch%02d.pth'%ep)).exists()
            m,ids,_=evaluation(data,test,alpha,bias,support)
            with np.load(out/(arm+'_test_queries.npz')) as z:
                assert np.array_equal(ids,z['selected_ids']),('selected test replay mismatch',cell,support,arm)
                np.testing.assert_allclose(m,z['metrics'],rtol=0,atol=0)
                starts=z['query_starts'].copy()
            raw[arm]=m
            vh,_=head_detail(data,val,alpha,bias,support);th,un=head_detail(data,test,alpha,bias,support)
            best=int(vh[...,0].mean((0,1)).argmin())
            for h in range(5):
                avg=th[:,:,h].mean((0,1));uu=un[:,:,h].mean((0,1))
                caseheads.append(dict(cell=cell,support=support,arm=arm,head=h+1,validation_selected=(h==best),validation_head_retmse=float(vh[:,:,h,0].mean()),
                    test_retmse=float(avg[0]),test_recall=float(avg[3]),exclusive_count=float(uu[0]),exclusive_oracle_count=float(uu[1]),exclusive_useful_1pct=float(uu[2]),exclusive_useful_5pct=float(uu[3]),exclusive_useful_10pct=float(uu[4]),
                    unique_useful_rate_denominator10=float(uu[3]/10),queries=32,channels=7))
            bb=bias.detach().cpu().numpy()
            for h in range(len(bb)):
                for k in range(h+1,len(bb)):
                    den=float(np.linalg.norm(bb[h])*np.linalg.norm(bb[k]))
                    casebias.append(dict(cell=cell,support=support,arm=arm,head_a=h+1,head_b=k+1,cosine=float(np.dot(bb[h],bb[k])/den) if den>0 else '',l2_distance=float(np.linalg.norm(bb[h]-bb[k])),zero_norm=den==0))
        for a,b in [('B1','B0'),('B2','B0'),('B2','B1')]:
            for scale in [1,2,4]:cis.append(dict(cell=cell,support=support,comparison=a+'-'+b,metric='RetMSE',**bootstrap(raw[a][...,0]-raw[b][...,0],starts,data.h*scale)))
        write_csv(out/'PAIRED_TIME_BLOCK_CI.csv',cis[-9:])
        write_csv(out/'HEAD_USEFULNESS.csv',caseheads);write_csv(out/'BIAS_SIMILARITY.csv',casebias)
        allheads.extend(caseheads);allbias.extend(casebias)
        with (out/'TEST_RESULTS.csv').open() as stream:
            allresults.extend([dict(cell=cell,support=support,**row) for row in csv.DictReader(stream)])
        resource=json.loads((out/'RESOURCE_SUMMARY.json').read_text());resources.append(dict(cell=cell,support=support,wall_seconds=resource['wall_seconds'],peak_allocated_bytes=resource['peak_allocated_bytes'],peak_reserved_bytes=resource['peak_reserved_bytes'],scope='cache+fixed 5epochs each B1/B2+validation+selected test; may share GPU with D4'))
        assert sha(f.row['checkpoint'])==proof['identity']['checkpoint']['checkpoint_sha256']
        files={p.name:sha(p) for p in out.glob('*.csv')};files.update({p.name:sha(p) for p in out.glob('*test_queries.npz')})
        complete=dict(status='PASS_PILOT_CELL_COMPLETE',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),cell=cell,support=support,seed=0,all_channels=7,
                      samples={'train':64,'val':32,'test':32},epochs_saved=list(range(6)),b1_b2_same_budget=True,source_checkpoint_unchanged=True,
                      selected_test_ids_and_metrics_exact_replay=True,files=files,provenance_sha256=sha(out/'PROVENANCE.json'))
        dump(out/'FINAL_GATE.json',complete);gates.append(complete)
        print('SOFTLAG FINAL CELL PASS',cell,support,flush=True)
    if support_filter!='REPORT':return
    write_csv(BASE/'TEST_RESULTS.csv',allresults);write_csv(BASE/'HEAD_DIAGNOSTICS.csv',allheads);write_csv(BASE/'BIAS_SIMILARITY.csv',allbias);write_csv(BASE/'PAIRED_TIME_BLOCK_CI.csv',cis);write_csv(BASE/'RESOURCE_SUMMARY.csv',resources)
    lines=['# SoftLag 최소 파일럿 최종 보고서','',
     '4개 조건 모두 검증 완료. 기존 V5 encoder와 query heads는 동결했고, seed0의 B1/B2 lag bias와 공통 alpha만 각각 5epoch 학습했다. 모든 epoch0–5를 보존하고 validation RetMSE 최소 epoch를 선택했다. Test는 선택 이후에만 평가했다.',
     '', '평가 범위는 각 조건 train64/validation32/test32개의 사전 고정 timestamp × 전체7채널이다. Full은 전체 candidate bank, P100은 기존 동일 후보100개를 유지했다. 전체 데이터셋 평가 또는 새 encoder 학습 결과로 해석하지 않는다. Solar는 재개하지 않았다.',
     '', '| 조건 | B0 RetMSE | B1 RetMSE | B2 RetMSE | B1 vs B0 개선% | B2 vs B1 개선% |', '|---|---:|---:|---:|---:|---:|']
    for cell,support in [('ETTh1_96','FULL'),('ETTh1_336','FULL'),('ETTh1_96','P100'),('ETTh1_336','P100')]:
        vals={x['method']:float(x['retmse']) for x in allresults if x['cell']==cell and x['support']==support and x['method'] in ['B0','B1','B2']}
        lines.append('| %s %s | %.12f | %.12f | %.12f | %.6f | %.6f |'%(cell,support,vals['B0'],vals['B1'],vals['B2'],100*(1-vals['B1']/vals['B0']),100*(1-vals['B2']/vals['B1'])))
    lines.extend(['','전체 판정: **INCONCLUSIVE**. 일관된 B2의 추가 이득은 관찰되지 않았다. H96 Full의 작은 RetMSE 개선은 AggMSE 개선을 동반하지 않았으며, H336은 RetMSE가 악화됐다. H96 P100은 validation이 epoch0을 선택하여 원래 모델과 동일하다. 표본·seed·parameter 수 차이 때문에 head별 lag 특화의 인과적 효과를 주장하지 않는다.',
      '', 'RetMSE는 선택한 후보 각각의 delta-last 복원 미래 MSE 평균이다. AggMSE/MAE는 선택10개 미래를 균등 평균한 예측의 오차다. Recall은 동일 support의 Individual-MSE Oracle10과의 겹침이다. Random은 seed0 고정 query별 무작위 unique Top10, Oracle은 진단 전용이며 LastRepeat와 함께 TEST_RESULTS.csv에 포함한다.',
      '', '과거 lag는 query[t]와 candidate[t+delta]를 비교한다. 모든 lag에 같은 query patch를 사용하며 미래·circular shift·padding을 쓰지 않는다. 실제 PAMR 구현을 발견하지 못해 사전 고정 raw patch cosine 정의를 사용했다. 원래 z-score negative future-MSE teacher와 mean-mixture KL을 그대로 호출했다.',
      '', 'alpha=0 점수/ordered Top10 재현, 두 step의 gradient, 전체 채널 batch, 마스크와 checkpoint 불변성을 검증했다. 후속 GPU replay에서 저장된 test 후보 ID와 지표가 정확히 일치했다. Head useful 기준은 query Oracle10 경계 + max(1e-8, 경계MSE×1/5/10%)이며 각각 보고했다. Bias cosine에서 zero norm은 미정의로 표시한다.',
      '', 'PAIRED_TIME_BLOCK_CI.csv는 H/2H/4H 시간 블록, 모든 채널 공동 resampling, seed0 2000회 percentile95%이다. 4개 미만 블록은 불충분으로 표기한다. 이는 여러 학습 seed의 불확실성이 아니며 탐색적 분석이다.',
      '', 'RESOURCE_SUMMARY.csv는 cache·학습·validation·선택 test의 wall time이며 GPU kernel time이 아니다. 병렬 D4 경합 가능성이 있어 단독 실행 벤치마크로 해석하지 않는다. 원래 runtime provenance UNKNOWN은 유지하며 현재 재현 PASS와 구분한다.',
      '', '상세 산출물: TEST_RESULTS.csv, 각 조건 VALIDATION_RESULTS.csv 및 선택 JSON, HEAD_DIAGNOSTICS.csv, BIAS_SIMILARITY.csv, 각 조건 LEARNED_LAG_PREFERENCES.csv, PAIRED_TIME_BLOCK_CI.csv, RESOURCE_SUMMARY.csv, 각 조건 FINAL_GATE.json.'])
    (BASE/'FINAL_REPORT_KO.md').write_text('\n'.join(lines),encoding='utf8')
    dump(BASE/'FINAL_GATE.json',dict(status='PASS_PILOT_COMPLETE',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),cells=4,verdict='INCONCLUSIVE',
        files={p.name:sha(p) for p in list(BASE.glob('*.csv'))+[BASE/'FINAL_REPORT_KO.md']},cell_gates=gates,original_models_unchanged=True,solar_resumed=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--support',choices=['FULL','P100','REPORT'],required=True);args=p.parse_args()
    try:main(args.support)
    except BaseException as e:
        dump(BASE/'00_Audit/FINALIZE_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
