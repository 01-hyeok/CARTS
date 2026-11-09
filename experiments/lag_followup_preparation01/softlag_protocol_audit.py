"""Read-only training evidence audit plus new required export/clarification files."""
import csv,datetime,hashlib,json,pickle
from pathlib import Path
import numpy as np
import torch
BASE=Path('/data/pjh_workspace/CARTS/results/TRACK-SOFTLAG-HEAD-PILOT01')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv_new(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row));text=[]
    import io
    s=io.StringIO();w=csv.DictWriter(s,fieldnames=fields);w.writeheader();w.writerows(rows);value=s.getvalue()
    if path.exists():assert path.read_text()==value,'Existing export preserved; differs'
    else:path.write_text(value)
def main():
    original=json.loads((BASE/'FINAL_GATE.json').read_text());assert original['status']=='PASS_PILOT_COMPLETE'
    for name,h in original['files'].items():assert sha(BASE/name)==h
    validations=[];preferences=[];arms=[];resources=[];comparison=[]
    for cell,support in [('ETTh1_96','FULL'),('ETTh1_336','FULL'),('ETTh1_96','P100'),('ETTh1_336','P100')]:
        out=BASE/(cell+'_'+support);proof=json.loads((out/'PROVENANCE.json').read_text());final=json.loads((out/'FINAL_GATE.json').read_text())
        assert final['selected_test_ids_and_metrics_exact_replay'] and final['samples']=={'train':64,'val':32,'test':32}
        cp=proof['identity']['checkpoint'];assert sha(Path(cp['checkpoint']))==cp['checkpoint_sha256']
        per={};selected=[]
        for arm in ['B1','B2']:
            states=[]
            for e in range(6):
                p=out/arm/('epoch%02d.pth'%e);z=torch.load(p,map_location='cpu',weights_only=False);assert z['epoch']==e and z['identity']==proof
                assert len(z['history'])==e+1
                if e==0:assert float(z['alpha'])==0 and not bool(z['bias'].any())
                else:
                    assert all(int(v['step'])==e*8 for v in z['optimizer']['state'].values())
                    assert z['first_bias_grad']==0 and z['later_bias_grad_max']>0
                states.append(hashlib.sha256(pickle.dumps(z['batch_rng'])).hexdigest())
            per[arm]=states
            choice=json.loads((out/arm/'SELECTION.json').read_text());hist=json.loads((out/arm/'VALIDATION_HISTORY.json').read_text())
            assert min(hist,key=lambda r:(r['val_retmse'],r['epoch']))==choice['selected'] and not choice['test_used']
            assert sha(Path(choice['checkpoint']))==choice['sha256'];selected.append(choice['selected'])
            arms.append(dict(cell=cell,support=support,arm=arm,trained_epochs=5,stored_epochs='0,1,2,3,4,5',optimizer_steps=40,
                selected_epoch=choice['selected']['epoch'],checkpoint_sha256=choice['sha256'],initial_alpha_bias_zero=True,
                bias_gradient_zero_at_first_step_then_nonzero=True,validation_only=True,original_checkpoint_unchanged=True,
                train_timestamps=64,validation_timestamps=32,test_timestamps=32,channels=7))
        assert per['B1']==per['B2'],'B1/B2 batch RNG order differs'
        for filename,target in [('VALIDATION_RESULTS.csv',validations),('LEARNED_LAG_PREFERENCES.csv',preferences)]:
            with (out/filename).open() as f:target.extend([dict(cell=cell,support=support,**r) for r in csv.DictReader(f)])
        raw={}
        for method in ['B0','B1','B2']:
            with np.load(out/(method+'_test_queries.npz')) as z:raw[method]=z['metrics'].copy()
        for a,b in [('B1','B0'),('B2','B0'),('B2','B1')]:
            da=raw[a][...,0];db=raw[b][...,0]
            comparison.append(dict(cell=cell,support=support,comparison=a+'-'+b,retmse_A=float(da.mean()),retmse_B=float(db.mean()),
                 relative_improvement_percent=float(100*(1-da.mean()/db.mean())),query_channel_win_rate=float((da<db).mean()),
                 timestamp_win_rate=float((da.mean(1)<db.mean(1)).mean()),query_channel_tie_rate=float((da==db).mean()),
                 query_channel_mse_delta=float((da-db).mean())))
        # These are explicit filesystem interval proxies, not measured GPU times.
        tv=max(p.stat().st_mtime for s in ['train','val'] for p in (out/'cache'/s).glob('*.json'))
        te=max(p.stat().st_mtime for p in (out/'cache/test').glob('*.json'))
        se=max((out/a/'SELECTION.json').stat().st_mtime for a in ['B1','B2'])
        begin=(out/'PROVENANCE.json').stat().st_mtime
        end=datetime.datetime.fromisoformat(json.loads((out/'COMPONENT_GATE.json').read_text())['at']).timestamp()
        r=json.loads((out/'RESOURCE_SUMMARY.json').read_text())
        resources.append(dict(cell=cell,support=support,total_wall_seconds_measured=r['wall_seconds'],
            train_val_cache_interval_proxy_seconds=tv-begin,test_cache_interval_proxy_seconds=te-se,
            B1_train_plus_validation_interval_proxy_seconds=(out/'B1/epoch05.pth').stat().st_mtime-(out/'B1/epoch00.pth').stat().st_mtime,
            B2_train_plus_validation_interval_proxy_seconds=(out/'B2/epoch05.pth').stat().st_mtime-(out/'B2/epoch00.pth').stat().st_mtime,
            final_test_and_save_interval_proxy_seconds=end-te,
            separate_training_kernel_time='UNMEASURED',separate_validation_wall_time='UNMEASURED',
            interval_definition='filesystem/JSON interval proxies; validation interleaved with train; neither GPU time nor isolated benchmark',
            peak_allocated_bytes=r['peak_allocated_bytes'],peak_reserved_bytes=r['peak_reserved_bytes']))
    csv_new(BASE/'VALIDATION_RESULTS.csv',validations);csv_new(BASE/'LEARNED_LAG_PREFERENCES.csv',preferences)
    csv_new(BASE/'PROTOCOL_AUDIT.csv',arms);csv_new(BASE/'PAIRED_COMPARISONS.csv',comparison);csv_new(BASE/'RESOURCE_PHASE_AUDIT.csv',resources)
    audit=['# SoftLag 프로토콜 교차 감사','',
      '첨부 libfile_f81cf8841d088191801987442e49b22d의 4조건과 실행 결과를 대조했다. 기존 V5 encoder·projection을 고정하고 B1/B2 bias와 공통 alpha만 5epoch, 각40optimizer step을 실행했다. 모든 epoch0–5가 있고 초기 alpha/bias=0, 첫 bias gradient=0 및 이후 nonzero를 확인했다.',
      '', '64/32/32 고정 timestamp와 전체7채널, Full 전체 Bank와 P100100개를 유지했다. B1/B2 각 epoch의 배치 RNG 상태 hash가 일치한다. validation RetMSE 최소(epoch0 허용) 선택과 checkpoint SHA, 선택 test의 ordered ID·지표 정확 재생 gate를 검증했다. 기존 checkpoint SHA는 그대로다.',
      '', '측정 한계: 원래 RESOURCE_SUMMARY의 총 wall time과 CUDA peak는 실측이다. 학습과 validation이 교차 실행되어 별도 kernel time/validation wall time은 기록하지 않았다. RESOURCE_PHASE_AUDIT의 세부 시간은 filesystem/JSON 구간 근사치로 명시했으며 순수 GPU 시간으로 표현하지 않는다.',
      '', '현재 Head Top10 overlap은 조건/arm별 전체 pair 평균을 저장했다. 개별 10개 pair matrix는 아직 별도 저장하지 않았다. head별 고유 유용 후보/성능은 HEAD_DIAGNOSTICS 및 조건별 HEAD_USEFULNESS에 있다. 따라서 세부 pair matrix까지 모두 마쳤다고 주장하지 않는다.',
      '', '고정 score와 raw-lag cache를 B1/B2에서 공유했다. 정규화는 전체 유효 candidate 축, teacher는 기존 negative z-score future-MSE, loss는 KL(teacher||mean mixture)이다. Lag 입력은 과거뿐이며 미래는 teacher/평가만 사용했다.',
      '', '추가 학습, 기존 checkpoint 변경, Solar 재개, 외부 업로드 또는 Git push는 하지 않았다.']
    p=BASE/'IMPLEMENTATION_AUDIT.md'
    if not p.exists():p.write_text('\n'.join(audit),encoding='utf8')
    report=(BASE/'FINAL_REPORT_KO.md').read_text(encoding='utf8')+'\n\n'+'\n'.join(audit)
    target=BASE/'FINAL_REPORT.md'
    if not target.exists():target.write_text(report,encoding='utf8')
    payload=dict(status='PASS_TRAIN_SELECTION_TEST_WITH_EXPLICIT_MEASUREMENT_LIMITS',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        original_final_gate_sha256=sha(BASE/'FINAL_GATE.json'),trained_conditions=4,trainable_arm_runs=8,all_epoch_checkpoints=48,
        equal_batch_rng_hashes=True,optimizer_steps_per_arm=40,remaining=['per-head-pair overlap matrix supplementary export'],
        unmeasured=['separate GPU training kernel time','separate validation wall time'],
        files={p.name:sha(p) for p in [BASE/'PROTOCOL_AUDIT.csv',BASE/'PAIRED_COMPARISONS.csv',BASE/'RESOURCE_PHASE_AUDIT.csv',BASE/'VALIDATION_RESULTS.csv',BASE/'LEARNED_LAG_PREFERENCES.csv',BASE/'IMPLEMENTATION_AUDIT.md',BASE/'FINAL_REPORT.md']})
    dest=BASE/'PROTOCOL_AUDIT_ADDENDUM.json'
    if not dest.exists():dest.write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload),flush=True)
if __name__=='__main__':main()
