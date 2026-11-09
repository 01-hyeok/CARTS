"""Durable sequential SoftLag expansion; one worker alongside diagnostic track."""
import sys,json,os,fcntl,subprocess,time,traceback,csv
from pathlib import Path
OWN=Path(__file__).resolve().parent
sys.path.insert(0,str(OWN));from contracts import *
PY='/data/pjh_workspace/ts-env/bin/python'
def status(s,**kw):dump(OUT/'CONTROLLER_STATUS.json',dict(status=s,at=now(),pid=os.getpid(),**kw))
def report(done,failed):
    rows=[]
    for c,s in REUSE+done:
        folder=(OLD if (c,s) in REUSE else OUT)/(c+'_'+s)
        with (folder/'TEST_RESULTS.csv').open() as f:rows.extend(dict(cell=c,support=s,reused=(c,s) in REUSE,source=str(folder/'TEST_RESULTS.csv'),**r) for r in csv.DictReader(f))
    snap=OUT/'summaries'/now().replace(':','');snap.mkdir(parents=True,exist_ok=True)
    with (snap/'TEST_RESULTS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# SoftLag 확장 실험 결과','',
        '완료 %d/16조건: 기존 4조건 재사용, 신규 %d조건 완료. 실패 %d조건.'%(4+len(done),len(done),len(failed)),
        '각 조건은 고정 train64/validation32/test32 timestamps와 전체 채널(ETTh1 7, Weather 21)이다. Full 후보 전체 또는 P100의 100개를 사용한다. 전체 query 평가나 encoder 재학습으로 해석하지 않는다.',
        'B0는 기존 frozen V5, B1은 공유 lag bias+alpha, B2는 5개 head lag bias+공통 alpha이다. 독립 V1 실험과 다르다. B1/B2는 동일 배치 순서로 5epoch/40step 학습하고 epoch0 포함 validation RetMSE 최소 epoch를 선택한다.',
        '', '| 조건 | support | B0 RetMSE | B1 RetMSE | B2 RetMSE | B1 epoch | B2 epoch |', '|---|---|---:|---:|---:|---:|---:|']
    for c,s in REUSE+done:
        x={r['method']:r for r in rows if r['cell']==c and r['support']==s}
        lines.append('| %s | %s | %s | %s | %s | %s | %s |'%(c,s,x['B0']['retmse'],x['B1']['retmse'],x['B2']['retmse'],x['B1']['selected_epoch'],x['B2']['selected_epoch']))
    lines.extend(['','TEST_RESULTS.csv는 Random/Oracle/LastRepeat 및 AggMSE/AggMAE/Recall도 포함한다. 조건별 HEAD_USEFULNESS, HEAD_PAIR_OVERLAP, BIAS_SIMILARITY, PAIRED_TIME_BLOCK_CI, LEARNED_LAG_PREFERENCES, PROTOCOL_AUDIT와 체크포인트를 별도 보존한다.',
        'CI는 동일 timestamp의 모든 채널을 공동 resampling하는 H/2H/4H block, seed0 2000회 percentile95%이며 4block 미만은 불충분으로 표기한다. 여러 학습 seed의 불확실성을 측정한 결과가 아니다.',
        '후보 미래는 teacher 및 평가에만 사용하고 lag 입력은 과거만 사용한다. Score normalization은 전체 support를 기준으로 한다. 원본 runtime provenance UNKNOWN과 현재 수치 재현 PASS를 구분한다.',
        'GPU wall time은 D1–D6와 공존하는 실행 시간이다. 단독 GPU 처리 성능, kernel time 또는 학습/validation 분리 시간은 미측정이다. Solar는 재개하지 않았다.',
        '종합 메커니즘 판정: INCONCLUSIVE. 제한된 고정 timestamp/seed0의 성능 차이만으로 head specialization의 인과적 효과나 전체 데이터 일반화를 확정하지 않는다.'])
    (snap/'REPORT_KO.md').write_text('\n'.join(lines),encoding='utf8')
    dump(snap/'GATE.json',dict(status='COMPLETE' if len(done)==12 and not failed else 'PARTIAL',reused=4,new_complete=len(done),expected_total=16,failed=failed,files={n:sha(snap/n) for n in ['TEST_RESULTS.csv','REPORT_KO.md']}))
    dump(OUT/'PROGRESS.json',dict(at=now(),complete=4+len(done),reused=4,new_complete=len(done),new_remaining=12-len(done)-len(failed),failed=failed,summary=str(snap)))
def main():
    (OUT/'logs').mkdir(parents=True,exist_ok=True)
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ready=json.loads((OUT/'00_Audit/CPU_PREFLIGHT.json').read_text());assert ready['status']=='PASS_CPU_CHECKPOINT_AND_REUSE'
    for n,h in ready['source_hashes'].items():assert sha(HERE/n)==h,('Source changed after CPU audit',n)
    for c,s in REUSE:validate_reuse(OLD/(c+'_'+s))
    done=[];failed=[];report(done,failed)
    for cell,support in NEW:
        out=OUT/(cell+'_'+support)
        if checked_gate(out/'FINAL_GATE.json','PASS_EXPANSION_CELL_COMPLETE'):done.append((cell,support));continue
        # Preserve the previous pilot's queued export, with no third GPU worker.
        while True:
            ds=json.loads((DIAG/'extensions_d5_d6/CONTROLLER_STATUS.json').read_text())
            ep=DIAG/'extensions_d5_d6/POST_EXPORT_STATUS.json';es=json.loads(ep.read_text()) if ep.exists() else {}
            if ds.get('status') not in ['COMPLETE_ALL_CASES','AVAILABLE_CASES_COMPLETE_REMAINING_DEPENDENCIES'] or es.get('status')=='SOFTLAG_PAIR_EXPORT_COMPLETE':break
            status('WAITING_ORIGINAL_PAIR_EXPORT_AT_BOUNDARY',cell=cell,support=support,export=es);time.sleep(15)
        py='/usr/bin/python3' if support=='FULL' else PY
        stages=['GATE']
        if not (out/'COMPONENT_GATE.json').exists():stages.append('TRAIN')
        stages.append('FINAL');okay=True
        for stage in stages:
            command=[py,'-u',str(OWN/'softlag_run.py'),cell,support,stage]
            log=OUT/'logs'/('%s_%s_%s.log'%(cell,support,stage))
            with log.open('a') as f:
                child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                status('RUNNING',cell=cell,support=support,phase=stage,worker_pid=child.pid,command=command,log=str(log),complete=4+len(done),failed=failed)
                rc=child.wait()
            if rc:
                failed.append(dict(cell=cell,support=support,phase=stage,returncode=rc,log=str(log)));okay=False;report(done,failed)
                # OOM/unknown ownership/numerical failure needs inspection; no serial fallback.
                status('BLOCKED_PRESERVED_RESULTS',failed=failed,complete=4+len(done));return
        if okay:
            assert checked_gate(out/'FINAL_GATE.json','PASS_EXPANSION_CELL_COMPLETE');done.append((cell,support));report(done,failed)
    status('COMPLETE_ALL_16',complete=16,reused=4,new_complete=12,failed=failed)
if __name__=='__main__':
    try:main()
    except BaseException as e:status('BLOCKED',error=repr(e),traceback=traceback.format_exc());raise
