"""Durable D4-boundary handoff, one GPU worker, CPU aggregation, explicit gaps."""
import datetime,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');HERE=Path(__file__).resolve().parent
BASE=ROOT/'results/TRACK-HEAD-SHIFT-DIAGNOSTIC01';OUT=BASE/'extensions_d5_d6'
ORDER=['ETTh1_96','ETTh1_336','ETTh1_720','Weather_96','ETTh1_192','Weather_192','Weather_336','Weather_720']
PY2='/data/pjh_workspace/ts-env/bin/python';PY1='/usr/bin/python3';GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def status(state_label,**kw):dump(OUT/'CONTROLLER_STATUS.json',dict(status=state_label,at=now(),pid=os.getpid(),**kw))
def runtime(cell,model,support):
    return PY2 if support=='P100' or (model in ['V0','V1'] and cell in ['ETTh1_96','ETTh1_720','Weather_96','Weather_720']) else PY1
def jobs():
    out=[]
    for cell in ORDER:
        for model in ['V5','V0','V1']:
            for training in ['FULL','P100']:out.append((cell,model,training,training))
        for model in ['V5','V0','V1']:
            for training in ['FULL','P100']:out.append((cell,model,training,'P100' if training=='FULL' else 'FULL'))
    return out
def ready_d4():
    state=BASE/'00_Audit/W720_RECOVERY_STATUS.json'
    if not state.exists():return False
    s=json.loads(state.read_text())
    if s['status']!='W720_D4_COMPONENTS_COMPLETE_STATISTICS_PENDING':return False
    for c in ORDER:
        for support in ['FULL','P100']:
            p=BASE/'D4'/('V5_'+c+'_'+support)/'COMPONENT_GATE.json'
            if not p.exists():return False
            g=json.loads(p.read_text())
            if g['status']!='D4_FEATURES_AND_RERANK_COMPLETE_STATISTICS_PENDING' or g['queries']!={'val':32,'test':32}:return False
    return True
def launch(command,log,job,phase):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        status('RUNNING',job=job,phase=phase,worker_pid=child.pid,command=command,log=str(log))
        code=child.wait()
    return code
def verified_gate(path,status_value):
    if not path.exists():return False
    g=json.loads(path.read_text());assert g['status']==status_value
    for p,h in g.get('files',{}).items():assert sha(path.parent/p)==h
    return True
def main():
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cpu=json.loads((OUT/'CPU_EXTENSION_READY_v2.json').read_text());assert cpu['status']=='PASS'
    for p,h in cpu['code_hashes'].items():assert sha(HERE/p)==h
    dump(OUT/'EXECUTION_QUEUE.json',dict(jobs=jobs(),native_cases=48,cross_support_cases=48,all_channels=True,seed=0,
        dependency='all 16 native V5 D4 feature components and W720 recovery exit; original D1/D2 CPU remains independent',
        one_gpu_worker=True,solar_resume=False,no_training=True,cross_support_missing_stored_reference='new exact canonical comparison at every sampled query; no serial production fallback'))
    while not ready_d4():status('WAITING_FOR_D4_BOUNDARY');time.sleep(15)
    # Lock lives across the queue; no old/new D4 controller may race this worker.
    boundary=(BASE/'00_Audit/d4_controller.lock').open('a')
    fcntl.flock(boundary,fcntl.LOCK_EX)
    complete=[];waiting=[];failed=[]
    for job in jobs():
        cell,model,training,evaluation=job;name=model+'_'+training+'_to_'+evaluation
        path=OUT/'complete_analysis'/cell/name/'ANALYSIS_GATE.json'
        if verified_gate(path,'PASS_D5_D6_CASE'):complete.append(job);continue
        py=runtime(cell,model,training);args=[cell,model,training,evaluation]
        smoke=OUT/'feature_smoke'/cell/name/'FEATURE_GATE.json';feature=OUT/'complete_features'/cell/name/'FEATURE_GATE.json'
        stages=[]
        if not verified_gate(smoke,'PASS_FIRST_TAIL_SMOKE_NOT_OFFICIAL'):stages.append(('SMOKE',[py,'-u',str(HERE/'d56_complete_run.py')]+args+['--smoke']))
        if not verified_gate(feature,'PASS_COMPLETE_D5_D6_FEATURES_AGGREGATION_PENDING'):stages.append(('FEATURES',[py,'-u',str(HERE/'d56_complete_run.py')]+args))
        stages.append(('CPU_AGGREGATION',[PY2,'-u',str(HERE/'d56_aggregate.py')]+args))
        success=True
        for phase,command in stages:
            log=OUT/'logs'/(cell+'_'+name+'_'+phase+'.log');code=launch(command,log,job,phase)
            if code:
                failed.append(dict(job=job,phase=phase,returncode=code,log=str(log)));dump(OUT/'FAILED_JOBS.json',failed);success=False
                # Keep numerical failures visible; a failed worker is never restarted.
                break
        if success:
            assert verified_gate(path,'PASS_D5_D6_CASE');complete.append(job)
        dump(OUT/'PROGRESS.json',dict(at=now(),complete=len(complete),native_complete=sum(j[2]==j[3] for j in complete),
            cross_support_complete=sum(j[2]!=j[3] for j in complete),failed=failed,waiting=waiting,total=96))
        # Reporting is a CPU child and occurs only after the GPU child exits.
        report=HERE/'d56_report.py'
        if report.exists():
            code=launch([PY2,'-u',str(report)],OUT/'logs/report.log',job,'PARTIAL_REPORT')
            if code:raise RuntimeError('Report verification failed; '+str(OUT/'logs/report.log'))
    status('COMPLETE_ALL_CASES' if not waiting and not failed else 'AVAILABLE_CASES_COMPLETE_REMAINING_DEPENDENCIES',complete=len(complete),failed=failed,waiting=waiting)
    return 0 if not failed else 1
if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:status('BLOCKED',error=repr(e),traceback=traceback.format_exc());raise
