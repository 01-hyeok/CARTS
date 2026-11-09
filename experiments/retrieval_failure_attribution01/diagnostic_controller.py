#!/usr/bin/env python3
"""Durable GPU1 queue: existing48 -> Oracle K supplement -> P100 -> Full -> summary.

The post-diagnostic Solar training authorization is a distinct gated queue;
never starts here without its own audited training implementation/smoke gates.
"""
import datetime,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');HERE=ROOT/'experiments/retrieval_failure_attribution01';OUT=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01'
PY='/data/pjh_workspace/ts-env/bin/python';LEGACY='/usr/bin/python3';GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
OLD=ROOT/'experiments/p100_head_retrieval_quality_legacy_pool02'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)
def current(**x):write(OUT/'00_Audit/CONTROLLER_STATUS.json',dict(at=now(),controller_pid=os.getpid(),**x))

def training_state():
    rows=[r for r in read(OLD/'configs/run_manifest.json') if r['seed']==0];assert len(rows)==48
    complete=sum((ROOT/r['output_path']/'COMPLETE.json').exists() for r in rows)
    active=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
        except (FileNotFoundError,PermissionError):continue
        if any(x.endswith('/runtime.py') or x.endswith('/runtime_vector.py') or x.endswith('/runtime_validation.py') for x in argv) and any('p100_head_retrieval_quality_legacy_pool02/' in x for x in argv):active.append(int(p.name))
    return complete,active

def gpu_apps():
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True).splitlines()
    return [r for r in rows if GPU in r]

def prerequisites():
    done,active=training_state()
    return done==48 and not active,dict(existing_training_complete=done,total=48,active_training_workers=active)

def run(command,label,env,phase):
    out=OUT/'logs';out.mkdir(exist_ok=True);path=out/(label+'.log')
    with path.open('a') as log:
        log.write('\nSTART '+now()+' '+repr(command)+'\n');log.flush()
        proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
        current(status='RUNNING',phase=phase,target=label,worker_pid=proc.pid,log=str(path),command=command,gpu_uuid=GPU)
        last_issue=0
        while proc.poll() is None:
            time.sleep(.5)
            age=time.time()-path.stat().st_mtime
            if phase=='FULL' and age>1800 and time.time()-last_issue>1800:
                issue=dict(status='PERFORMANCE_STALL_OBSERVATION',at=now(),target=label,worker_pid=proc.pid,log=str(path),seconds_without_log=age,action='reported; worker not killed')
                write(OUT/'00_Audit/IMMEDIATE_ISSUE.json',issue);print('FULL ISSUE',json.dumps(issue),flush=True);last_issue=time.time()
        assert proc.returncode==0,(label,'worker failed',proc.returncode,str(path))

def main():
    OUT.mkdir(exist_ok=True);lock=(OUT/'diagnostic.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=read(HERE/'PROTOCOL.json');cells=protocol['cell_order'];assert len(cells)==12
    gate=read(HERE/'READY_GATE.json');assert gate['status']=='CPU_READY_REAL_GPU_GATE_AT_EXECUTION'
    import hashlib
    for name,digest in gate['source_sha256'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,(name,'source changed after tests')
    while True:
        ready,state=prerequisites()
        apps=gpu_apps() if ready else []
        if ready and not apps:break
        current(status='WAITING_EXISTING_TRAINING',gpu_apps=apps,**state)
        print('WAIT',now(),state,'gpu_apps',apps,flush=True);time.sleep(30)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES=GPU)
    subprocess.run([PY,str(HERE/'phase0.py')],cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),check=True)
    oracle_out=ROOT/'results/TRACK-P100-HEAD-RETRIEVAL-QUALITY01-ORACLE-K-SUPPLEMENT01'
    if not (oracle_out/'SMOKE_GATE.json').exists():
        run([PY,'-u',str(HERE/'oracle_k_completion.py'),'--smoke'],'oracle_k_smoke',env,'ORACLE_SMOKE')
    for cell in cells:
        while gpu_apps():
            current(status='WAITING_GPU1_RELEASE',phase='ORACLE_K',target=cell,gpu_apps=gpu_apps());time.sleep(30)
        run([PY,'-u',str(HERE/'oracle_k_completion.py'),cell],'ORACLE_'+cell,env,'ORACLE_K')
        assert (oracle_out/cell/'COMPLETE.json').exists()
    run([PY,str(HERE/'oracle_k_summary.py')],'oracle_k_summary',dict(env,CUDA_VISIBLE_DEVICES=''),'ORACLE_SUMMARY')
    assert read(oracle_out/'FINAL_GATE.json')['status']=='PASS'
    for support in ['P100','FULL']:
        if support=='FULL':assert all((OUT/'P100'/c/'COMPLETE.json').exists() for c in cells)
        for cell in cells:
            ready,state=prerequisites();assert ready,state
            while gpu_apps():
                current(status='WAITING_GPU1_RELEASE',phase=support,target=cell,gpu_apps=gpu_apps());time.sleep(30)
            run([PY if support=='P100' else LEGACY,'-u',str(HERE/'diagnostic_run.py'),support,cell],support+'_'+cell,env,support)
            assert (OUT/support/cell/'COMPLETE.json').exists()
            print('VALIDATED',support,cell,now(),flush=True)
    run([PY,str(HERE/'diagnostic_report.py')],'final_summary',dict(env,CUDA_VISIBLE_DEVICES=''),'CPU_SUMMARY')
    assert read(OUT/'07_Summary/SUMMARY_GATE.json')['status']=='PASS'
    post=read(HERE/'POST_DIAGNOSTIC_SOLAR_TRAINING_QUEUE.json')
    post.update(status='PREDECESSORS_COMPLETE_TRAINING_IMPLEMENTATION_GATE_PENDING',predecessors_completed_at=now(),actual_training_started=False)
    write(HERE/'POST_DIAGNOSTIC_SOLAR_TRAINING_QUEUE.json',post)
    current(status='DIAGNOSTICS_COMPLETE_POST_TRAINING_GATE_PENDING',missing_full_solar_models=8,post_training_queue=str(HERE/'POST_DIAGNOSTIC_SOLAR_TRAINING_QUEUE.json'))

if __name__=='__main__':
    try:main()
    except BlockingIOError:
        print('Existing diagnostic controller owns lock; no second controller started',flush=True)
    except BaseException as e:
        current(status='FAILED',error=repr(e),traceback=traceback.format_exc())
        write(OUT/'00_Audit/IMMEDIATE_ISSUE.json',dict(status='FAILED',at=now(),error=repr(e),traceback=traceback.format_exc()))
        raise
