"""Wait for the existing controller/recovery boundary; probe only, no blind retry."""
import fcntl,json,os,subprocess,time,traceback,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/data/pjh_workspace/CARTS');BASE=ROOT/'results/TRACK-HEAD-SHIFT-DIAGNOSTIC01'
EXT=BASE/'extensions_d5_d6';OUT=EXT/'cublas_recovery02'
GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state(label,**kw):
    from datetime import datetime,timezone
    p=OUT/'CURRENT.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(status=label,at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**kw),indent=2));t.replace(p)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ready=json.loads((OUT/'CPU_READY.json').read_text());assert ready['status']=='PASS'
    for n,h in ready['source_hashes'].items():assert sha(HERE/n)==h
    state('WAITING_EXISTING_D56_BOUNDARY',gpu_used=False)
    boundary=(BASE/'00_Audit/d4_controller.lock').open('a');fcntl.flock(boundary,fcntl.LOCK_EX)
    current=json.loads((EXT/'CONTROLLER_STATUS.json').read_text())
    assert current['status'] in ['COMPLETE_ALL_CASES','AVAILABLE_CASES_COMPLETE_REMAINING_DEPENDENCIES'],current
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
    assert not any(line.startswith(GPU) for line in apps.splitlines()),'GPU1 not idle; no other job stopped'
    cmd=['/data/pjh_workspace/ts-env/bin/python','-u',str(HERE/'probe.py')]
    log=OUT/'probe.log'
    with log.open('a') as f:
        child=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        state('RUNNING_EXACT_TAIL_PROBE',worker_pid=child.pid,command=cmd,log=str(log))
        code=child.wait()
    state('PROBE_COMPLETE_PRODUCTION_STILL_PENDING' if code==0 else 'PROBE_BLOCKED',returncode=code,log=str(log),gpu_used=False)
if __name__=='__main__':
    try:main()
    except BaseException as error:state('BLOCKED',error=repr(error),traceback=traceback.format_exc(),production_activated=False);raise
