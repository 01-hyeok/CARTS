"""Serialize two read-only GPU diagnoses after the current full-population queue."""
import fcntl,json,os,subprocess,hashlib,traceback
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=Path('/data/pjh_workspace/CARTS')
BASE=ROOT/'results/TRACK-HEAD-SHIFT-DIAGNOSTIC01';OUT=BASE/'extensions_d5_d6/cublas_recovery03'
GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
def state(s,**kw):
    p=OUT/'CURRENT.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(status=s,at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**kw),indent=2));t.replace(p)
def main():
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ready=json.loads((OUT/'CPU_READY.json').read_text());assert ready['status']=='PASS'
    for n,h in ready['source_hashes'].items():assert hashlib.sha256((HERE/n).read_bytes()).hexdigest()==h
    state('WAITING_ACTIVE_FULL_POPULATION_BOUNDARY',gpu_used=False)
    boundary=(BASE/'00_Audit/d4_controller.lock').open('a');fcntl.flock(boundary,fcntl.LOCK_EX)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
    assert not any(l.startswith(GPU) for l in apps.splitlines()),'GPU1 is occupied; no process was stopped'
    results=[]
    for script,python in [('probe.py','/data/pjh_workspace/ts-env/bin/python'),('population_probe.py','/usr/bin/python3')]:
        for n,h in ready['source_hashes'].items():assert hashlib.sha256((HERE/n).read_bytes()).hexdigest()==h,'Source changed while waiting: '+n
        log=OUT/(script+'.log');cmd=[python,'-u',str(HERE/script)]
        with log.open('a') as f:
            child=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            state('RUNNING_GPU_DIAGNOSIS',script=script,worker_pid=child.pid,command=cmd,log=str(log));rc=child.wait()
        results.append(dict(script=script,returncode=rc,log=str(log)))
    state('DIAGNOSES_FINISHED_PRODUCTION_STILL_UNCHANGED',results=results,gpu_used=False)
if __name__=='__main__':
    try:main()
    except BaseException as e:state('BLOCKED',error=repr(e),traceback=traceback.format_exc(),production_activated=False);raise
