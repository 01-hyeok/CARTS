import os,json,fcntl,subprocess,traceback,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/data/pjh_workspace/CARTS');HERE=Path(__file__).resolve().parent;BASE=ROOT/'results/TRACK-HEAD-SHIFT-DIAGNOSTIC01';OUT=BASE/'full_population_completion/population_recovery05';GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
def state(s,**kw):
    p=OUT/'CURRENT.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(status=s,at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**kw),indent=2));t.replace(p)
def main():
    own=(OUT/'controller.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state('WAITING_WEATHER_RECOVERY_BOUNDARY',gpu_used=False)
    lock=(BASE/'00_Audit/d4_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    ready=json.loads((OUT/'CPU_READY.json').read_text());assert ready['status']=='PASS'
    for n,h in ready['source_hashes'].items():assert hashlib.sha256((HERE/n).read_bytes()).hexdigest()==h
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert GPU not in apps
    cmd=['/usr/bin/python3','-u',str(HERE/'probe.py')];log=OUT/'probe.log'
    with log.open('a') as f:
        p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        state('RUNNING_GPU_DIAGNOSIS',worker_pid=p.pid,command=cmd,log=str(log));rc=p.wait()
    state('DIAGNOSIS_FINISHED_PRODUCTION_UNCHANGED',returncode=rc,log=str(log))
if __name__=='__main__':
    try:main()
    except BaseException as e:state('BLOCKED',error=repr(e),traceback=traceback.format_exc());raise
