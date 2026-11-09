"""CPU waiter: export remaining tiny SoftLag matrix only after GPU queue exits."""
import fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');HERE=Path(__file__).resolve().parent
BASE=ROOT/'results/TRACK-HEAD-SHIFT-DIAGNOSTIC01';OUT=BASE/'extensions_d5_d6';SOFT=ROOT/'results/TRACK-SOFTLAG-HEAD-PILOT01'
def write(v):
    p=OUT/'POST_EXPORT_STATUS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def main():
    lock=(OUT/'post_export.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        p=OUT/'CONTROLLER_STATUS.json';s=json.loads(p.read_text()) if p.exists() else {}
        if s.get('status') in ['COMPLETE_ALL_CASES','AVAILABLE_CASES_COMPLETE_REMAINING_DEPENDENCIES']:break
        write(dict(status='WAITING_D56_GPU_QUEUE_EXIT',pid=os.getpid(),gpu_used=False));time.sleep(15)
    boundary=(BASE/'00_Audit/d4_controller.lock').open('a');fcntl.flock(boundary,fcntl.LOCK_EX)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for support,py in [('FULL','/usr/bin/python3'),('P100','/data/pjh_workspace/ts-env/bin/python'),('REPORT','/data/pjh_workspace/ts-env/bin/python')]:
        command=[py,'-u',str(HERE/'softlag_pair_export.py'),support];log=SOFT/'00_Audit'/('PAIR_EXPORT_'+support+'.log')
        with log.open('a') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            write(dict(status='RUNNING_PAIR_EXPORT',support=support,worker_pid=child.pid,command=command,log=str(log)))
            code=child.wait()
        if code:write(dict(status='BLOCKED',returncode=code,log=str(log)));return
    write(dict(status='SOFTLAG_PAIR_EXPORT_COMPLETE',gpu_used=False))
if __name__=='__main__':main()
