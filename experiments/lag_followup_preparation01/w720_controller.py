"""Finish only missing W720 native gates/D4; never launches Solar or old queues."""
import datetime,fcntl,json,os,subprocess,sys
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');HERE=Path(__file__).resolve().parent
BASE=ROOT/'results/TRACK-HEAD-SHIFT-DIAGNOSTIC01';AUDIT=BASE/'00_Audit'
def dump(v):
    p=AUDIT/'W720_RECOVERY_STATUS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def main():
    lock=(AUDIT/'d4_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LAG_TRACK='TRACK-HEAD-SHIFT-DIAGNOSTIC01')
    for support in ['FULL','P100']:
        py='/usr/bin/python3' if support=='FULL' else '/data/pjh_workspace/ts-env/bin/python'
        for phase in ['reproduce','smoke','d4']:
            log=BASE/'logs'/('W720_recovery_'+support+'_'+phase+'.log')
            command=[py,'-u',str(HERE/'complete_w720.py'),support,phase]
            with log.open('a') as stream:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                dump(dict(status='RUNNING',controller_pid=os.getpid(),worker_pid=child.pid,support=support,phase=phase,command=command,log=str(log),at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
                code=child.wait()
            if code:
                dump(dict(status='BLOCKED',returncode=code,log=str(log),support=support,phase=phase,at=datetime.datetime.now(datetime.timezone.utc).isoformat()));return code
    dump(dict(status='W720_D4_COMPONENTS_COMPLETE_STATISTICS_PENDING',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_results_preserved=True));return 0
if __name__=='__main__':sys.exit(main())
