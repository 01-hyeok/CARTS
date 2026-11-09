"""Durable serial queue; never signal other experiments or borrow another GPU."""
import os,sys,subprocess,time,json,fcntl,traceback
from experiment_common import *

def resources_ready():
    old=ROOT/'results/TRACK-SCORE-FUSION-ABLATION01/CONTROLLER_STATUS.json'
    status=json.loads(old.read_text())
    if status.get('status')!='COMPLETE':return False,dict(reason='existing score fusion not COMPLETE',existing=status)
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],universal_newlines=True)
    occupied=[x for x in raw.splitlines() if x.startswith(GPU)]
    if occupied:return False,dict(reason='GPU1 occupied',processes=occupied)
    return True,dict(reason='existing score fusion COMPLETE and GPU1 has no compute processes',checked_at=now())

def main():
    OUT.mkdir(exist_ok=True);lock=(OUT/'CONTROLLER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ready=json.loads((OUT/'CPU_IMPLEMENTATION_GATE.json').read_text());assert ready['status']=='PASS'
    for n,h in ready['source_hashes'].items():assert sha(HERE/n)==h,'New implementation changed after CPU gate'
    for support in ('P100','FULL'):
        if support=='FULL':
            from summarize import summarize
            summarize()
            assert all((OUT/'P100'/c/'COMPLETE.json').exists() for c in CELLS)
        for cell in CELLS:
            dest=OUT/support/cell
            if (dest/'COMPLETE.json').exists():
                m=json.loads((dest/'COMPLETE.json').read_text());assert m['status']=='PASS_ALL_QUERY_TRAIN_VAL_TEST'
                continue
            while True:
                cpu=json.loads((OUT/'CPU_PREPARATION_STATUS.json').read_text())
                if cpu['status']=='BLOCKED':raise RuntimeError(('CPU preparation blocked',cpu))
                feature_ready=(OUT/'features'/cell/'COMPLETE.json').exists()
                ok,detail=resources_ready()
                if ok and feature_ready:break
                dump(OUT/'CONTROLLER_STATUS.json',dict(status='WAITING_RESOURCE_OR_FEATURES',cell=cell,support=support,
                     feature_ready=feature_ready,resource=detail,at=now(),pid=os.getpid(),no_gpu_worker_started=True))
                time.sleep(30)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            py='/data/pjh_workspace/ts-env/bin/python' if support=='P100' else '/usr/bin/python3'
            cmd=[py,'-u',str(HERE/'run_experiment.py'),cell,support]
            logs=OUT/'logs';logs.mkdir(exist_ok=True);log=logs/(support+'_'+cell+'_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S')+'.log')
            with log.open('x') as stream:
                p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                dump(OUT/'CONTROLLER_STATUS.json',dict(status='RUNNING_GATES_THEN_TRAIN',cell=cell,support=support,worker_pid=p.pid,controller_pid=os.getpid(),command=cmd,log=str(log),at=now()))
                rc=p.wait()
            assert rc==0,dict(cell=cell,support=support,returncode=rc,log=str(log))
            assert (dest/'COMPLETE.json').exists()
            from summarize import summarize
            summarize()
    dump(OUT/'CONTROLLER_STATUS.json',dict(status='COMPLETE',at=now(),conditions=16,seed=0))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        dump(OUT/'CONTROLLER_STATUS.json',dict(status='BLOCKED',at=now(),error=repr(e),traceback=traceback.format_exc(),preserved=True));raise
