"""One recovery supervisor: resume selected evaluation, then original queues."""
import sys,os,json,subprocess,time,fcntl,traceback,importlib.util
from pathlib import Path
RECOVERY=Path(__file__).resolve().parent;sys.path.insert(0,str(RECOVERY.parent))
from experiment_common import *
import controller as native

def main():
    root=OUT/'recovery01';lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gate=json.loads((root/'COMPATIBILITY_GATE.json').read_text());assert gate['status']=='PASS'
    assert gate['compat_sha256']==sha(RECOVERY/'compat.py')
    sealed=json.loads((root/'SOURCE_SEAL.json').read_text())
    assert sealed['sources']=={p.name:sha(p) for p in RECOVERY.glob('*.py')}
    # Reacquire original locks to prove no original queue is alive; native.main
    # subsequently owns its own lock while it runs.
    for path in (OUT/'CONTROLLER.lock',OUT/'supplemental/controller.lock',OUT/'GPU.lock'):
        with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dump(OUT/'supplemental/CONTROLLER_STATUS.json',dict(status='WAITING_RECOVERY_NATIVE16',at=now(),supervisor_pid=os.getpid()))
    if not (OUT/'FULL/ETTh1_96/COMPLETE.json').exists():
        ok,reason=native.resources_ready();assert ok,reason
        log=root/'evaluation_resume.log';assert not log.exists()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with log.open('x') as f:
            p=subprocess.Popen(['/usr/bin/python3','-u',str(RECOVERY/'resume.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            dump(OUT/'CONTROLLER_STATUS.json',dict(status='RECOVERY_FINAL_EVALUATION',cell='ETTh1_96',support='FULL',worker_pid=p.pid,controller_pid=os.getpid(),log=str(log),progress_path=str(root/'evaluation/ETTh1_96_FULL/CURRENT.json'),at=now()))
            assert p.wait()==0,'Recovery evaluation failed; see '+str(log)
    real_popen=subprocess.Popen
    def redirected(cmd,*args,**kwargs):
        if isinstance(cmd,list) and len(cmd)>2:
            if cmd[2]==str(RECOVERY.parent/'run_experiment.py'):cmd=cmd[:2]+[str(RECOVERY/'entry.py'),'native']+cmd[3:]
            elif cmd[2]==str(RECOVERY.parent/'supplemental/run_supplement.py'):cmd=cmd[:2]+[str(RECOVERY/'entry.py'),'supplement']+cmd[3:]
        return real_popen(cmd,*args,**kwargs)
    subprocess.Popen=redirected
    import summarize,report_notes
    old_summary=summarize.summarize
    def summary():
        old_summary();report_notes.annotate(OUT)
    summarize.summarize=summary
    native.main()
    spec=importlib.util.spec_from_file_location('original_supp_controller',RECOVERY.parent/'supplemental/controller.py')
    supp=importlib.util.module_from_spec(spec);spec.loader.exec_module(supp);supp.main()
    dump(root/'COMPLETE.json',dict(status='ALL_QUEUES_COMPLETE',at=now()))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        info=dict(status='BLOCKED',at=now(),error=repr(e),traceback=traceback.format_exc(),preserved=True,recovery='recovery01')
        dump(OUT/'recovery01/FAILURE.json',info);dump(OUT/'CONTROLLER_STATUS.json',info);raise
