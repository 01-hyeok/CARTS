"""Single isolated serial supervisor: resume + remaining5 + pool8 + cross16 + final gate."""
from recovery_runtime import *
import time,traceback
from recovery02_validate import targets,validate_job
def launch(mode,args,path,label):
    seal_check();preserved(False);gpu_ready()
    py='/usr/bin/python3' if (mode in ('resume','native') or label['training']=='FULL') else '/data/pjh_workspace/ts-env/bin/python'
    cmd=[py,'-B','-u',str(CODE/'entry.py'),mode]+args
    log=OUT/'logs'/(mode+'_'+label['cell']+'_'+(label['pool'] or label['training'])+'_'+datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')+'.log');log.parent.mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with log.open('x') as f:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        status=dict(status='RUNNING',mode=mode,label=label,cell=label['cell'],support=label['support'],worker_pid=p.pid,controller_pid=os.getpid(),command=cmd,output=str(path),log=str(log),at=now())
        dump(OUT/'CONTROLLER_STATUS.json',status)
        if mode in ('pool','cross'):dump(OUT/'supplemental/CONTROLLER_STATUS.json',status)
        rc=p.wait()
    assert rc==0,('Worker failed',rc,str(log))
    validate_job(path,label,True)
def main():
    seal_check();protect();assert read(OUT/'PREFLIGHT.json')['status']=='PASS'
    assert read(OUT/'PREFLIGHT.json')['source_seal_sha256']==sha(OUT/'SOURCE_SEAL.json')
    locks=[hold(OUT/'controller.lock'),hold(ORIGINAL/'CONTROLLER.lock'),hold(ORIGINAL/'recovery01/controller.lock'),hold(ORIGINAL/'supplemental/controller.lock')]
    dump(OUT/'supplemental/CONTROLLER_STATUS.json',dict(status='WAITING_RECOVERY_NATIVE16',at=now()))
    for path,label in targets():
        if (path/'COMPLETE.json').exists():validate_job(path,label,False);continue
        assert not path.exists(),('Partial output requires explicit recovery review',str(path))
        mode=label['mode'];args=[label['cell'],label['training']]
        if mode=='native' and label['cell']=='Weather_96' and label['support']=='FULL':mode='resume';args=[]
        if label['pool']:args.append(label['pool'])
        while True:
            try:gpu_ready();break
            except AssertionError as e:
                dump(OUT/'CONTROLLER_STATUS.json',dict(status='WAITING_GPU_NO_SIGNALS',reason=str(e),next=label,controller_pid=os.getpid(),at=now()));time.sleep(30)
        launch(mode,args,path,label)
    dump(OUT/'supplemental/CONTROLLER_STATUS.json',dict(status='COMPLETE',jobs=24,at=now()))
    dump(OUT/'CONTROLLER_STATUS.json',dict(status='FINAL_VALIDATION_AND_REPORTS',at=now(),controller_pid=os.getpid()))
    import finalize;finalize.run()
    from readiness import check
    ready=check();dump(OUT/'COMPLETE.json',dict(status='ALL_QUEUES_AND_FINAL_REPORTS_VERIFIED',readiness=ready,at=now()))
    dump(OUT/'CONTROLLER_STATUS.json',dict(status='COMPLETE',native=16,pool=8,cross=16,final_pass=True,at=now()))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        dump(OUT/'CONTROLLER_STATUS.json',dict(status='BLOCKED',error=repr(e),traceback=traceback.format_exc(),at=now()));raise
