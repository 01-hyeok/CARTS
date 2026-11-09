"""Continue Full/cross-support only after the original P100 queue completes."""
import os,sys,time,json,hashlib,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');HERE=Path(__file__).resolve().parent;OUT=ROOT/'results/TRACK-PREDICTIVE-SCORE-CALIBRATION01';EXT=OUT/'full_extension01';OLD=ROOT/'results/TRACK-LEARNED-DYNAMICS-CROSS01'
GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec';CELLS=['ETTh1_96','ETTh1_336','ETTh1_720','Weather_96','Weather_336','Weather_720','ETTh1_192','Weather_192']
def read(path):return json.loads(path.read_text()) if path.exists() else {}
def status(phase,**kw):
    p=EXT/'QUEUE_STATUS.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(phase=phase,at=time.time(),pid=os.getpid(),solar_started=False,**kw),indent=2));os.replace(tmp,p)
def sealed():
    manifest=read(EXT/'SOURCE_SEAL.json')
    assert manifest['files']=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    for rel,digest in manifest['parent_dependencies'].items():assert hashlib.sha256((HERE.parent/rel).read_bytes()).hexdigest()==digest
def gpu_idle():
    data=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    index,uuid,memory,util=[x.strip() for x in data];assert index=='1' and uuid==GPU
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    return int(memory)<128 and int(util)==0 and not any(x.startswith(GPU) for x in procs.splitlines())
def execute(script,args,python='/usr/bin/python3'):
    sealed()
    while not gpu_idle():status('WAITING_GPU_RELEASE',next_script=script,next_args=args);time.sleep(30)
    log=EXT/'logs'/('%s_%s_%d.log'%(script.replace('.py',''),'_'.join(x.replace('--','') for x in args),int(time.time())));log.parent.mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with log.open('x') as stream:
        p=subprocess.Popen([python,'-u',str(HERE/script)]+args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        status('RUNNING',script=script,args=args,worker_pid=p.pid,log=str(log));code=p.wait()
    assert code==0,'Worker failed; preserve output and inspect '+str(log)
def main():
    lock=(EXT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);sealed()
    while True:
        parent=read(OUT/'QUEUE_STATUS.json')
        if parent.get('phase')=='P100_P1_P2_COMPLETE_FULL_IMPLEMENTATION_PENDING':break
        status('WAITING_P100_QUEUE',parent=parent);time.sleep(60)
    assert all(read(OUT/'runs/P100'/c/'P2/COMPLETE.json').get('status')=='PASS_ALL_QUERY_P100' for c in CELLS)
    assert read(OLD/'recovery01/COMPLETE.json').get('status')=='ALL_QUEUES_COMPLETE'
    while not gpu_idle():status('WAITING_GPU1_AFTER_P100');time.sleep(30)
    time.sleep(30);assert gpu_idle()
    oldlock=(OLD/'GPU.lock').open('a')
    while True:
        try:fcntl.flock(oldlock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:status('WAITING_EXISTING_GPU_LOCK');time.sleep(30)
    for cell in CELLS:
        if not (EXT/'P0'/cell/'COMPLETE.json').exists():execute('full_p0.py',[cell])
        assert read(EXT/'P0'/cell/'COMPLETE.json').get('status')=='PASS_FULL_P0_ALL_QUERY'
    for phase in ('P1','P2'):
        for cell in CELLS:
            gate=OUT/'gate_runs/FULL'/cell/phase/'COMPLETE.json'
            if not gate.exists():execute('full_train.py',[cell,phase,'--gate-only'])
            assert read(gate).get('status')=='PASS_FULL_ACTUAL_INTEGRATION_ONLY'
            target=OUT/'runs/FULL'/cell/phase/'COMPLETE.json'
            if not target.exists():execute('full_train.py',[cell,phase])
            assert read(target).get('status')=='PASS_ALL_QUERY_FULL'
    for training in ('P100','FULL'):
        evaluation='FULL' if training=='P100' else 'P100'
        python='/data/pjh_workspace/ts-env/bin/python' if training=='P100' else '/usr/bin/python3'
        for phase in ('P1','P2'):
            for cell in CELLS:
                target=OUT/'cross_support'/('%s_to_%s'%(training,evaluation))/cell/phase/'COMPLETE.json'
                if not target.exists():execute('cross_support.py',[cell,training,phase],python)
                assert read(target).get('status')=='PASS_FROZEN_CROSS_SUPPORT'
    status('FULL_AND_CROSS_COMPLETE_REPORT_AND_P3_P4_DECISION_PENDING',full_native_conditions=8,cross_support_phase_conditions=32,p3_started=False,p4_started=False)
if __name__=='__main__':
    try:main()
    except BaseException as e:status('BLOCKED_PRESERVED',error=repr(e),traceback=traceback.format_exc());raise
