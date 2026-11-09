"""Fail-closed sequential P100 queue. Never stops old work or starts Solar."""
import os,sys,json,time,subprocess,fcntl,traceback,hashlib
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/TRACK-PREDICTIVE-SCORE-CALIBRATION01';OLD=ROOT/'results/TRACK-LEARNED-DYNAMICS-CROSS01'
GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec';PY='/data/pjh_workspace/ts-env/bin/python'
CELLS=['ETTh1_96','ETTh1_336','ETTh1_720','Weather_96','Weather_336','Weather_720','ETTh1_192','Weather_192']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,info):
    p=OUT/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(dict(at=time.time(),**info),indent=2));os.replace(tmp,p)
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def status(phase,**kw):write('QUEUE_STATUS.json',dict(phase=phase,pid=os.getpid(),solar_started=False,**kw))
def sealed():
    seal=read(OUT/'EXECUTION_SOURCE_SEAL.json')
    assert seal['files']=={p.name:sha(p) for p in HERE.glob('*.py')},'Source changed after reviewed seal; do not launch'
def gpu_idle():
    raw=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    index,uuid,used,util=[x.strip() for x in raw.split(',')];assert index=='1' and uuid==GPU
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    return not any(line.startswith(GPU) for line in apps.splitlines()) and int(used)<128 and int(util)==0
def prerequisites():
    assert read(OUT/'CPU_SYNTHETIC_TESTS.json').get('passed')
    assert read(OUT/'P2_CPU_TESTS.json').get('passed')
    assert read(OUT/'P1_CPU_INTEGRATION.json').get('status')=='PASS_CPU_ACTUAL_TRAIN_BLOCK_INTEGRATION'
    assert read(OUT/'P0_CPU_COMPLETE.json').get('status')=='PARTIAL_P0_COMPLETE'
    assert read(OUT/'REGIME_CPU_COMPLETE.json').get('status')=='PASS_ALL_AVAILABLE_COMPLETED_CONDITIONS'
    local=read(OUT/'LOCAL_RANKING_CPU_COMPLETE.json')
    assert local.get('status')=='P100_LOCAL_RANKING_COMPLETE'
    assert len(local['replay'])==16
    # CPU/GPU numerical mismatch is explicitly pending canonical GPU replay,
    # never silently accepted as baseline reproduction.
def execute(script,args,gpu):
    sealed();log=OUT/'logs'/('%s_%s_%d.log'%(script.replace('.py',''),'_'.join(args),int(time.time())));log.parent.mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU if gpu else '',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if gpu:assert gpu_idle(),'Another GPU workload appeared; fail closed'
    with log.open('x') as stream:
        worker=subprocess.Popen([PY,'-u',str(HERE/script)]+args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        status('RUNNING',script=script,args=args,worker_pid=worker.pid,gpu=gpu,log=str(log))
        assert worker.wait()==0,'Worker failed; preserve outputs: '+str(log)
def main():
    lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);sealed()
    while True:
        legacy=read(OLD/'recovery01/COMPLETE.json');native=read(OLD/'CONTROLLER_STATUS.json');supp=read(OLD/'supplemental/CONTROLLER_STATUS.json')
        if native.get('status')=='BLOCKED' or supp.get('status')=='BLOCKED':
            status('WAITING_LEGACY_BLOCKER',legacy_native=native,legacy_supplement=supp)
        elif legacy.get('status')=='ALL_QUEUES_COMPLETE' and native.get('status')=='COMPLETE' and supp.get('status')=='COMPLETE' and gpu_idle():
            break
        else:status('WAITING_LEGACY_QUEUES_AND_GPU1',legacy_native=native,legacy_supplement=supp)
        time.sleep(60)
    prerequisites();time.sleep(30);assert gpu_idle(),'GPU did not remain idle'
    # This is the same advisory lock old native workers use; never cancel them.
    oldlock=(OLD/'GPU.lock').open('a');fcntl.flock(oldlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for cell in CELLS:
        target=OUT/'canonical_cache/P100'/cell/'COMPLETE.json'
        if not target.exists():execute('prepare_gpu.py',[cell],True)
        assert read(target).get('status')=='PASS_ALL_QUERY_P100_HEAD_CACHE'
    if not (OUT/'LOCAL_RANKING_GPU_COMPLETE.json').exists():execute('local_ranking_gpu.py',[],True)
    local=read(OUT/'LOCAL_RANKING_GPU_COMPLETE.json')
    assert local.get('status')=='P100_LOCAL_RANKING_COMPLETE_GPU'
    assert len(local['replay'])==16 and all(not any(r['ordered_id_mismatches'].values()) for r in local['replay'])
    if not (OUT/'HEAD_DIAGNOSTICS_COMPLETE.json').exists():execute('head_diagnostics_gpu.py',[],True)
    assert read(OUT/'HEAD_DIAGNOSTICS_COMPLETE.json').get('status')=='PASS_P100_ALL8'
    write('P100_P0_GATE.json',dict(status='PASS',scope='P100 all8 only; Full P0 remains pending',evidence=['P0_CPU_COMPLETE.json','REGIME_CPU_COMPLETE.json','LOCAL_RANKING_GPU_COMPLETE.json','HEAD_DIAGNOSTICS_COMPLETE.json','canonical_cache/P100/*/COMPLETE.json'],test_used_for_method_selection=False))
    for phase in ('P1','P2'):
        if phase=='P2':assert all(read(OUT/'runs/P100'/c/'P1/COMPLETE.json').get('status')=='PASS_ALL_QUERY_P100' for c in CELLS)
        for cell in CELLS:
            target=OUT/'runs/P100'/cell/phase/'COMPLETE.json'
            if not target.exists():execute('train_p100.py',[cell,phase],True)
            assert read(target).get('status')=='PASS_ALL_QUERY_P100'
    status('P100_P1_P2_COMPLETE_FULL_IMPLEMENTATION_PENDING',full_started=False,p3_started=False,p4_started=False)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        status('BLOCKED_PRESERVED',error=repr(error),traceback=traceback.format_exc());raise
