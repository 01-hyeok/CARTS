"""Durable CPU waiter; one diagnostic GPU job only after bounded queue exits."""
import os,sys,json,time,fcntl,subprocess,traceback,datetime
from pathlib import Path
from coverage_gate import require_safe_boundary, sha
ROOT=Path('/data/pjh_workspace/CARTS')
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01'
SUPPLEMENT=ROOT/'results/TRACK-RETRIEVAL-DIAGNOSTIC-SUPPLEMENT01'
OUT=SOURCE/'00_Audit/TAIL06'
GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
BOUND_CONTROLLER=12449


def read(p):return json.loads(p.read_text())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(x):
    x=dict(x,at=now(),pid=os.getpid());p=OUT/'BOUNDARY_CURRENT.json';t=p.with_suffix('.tmp')
    t.write_text(json.dumps(x,indent=2));t.replace(p)
def alive(pid):
    p=Path('/proc/%d/stat'%pid)
    return p.exists() and p.read_text().split(') ')[1].split()[0]!='Z'
def gpu_busy():
    return GPU in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)


def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'boundary.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ready=read(HERE/'CPU_READY.json')
    assert ready['status']=='CPU_PASS_GPU_UNTESTED'
    assert ready['source_hashes']=={p.name:sha(p) for p in HERE.glob('*.py')}
    assert read(SUPPLEMENT/'P100_DELIVERABLE_GATE.json')['status']=='PASS'
    while alive(BOUND_CONTROLLER):
        args=Path('/proc/%d/cmdline'%BOUND_CONTROLLER).read_bytes()
        assert b'retrieval_all_channel_tail05/full_channel_controller.py' in args
        write(dict(status='WAITING_BOUNDARY_NO_GPU',controller_pid=BOUND_CONTROLLER,
                   worker_started=False,solar_training_release=False))
        time.sleep(30)
    # The running controller holds this same lock until it exits.
    common=(SOURCE/'diagnostic.lock').open('a');fcntl.flock(common,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cells=['ETTh1_96','ETTh1_192','ETTh1_336','ETTh1_720','Weather_96','Weather_192','Weather_336']
    require_safe_boundary(read(SOURCE/'00_Audit/TAIL05/BOUNDED_QUEUE_STOP.json'),
                          {c:read(SUPPLEMENT/'FULL'/c/'DELIVERABLE_GATE.json') for c in cells},
                          {c:sha(SOURCE/'FULL'/c/'COMPLETE.json') for c in cells},
                          alive(BOUND_CONTROLLER),gpu_busy())
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    log=OUT/'WEATHER720_NUMERIC_DIAGNOSIS.log'
    with log.open('x') as stream:
        worker=subprocess.Popen(['/data/pjh_workspace/ts-env/bin/python','-u',str(HERE/'diagnose_weather720.py')],
                                cwd=str(ROOT),env=env,stdout=stream,stderr=subprocess.STDOUT)
        write(dict(status='RUNNING_ISOLATED_NUMERIC_DIAGNOSIS',worker_pid=worker.pid,log=str(log),
                   preceding_seven_cells_delivered=True,production_release=False))
        code=worker.wait()
    assert code==0,('Numeric diagnostic failed',code,str(log))
    assert read(OUT/'WEATHER720_DIAGNOSIS.json')['status']=='DIAGNOSIS_COMPLETE_NOT_A_RELEASE_GATE'
    write(dict(status='DIAGNOSIS_COMPLETE_WEATHER720_STILL_BLOCKED',log=str(log),production_release=False,
               solar_training_release=False))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(dict(status='BLOCKED',error=repr(error),traceback=traceback.format_exc(),
                   production_release=False,solar_training_release=False))
        raise
