"""Validated Weather720 main -> CPU supplement -> GPU supplement -> final gate."""
import os,sys,fcntl,subprocess,traceback
from pathlib import Path
import gate
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(gate.ROOT/'experiments/retrieval_diagnostic_supplement01'))
import supplement_controller as queue
from supplement_common import read,dump,sha,SOURCE,OUT
old=queue.old

def main():
    gate.verify()
    assert read(OUT/'P100_DELIVERABLE_GATE.json')['status']=='PASS'
    for cell in ['ETTh1_96','ETTh1_192','ETTh1_336','ETTh1_720','Weather_96','Weather_192','Weather_336']:
        done=read(OUT/'FULL'/cell/'DELIVERABLE_GATE.json')
        assert done['status']=='PASS' and done['source_complete_sha256']==sha(SOURCE/'FULL'/cell/'COMPLETE.json')
    assert old.prerequisites()[0]
    lock=(SOURCE/'diagnostic.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not old.gpu_apps(),'GPU1 is in use; no competing production start'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=old.GPU,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    cpu=dict(env,CUDA_VISIBLE_DEVICES='')
    queue.launch([old.LEGACY,'-u',str(HERE/'resume_entry.py'),'FULL','Weather_720'],'RESUME09_FULL_Weather_720',env,'FULL',True)
    queue.launch([old.PY,str(queue.HERE/'supplement_cpu.py'),'FULL','Weather_720'],'RESUME09_CPU_Weather_720',cpu,'CPU_COMPLETENESS')
    queue.launch([old.LEGACY,'-u',str(HERE/'supplement_entry.py'),'FULL','Weather_720'],'RESUME09_SUPPLEMENT_Weather_720',env,'FULL',True)
    queue.launch([old.PY,str(queue.HERE/'supplement_report.py'),'--cell','FULL','Weather_720'],'RESUME09_GATE_Weather_720',cpu,'CPU_COMPLETENESS')
    final=read(OUT/'FULL/Weather_720/DELIVERABLE_GATE.json')
    assert final['status']=='PASS' and final['source_complete_sha256']==sha(SOURCE/'FULL/Weather_720/COMPLETE.json')
    # The next cells require explicit 137-channel data-path/memory gates.
    # Do not release Solar training or silently use any channel fallback.
    result=dict(status='WEATHER720_DELIVERED_SOLAR_DATA_BATCH_PREFLIGHT_PENDING',completed_full_cells=8,
                next_cells=['Solar_96','Solar_192','Solar_336','Solar_720'],solar_training_release=False,
                controller_pid=os.getpid(),gate_sha256=sha(OUT/'FULL/Weather_720/DELIVERABLE_GATE.json'))
    dump(SOURCE/'00_Audit/RESUME09/COMPLETE.json',result);old.current(**{k:v for k,v in result.items() if k!='controller_pid'})

if __name__=='__main__':
    try:main()
    except BaseException as error:
        issue=dict(status='FAILED',controller_pid=os.getpid(),error=repr(error),traceback=traceback.format_exc(),solar_training_release=False)
        dump(SOURCE/'00_Audit/RESUME09/FAILED.json',issue);old.current(status='FAILED',error=repr(error));raise
