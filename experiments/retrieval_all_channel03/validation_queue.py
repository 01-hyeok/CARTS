"""Serial validation only. Never activates or releases production controllers."""
import os,sys,time,subprocess,json
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/ALL_CHANNEL03'
def write(d):
    p=OUT/'VALIDATION_QUEUE_CURRENT.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def check(name):
    d=json.loads((OUT/name).read_text());assert d['status']=='PASS',(name,d['status'])
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
    while Path('/proc/4172922').exists():
        write(dict(status='WAITING_CURRENT_H96_GATE',controller_pid=os.getpid(),worker_pid=4172922));time.sleep(5)
    check('Weather_96_V5_PACKED_LT_FULL_GATE.json')
    for script,cell,gate in [('validate_packed_lt.py','Weather_720','Weather_720_V5_PACKED_LT_FULL_GATE.json'),
                             ('validate_metric_batch.py','Weather_96','Weather_96_METRIC_BATCH_GATE.json'),
                             ('validate_metric_batch.py','Weather_720','Weather_720_METRIC_BATCH_GATE.json')]:
        logpath=OUT/(cell+'_'+script+'.log')
        with logpath.open('x') as log:
            p=subprocess.Popen(['/usr/bin/python3','-u',str(HERE/script),'--cell',cell],cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
            write(dict(status='RUNNING_VALIDATION_ONLY',controller_pid=os.getpid(),worker_pid=p.pid,cell=cell,script=script,log=str(logpath)))
            result=p.wait()
        assert result==0,(script,cell,result);check(gate)
    write(dict(status='PASS_GPU_VALIDATIONS',controller_pid=os.getpid(),production_activated=False))
if __name__=='__main__':
    try:main()
    except BaseException as e:write(dict(status='FAILED',controller_pid=os.getpid(),error=repr(e),production_activated=False));raise
