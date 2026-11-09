"""Verified saved-array D1/D2 additions, isolated from all GPU processes."""
import datetime,fcntl,json,os,traceback
from pathlib import Path
from .d2_saved_blocks import run

ROOT=Path('/data/pjh_workspace/CARTS')
BASE=ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02'
OUT=ROOT/'results/TRACK-HEAD-SHIFT-DIAGNOSTIC01'

def state(status,**kwargs):
    p=OUT/'00_Audit/CPU_QUEUE_STATUS.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(status=status,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),**kwargs),indent=2));tmp.replace(p)

def main():
    lock=(OUT/'00_Audit/cpu_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    done=0;missing=[]
    for cell in ['ETTh1_96','ETTh1_192','ETTh1_336','ETTh1_720','Weather_96','Weather_192','Weather_336','Weather_720']:
        gate=BASE/'cells'/cell/'CELL_ARTIFACT_GATE.json'
        if not gate.exists():missing.append(dict(cell=cell,reason='Mechanism02 integrated gate absent'));continue
        for training in ['P100','FULL']:
            base=BASE/'checkpoint_runs'/cell/('V5_'+training)
            for variant in ['FULL','P100_CONTROLLED']:
                for split in ['val','test']:
                    dest=OUT/'D2_SAVED_ARRAYS'/cell/('V5_'+training)/variant/split
                    if (dest/'CPU_SET_GATE.json').exists():done+=1;continue
                    state('RUNNING',cell=cell,training_support=training,variant=variant,split=split,completed=done,gpu_used=False)
                    run(base,dest,split,variant);done+=1
            missing.append(dict(cell=cell,training_support=training,reason='P100_NATIVE head arrays unavailable; additional inference required. Lost-candidate mixture rank/all-head score/probabilities remain pending.'))
    state('SAVED_ARRAY_COMPONENTS_COMPLETE_NOT_FULL_D1_D4',completed=done,pending=missing,gpu_used=False)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        state('BLOCKED',error=repr(e),traceback=traceback.format_exc(),preserved=True);raise
