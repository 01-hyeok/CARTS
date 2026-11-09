"""One observer, no worker launch, signal, or finalization."""
from recovery_runtime import *
import time,traceback
from recovery02_validate import targets
def main():
    seal_check();protect()
    locks=[hold(OUT/'monitor.lock'),hold(ORIGINAL/'recovery01/monitor.lock')]
    while True:
        state=read(OUT/'CONTROLLER_STATUS.json') if (OUT/'CONTROLLER_STATUS.json').exists() else {}
        done={m:[] for m in ('native','pool','cross')}
        for p,l in targets():
            if (p/'COMPLETE.json').exists():done[l['mode']].append(l)
        current={}
        if state.get('output'):
            p=Path(state['output']);current={n:read(p/n) for n in ('CURRENT.json','EPOCH_STATE.json') if (p/n).exists()}
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],universal_newlines=True)
        record=dict(status='TRACKING',at=now(),pid=os.getpid(),controller=state,completed=done,current=current,gpu_processes=gpu.splitlines())
        dump(OUT/'MONITOR_STATUS.json',record)
        with (OUT/'RESOURCE_SAMPLES.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if state.get('status') in ('BLOCKED','COMPLETE'):return
        time.sleep(30)
if __name__=='__main__':
    try:main()
    except BaseException as e:dump(OUT/'MONITOR_STATUS.json',dict(status='MONITOR_ERROR',error=repr(e),traceback=traceback.format_exc(),at=now()));raise
