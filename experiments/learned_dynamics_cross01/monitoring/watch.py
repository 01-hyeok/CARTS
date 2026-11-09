"""CPU-only durable progress/resource evidence; never controls GPU workers."""
import sys,os,json,time,subprocess,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from experiment_common import OUT,dump,now,GPU

def read(p):
    try:return json.loads(p.read_text())
    except FileNotFoundError:return None
def main():
    last=None
    while True:
        primary=read(OUT/'CONTROLLER_STATUS.json');supplement=read(OUT/'supplemental/CONTROLLER_STATUS.json')
        completed=[str(p.parent.relative_to(OUT)) for support in ('P100','FULL') for p in (OUT/support).glob('*/COMPLETE.json')]
        extra=[str(p.parent.relative_to(OUT)) for kind in ('pool','cross') for p in (OUT/'supplemental'/kind).glob('*/*/COMPLETE.json')]
        state=dict(at=now(),primary=primary,supplement=supplement,completed_native=completed,completed_supplement=extra,pid=os.getpid())
        if primary and primary.get('cell'):
            base=OUT/primary['support']/primary['cell'];state['current']=read(base/'CURRENT.json');state['epoch']=read(base/'EPOCH_STATE.json')
        state['gpu1']=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],universal_newlines=True).strip()
        applications=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],universal_newlines=True)
        state['gpu1_compute_processes']=[x for x in applications.splitlines() if x.startswith(GPU)]
        with (OUT/'RESOURCE_SAMPLES.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
        signature=json.dumps(dict(primary=primary,supplement=supplement,current=state.get('current'),completed=completed,extra=extra),sort_keys=True)
        if signature!=last:
            with (OUT/'PROGRESS_EVENTS.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
            last=signature
        if any(s and s.get('status')=='BLOCKED' for s in (primary,supplement)):
            state['status']='BLOCKED_REQUIRES_REVIEW';dump(OUT/'MONITOR_STATUS.json',state);return
        if len(completed)==16 and len(extra)==24:
            state['status']='FINAL_REPORT_GENERATION';dump(OUT/'MONITOR_STATUS.json',state)
            from finalize_all import main as finish
            finish();state['status']='ALL_COMPARISONS_AND_FINAL_REPORTS_COMPLETE';dump(OUT/'MONITOR_STATUS.json',state);return
        state['status']='TRACKING';dump(OUT/'MONITOR_STATUS.json',state);time.sleep(30)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        dump(OUT/'MONITOR_STATUS.json',dict(status='MONITOR_ERROR',error=repr(e),traceback=traceback.format_exc(),at=now(),workers_untouched=True));raise
