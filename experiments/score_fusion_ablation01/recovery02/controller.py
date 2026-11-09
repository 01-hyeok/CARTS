"""Continue only unstarted jobs through the reviewed wrapper; original controller preserved."""
import sys,importlib.util,json,hashlib,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
spec=importlib.util.spec_from_file_location('original_controller',HERE.parent/'controller.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
ready=json.loads((c.OUT/'recovery02/CPU_READY.json').read_text());assert ready['status']=='PASS'
for n,h in ready['source_hashes'].items():assert c.sha(HERE/n)==h
original=c.subprocess.Popen
original_call=c.subprocess.call
did_launch_evaluation=False
def launch(cmd,*args,**kw):
    global did_launch_evaluation
    if isinstance(cmd,list):
        if str(c.HERE/'evaluate.py') in cmd:did_launch_evaluation=True
        cmd[:]=[str(HERE/'run.py') if x==str(c.HERE/'evaluate.py') else str(HERE/'summarize.py') if x==str(c.HERE/'summarize.py') else x for x in cmd]
    return original(cmd,*args,**kw)
c.subprocess.Popen=launch
def summarize_completed_once(cmd,*args,**kw):
    if isinstance(cmd,list) and str(c.HERE/'summarize.py') in cmd and not did_launch_evaluation:
        # Completed blocks still pass c.valid() hashes; avoid identical report
        # regeneration eight times when resuming after all ETTh1 is finished.
        return 0
    return original_call(cmd,*args,**kw)
c.subprocess.call=summarize_completed_once
if __name__=='__main__':
    try:
        c.dump(c.OUT/'CONTROLLER_STATUS.json',dict(status='RESUMING_VERIFY_COMPLETED_HASHES',at=c.now(),pid=c.os.getpid(),no_gpu_worker_started=True))
        c.main()
    except BaseException as e:c.dump(c.OUT/'CONTROLLER_STATUS.json',dict(status='BLOCKED',at=c.now(),error=repr(e),traceback=traceback.format_exc(),preserved=True));raise
