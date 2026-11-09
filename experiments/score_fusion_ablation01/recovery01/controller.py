"""Continue only unstarted jobs through the reviewed wrapper; original controller preserved."""
import sys,importlib.util,json,hashlib,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
spec=importlib.util.spec_from_file_location('original_controller',HERE.parent/'controller.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
ready=json.loads((c.OUT/'recovery01/CPU_READY.json').read_text());assert ready['status']=='PASS'
for n,h in ready['source_hashes'].items():assert c.sha(HERE/n)==h
original=c.subprocess.Popen
def launch(cmd,*args,**kw):
    if isinstance(cmd,list):
        cmd[:]=[str(HERE/'run.py') if x==str(c.HERE/'evaluate.py') else str(HERE/'summarize.py') if x==str(c.HERE/'summarize.py') else x for x in cmd]
    return original(cmd,*args,**kw)
c.subprocess.Popen=launch
if __name__=='__main__':
    try:c.main()
    except BaseException as e:c.dump(c.OUT/'CONTROLLER_STATUS.json',dict(status='BLOCKED',at=c.now(),error=repr(e),traceback=traceback.format_exc(),preserved=True));raise
