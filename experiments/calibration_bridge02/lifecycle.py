"""Durable at-most-once stage claims; no stale-claim takeover."""
import datetime,hashlib,json,os,sys,uuid
from pathlib import Path

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def identity(pid, proc=Path('/proc')):
    p=proc/str(pid)
    try:
        # comm may contain spaces/parentheses; field 22 follows the final ')'.
        stat=(p/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
        command=(p/'cmdline').read_bytes()
        return dict(pid=pid,boot_id=(proc/'sys/kernel/random/boot_id').read_text().strip(),start_ticks=int(fields[19]),state=fields[0],command=command.decode(errors='replace').split('\0')[:-1],command_sha256=hashlib.sha256(command).hexdigest())
    except (FileNotFoundError,ProcessLookupError):return None
def same_process(saved,current):
    return bool(current and all(saved[k]==current[k] for k in ('pid','boot_id','start_ticks','command_sha256')) and current['state'] not in ('Z','X'))
def classify(saved,proc=Path('/proc')):
    current=identity(saved['pid'],proc)
    return 'SAME_PROCESS_LIVE' if same_process(saved,current) else ('PID_REUSED_OR_CHANGED' if current else 'PROCESS_ABSENT')
def exclusive_json(path,value):
    # O_EXCL is the claim. A crash leaving partial JSON is also a permanent claim.
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
def atomic_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    exclusive_json(tmp,value);os.replace(tmp,path)
class Stage:
    def __init__(self,root,stage,bindings):
        self.root=Path(root);self.stage=stage;self.run_id=uuid.uuid4().hex;self.workers=[]
        who=identity(os.getpid())
        self.record=dict(stage=stage,run_id=self.run_id,started_at=now(),supervisor=who,command=who['command'],bindings=bindings)
        exclusive_json(self.root/(stage.upper()+'_STARTED.json'),self.record)
        self.directory=self.root/('launch_'+stage);self.directory.mkdir(exist_ok=False)
        self.update('RUNNING')
    def update(self,status,**extra):
        atomic_json(self.directory/'STATE.json',dict(self.record,status=status,at=now(),workers=self.workers,**extra))
    def started(self,p,cmd,step):
        who=identity(p.pid)
        self.workers.append(dict(pid=p.pid,identity=who,command=cmd,step=step,status='RUNNING',started_at=now()))
        self.update('RUNNING')
    def ended(self,code):
        self.workers[-1].update(status='COMPLETE' if code==0 else 'FAILED',exit_code=code,ended_at=now(),wait_reaped=True)
        self.update('RUNNING' if code==0 else 'FAILED')
    def complete(self):self.update('COMPLETE',ended_at=now())
    def failed(self,error):self.update('FAILED',ended_at=now(),error=repr(error),automatic_retry=False)
def prior_finished(root,stage):
    root=Path(root);start=json.loads((root/(stage.upper()+'_STARTED.json')).read_text());state=json.loads((root/('launch_'+stage)/'STATE.json').read_text())
    if state['run_id']!=start['run_id'] or state['status']!='COMPLETE':raise ValueError('Prior stage is not complete')
    if same_process(start['supervisor'],identity(start['supervisor']['pid'])):raise ValueError('Prior supervisor still live')
    for w in state['workers']:
        if w['status']!='COMPLETE' or w.get('exit_code')!=0 or not w.get('wait_reaped'):raise ValueError('Prior worker not successfully reaped')
        if w['identity'] and same_process(w['identity'],identity(w['pid'])):raise ValueError('Prior worker still live')
    return state
