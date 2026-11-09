"""Resume original eight-run and follow-up queue with explicit shared-GPU guard.

Completed runs are authenticated and skipped. Original code and completed
provenance are immutable. H720 waits when its verified full shape cannot coexist.
"""
import os,subprocess,sys,time,traceback
import parallel_policy as p
sys.path.insert(0,str(p.ORIGINAL))
import solar_full_controller as original
t=original.t

def wait_budget(cell,label):
    while True:
        try:return p.budget(t.read(t.OUT/'SMOKE_GATE.json'),cell,p.snapshot())
        except AssertionError as exc:
            if not str(exc).startswith("('WAIT_RESOURCE"):
                raise
            original.status(status='WAITING_VERIFIED_MEMORY_HEADROOM',target=label,reason=str(exc),scientific_conditions_unchanged=True)
            time.sleep(30)

def run(command,label,env):
    path=t.OUT/'logs'/(label+'_parallel13.log')
    with path.open('a') as f:
        f.write('\nSTART '+t.rt.now()+' '+repr(command)+'\n');f.flush()
        proc=subprocess.Popen(command,cwd=t.ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        original.status(status='RUNNING',target=label,worker_pid=proc.pid,log=str(path),gpu_uuid=t.GPU,shared_gpu_authorized=True,wrapper_hashes=p.wrapper_hashes())
        assert proc.wait()==0,(label,'worker failed',str(path))

def launch(command,label,env):
    if command[2].endswith('/solar_full_train.py') and '--cell' in command:
        cell=command[command.index('--cell')+1];heads=int(command[command.index('--heads')+1])
        if (t.OUT/cell/f'V{heads}'/'seed0/COMPLETE.json').exists():
            row=p.validate_completed(t,cell,heads);t.dump(p.EVIDENCE/(cell+f'_V{heads}_PRESERVED.json'),row);return
        wait_budget(cell,label)
        gate=p.EVIDENCE/(cell+'_COEXISTENCE_SMOKE.json')
        if not gate.exists():
            run([original.PY,'-u',str(p.HERE/'solar_full_train_parallel.py'),'smoke',cell],label+'_coexistence_smoke',env)
        wait_budget(cell,label)
        run([original.PY,'-u',str(p.HERE/'solar_full_train_parallel.py'),'train',cell,'--heads',str(heads)],label,env)
    else:
        # Original diagnostic implementation retains its exclusive guard. Its
        # safe wait is preserved; no silently patched supplemental inference.
        while original.apps() or '/experiments/retrieval_failure_mechanism02/controller.py' in p.cmd(p.read(p.MECH).get('controller_pid',-1)):
            original.status(status='WAITING_GPU1_RELEASE_FOR_ORIGINAL_FOLLOWUP',target=label)
            time.sleep(30)
        run(command,label,env)

if __name__=='__main__':
    try:
        p.EVIDENCE.mkdir(parents=True,exist_ok=True)
        ready=t.read(p.EVIDENCE/'CPU_GATE.json');assert ready['status']=='PASS' and ready['wrapper_hashes']==p.wrapper_hashes()
        original.launch=launch
        original.main()
    except BlockingIOError:print('Existing controller; no duplicate started',flush=True)
    except BaseException as exc:
        original.status(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
        t.dump(p.EVIDENCE/'CONTROLLER_FAILURE.json',dict(at=t.rt.now(),error=repr(exc),traceback=traceback.format_exc()));raise
