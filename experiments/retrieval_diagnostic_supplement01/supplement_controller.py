"""Resume sequential work, close P100 deliverables before Full, release Solar last."""
import argparse,fcntl,os,signal,subprocess,sys,time,traceback
from supplement_common import *
sys.path.insert(0,str(ORIGINAL))
import diagnostic_controller as old
PY=old.PY;LEGACY=old.LEGACY

def command(pid):
    try:return [s.decode() for s in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if s]
    except FileNotFoundError:return []
def process_state(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().split(') ')[1].split()[0]
    except FileNotFoundError:return None
def wait_active_child(handoff):
    worker=handoff['worker_pid'];controller=handoff['controller_pid']
    assert command(controller)==handoff['controller_command'] and process_state(controller)=='T'
    while process_state(worker) not in (None,'Z'):
        assert command(worker)==handoff['worker_command'],'worker PID reused or changed'
        dump(OUT/'CONTROLLER_STATUS.json',dict(status='WAITING_UNINTERRUPTED_EXISTING_WORKER',controller_pid=os.getpid(),worker_pid=worker,cell=handoff['cell']))
        time.sleep(5)
    assert (SOURCE/'P100'/handoff['cell']/'COMPLETE.json').exists(),'existing worker did not complete; leave old controller held'
    assert command(controller)==handoff['controller_command'] and process_state(controller)=='T'
    # Signal this verified CPU queue process only, never its worker/process group.
    os.kill(controller,signal.SIGTERM);os.kill(controller,signal.SIGCONT)
    for _ in range(50):
        if process_state(controller) in (None,'Z'):break
        time.sleep(.1)
    assert process_state(controller) in (None,'Z')
    dump(OUT/'HANDOFF_COMPLETE.json',dict(status='PASS',old_controller=controller,preserved_worker=worker,worker_completed=True,new_controller=os.getpid()))

def launch(cmd,label,env,phase,gpu=False):
    if gpu:
        while old.gpu_apps():
            old.current(status='WAITING_GPU1_RELEASE',phase=phase,target=label,gpu_apps=old.gpu_apps());time.sleep(15)
    old.run(cmd,label,env,phase)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    gate=read(HERE/'READY_GATE.json');assert gate['status']=='CPU_PASS_GPU_GATE_AT_EXECUTION' and gate['source_hashes']==source_hashes()
    handoff=read(OUT/'HANDOFF_PLAN.json')
    if not (OUT/'HANDOFF_COMPLETE.json').exists():wait_active_child(handoff)
    lock=(SOURCE/'diagnostic.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert old.prerequisites()[0]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES=old.GPU)
    cpu=dict(env,CUDA_VISIBLE_DEVICES='')
    assert read(ROOT/'results/TRACK-P100-HEAD-RETRIEVAL-QUALITY01-ORACLE-K-SUPPLEMENT01/FINAL_GATE.json')['status']=='PASS'
    for support in ['P100','FULL']:
        if support=='FULL':assert read(OUT/'P100_DELIVERABLE_GATE.json')['status']=='PASS'
        for cell in CELLS:
            # Keep already-completed main computations; validate model-file SHA.
            original=SOURCE/support/cell/'COMPLETE.json'
            if original.exists():
                done=read(original)
                for m in done['models']:assert sha(m['checkpoint'])==m['sha256']
            else:
                launch([PY if support=='P100' else LEGACY,'-u',str(ORIGINAL/'diagnostic_run.py'),support,cell],support+'_'+cell,env,support,True)
            launch([PY,str(HERE/'supplement_cpu.py'),support,cell],f'SUPPLEMENT_CPU_{support}_{cell}',cpu,'CPU_COMPLETENESS')
            launch([PY if support=='P100' else LEGACY,'-u',str(HERE/'supplement_inference.py'),support,cell],f'SUPPLEMENT_{support}_{cell}',env,'FULL' if support=='FULL' else 'P100_DETAIL',True)
            launch([PY,str(HERE/'supplement_report.py'),'--cell',support,cell],f'DELIVERABLE_{support}_{cell}',cpu,'CPU_COMPLETENESS')
        launch([PY,str(HERE/'supplement_report.py'),'--phase',support],f'{support}_DELIVERABLE_GATE',cpu,'CPU_COMPLETENESS')
    launch([PY,str(ORIGINAL/'diagnostic_report.py')],'final_summary',cpu,'CPU_SUMMARY')
    launch([PY,str(HERE/'supplement_report.py'),'--final'],'expanded_final_summary',cpu,'CPU_SUMMARY')
    assert read(OUT/'FINAL_DELIVERABLE_GATE.json')['status']=='PASS'
    post=read(ORIGINAL/'POST_DIAGNOSTIC_SOLAR_TRAINING_QUEUE.json')
    post.update(status='ALL_DELIVERABLES_COMPLETE_SOLAR_GPU_SMOKE_PENDING',deliverable_gate=str(OUT/'FINAL_DELIVERABLE_GATE.json'),actual_training_started=False)
    dump(ORIGINAL/'POST_DIAGNOSTIC_SOLAR_TRAINING_QUEUE.json',post)
    old.current(status='DIAGNOSTICS_COMPLETE_POST_TRAINING_GATE_PENDING',expanded_deliverable_gate=str(OUT/'FINAL_DELIVERABLE_GATE.json'),missing_full_solar_models=8)
    solar=ROOT/'results/TRACK-SOLAR-FULL-MISSING-CHECKPOINTS01';held=read(solar/'DELIVERABLE_COMPLETENESS_HOLD.json')
    pid=4088382;cmd=command(pid)
    assert any(x.endswith('/solar_full_missing_checkpoints01/solar_full_controller.py') for x in cmd) and process_state(pid)=='T','Solar waiter identity/state changed; do not signal'
    dump(solar/'DELIVERABLE_COMPLETENESS_RELEASE.json',dict(status='RELEASED_AFTER_EXPANDED_FINAL_GATE',pid=pid,gate_sha256=sha(OUT/'FINAL_DELIVERABLE_GATE.json'),all_prior_deliverables_complete=True))
    os.kill(pid,signal.SIGCONT)
    dump(OUT/'CONTROLLER_STATUS.json',dict(status='COMPLETE_SOLAR_WAITER_RELEASED',controller_pid=os.getpid(),solar_waiter_pid=pid))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        issue=dict(status='FAILED',controller_pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc())
        dump(OUT/'IMMEDIATE_ISSUE.json',issue);old.current(status='FAILED',error=repr(exc),traceback=traceback.format_exc());raise
