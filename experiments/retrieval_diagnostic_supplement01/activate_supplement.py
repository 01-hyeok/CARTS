"""One-time CPU-controller handoff; active GPU worker is never signalled."""
import os,signal,subprocess,time
from supplement_common import *

def cmd(pid):return [x.decode() for x in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if x]
def state(pid):return (Path('/proc')/str(pid)/'stat').read_text().split(') ')[1].split()[0]

def main():
    assert not (OUT/'HANDOFF_PLAN.json').exists(),'Handoff already exists; verify rather than duplicate'
    ready=read(HERE/'READY_GATE.json');assert ready['status']=='CPU_PASS_GPU_GATE_AT_EXECUTION' and ready['source_hashes']==source_hashes()
    current=read(SOURCE/'00_Audit/CONTROLLER_STATUS.json');controller=current['controller_pid'];worker=current['worker_pid']
    assert current['phase']=='P100' and current['status']=='RUNNING'
    cc=cmd(controller);wc=cmd(worker)
    assert str(ORIGINAL/'diagnostic_controller.py') in cc
    assert str(ORIGINAL/'diagnostic_run.py') in wc and wc[-2]=='P100'
    assert int((Path('/proc')/str(worker)/'stat').read_text().split(') ')[1].split()[1])==controller
    assert state(worker) not in ['T','Z']
    original_hashes=read(ORIGINAL/'READY_GATE.json')['source_sha256']
    for name,digest in original_hashes.items():assert sha(ORIGINAL/name)==digest
    # Freeze only the parent queue so it cannot auto-enter Full; its GPU child
    # continues unchanged and will finish normally. New queue waits for it.
    os.kill(controller,signal.SIGSTOP)
    time.sleep(.2);assert state(controller)=='T' and state(worker)!='T'
    plan=dict(status='OLD_CPU_QUEUE_HELD_ACTIVE_GPU_WORKER_UNCHANGED',controller_pid=controller,controller_command=cc,worker_pid=worker,worker_command=wc,cell=wc[-1],original_worker_log=current['log'],new_order='finish current worker -> P100 detail gates/all remaining P100 -> P100 deliverable gate -> Full+details -> final deliverable gate -> Solar Full8',worker_signalled=False)
    dump(OUT/'HANDOFF_PLAN.json',plan)
    logs=OUT/'logs';logs.mkdir(exist_ok=True);path=logs/'controller.log'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with path.open('a') as f:p=subprocess.Popen(['/data/pjh_workspace/ts-env/bin/python','-u',str(HERE/'supplement_controller.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    dump(OUT/'ACTIVATION.json',dict(status='NEW_CPU_QUEUE_WAITING_CURRENT_WORKER',controller_pid=p.pid,preserved_gpu_worker=worker,log=str(path)))
    print('ACTIVATED',p.pid,'GPU worker preserved',worker,'log',path,flush=True)

if __name__=='__main__':main()
