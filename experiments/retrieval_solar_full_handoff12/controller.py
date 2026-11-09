"""CPU-boundary replacement; existing GPU worker finishes untouched."""
import os,sys,time,signal,fcntl,traceback
from pathlib import Path
import held_gate as hg
ROOT=Path('/data/pjh_workspace/CARTS');HERE=Path(__file__).resolve().parent
AUDIT=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/SOLAR_HANDOFF12'
PRIOR=ROOT/'experiments/retrieval_solar_full_diagnostic11'
sys.path.insert(0,str(ROOT/'experiments/retrieval_diagnostic_supplement01'))
import supplement_controller as queue
from supplement_common import read,dump,sha,SOURCE,OUT,ORIGINAL
old=queue.old;gate=hg.load_module(PRIOR/'gate.py','unchanged_diagnostic_gate')

def snapshot(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return None
    st=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,state=st[0],start_ticks=st[19],exit_code=int(st[49]),command=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x])
def sources():
    ready=read(HERE/'CPU_READY.json');assert ready['status']=='PASS'
    for p,h in ready['sources'].items():assert sha(ROOT/p)==h,('CPU handoff source changed',p)
    assert sha(HERE/'PLAN.json')==ready['plan_sha256'];return ready
def status(**kw):dump(AUDIT/'STATUS.json',dict(at=old.now(),controller_pid=os.getpid(),**kw))
def terminate_stopped(pid):
    os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
    for _ in range(100):
        st=snapshot(pid)
        if st is None or st['state']=='Z':return
        time.sleep(.1)
    raise RuntimeError('Verified stopped CPU controller did not terminate; no handoff')
def boundary(plan):
    while True:
        sources();c=snapshot(plan['controller']['pid']);w=snapshot(plan['worker']['pid'])
        assert c is not None and w is not None,'Unexpected predecessor disappearance; inspect before continuing'
        assert c['command']==plan['controller']['command'] and c['start_ticks']==plan['controller']['start_ticks'] and c['state']=='T'
        assert w['start_ticks']==plan['worker']['start_ticks']
        if w['state']=='Z':break
        assert w['command']==plan['worker']['command']
        status(status='WAITING_UNINTERRUPTED_MAIN_WORKER',worker_pid=w['pid'],cell=plan['cell'],gpu_work_started_here=False)
        time.sleep(15)
    children=(Path('/proc')/str(c['pid'])/'task'/str(c['pid'])/'children').read_text().split()
    hg.validate_boundary(plan,c,w,children)
    p=SOURCE/'FULL'/plan['cell']/'COMPLETE.json';done=read(p);assert done['status']=='COMPLETE_WITH_MISSING_MODELS'
    e=plan['expected'][plan['cell']];assert done['query_counts']==e['counts'] and done['channels']==e['channels']
    assert not (gate.AUDIT/plan['cell']/'main_FAILED.json').exists()
    terminate_stopped(c['pid'])
    dump(AUDIT/'BOUNDARY_HANDOFF.json',dict(status='PASS',at=old.now(),old_controller=c['pid'],uninterrupted_worker=w['pid'],completed_cell=plan['cell'],source_complete_sha256=sha(p)))
def release_training(plan):
    old.current(status='DIAGNOSTICS_COMPLETE_POST_TRAINING_GATE_PENDING',missing_full_solar_models=8)
    proof=hg.verify_all(ROOT,plan['expected'])
    legacy=ROOT/'results/TRACK-SOLAR-FULL-MISSING-CHECKPOINTS01'
    hold=read(legacy/'DELIVERABLE_COMPLETENESS_HOLD.json');assert sha(legacy/'DELIVERABLE_COMPLETENESS_HOLD.json')==plan['hold_sha256']
    current=read(legacy/'CONTROLLER_STATUS.json');pin=plan['held_waiter'];actual=snapshot(pin['pid']);assert actual is not None
    children=(Path('/proc')/str(pin['pid'])/'task'/str(pin['pid'])/'children').read_text().split()
    hg.validate_held_target(hold,current,pin,actual,children,proof)
    # Retire only the obsolete held CPU queue after all original hold reasons
    # have passed. The replacement keeps its independent artifact + GPU gates.
    terminate_stopped(pin['pid']);assert not (snapshot(pin['pid']) or {}).get('command')
    replacement=snapshot(plan['replacement']['pid']);assert replacement and replacement['command']==plan['replacement']['command'] and replacement['start_ticks']==plan['replacement']['start_ticks']
    assert not (Path('/proc')/str(replacement['pid'])/'task'/str(replacement['pid'])/'children').read_text().strip()
    path=legacy/'DELIVERABLE_COMPLETENESS_RELEASE.json';assert not path.exists()
    dump(AUDIT/'FINAL_PREREQUISITE_PROOF.json',proof)
    dump(path,dict(status='RELEASED_AFTER_EXPANDED_FINAL_GATE',pid=pin['pid'],gate_sha256=proof['final_gate_sha256'],all_prior_deliverables_complete=True,held_reason_verified=True,retired_old_waiter=True))
    dump(AUDIT/'COMPLETE.json',dict(status='PASS',at=old.now(),diagnostic_cells=24,training_started_here=False,replacement_waiter=replacement['pid']))
def main():
    sources();own=(AUDIT/'controller.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(HERE/'PLAN.json');boundary(plan)
    lock=(SOURCE/'diagnostic.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert old.prerequisites()[0];gate.predecessors()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=old.GPU,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');cpu=dict(env,CUDA_VISIBLE_DEVICES='')
    for cell in gate.SOLAR:
        sources();gate.source_check()
        if (OUT/'FULL'/cell/'DELIVERABLE_GATE.json').exists():gate.delivered(cell);continue
        if not (gate.AUDIT/cell/'GPU_PREFLIGHT.json').exists():queue.launch([old.LEGACY,'-u',str(PRIOR/'preflight.py'),cell],'SOLAR_HANDOFF12_PREFLIGHT_'+cell,env,'FULL',True)
        gate.preflight(cell)
        if not (SOURCE/'FULL'/cell/'COMPLETE.json').exists():queue.launch([old.LEGACY,'-u',str(PRIOR/'entry.py'),'main',cell],'SOLAR_HANDOFF12_MAIN_'+cell,env,'FULL',True)
        if not (OUT/'FULL'/cell/'CPU_GATE.json').exists():queue.launch([old.PY,str(queue.HERE/'supplement_cpu.py'),'FULL',cell],'SOLAR_HANDOFF12_CPU_'+cell,cpu,'CPU_COMPLETENESS')
        if not (OUT/'FULL'/cell/'INFERENCE_GATE.json').exists():queue.launch([old.LEGACY,'-u',str(PRIOR/'entry.py'),'supplement',cell],'SOLAR_HANDOFF12_DETAIL_'+cell,env,'FULL',True)
        queue.launch([old.PY,str(queue.HERE/'supplement_report.py'),'--cell','FULL',cell],'SOLAR_HANDOFF12_GATE_'+cell,cpu,'CPU_COMPLETENESS');gate.delivered(cell)
    queue.launch([old.PY,str(queue.HERE/'supplement_report.py'),'--phase','FULL'],'SOLAR_HANDOFF12_FULL_GATE',cpu,'CPU_SUMMARY')
    queue.launch([old.PY,str(ORIGINAL/'diagnostic_report.py')],'SOLAR_HANDOFF12_SUMMARY',cpu,'CPU_SUMMARY')
    queue.launch([old.PY,str(queue.HERE/'supplement_report.py'),'--final'],'SOLAR_HANDOFF12_FINAL_GATE',cpu,'CPU_SUMMARY')
    sources();release_training(plan)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        issue=dict(status='FAILED',at=old.now(),controller_pid=os.getpid(),error=repr(e),traceback=traceback.format_exc())
        dump(AUDIT/'FAILED.json',issue);status(**{k:v for k,v in issue.items() if k not in ['controller_pid','at']});raise
