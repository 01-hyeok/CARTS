"""Weather terminal gate -> Solar137 GPU preflight -> four ordered data-only cells."""
import os,sys,time,fcntl,traceback,signal
import gate
sys.path.insert(0,str(gate.ROOT/'experiments/retrieval_diagnostic_supplement01'))
import supplement_controller as queue
from supplement_common import read,dump,sha,SOURCE,OUT,ORIGINAL
old=queue.old

def main():
    gate.source_check();gate.AUDIT.mkdir(parents=True,exist_ok=True)
    own=(gate.AUDIT/'controller.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (SOURCE/'00_Audit/RESUME09/COMPLETE.json').exists() or gate.process_alive(42420):
        assert not (SOURCE/'00_Audit/RESUME09/FAILED.json').exists(),'Predecessor failed; no Solar start'
        dump(gate.AUDIT/'STATUS.json',dict(status='WAITING_WEATHER720_FINAL_GATE_AND_CONTROLLER_EXIT',pid=os.getpid(),at=old.now()))
        time.sleep(15)
    previous=gate.predecessors()
    lock=(SOURCE/'diagnostic.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert old.prerequisites()[0]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=old.GPU,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    cpu=dict(env,CUDA_VISIBLE_DEVICES='')
    dump(gate.AUDIT/'HANDOFF.json',dict(status='PASS',controller_pid=os.getpid(),at=old.now(),previous_deliverable_hashes=previous))
    for cell in gate.SOLAR:
        gate.source_check()
        if (OUT/'FULL'/cell/'DELIVERABLE_GATE.json').exists():gate.delivered(cell);continue
        queue.launch([old.LEGACY,'-u',str(gate.HERE/'preflight.py'),cell],'SOLAR_DIAG11_PREFLIGHT_'+cell,env,'FULL',True)
        gate.preflight(cell)
        if not (SOURCE/'FULL'/cell/'COMPLETE.json').exists():
            queue.launch([old.LEGACY,'-u',str(gate.HERE/'entry.py'),'main',cell],'SOLAR_DIAG11_MAIN_'+cell,env,'FULL',True)
        queue.launch([old.PY,str(queue.HERE/'supplement_cpu.py'),'FULL',cell],'SOLAR_DIAG11_CPU_'+cell,cpu,'CPU_COMPLETENESS')
        queue.launch([old.LEGACY,'-u',str(gate.HERE/'entry.py'),'supplement',cell],'SOLAR_DIAG11_DETAIL_'+cell,env,'FULL',True)
        queue.launch([old.PY,str(queue.HERE/'supplement_report.py'),'--cell','FULL',cell],'SOLAR_DIAG11_GATE_'+cell,cpu,'CPU_COMPLETENESS')
        gate.delivered(cell)
    queue.launch([old.PY,str(queue.HERE/'supplement_report.py'),'--phase','FULL'],'SOLAR_DIAG11_FULL_GATE',cpu,'CPU_SUMMARY')
    queue.launch([old.PY,str(ORIGINAL/'diagnostic_report.py')],'SOLAR_DIAG11_SUMMARY',cpu,'CPU_SUMMARY')
    queue.launch([old.PY,str(queue.HERE/'supplement_report.py'),'--final'],'SOLAR_DIAG11_FINAL_GATE',cpu,'CPU_SUMMARY')
    # All artifact gates, not merely status strings, precede any training release.
    old.current(status='DIAGNOSTICS_COMPLETE_POST_TRAINING_GATE_PENDING',missing_full_solar_models=8)
    sys.path.insert(0,str(gate.ROOT/'experiments/solar_full_all_channel04'))
    import prerequisites
    proof=prerequisites.verify(require_release=False)
    obsolete=4088382;p=gate.Path('/proc')/str(obsolete)
    if gate.process_alive(obsolete):
        cmd=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
        children=(p/'task'/str(obsolete)/'children').read_text().split()
        state=read(gate.ROOT/'results/TRACK-SOLAR-FULL-MISSING-CHECKPOINTS01/CONTROLLER_STATUS.json')
        gate.validate_old_waiter(cmd,children,state)
        # Terminate only this verified obsolete childless CPU waiter. SIGCONT
        # delivers its pending SIGTERM if it was stopped; no worker is touched.
        os.kill(obsolete,signal.SIGTERM);os.kill(obsolete,signal.SIGCONT)
        for _ in range(50):
            if not gate.process_alive(obsolete):break
            time.sleep(.1)
        assert not gate.process_alive(obsolete),'Old CPU waiter termination uncertain; no release'
    release=gate.ROOT/'results/TRACK-SOLAR-FULL-MISSING-CHECKPOINTS01/DELIVERABLE_COMPLETENESS_RELEASE.json'
    assert not release.exists(),'Preserve any existing release; investigate before replacement'
    dump(release,dict(status='RELEASED_AFTER_EXPANDED_FINAL_GATE',pid=obsolete,gate_sha256=proof['final_gate_sha256'],all_prior_deliverables_complete=True))
    dump(gate.AUDIT/'COMPLETE.json',dict(status='PASS',at=old.now(),full_cells=12,training_started_here=False,solar_training_waiter=4182673,prerequisites=proof))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        issue=dict(status='FAILED',at=old.now(),controller_pid=os.getpid(),error=repr(e),traceback=traceback.format_exc())
        dump(gate.AUDIT/'FAILED.json',issue);dump(SOURCE/'00_Audit/IMMEDIATE_ISSUE.json',issue)
        old.current(status='FAILED',error=repr(e),failed_record=str(gate.AUDIT/'FAILED.json'));raise
