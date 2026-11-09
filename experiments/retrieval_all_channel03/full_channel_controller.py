"""P100-gated Full resume; preserve canonical model-specific inference runtimes."""
import sys,os,fcntl,signal,traceback
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments/retrieval_diagnostic_supplement01'),str(ROOT/'experiments/retrieval_full_precision_repair02')]
import supplement_controller as queue
from supplement_common import read,dump,sha,CELLS,SOURCE,OUT,ORIGINAL
import diagnostic_controller as old
from full_precision_policy import uses_score_cache
def hashes():return {p.name:sha(p) for p in HERE.glob('*.py')}
def main():
    gate=read(HERE/'READY_GATE.json');assert gate['status']=='PASS' and gate['source_hashes']==hashes()
    assert read(OUT/'P100_DELIVERABLE_GATE.json')['status']=='PASS'
    assert read(SOURCE/'00_Audit/ALL_CHANNEL03/VALIDATION_QUEUE_V3_CURRENT.json')['status']=='PASS_GPU_VALIDATIONS'
    assert old.prerequisites()[0]
    for c in CELLS:
        a=read(SOURCE/'P100'/c/'COMPLETE.json');g=read(OUT/'P100'/c/'DELIVERABLE_GATE.json')
        assert g['status']=='PASS' and g['source_complete_sha256']==sha(SOURCE/'P100'/c/'COMPLETE.json')
        for m in a['models']:assert sha(m['checkpoint'])==m['sha256']
    lock=(SOURCE/'diagnostic.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES=old.GPU)
    cpu=dict(env,CUDA_VISIBLE_DEVICES='')
    for c in CELLS:
        original=SOURCE/'FULL'/c/'COMPLETE.json'
        if original.exists():
            for m in read(original)['models']:assert sha(m['checkpoint'])==m['sha256']
            delivered=OUT/'FULL'/c/'DELIVERABLE_GATE.json'
            if delivered.exists():
                d=read(delivered);assert d['status']=='PASS' and d['source_complete_sha256']==sha(original)
                continue
        else:
            if not c.startswith('Solar_') and uses_score_cache(c):
                queue.launch([old.PY,'-u',str(HERE/'full_channel_v0_export.py'),c],f'FULL_V0_CANONICAL_{c}',env,'FULL',True)
            queue.launch([old.LEGACY,'-u',str(HERE/'full_channel_run.py'),'FULL',c],f'FULL_{c}',env,'FULL',True)
        queue.launch([old.PY,str(queue.HERE/'supplement_cpu.py'),'FULL',c],f'SUPPLEMENT_CPU_FULL_{c}',cpu,'CPU_COMPLETENESS')
        queue.launch([old.LEGACY,'-u',str(HERE/'full_channel_supplement.py'),'FULL',c],f'SUPPLEMENT_FULL_{c}',env,'FULL',True)
        queue.launch([old.PY,str(queue.HERE/'supplement_report.py'),'--cell','FULL',c],f'DELIVERABLE_FULL_{c}',cpu,'CPU_COMPLETENESS')
    queue.launch([old.PY,str(queue.HERE/'supplement_report.py'),'--phase','FULL'],'FULL_DELIVERABLE_GATE',cpu,'CPU_COMPLETENESS')
    queue.launch([old.PY,str(ORIGINAL/'diagnostic_report.py')],'final_summary',cpu,'CPU_SUMMARY')
    queue.launch([old.PY,str(queue.HERE/'supplement_report.py'),'--final'],'expanded_final_summary',cpu,'CPU_SUMMARY')
    assert read(OUT/'FINAL_DELIVERABLE_GATE.json')['status']=='PASS'
    post=read(ORIGINAL/'POST_DIAGNOSTIC_SOLAR_TRAINING_QUEUE.json')
    post.update(status='ALL_DELIVERABLES_COMPLETE_SOLAR_GPU_SMOKE_PENDING',deliverable_gate=str(OUT/'FINAL_DELIVERABLE_GATE.json'),actual_training_started=False)
    dump(ORIGINAL/'POST_DIAGNOSTIC_SOLAR_TRAINING_QUEUE.json',post)
    solar=ROOT/'results/TRACK-SOLAR-FULL-MISSING-CHECKPOINTS01';pid=4088382
    assert any(x.endswith('/solar_full_missing_checkpoints01/solar_full_controller.py') for x in queue.command(pid)) and queue.process_state(pid)=='T'
    dump(solar/'DELIVERABLE_COMPLETENESS_RELEASE.json',dict(status='RELEASED_AFTER_EXPANDED_FINAL_GATE',pid=pid,gate_sha256=sha(OUT/'FINAL_DELIVERABLE_GATE.json'),all_prior_deliverables_complete=True))
    os.kill(pid,signal.SIGCONT)
    old.current(status='DIAGNOSTICS_COMPLETE_SOLAR_WAITER_RELEASED',solar_waiter_pid=pid)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        issue=dict(status='FAILED',at=old.now(),controller_pid=os.getpid(),error=repr(e),traceback=traceback.format_exc())
        dump(SOURCE/'00_Audit/IMMEDIATE_ISSUE.json',issue);dump(OUT/'IMMEDIATE_ISSUE.json',issue);old.current(status='FAILED',error=repr(e),traceback=traceback.format_exc());raise
