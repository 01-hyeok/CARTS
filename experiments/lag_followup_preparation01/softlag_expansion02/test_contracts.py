import json,hashlib
from pathlib import Path
import pytest
from contracts import *

def test_scope_exact():
    assert len(JOBS)==len(set(JOBS))==16 and len(NEW)==12 and len(REUSE)==4
    assert set(NEW).isdisjoint(REUSE)
    assert all((d+'_'+str(h),s) in JOBS for d in ['ETTh1','Weather'] for h in [96,192,336,720] for s in ['P100','FULL'])
@pytest.mark.parametrize('controller,worker,phase',[('d56_controller_v2.py','d56_complete_run.py','FEATURES'),('d123_completion_queue.py','d123_full_population.py','FULL_QUERY_INFERENCE')])
def test_exact_authorized_owner(controller,worker,phase):
    assert validate_other(dict(status='RUNNING',pid=1,worker_pid=2,phase=phase),str(HERE/worker),str(HERE/controller))
@pytest.mark.parametrize('bad',[dict(status='WAITING',pid=1,worker_pid=2,phase='FEATURES'),dict(status='RUNNING',pid=1,worker_pid=2,phase='CPU_AGGREGATION'),dict(status='RUNNING',pid=1,phase='FEATURES')])
def test_reject_stale_or_cpu_owners(bad):
    with pytest.raises(AssertionError):validate_other(bad,str(HERE/'d56_complete_run.py'),str(HERE/'d56_controller_v2.py'))
def test_reject_unrelated_worker():
    with pytest.raises(AssertionError):validate_other(dict(status='RUNNING',pid=1,worker_pid=2,phase='FEATURES'),'/tmp/other.py',str(HERE/'d56_controller_v2.py'))
def test_tampered_gate_rejected(tmp_path):
    p=tmp_path/'x.csv';p.write_text('original');dump(tmp_path/'GATE.json',dict(status='PASS',files={'x.csv':sha(p)}))
    assert checked_gate(tmp_path/'GATE.json','PASS')
    p.write_text('tampered')
    with pytest.raises(AssertionError):checked_gate(tmp_path/'GATE.json','PASS')
def test_old_scope_cannot_launch_new_worker():
    text=(Path(__file__).parent/'softlag_run.py').read_text()
    assert 'assert (cell,support) in NEW' in text
    assert 'set_per_process_memory_fraction(.40' in text
    assert "before['free_mib']>=24576" in text
