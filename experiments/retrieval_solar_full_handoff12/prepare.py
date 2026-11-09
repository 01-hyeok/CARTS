"""CPU-only preflight; never changes a running process or releases training."""
import json,os,sys,xml.etree.ElementTree as ET
from pathlib import Path
import held_gate as h
import controller as c

def main():
    assert not (c.HERE/'PLAN.json').exists(),'Preserve the existing plan'
    tests=ET.parse(c.AUDIT/'pytest.xml').getroot()
    suites=list(tests.iter('testsuite'))
    assert sum(int(s.get('tests','0')) for s in suites)>=39
    assert all(int(s.get(k,'0'))==0 for s in suites for k in ('failures','errors','skipped'))
    c.gate.source_check();c.gate.predecessors()
    expected=h.expected_from_p100(c.ROOT);checked=[];pending=[];hashes={}
    for support in ('P100','FULL'):
        for cell in h.CELLS:
            p=c.OUT/support/cell/'DELIVERABLE_GATE.json'
            if not p.exists():pending.append(support+'/'+cell);continue
            hashes.update(h.verify_cell(c.ROOT,support,cell,expected[cell]));checked.append(support+'/'+cell)
    assert pending==['FULL/Solar_192','FULL/Solar_336','FULL/Solar_720']
    legacy=c.ROOT/'results/TRACK-SOLAR-FULL-MISSING-CHECKPOINTS01'
    hold=h.read(legacy/'DELIVERABLE_COMPLETENESS_HOLD.json')
    assert hold==h.read(legacy/'CONTROLLER_STATUS.json') and hold['status']==h.HELD and not hold['actual_training_started']
    assert not (legacy/'DELIVERABLE_COMPLETENESS_RELEASE.json').exists()
    assert not (c.OUT/'FINAL_DELIVERABLE_GATE.json').exists()
    before={name:c.snapshot(pid) for name,pid in [('controller',54557),('worker',70152),('held_waiter',4088382),('replacement',4182673)]}
    assert all(before.values())
    assert before['controller']['state']=='S' and before['held_waiter']['state']=='T'
    assert before['worker']['command'][-2:]==['main','Solar_192']
    assert (Path('/proc/54557/task/54557/children')).read_text().split()==['70152']
    for pid in (4088382,4182673):assert not Path(f'/proc/{pid}/task/{pid}/children').read_text().strip()
    replacement_state=h.read(c.ROOT/'results/TRACK-SOLAR-FULL-ALL-CHANNEL04/CONTROLLER_STATUS.json')
    assert replacement_state['status']=='WAITING_ALL_PRIOR_DIAGNOSTICS' and not replacement_state['actual_training_started']
    # Demonstrate the old predicate rejects this real held state; no signaling.
    try:c.gate.validate_old_waiter(before['held_waiter']['command'],[],hold)
    except AssertionError:old_rejection=True
    else:raise AssertionError('The old mismatch no longer exists; review instead of replacing')
    try:h.verify_all(c.ROOT,expected)
    except (AssertionError,FileNotFoundError,KeyError) as exc:blocked=repr(exc)
    else:raise AssertionError('Unexpected complete global gate; inspect before activation')
    plan=dict(cell='Solar_192',expected=expected,hold_sha256=h.sha(legacy/'DELIVERABLE_COMPLETENESS_HOLD.json'),verified_existing_artifact_hashes=hashes,**before)
    c.dump(c.HERE/'PLAN.json',plan)
    sources=dict(c.gate.source_check()['sources'])
    for p in c.HERE.glob('*.py'):sources[str(p.relative_to(c.ROOT))]=h.sha(p)
    for p in (c.ROOT/'experiments/solar_full_all_channel04').glob('*.py'):sources[str(p.relative_to(c.ROOT))]=h.sha(p)
    c.dump(c.AUDIT/'PREACTIVATION_AUDIT.json',dict(status='PASS',at=c.old.now(),existing_cells_verified=checked,pending_cells=pending,old_status_predicate_rejection_reproduced=old_rejection,global_gate_currently_blocked=blocked,all_prior_outputs_preserved=True,training_released=False,processes_before=before))
    c.dump(c.HERE/'CPU_READY.json',dict(status='PASS',at=c.old.now(),sources=sources,plan_sha256=h.sha(c.HERE/'PLAN.json'),tests=39,existing_cells_verified=len(checked),training_gate_still_blocked=True))
    print(json.dumps(dict(status='CPU_READY',verified_cells=len(checked),pending=pending,global_gate_blocked=blocked)))

if __name__=='__main__':main()
