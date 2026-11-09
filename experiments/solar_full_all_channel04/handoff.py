"""Replace only the known obsolete CPU waiter, after its owner has exited."""
import os,signal,time
from pathlib import Path
import prerequisites as gate

OLD_SUFFIX='/experiments/solar_full_missing_checkpoints01/solar_full_controller.py'
def command(pid):
    p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().decode().split('\0') if p.exists() else []
def process_exists(pid):
    p=Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().rsplit(')',1)[1].split()[0]!='Z'
def validate_target(cmd,children):
    assert any(x.endswith(OLD_SUFFIX) for x in cmd),'Refuse to touch any unrelated process'
    assert not children,'Old waiter has a child; stop and investigate instead of terminating work'

def retire_old_waiter(root=gate.ROOT):
    proof=gate.verify(root);root=Path(root)
    state=gate.read(root/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/CONTROLLER_STATUS.json')
    owner=state.get('controller_pid')
    if owner and process_exists(owner):return None # running Full controller is never interrupted
    rel=gate.read(root/'results/TRACK-SOLAR-FULL-MISSING-CHECKPOINTS01/DELIVERABLE_COMPLETENESS_RELEASE.json');pid=int(rel['pid'])
    cmd=command(pid)
    if not cmd:return dict(status='OLD_WAITER_ALREADY_EXITED',pid=pid,prerequisite=proof)
    children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split();validate_target(cmd,children)
    oldstate=gate.read(root/'results/TRACK-SOLAR-FULL-MISSING-CHECKPOINTS01/CONTROLLER_STATUS.json')
    assert oldstate['status']=='WAITING_ALL_PRIOR_DIAGNOSTICS' and not oldstate.get('actual_training_started',False)
    # This obsolete waiter cannot pass the new completion status. Its childless
    # waiting state is independently checked after release, before termination.
    os.kill(pid,signal.SIGTERM)
    for _ in range(20):
        if not command(pid):break
        time.sleep(.25)
    assert not command(pid),'Waiter termination unconfirmed; do not acquire GPU'
    return dict(status='RETIRED_OBSOLETE_CPU_WAITER',pid=pid,command=cmd,prerequisite=proof)
