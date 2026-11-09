"""CPU watchdog waits for an existing worker to finish, probes, then resumes.

Suspends ONLY the verified controller while it is blocked in waitpid. Its live
worker is never signalled. No replacement controller or production patch here.
The independent watchdog resumes the original controller on helper failure.
"""
import argparse
import fcntl
import json
import os
import signal
import shlex
import subprocess
import sys
import time
from pathlib import Path
from cache_store import atomic_json,file_sha

ROOT=Path('/data/pjh_workspace/CARTS')
BASE=ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02'
CONTROLLER=ROOT/'experiments/retrieval_failure_mechanism02/controller.py'
WORKER=ROOT/'experiments/retrieval_failure_mechanism02/worker.py'
UUID='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'


def process(pid):
    p=Path('/proc')/str(pid)
    try:
        stat=(p/'stat').read_text();tail=stat[stat.rfind(')')+2:].split()
        return dict(pid=pid,start=tail[19],state=tail[0],ppid=int(tail[1]),exit_code=int(tail[49]),
            cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),
            wchan=(p/'wchan').read_text().strip(),children=(p/'task'/str(pid)/'children').read_text().split())
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None


def same_process(expected):
    actual=process(expected['pid'])
    return actual if actual and actual['start']==expected['start'] and (actual['cmd']==expected['cmd'] or actual['state']=='Z') else None


def validate_pair(owner,worker,status):
    assert str(CONTROLLER) in owner['cmd'] and '--mode all' in owner['cmd']
    assert owner['wchan']=='do_wait' and owner['state']=='S'
    assert owner['children']==[str(worker['pid'])] and worker['ppid']==owner['pid']
    assert str(WORKER) in worker['cmd'] and '--mode analyze' in worker['cmd']
    assert shlex.split(worker['cmd'])==status['command']
    assert status['controller_pid']==owner['pid'] and status['worker_pid']==worker['pid']
    assert status['phase']=='MECHANISM_DIAGNOSTICS'


def resume(owner):
    actual=same_process(owner)
    if actual and actual['state'] in ('T','t'):
        os.kill(owner['pid'],signal.SIGCONT)
        return True
    return False


def owned_probes(out):
    matches=[]
    script=str(Path(__file__).with_name('gpu_oracle_benchmark.py'))
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        info=process(int(p.name))
        if not info or info['state']=='Z':continue
        args=shlex.split(info['cmd'])
        if script in args and '--out' in args:
            target=args[args.index('--out')+1]
            if str(Path(target).parent)==str(Path(out)):matches.append(info)
    return matches


def safe_resume(owner,out):
    # Even if the coordinator dies during a probe, never resume production
    # alongside a still-running owned GPU benchmark. No process is killed.
    while owned_probes(out):
        atomic_json(Path(out)/'WAIT_OWNED_PROBE_EXIT.json',dict(at=time.time(),reason='prevent third GPU job'))
        time.sleep(2)
    return resume(owner)


def watchdog(out):
    out=Path(out);proof=json.loads((out/'OWNER.json').read_text());owner=proof['controller']
    atomic_json(out/'WATCHDOG_READY.json',dict(pid=os.getpid(),at=time.time()))
    while not (out/'FINISHED.json').exists():
        beat=out/'HEARTBEAT.json'
        stale=not beat.exists() or time.time()-beat.stat().st_mtime>30
        if stale or time.time()>proof['absolute_deadline']:
            did=safe_resume(owner,out);atomic_json(out/'WATCHDOG_RESUME.json',dict(resumed=did,at=time.time(),reason='heartbeat/deadline'))
            return
        time.sleep(2)


def verify_finished_worker(worker,target):
    assert worker['state']=='Z','waitpid owner paused; child must finish normally before probe'
    assert worker['exit_code']==0,'original worker did not exit successfully'
    cell,model,support=target.rsplit('_',2)
    root=BASE/'checkpoint_runs'/cell/(model+'_'+support)
    core=json.loads((root/'CORE_COMPLETE.json').read_text())
    assert core['status']=='CORE_COMPLETE_NOT_ALL_PHASES' and core['files']
    for split,n in core['query_counts'].items():
        g=json.loads((root/split/'SPLIT_GATE.json').read_text())
        assert g['status']=='PASS' and g['queries']==n and g['units']==n*core['channels']
    for name,h in core['files'].items():assert file_sha(root/name)==h
    return dict(path=str(root/'CORE_COMPLETE.json'),sha256=file_sha(root/'CORE_COMPLETE.json'),
                query_counts=core['query_counts'],channels=core['channels'])


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    lock=(BASE/'00_Audit/cache14_boundary_probe.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=json.loads((BASE/'00_Audit/CONTROLLER_STATUS.json').read_text())
    owner=process(status['controller_pid']);child=process(status['worker_pid']);validate_pair(owner,child,status)
    assert status['target']=='Weather_192_V0_P100','only explicitly inspected current boundary'
    assert not (BASE/'checkpoint_runs/Weather_192/V0_P100/CORE_COMPLETE.json').exists()
    code_hashes={p.name:file_sha(p) for p in (ROOT/'experiments/retrieval_failure_mechanism02').glob('*.py')}
    atomic_json(out/'OWNER.json',dict(controller=owner,worker=child,status=status,live_source_hashes=code_hashes,absolute_deadline=time.time()+3*3600))
    atomic_json(out/'HEARTBEAT.json',dict(at=time.time(),phase='PREPARING'))
    wd=subprocess.Popen([sys.executable,str(Path(__file__)), '--watchdog',str(out)],start_new_session=True,
        stdout=(out/'watchdog.log').open('a'),stderr=subprocess.STDOUT)
    deadline=time.time()+10
    while not (out/'WATCHDOG_READY.json').exists():
        assert time.time()<deadline and wd.poll() is None;time.sleep(.1)
    suspended=False
    try:
        validate_pair(same_process(owner),same_process(child),json.loads((BASE/'00_Audit/CONTROLLER_STATUS.json').read_text()))
        os.kill(owner['pid'],signal.SIGSTOP);suspended=True
        time.sleep(.1)
        stopped=same_process(owner);running=same_process(child)
        assert stopped and stopped['state'] in ('T','t') and stopped['children']==[str(child['pid'])]
        assert running and running['state'] not in ('T','t','Z')
        atomic_json(out/'STATE.json',dict(status='CONTROLLER_DISPATCH_HELD_WORKER_UNTOUCHED',controller_pid=owner['pid'],worker_pid=child['pid'],at=time.time()))
        while True:
            atomic_json(out/'HEARTBEAT.json',dict(at=time.time(),phase='WAIT_CURRENT_WORKER'))
            assert not (out/'WATCHDOG_RESUME.json').exists(),'watchdog already resumed queue'
            assert same_process(owner)['state'] in ('T','t')
            running=same_process(child)
            assert running is not None,'unexpected worker disappearance'
            if running['state']=='Z':break
            assert time.time()<json.loads((out/'OWNER.json').read_text())['absolute_deadline'];time.sleep(2)
        # Keep heartbeat alive while potentially large immutable artifacts hash.
        # Full validation runs in a separate CPU child, not the GPU benchmark.
        verifier=subprocess.Popen([sys.executable,str(Path(__file__)),'--verify',str(out)],stdout=(out/'verify.log').open('w'),stderr=subprocess.STDOUT)
        while verifier.poll() is None:
            atomic_json(out/'HEARTBEAT.json',dict(at=time.time(),phase='VERIFY_COMPLETED_CORE'));time.sleep(2)
        assert verifier.returncode==0,'original worker not complete; resume controller without probe'
        assert code_hashes=={p.name:file_sha(p) for p in (ROOT/'experiments/retrieval_failure_mechanism02').glob('*.py')}
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        for model in ['V1','V5']:
            log=(out/(model+'.log')).open('w')
            p=subprocess.Popen(['/data/pjh_workspace/ts-env/bin/python',str(Path(__file__).with_name('gpu_oracle_benchmark.py')),
                '--cell','Weather_192','--model',model,'--support','P100','--out',str(out/model)],env=env,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
            atomic_json(out/'STATE.json',dict(status='GPU_PROBE_ONLY_NO_PRODUCTION_PATCH',probe_pid=p.pid,model=model,at=time.time()))
            begin=time.time()
            while p.poll() is None:
                atomic_json(out/'HEARTBEAT.json',dict(at=time.time(),phase='GPU_PROBE',probe_pid=p.pid))
                # Do not resume while own GPU child is alive. Normal bounded
                # benchmark raises after its completed block if over budget.
                if time.time()-begin>300:
                    atomic_json(out/'PROBE_SLOW.json',dict(at=time.time(),pid=p.pid,action='await safe process exit; no concurrent dispatch'))
                time.sleep(2)
            log.close()
            assert p.returncode==0,'GPU probe failed; keep original production path'
        atomic_json(out/'PROBES_COMPLETE.json',dict(status='GPU_PROBES_FINISHED_NOT_APPLIED',at=time.time()))
    except BaseException as e:
        import traceback
        atomic_json(out/'FAILED.json',dict(error=repr(e),traceback=traceback.format_exc(),at=time.time()))
    finally:
        if suspended:safe_resume(owner,out)
        atomic_json(out/'FINISHED.json',dict(status='ORIGINAL_CONTROLLER_RESUMED_NO_CODE_CHANGES',at=time.time(),controller_pid=owner['pid']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out');p.add_argument('--watchdog');p.add_argument('--verify');a=p.parse_args()
    if a.watchdog:watchdog(a.watchdog)
    elif a.verify:
        out=Path(a.verify);proof=json.loads((out/'OWNER.json').read_text())
        atomic_json(out/'CURRENT_CORE_VERIFIED.json',verify_finished_worker(same_process(proof['worker']),proof['status']['target']))
    else:run(a.out)
