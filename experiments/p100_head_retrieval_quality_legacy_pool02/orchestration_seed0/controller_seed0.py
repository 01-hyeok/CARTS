"""Seed0-only queue; adopts the already running worker without restarting training."""
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from runtime import HERE, ROOT, OUT, GPU_UUID, config, write_json, now, source_hashes, sha256_file
SCOPE_DIR=Path(__file__).resolve().parent
TOTAL=48
CANCELLED=96

def proc_identity(pid):
    path=Path(f'/proc/{pid}')
    try:
        stat=(path/'stat').read_text().split(') ',1)[1].split()
        if stat[0]=='Z':return None
        return dict(pid=pid,start_ticks=stat[19],argv=[x.decode() for x in (path/'cmdline').read_bytes().split(b'\0') if x])
    except (FileNotFoundError,ProcessLookupError):return None

def track_workers():
    workers=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:identity=proc_identity(int(path.name))
        except PermissionError:continue
        if identity and str(HERE/'runtime.py') in identity['argv']:workers.append(identity)
    return workers

def validate_manifest(rows):
    expected=[f'{d}_{h}_{m}_seed0' for d in ('ETTh1','Weather','Solar') for h in (96,192,336,720) for m in ('V0','V1','V2','V5')]
    assert len(rows)==TOTAL and [r['run_id'] for r in rows]==expected
    assert all(r['seed']==0 for r in rows),'Only user-authorized seed0 is allowed'
    return rows

def save_manifest(rows):
    validate_manifest(rows)
    write_json(HERE/'configs/run_manifest.json',rows)
    path=HERE/'configs/run_manifest.csv';tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    os.replace(tmp,path)

def progress(rows):
    completed=[r['run_id'] for r in rows if r['status']=='COMPLETE']
    write_json(OUT/'progress.json',dict(completed=len(completed),total=TOTAL,completed_runs=completed,
        running=sum(r['status']=='RUNNING' for r in rows),failed=sum(r['status']=='FAILED' for r in rows),
        not_started=sum(r['status']=='NOT_STARTED' for r in rows),cancelled_by_user=CANCELLED,active_seeds=[0],updated_at=now()))

def execute(command,label,status):
    if status in ('STAGE1_RUNNING','POOL_GENERATION'):
        assert not track_workers(),'An existing runtime worker must not be duplicated'
        if status=='STAGE1_RUNNING':assert label.endswith('_seed0')
    log=OUT/'logs'/f'{label}.log';log.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPU_UUID,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with log.open('a') as output:
        output.write(f'\nSTART {now()} COMMAND {command!r}\n');output.flush()
        worker=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
        write_json(OUT/'current.json',dict(status=status,target=label,controller_pid=os.getpid(),worker_pid=worker.pid,
            log=str(log),command=command,gpu_uuid=GPU_UUID,physical_gpu_index=1,total=TOTAL,cancelled_by_user=CANCELLED,active_seeds=[0],updated_at=now()))
        print(f'START {label} worker_pid={worker.pid} log={log}',flush=True)
        rc=worker.wait()
    if rc:
        write_json(OUT/'current.json',dict(status='FAILED',target=label,returncode=rc,log=str(log),total=TOTAL,cancelled_by_user=CANCELLED,updated_at=now()))
        raise RuntimeError(f'{label} failed rc={rc}; queue stopped, outputs preserved')

def adopt_existing(rows):
    path=OUT/'SEED0_ADOPTION.json'
    if not path.exists():
        assert not track_workers(),'Unaccounted existing worker'
        return
    adoption=json.loads(path.read_text())
    row=next(r for r in rows if r['run_id']==adoption['run_id'])
    identity=adoption['identity'];pid=identity['pid']
    while True:
        active=proc_identity(pid)
        if active is None:break
        assert active==identity,'Worker PID identity changed; refusing intervention'
        assert track_workers()==[identity],'Unexpected concurrent runtime worker'
        current=dict(adoption['previous_current'],controller_pid=os.getpid(),worker_pid=pid,
            status='STAGE1_RUNNING',adopted_existing_worker=True,total=TOTAL,cancelled_by_user=CANCELLED,
            active_seeds=[0],updated_at=now())
        write_json(OUT/'current.json',current);progress(rows)
        print(f'ADOPTED RUNNING {row["run_id"]} pid={pid}; no restart',flush=True)
        time.sleep(30)
    done=ROOT/row['output_path']/'COMPLETE.json'
    assert done.exists(),'Adopted worker ended without COMPLETE; preserved artifacts, queue remains stopped'
    payload=json.loads(done.read_text())
    assert payload['status']=='COMPLETE' and payload['run_id']==row['run_id']
    assert sha256_file(payload['selected_checkpoint'])==payload['checkpoint_sha256']
    row['status']='COMPLETE';row['blocked_by']=None;save_manifest(rows);progress(rows)
    write_json(OUT/'SEED0_ADOPTION_COMPLETE.json',dict(run_id=row['run_id'],worker_pid=pid,completed_at=now(),worker_restarted=False))
    print('ADOPTION COMPLETE',row['run_id'],flush=True)

def main():
    cfg=config()
    assert cfg['execution']['seed_order']==[0] and cfg['execution']['total_runs']==TOTAL
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU_UUID
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (OUT/'controller.pid').write_text(str(os.getpid())+'\n')
        gate=json.loads((HERE/'reports/RUNTIME_GATES.json').read_text())
        assert gate['status']=='PASS' and gate['source_hashes']==source_hashes()
        rows=validate_manifest(json.loads((HERE/'configs/run_manifest.json').read_text()))
        adopt_existing(rows)
        for row in rows:
            if (OUT/'PAUSE_REQUESTED.json').exists():
                progress(rows)
                write_json(OUT/'current.json',dict(status='PAUSED_AT_RUN_BOUNDARY',next_run=row['run_id'],total=TOTAL,cancelled_by_user=CANCELLED,updated_at=now()))
                return
            cell=f"{row['dataset']}_{row['horizon']}";done=ROOT/row['output_path']/'COMPLETE.json'
            if done.exists():
                payload=json.loads(done.read_text())
                assert payload['status']=='COMPLETE' and payload['run_id']==row['run_id']
                assert sha256_file(payload['selected_checkpoint'])==payload['checkpoint_sha256']
            else:
                if not (OUT/'pools'/cell/'COMPLETE.json').exists():
                    execute([sys.executable,'-u',str(HERE/'runtime.py'),'pool',cell],f'pool_{cell}','POOL_GENERATION')
                if (OUT/'PAUSE_REQUESTED.json').exists():
                    progress(rows);return
                row['status']='RUNNING';row['blocked_by']=None;save_manifest(rows);progress(rows)
                try:execute([sys.executable,'-u',str(HERE/'runtime.py'),'train',row['run_id']],row['run_id'],'STAGE1_RUNNING')
                except Exception:
                    row['status']='FAILED';save_manifest(rows);progress(rows);raise
                assert done.exists()
                execute([sys.executable,'-u',str(SCOPE_DIR/'summarize_seed0.py')],'summary_seed0','CPU_SUMMARY')
            row['status']='COMPLETE';row['blocked_by']=None;save_manifest(rows);progress(rows)
        execute([sys.executable,'-u',str(SCOPE_DIR/'summarize_seed0.py')],'summary_seed0','CPU_SUMMARY')
        write_json(OUT/'current.json',dict(status='COMPLETE',completed=TOTAL,total=TOTAL,cancelled_by_user=CANCELLED,active_seeds=[0],stage2=False,updated_at=now()))
        print('ALL 48 SEED0 STAGE1 RUNS COMPLETE; seeds1/2 cancelled by user',flush=True)

if __name__=='__main__':main()
