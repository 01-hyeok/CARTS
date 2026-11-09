"""One durable sequential process, one GPU, no Stage2, resumable epoch checkpoints."""
import fcntl
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from runtime import HERE, ROOT, OUT, GPU_UUID, config, write_json, now, source_hashes, sha256_file

def save_manifest(rows):
    write_json(HERE/'configs/run_manifest.json',rows)
    path=HERE/'configs/run_manifest.csv';tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    os.replace(tmp,path)

def execute(command,label,status):
    log=OUT/'logs'/f'{label}.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=GPU_UUID,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with log.open('a') as output:
        output.write(f'\nSTART {now()} COMMAND {command!r}\n');output.flush()
        worker=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
        write_json(OUT/'current.json',dict(status=status,target=label,controller_pid=os.getpid(),worker_pid=worker.pid,
                   log=str(log),command=command,gpu_uuid=GPU_UUID,physical_gpu_index=1,updated_at=now()))
        print(f'START {label} worker_pid={worker.pid} log={log}',flush=True)
        rc=worker.wait()
    if rc:
        write_json(OUT/'current.json',dict(status='FAILED',target=label,returncode=rc,log=str(log),updated_at=now()))
        raise RuntimeError(f'{label} failed rc={rc}; partial results preserved; queue stopped')

def main():
    cfg=config()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU_UUID
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (OUT/'controller.pid').write_text(str(os.getpid())+'\n')
    gates=json.loads((HERE/'reports/RUNTIME_GATES.json').read_text())
    assert gates['status']=='PASS' and gates['source_hashes']==source_hashes()
    rows=json.loads((HERE/'configs/run_manifest.json').read_text())
    assert len(rows)==144 and [r['seed'] for r in rows]==[0]*48+[1]*48+[2]*48
    completed=[]
    for row in rows:
        cell=f"{row['dataset']}_{row['horizon']}"
        done=ROOT/row['output_path']/'COMPLETE.json'
        if done.exists():
            payload=json.loads(done.read_text())
            assert payload['status']=='COMPLETE' and payload['run_id']==row['run_id']
            assert sha256_file(payload['selected_checkpoint'])==payload['checkpoint_sha256']
        else:
            if not (OUT/'pools'/cell/'COMPLETE.json').exists():
                execute([sys.executable,'-u',str(HERE/'runtime.py'),'pool',cell],f'pool_{cell}','POOL_GENERATION')
            row['status']='RUNNING';row['blocked_by']=None;save_manifest(rows)
            try:
                execute([sys.executable,'-u',str(HERE/'runtime.py'),'train',row['run_id']],row['run_id'],'STAGE1_RUNNING')
            except Exception:
                row['status']='FAILED';save_manifest(rows)
                raise
            assert done.exists()
        row['status']='COMPLETE';row['blocked_by']=None;save_manifest(rows)
        completed.append(row['run_id'])
        write_json(OUT/'progress.json',dict(completed=len(completed),total=144,completed_runs=completed,updated_at=now()))
        execute([sys.executable,'-u',str(HERE/'summarize.py')], 'summary', 'CPU_SUMMARY')
    write_json(OUT/'current.json',dict(status='COMPLETE',completed=144,total=144,stage2=False,updated_at=now()))
    print('ALL 144 STAGE1 RUNS COMPLETE',flush=True)

if __name__=='__main__':
    main()
