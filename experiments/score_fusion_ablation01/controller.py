"""One durable worker, strict cell order, no training, immutable old artifacts."""
import os,sys,json,subprocess,hashlib,fcntl,datetime,traceback,time
from pathlib import Path
from fusion_config import ROOT,HERE,OUT,GPU,CELLS,PROTOCOL

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(p)
def valid(path):
    if not path.exists():return False
    g=json.loads(path.read_text());assert g['status']=='PASS_COMPLETE_ALL_QUERY_SCORE_FUSION'
    assert g['e0_ordered_id_mismatches']==0
    for n,h in g['files'].items():assert sha(path.parent/n)==h
    for n,h in g['identity']['source_hashes'].items():assert sha(ROOT/n)==h
    return True
def main():
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ready=json.loads((OUT/'CPU_READY.json').read_text());assert ready['status']=='PASS'
    for n,h in ready['source_hashes'].items():assert sha(HERE/n)==h
    jobs=[(c,s) for c in CELLS for s in ('FULL','P100')]
    dump(OUT/'EXECUTION_QUEUE.json',dict(jobs=jobs,evaluation_supports=['FULL','P100'],total_combinations=32,protocol=PROTOCOL,source_hashes=ready['source_hashes']))
    for cell,training in jobs:
        dest=OUT/'checkpoints'/cell/('V5_'+training)
        if not valid(dest/'COMPLETE.json'):
            py='/usr/bin/python3' if training=='FULL' else '/data/pjh_workspace/ts-env/bin/python'
            cmd=[py,'-u',str(HERE/'evaluate.py'),cell,training]
            log=OUT/'logs'/(cell+'_'+training+'_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S')+'.log');log.parent.mkdir(exist_ok=True)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            with log.open('w') as stream:
                p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                dump(OUT/'CONTROLLER_STATUS.json',dict(status='RUNNING',cell=cell,training_support=training,worker_pid=p.pid,controller_pid=os.getpid(),at=now(),command=cmd,log=str(log)))
                rc=p.wait()
            assert rc==0,dict(cell=cell,training=training,returncode=rc,log=str(log))
            assert valid(dest/'COMPLETE.json')
        rc=subprocess.call(['/data/pjh_workspace/ts-env/bin/python',str(HERE/'summarize.py')],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        assert rc==0,'summary failed; do not mark complete'
    final=json.loads((OUT/'FINAL_GATE.json').read_text());assert final['status']=='PASS_ALL_32_COMBINATIONS'
    dump(OUT/'CONTROLLER_STATUS.json',dict(status='COMPLETE',at=now(),pid=os.getpid(),checkpoints=16,evaluation_combinations=32,final_gate_sha256=sha(OUT/'FINAL_GATE.json')))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        dump(OUT/'CONTROLLER_STATUS.json',dict(status='BLOCKED',at=now(),pid=os.getpid(),error=repr(e),traceback=traceback.format_exc(),existing_results_preserved=True));raise
