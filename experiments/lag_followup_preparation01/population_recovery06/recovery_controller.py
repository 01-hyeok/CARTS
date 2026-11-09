"""Durable final missing-cell recovery. Old failures and all complete cells survive."""
import sys,os,json,fcntl,subprocess,traceback,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from d56_aggregate import sha,dump,now
ROOT=Path('/data/pjh_workspace/CARTS');BASE=ROOT/'results/TRACK-HEAD-SHIFT-DIAGNOSTIC01';POP=BASE/'full_population_completion';OUT=POP/'population_recovery06'
GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec';PY='/data/pjh_workspace/ts-env/bin/python'

def state(s,**kw):dump(OUT/'RECOVERY_CURRENT.json',dict(status=s,at=now(),pid=os.getpid(),**kw))

def verified(p,status):
    g=json.loads(p.read_text());assert g['status']==status
    for n,h in g['files'].items():assert sha(p.parent/n)==h
    return g

def main():
    own=(OUT/'recovery.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    boundary=(BASE/'00_Audit/d4_controller.lock').open('a');fcntl.flock(boundary,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ready=json.loads((OUT/'CPU_READY.json').read_text());assert ready['status']=='PASS'
    for n,h in ready['source_hashes'].items():assert sha(HERE/n)==h
    g=json.loads((OUT/'VALIDATION_GATE.json').read_text());assert g['status']=='PASS_ALL_QUERY_NATIVE_AND_CONTROLLED_ORDERED_IDS'
    for n,h in g['patch']['source_hashes'].items():assert sha(HERE/n)==h
    assert sha(HERE/'validate.py')==g['source_sha256']
    assert GPU not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    marker=OUT/'PRESERVATION.json';preserved=OUT/'preserved_failure'
    if not marker.exists():
        rows=[]
        for folder in ['smoke','raw','analysis']:
            p=POP/folder/'ETTh1_96/V5_FULL';target=preserved/folder/'ETTh1_96/V5_FULL'
            if not p.exists():continue
            assert POP.resolve() in p.resolve().parents and not any(p.glob('*GATE.json'))
            target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists();p.rename(target)
            rows.append(dict(original=str(p),preserved=str(target)))
        preserved.mkdir(parents=True,exist_ok=True)
        for name in ['CONTROLLER_STATUS.json','PROGRESS.json']:
            p=POP/name
            if p.exists():shutil.copyfile(p,preserved/name)
        dump(marker,dict(at=now(),rows=rows,completed_cells_modified=False))
    stages=[('SMOKE',['/usr/bin/python3','-u',str(HERE/'recover_run.py'),'--smoke'],POP/'smoke/ETTh1_96/V5_FULL/GATE.json','PASS_FIRST_TAIL_FULL_POPULATION_SMOKE',True),
            ('FULL_QUERY_INFERENCE',['/usr/bin/python3','-u',str(HERE/'recover_run.py')],POP/'raw/ETTh1_96/V5_FULL/GATE.json','PASS_FULL_QUERY_D123_RAW',True),
            ('CPU_AGGREGATION',[PY,'-u',str(HERE.parent/'d123_full_aggregate.py'),'ETTh1_96','FULL'],POP/'analysis/ETTh1_96/V5_FULL/DELIVERABLE_GATE.json','PASS_FULL_QUERY_D1_D2_D3',False)]
    for phase,cmd,gate,status,gpu in stages:
        if gate.exists():verified(gate,status);continue
        log=OUT/(phase+'.log')
        with log.open('a') as f:
            p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU if gpu else '',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            state('RUNNING',phase=phase,worker_pid=p.pid,command=cmd,log=str(log));rc=p.wait()
        assert rc==0,(phase,rc);verified(gate,status)
    paths=list((POP/'analysis').glob('*/*/DELIVERABLE_GATE.json'));assert len(paths)==16
    for p in paths:verified(p,'PASS_FULL_QUERY_D1_D2_D3')
    dump(OUT/'RESOLUTION_GATE.json',dict(status='PASS_MISSING_CELL_RECOVERED_ALL_16_VERIFIED',at=now(),validation_sha256=sha(OUT/'VALIDATION_GATE.json'),
        deliverable_sha256=sha(POP/'analysis/ETTh1_96/V5_FULL/DELIVERABLE_GATE.json'),completed_cells=16,original_failures_preserved=True))
    dump(POP/'PROGRESS.json',dict(at=now(),complete=16,expected=16,failed=[],resolution_manifest=str(OUT/'RESOLUTION_GATE.json')))
    dump(POP/'CONTROLLER_STATUS.json',dict(status='FULL_QUERY_D123_COMPLETE',at=now(),complete=16,failed=[],resolution_manifest=str(OUT/'RESOLUTION_GATE.json')))
    state('PASS_MISSING_CELL_RECOVERED_ALL_16_VERIFIED',finalizer='Existing CPU finalizer continues independently; not yet asserted COMPLETE')

if __name__=='__main__':
    try:main()
    except BaseException as e:state('BLOCKED',error=repr(e),traceback=traceback.format_exc());raise
