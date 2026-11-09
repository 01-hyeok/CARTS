"""Load frozen selection without fitting; write evaluation in a new directory."""
import sys,os,time,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from experiment_common import *
import run_experiment as main
from train_models import Features,Arm
import torch

def restored(dest):
    source=OUT/'FULL/ETTh1_96'
    ex=main.Experiment.__new__(main.Experiment)
    ex.cell='ETTh1_96';ex.support='FULL';ex.dest=dest
    ex.d,ex.f,ex.prior,ex.identity,ex.before=main.ev.setup(ex.cell,ex.support)
    ex.features=Features(ex.d,OUT/'features'/ex.cell)
    ex.cache={};ex.arms={};ex.opts={};ex.history=[];ex.steps=0
    ex.started=time.perf_counter();ex.raw_time=0;ex.scoring_time=0
    selected=json.loads((source/'SELECTION.json').read_text());state=selected['arms']
    checkpoints=[]
    for name in ARMS:
        p=source/'checkpoints'/name/('epoch%02d.pt'%state[name]['best_epoch'])
        z=load(p);assert z['epoch']==state[name]['best_epoch']
        arm=Arm(name).to(ex.d.device);arm.load_state_dict(z['model']);arm.eval()
        ex.arms[name]=arm;checkpoints.append(dict(arm=name,path=str(p),sha256=sha(p)))
    return ex,state,checkpoints

def run():
    import compat;compat.install()
    root=OUT/'recovery01';source=OUT/'FULL/ETTh1_96';dest=root/'evaluation/ETTh1_96_FULL'
    assert not dest.exists(),'No overwrite: inspect partial recovery explicitly'
    dest.mkdir(parents=True)
    gate=json.loads((root/'COMPATIBILITY_GATE.json').read_text());assert gate['status']=='PASS'
    assert gate['compat_sha256']==sha(HERE/'compat.py')
    manifest=json.loads((root/'PRESERVATION_MANIFEST.json').read_text())
    for path,h in manifest['files'].items():assert sha(ROOT/path)==h,path
    ex,state,cp=restored(dest)
    dump(dest/'RECOVERY_IDENTITY.json',dict(checkpoints=cp,no_refit=True,no_reselection=True,original_source_hashes=sources(),compat_sha256=sha(HERE/'compat.py'),at=now()))
    ex.gpu_gate()
    from final_evaluation import finish
    finish(ex,state)
    for path,h in manifest['files'].items():assert sha(ROOT/path)==h,path
    # Publish only missing evaluation outputs. Keep every original file in place.
    for name in ('final_raw','FULL_RESULTS.json','FULL_RESULTS.csv','FINAL_VAL_GATE.json','FINAL_TEST_GATE.json'):
        target=source/name;assert not target.exists(),str(target)
        target.symlink_to(dest/name,target_is_directory=(dest/name).is_dir())
    assert json.loads((source/'LEARNED_PARAMETERS.json').read_text())==json.loads((dest/'LEARNED_PARAMETERS.json').read_text())
    history=json.loads((source/'TRAIN_HISTORY.json').read_text())
    old_seconds=max(r.get('seconds',0) for r in history)
    resource=[]
    for line in (OUT/'RESOURCE_SAMPLES.jsonl').open():
        r=json.loads(line)
        if r.get('primary',{}).get('worker_pid')==486937:resource.append(r.get('gpu1'))
    record=dict(status='PASS_ALL_QUERY_TRAIN_VAL_TEST',at=now(),cell=ex.cell,support=ex.support,selection=state,
        source_hashes=sources(),identity=json.loads((source/'IDENTITY.json').read_text()),queries={s:len(d) for s,d in ex.d.datasets.items()},channels=ex.d.c,
        elapsed_seconds=old_seconds+time.perf_counter()-ex.started,raw_encoding_distance_seconds=ex.raw_time,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        recovery=dict(no_refit=True,no_reselection=True,evaluation_path=str(dest),original_elapsed_from_last_history=old_seconds,
        peak_memory_scope='recovery evaluation only; original training allocator peak unavailable after crash; original nvidia-smi samples preserved',
        original_resource_sample_count=len(resource),preservation_manifest_sha256=sha(root/'PRESERVATION_MANIFEST.json')))
    dump(dest/'COMPLETE.json',record);dump(source/'COMPLETE.json',record)

if __name__=='__main__':
    import fcntl
    lock=(OUT/'GPU.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run()
