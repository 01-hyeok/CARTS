"""CPU-only expansion scope and strict coexistence contracts."""
import hashlib,json,os,tempfile,datetime
from pathlib import Path

ROOT=Path('/data/pjh_workspace/CARTS')
HERE=ROOT/'experiments/lag_followup_preparation01'
OUT=ROOT/'results/TRACK-SOFTLAG-HEAD-EXPANSION02'
OLD=ROOT/'results/TRACK-SOFTLAG-HEAD-PILOT01'
DIAG=ROOT/'results/TRACK-HEAD-SHIFT-DIAGNOSTIC01'
GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
CELLS=[d+'_'+str(h) for d in ['ETTh1','Weather'] for h in [96,192,336,720]]
JOBS=[(c,s) for c in CELLS for s in ['FULL','P100']]
REUSE=[(c,s) for c,s in JOBS if c in ['ETTh1_96','ETTh1_336']]
NEW=[j for j in JOBS if j not in REUSE]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()
def dump(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=str(p.parent),prefix=p.name+'.')
    with os.fdopen(fd,'w') as f:json.dump(o,f,indent=2,allow_nan=False)
    os.replace(tmp,p)
def validate_other(state,cmd,parent_cmd):
    """No broad Python/GPU allowlist: only our two known diagnostic queues."""
    assert state.get('status')=='RUNNING'
    assert state.get('worker_pid') and state.get('pid')
    allowed={
        'd56_controller_v2.py':('d56_complete_run.py',{'SMOKE','FEATURES'}),
        'd123_completion_queue.py':('d123_full_population.py',{'SMOKE','FULL_QUERY_INFERENCE'})}
    for controller,(worker,phases) in allowed.items():
        if str(HERE/controller) in parent_cmd and str(HERE/worker) in cmd:
            assert state.get('phase') in phases
            return True
    raise AssertionError('GPU owner is not an approved D1-D6 worker')
def checked_gate(p,status):
    p=Path(p)
    if not p.exists():return None
    g=json.loads(p.read_text());assert g['status']==status
    for n,h in g.get('files',{}).items():assert sha(p.parent/n)==h,(p,n)
    return g
def validate_reuse(out):
    g=checked_gate(out/'FINAL_GATE.json','PASS_PILOT_CELL_COMPLETE')
    assert g and g['samples']=={'train':64,'val':32,'test':32}
    assert g['epochs_saved']==list(range(6)) and g['all_channels']==7
    assert g['selected_test_ids_and_metrics_exact_replay'] and g['source_checkpoint_unchanged']
    assert sha(out/'PROVENANCE.json')==g['provenance_sha256']
    p=json.loads((out/'PROVENANCE.json').read_text());cp=p['identity']['checkpoint']
    assert sha(cp['checkpoint'])==cp['checkpoint_sha256']
    return dict(cell=g['cell'],support=g['support'],path=str(out),gate_sha256=sha(out/'FINAL_GATE.json'),checkpoint_sha256=cp['checkpoint_sha256'])
