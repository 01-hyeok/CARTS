"""CPU-only, artifact-bound Solar diagnostic handoff. No model training."""
import hashlib,json
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS')
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01'
SUPP=ROOT/'results/TRACK-RETRIEVAL-DIAGNOSTIC-SUPPLEMENT01'
AUDIT=SOURCE/'00_Audit/SOLAR_DIAG11'
SOLAR=[f'Solar_{h}' for h in (96,192,336,720)]
PRIOR=[f'{d}_{h}' for d in ('ETTh1','Weather') for h in (96,192,336,720)]
EXPECTED={96:(36601,5161,10417),192:(36409,5065,10321),336:(36121,4921,10177),720:(35353,4537,9793)}

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def process_alive(pid):
    p=Path('/proc')/str(pid)/'stat'
    return p.exists() and p.read_text().rsplit(')',1)[1].split()[0]!='Z'
def source_check():
    proof=read(HERE/'CPU_READY.json')
    assert proof['status']=='PASS'
    for p,h in proof['sources'].items():assert sha(ROOT/p)==h,('source changed',p)
    return proof
def validate_cell_gate(done,cpu,inf,final,digests):
    assert done['status'] in ('COMPLETE','COMPLETE_WITH_MISSING_MODELS')
    for x in (cpu,inf,final):assert x['status']=='PASS' and x['source_complete_sha256']==digests['complete']
    assert final['cpu_gate_sha256']==digests['cpu'] and final['inference_gate_sha256']==digests['inference']
    assert not final['deferred'] and cpu['all_query_channels']
    splits={r['split']:r for r in inf['splits']};assert set(splits)=={'val','test'}
    for s,g in splits.items():
        assert g['status']=='PASS' and g['all_methods_ordered_top10_and_oracle_ids_exact']
        assert g['queries']==done['query_counts'][s] and g['channels']==done['channels']
        assert g['units']==g['queries']*g['channels']
def delivered(cell):
    src=SOURCE/'FULL'/cell/'COMPLETE.json';dst=SUPP/'FULL'/cell
    files=[src,dst/'CPU_GATE.json',dst/'INFERENCE_GATE.json',dst/'DELIVERABLE_GATE.json']
    validate_cell_gate(*(read(p) for p in files),dict(complete=sha(files[0]),cpu=sha(files[1]),inference=sha(files[2])))
    return sha(files[-1])
def predecessors():
    prior=read(SOURCE/'00_Audit/RESUME09/COMPLETE.json')
    assert prior['status']=='WEATHER720_DELIVERED_SOLAR_DATA_BATCH_PREFLIGHT_PENDING'
    assert not process_alive(prior['controller_pid']),'Weather controller is still active'
    pg=read(SUPP/'P100_DELIVERABLE_GATE.json');assert pg['status']=='PASS' and len(pg['cells'])==12
    assert {r['cell'] for r in pg['cells']}==set(PRIOR+SOLAR)
    for row in pg['cells']:
        src=SOURCE/'P100'/row['cell']/'COMPLETE.json'
        assert row['status']=='PASS' and row['source_complete_sha256']==sha(src)
        assert row==read(SUPP/'P100'/row['cell']/'DELIVERABLE_GATE.json')
    return {c:delivered(c) for c in PRIOR}
def validate_shape(cell,channels,bank,counts):
    h=int(cell.rsplit('_',1)[1]);n,v,t=EXPECTED[h]
    assert channels==137 and bank==n
    assert counts=={'train':n,'val':v,'test':t}
def preflight(cell):
    p=AUDIT/cell/'GPU_PREFLIGHT.json';a=read(p)
    assert a['status']=='PASS' and a['cell']==cell and a['all_channels_single_batch']
    validate_shape(cell,a['channels'],a['full_candidate_count'],a['query_counts'])
    assert a['source_hashes']==source_check()['sources']
    assert a['missing_full_models']==['V0','V5'] and not a['learned_model_reproduction_claimed']
    assert a['canonical_ordered_id_mismatches']==0
    return a
def validate_old_waiter(command,children,state):
    assert any(x.endswith('/experiments/solar_full_missing_checkpoints01/solar_full_controller.py') for x in command)
    assert not children and state['status']=='WAITING_ALL_PRIOR_DIAGNOSTICS'
    assert not state.get('actual_training_started',False)
