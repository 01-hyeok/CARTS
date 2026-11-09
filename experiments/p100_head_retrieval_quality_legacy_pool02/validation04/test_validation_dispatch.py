import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import runtime_validation as rv
import controller_validation as ctl

def test_only_nondetailed_validation_is_accelerated(monkeypatch,tmp_path):
    called=[]
    monkeypatch.setattr(rv.rt.torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(rv.graphed_validation,'evaluate',lambda *a,**k:(called.append('graph') or {'validation_acceleration':{'probe':0}}))
    monkeypatch.setattr(rv.rt,'evaluate',lambda *a,**k:(called.append('canonical') or {}))
    monkeypatch.setattr(rv,'CURRENT_DEST',tmp_path);monkeypatch.setattr(rv,'VALIDATION_CALL',0)
    rv.evaluate(None,None,None,'val');assert called==['graph']
    rv.evaluate(None,None,None,'val',detailed=True);assert called[-1]=='canonical'
    rv.evaluate(None,None,None,'test');assert called[-1]=='canonical'
    rv.evaluate(None,None,None,'val',baselines=True);assert called[-1]=='canonical'
    assert (tmp_path/'validation_acceleration/call01.json').exists()

def test_controller_routes_remaining_solar(monkeypatch):
    seen=[];monkeypatch.setattr(ctl,'execute_original',lambda cmd,*args:seen.append(cmd))
    for horizon in (192,336,720):
        for model in ('V0','V1','V2','V5'):
            run=f'Solar_{horizon}_{model}_seed0';ctl.execute(['python','-u','old.py','train',run],run,'STAGE1_RUNNING')
            assert seen[-1][2]==str(ctl.HERE/'runtime_validation.py')
    assert len(seen)==12

def test_exact_gate_source_pins():
    g=json.loads((rv.LOCAL/'GRAPH_GATES.json').read_text())
    assert g['status']=='PASS'
    assert g['graph_source_sha256']==rv.rt.sha256_file(rv.LOCAL/'graphed_validation.py')
    assert g['cache_source_sha256']==rv.rt.sha256_file(rv.LOCAL/'cached_validation.py')
    assert all(x['score_mismatch_batches']==x['pick_mismatch_batches']==0 for x in g['cases'])

def test_existing_source_unchanged():
    g=json.loads((rv.rt.HERE/'reports/RUNTIME_GATES.json').read_text())
    assert g['source_hashes']==rv.rt.source_hashes()

def test_seed_scope_unchanged():
    rows=json.loads((rv.rt.HERE/'configs/run_manifest.json').read_text())
    assert len(rows)==48 and all(x['seed']==0 and x['train_epochs']==10 and x['logical_batch_size']==32 for x in rows)
