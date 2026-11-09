"""CPU-only strict predecessor verification; never starts or stops a process."""
import hashlib,json
from pathlib import Path

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def passed(p):
    x=read(p);assert x['status']=='PASS',str(p);return x

def verify_solar(root):
    root=Path(root);out=root/'results/TRACK-SOLAR-FULL-ALL-CHANNEL04'
    final=passed(out/'FINAL_GATE.json')
    assert final['trained_models']==8 and final['completed_diagnostic_cells']==4
    assert final['seed']==0 and final['stage2'] is False
    state=read(out/'CONTROLLER_STATUS.json');assert state['status']=='COMPLETE'
    records=[]
    for h in (96,192,336,720):
        cell=f'Solar_{h}'
        for model in ('V0','V5'):
            dest=out/cell/model/'seed0';done=read(dest/'COMPLETE.json')
            assert done['status']=='COMPLETE' and done['seed']==0
            cp=Path(done['checkpoint']);assert sha(cp)==done['checkpoint_sha256']
            hist=read(dest/'history.json');assert [x['epoch'] for x in hist]==list(range(1,11))
            for ep in hist:
                assert (cp.parent/f"epoch{ep['epoch']:02d}.pth").exists()
                assert ep['val']['channels']==137 and ep['val']['all_channel_batch']
            sel=read(dest/'checkpoint_selection.json')
            assert sel['test_used_for_selection'] is False
            best=min(hist,key=lambda x:(x['val']['retmse10'],x['epoch']))
            assert sel['epoch']==best['epoch'] and sel['sha256']==done['checkpoint_sha256']
            assert done['test']['channels']==137
            assert done['test']['units']==done['test']['queries']*137
            records.append(dict(cell=cell,model=model,checkpoint_sha256=done['checkpoint_sha256']))
        diag=root/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01-SOLAR-ALL-CHANNEL04/FULL'/cell
        source=diag/'COMPLETE.json';comp=read(source)
        assert comp['status']=='COMPLETE' and not comp['missing_models']
        assert comp['channels']==137 and len(comp['models'])==2
        rep=passed(diag/'CANONICAL_REPRODUCTION.json');assert len(rep['rows'])==4
        assert {(x['model'].split('_')[0],x['split']) for x in rep['rows']}=={(m,s) for m in ('V0','V5') for s in ('val','test')}
        for row in rep['rows']:
            assert row['status']=='PASS' and row['ordered_ids_exact'] and row['predictions_exact']
            assert row['channels']==137 and row['queries']==comp['query_counts'][row['split']]
        supp=root/'results/TRACK-SOLAR-FULL-ALL-CHANNEL04-DIAGNOSTIC-SUPPLEMENT/FULL'/cell
        cg=passed(supp/'CPU_GATE.json');ig=passed(supp/'INFERENCE_GATE.json');dg=passed(supp/'DELIVERABLE_GATE.json')
        for gate in (cg,ig,dg):assert gate['source_complete_sha256']==sha(source)
        assert not dg['deferred'] and cg['all_query_channels']
        assert dg['cpu_gate_sha256']==sha(supp/'CPU_GATE.json')
        assert dg['inference_gate_sha256']==sha(supp/'INFERENCE_GATE.json')
        assert {x['split'] for x in ig['splits']}=={'val','test'}
        for row in ig['splits']:
            assert row['status']=='PASS' and row['all_methods_ordered_top10_and_oracle_ids_exact']
            assert row['channels']==137 and row['queries']==comp['query_counts'][row['split']]
            assert row['units']==row['queries']*137
    report=out/'FINAL_REPORT_KO.md';assert report.exists() and report.stat().st_size>0
    return dict(status='PASS',trained_models=records,final_gate_sha256=sha(out/'FINAL_GATE.json'),
                report_sha256=sha(report),controller_pid=state.get('controller_pid'))

def verify(root):
    # Existing 48 training + 24 diagnostics + their original summaries remain prerequisites.
    import importlib.util
    p=Path(root)/'experiments/solar_full_all_channel04/prerequisites.py'
    spec=importlib.util.spec_from_file_location('mechanism02_prior_prerequisites',p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    prior=mod.verify(root)
    solar=verify_solar(root)
    return dict(status='PASS',prior=prior,solar=solar)
