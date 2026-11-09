"""CPU-only artifact-bound handoff. A terminal status string alone never releases GPU."""
import hashlib,json
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS')
CELLS=[f'{d}_{h}' for d in ('ETTh1','Weather','Solar') for h in (96,192,336,720)]
TERMINAL={'DIAGNOSTICS_COMPLETE_SOLAR_WAITER_RELEASED','DIAGNOSTICS_COMPLETE_POST_TRAINING_GATE_PENDING'}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def passed(p):
    a=read(p);assert a['status']=='PASS',str(p);return a

def verify(root=ROOT,require_release=True):
    root=Path(root);diag=root/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01';supp=root/'results/TRACK-RETRIEVAL-DIAGNOSTIC-SUPPLEMENT01'
    manifest=root/'experiments/p100_head_retrieval_quality_legacy_pool02/configs/run_manifest.json'
    rows=[r for r in read(manifest) if r['seed']==0];assert len(rows)==48
    assert len({r['run_id'] for r in rows})==48
    for r in rows:
        assert r['status']=='COMPLETE'
        assert read(root/r['output_path']/'COMPLETE.json')['status']=='COMPLETE'
    passed(root/'results/TRACK-P100-HEAD-RETRIEVAL-QUALITY01-ORACLE-K-SUPPLEMENT01/FINAL_GATE.json')
    for support in ('P100','FULL'):
        phase=passed(supp/f'{support}_DELIVERABLE_GATE.json')
        assert len(phase['cells'])==12
        indexed={r['cell']:r for r in phase['cells']};assert set(indexed)==set(CELLS)
        for cell in CELLS:
            src=diag/support/cell/'COMPLETE.json';done=read(src);dest=supp/support/cell
            assert done['status'] in ('COMPLETE','COMPLETE_WITH_MISSING_MODELS')
            expected=sha(src);cpu=passed(dest/'CPU_GATE.json');inf=passed(dest/'INFERENCE_GATE.json');gate=passed(dest/'DELIVERABLE_GATE.json')
            for g in (cpu,inf,gate):assert g['source_complete_sha256']==expected,(support,cell,'stale artifact')
            assert cpu['all_query_channels']
            assert gate==indexed[cell],(support,cell,'phase gate does not bind cell gate')
            assert gate['cpu_gate_sha256']==sha(dest/'CPU_GATE.json')
            assert gate['inference_gate_sha256']==sha(dest/'INFERENCE_GATE.json')
            if support=='FULL':assert not gate['deferred']
            splits={r['split']:r for r in inf['splits']};assert set(splits)=={'val','test'}
            for split,g in splits.items():
                assert g['status']=='PASS' and g['all_methods_ordered_top10_and_oracle_ids_exact']
                assert g['queries']==done['query_counts'][split] and g['channels']==done['channels']
                assert g['units']==g['queries']*g['channels']
    summary=diag/'07_Summary/SUMMARY_GATE.json';passed(summary)
    finalpath=supp/'FINAL_DELIVERABLE_GATE.json';final=passed(finalpath)
    assert final['cells']==24 and final['all_nonmissing_requested_details_complete']
    assert final['original_summary_sha256']==sha(summary)
    assert final['p100_gate_sha256']==sha(supp/'P100_DELIVERABLE_GATE.json')
    assert final['full_gate_sha256']==sha(supp/'FULL_DELIVERABLE_GATE.json')
    state=read(diag/'00_Audit/CONTROLLER_STATUS.json');assert state['status'] in TERMINAL
    if require_release:
        rel=read(root/'results/TRACK-SOLAR-FULL-MISSING-CHECKPOINTS01/DELIVERABLE_COMPLETENESS_RELEASE.json')
        assert rel['status']=='RELEASED_AFTER_EXPANDED_FINAL_GATE' and rel['all_prior_deliverables_complete']
        assert rel['gate_sha256']==sha(finalpath)
    return dict(status='PASS',training_runs=48,diagnostic_cells=24,final_gate_sha256=sha(finalpath),controller_status=state['status'])
