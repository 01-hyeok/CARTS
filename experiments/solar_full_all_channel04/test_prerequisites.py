import json
from pathlib import Path
import pytest
import prerequisites as p
def put(path,x):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x))
def fixture(root,status='DIAGNOSTICS_COMPLETE_SOLAR_WAITER_RELEASED'):
    d=root/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01';s=root/'results/TRACK-RETRIEVAL-DIAGNOSTIC-SUPPLEMENT01'
    rows=[]
    for i in range(48):
        rows.append(dict(run_id=str(i),seed=0,status='COMPLETE',output_path='training/'+str(i)));put(root/'training'/str(i)/'COMPLETE.json',dict(status='COMPLETE'))
    put(root/'experiments/p100_head_retrieval_quality_legacy_pool02/configs/run_manifest.json',rows)
    put(root/'results/TRACK-P100-HEAD-RETRIEVAL-QUALITY01-ORACLE-K-SUPPLEMENT01/FINAL_GATE.json',dict(status='PASS'))
    for support in ('P100','FULL'):
        phase=[]
        for cell in p.CELLS:
            src=d/support/cell/'COMPLETE.json';put(src,dict(status='COMPLETE',query_counts=dict(val=8,test=9),channels=137));h=p.sha(src)
            path=s/support/cell;cpu=dict(status='PASS',source_complete_sha256=h,all_query_channels=True)
            inf=dict(status='PASS',source_complete_sha256=h,splits=[dict(split=k,status='PASS',queries=n,channels=137,units=n*137,all_methods_ordered_top10_and_oracle_ids_exact=True) for k,n in [('val',8),('test',9)]])
            put(path/'CPU_GATE.json',cpu);put(path/'INFERENCE_GATE.json',inf)
            gate=dict(status='PASS',cell=cell,source_complete_sha256=h,cpu_gate_sha256=p.sha(path/'CPU_GATE.json'),inference_gate_sha256=p.sha(path/'INFERENCE_GATE.json'),deferred=[])
            put(path/'DELIVERABLE_GATE.json',gate);phase.append(gate)
        put(s/(support+'_DELIVERABLE_GATE.json'),dict(status='PASS',cells=phase))
    put(d/'07_Summary/SUMMARY_GATE.json',dict(status='PASS'))
    put(s/'FINAL_DELIVERABLE_GATE.json',dict(status='PASS',cells=24,all_nonmissing_requested_details_complete=True,
        original_summary_sha256=p.sha(d/'07_Summary/SUMMARY_GATE.json'),p100_gate_sha256=p.sha(s/'P100_DELIVERABLE_GATE.json'),full_gate_sha256=p.sha(s/'FULL_DELIVERABLE_GATE.json')))
    put(d/'00_Audit/CONTROLLER_STATUS.json',dict(status=status))
    put(root/'results/TRACK-SOLAR-FULL-MISSING-CHECKPOINTS01/DELIVERABLE_COMPLETENESS_RELEASE.json',dict(status='RELEASED_AFTER_EXPANDED_FINAL_GATE',all_prior_deliverables_complete=True,gate_sha256=p.sha(s/'FINAL_DELIVERABLE_GATE.json')))
    return d,s
@pytest.mark.parametrize('status',sorted(p.TERMINAL))
def test_both_terminal_spellings_need_artifacts(tmp_path,status):
    fixture(tmp_path,status);assert p.verify(tmp_path)['diagnostic_cells']==24
@pytest.mark.parametrize('kind',['status_only','missing_full_cell','stale_source','stale_cpu','short_query_count','bad_summary','missing_release','running'])
def test_reject_premature_or_stale_release(tmp_path,kind):
    d,s=fixture(tmp_path)
    if kind=='status_only':(s/'FINAL_DELIVERABLE_GATE.json').unlink()
    if kind=='missing_full_cell':(d/'FULL/Solar_720/COMPLETE.json').unlink()
    if kind=='stale_source':put(d/'FULL/Solar_720/COMPLETE.json',dict(status='COMPLETE',extra=True))
    if kind=='stale_cpu':put(s/'FULL/Solar_720/CPU_GATE.json',dict(status='PASS',source_complete_sha256='bad',all_query_channels=True))
    if kind=='short_query_count':
        x=p.read(s/'FULL/Solar_720/INFERENCE_GATE.json');x['splits'][0]['queries']=1;put(s/'FULL/Solar_720/INFERENCE_GATE.json',x)
    if kind=='bad_summary':put(d/'07_Summary/SUMMARY_GATE.json',dict(status='PASS',edited=True))
    if kind=='missing_release':(tmp_path/'results/TRACK-SOLAR-FULL-MISSING-CHECKPOINTS01/DELIVERABLE_COMPLETENESS_RELEASE.json').unlink()
    if kind=='running':put(d/'00_Audit/CONTROLLER_STATUS.json',dict(status='RUNNING'))
    with pytest.raises((AssertionError,FileNotFoundError,KeyError)):p.verify(tmp_path)

def test_handoff_never_targets_full_worker_or_active_child():
    from handoff import validate_target
    with pytest.raises(AssertionError):validate_target(['python','/experiments/retrieval_all_channel03/full_channel_run.py'],[])
    with pytest.raises(AssertionError):validate_target(['python','/data/pjh_workspace/CARTS/experiments/solar_full_missing_checkpoints01/solar_full_controller.py'],['123'])
    validate_target(['python','/data/pjh_workspace/CARTS/experiments/solar_full_missing_checkpoints01/solar_full_controller.py'],[])
