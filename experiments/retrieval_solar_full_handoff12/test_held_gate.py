import copy,csv,json
from pathlib import Path
import pytest
import held_gate as g

BASE=Path(__file__).resolve().parents[1]
PREREQ=BASE/('solar_batch04' if (BASE/'solar_batch04').exists() else 'solar_full_all_channel04')/'prerequisites.py'
prior=g.load_module(PREREQ,'prior_prerequisite_tests')

def put(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))

def table(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def fixture(root):
    d=root/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01';s=root/'results/TRACK-RETRIEVAL-DIAGNOSTIC-SUPPLEMENT01'
    runs=[]
    for i in range(48):
        runs.append(dict(run_id=str(i),seed=0,status='COMPLETE',output_path='training/'+str(i)))
        put(root/'training'/str(i)/'COMPLETE.json',dict(status='COMPLETE'))
    put(root/'experiments/p100_head_retrieval_quality_legacy_pool02/configs/run_manifest.json',runs)
    put(root/'results/TRACK-P100-HEAD-RETRIEVAL-QUALITY01-ORACLE-K-SUPPLEMENT01/FINAL_GATE.json',dict(status='PASS'))
    for support in ['P100','FULL']:
        phase=[]
        for cell in g.CELLS:
            c=137 if cell.startswith('Solar') else 21 if cell.startswith('Weather') else 7
            counts=dict(val=8,test=9);src=d/support/cell;dst=s/support/cell;models=[]
            if support!='FULL' or not cell.startswith('Solar'):
                for v in ['V0','V5']:
                    ck=root/'checkpoints'/support/cell/(v+'.bin');put(ck,dict(model=v))
                    models.append(dict(model=v+('_P100Trained' if support=='P100' else '_FullTrained'),checkpoint=str(ck),sha256=g.sha(ck),training_support=support,evaluation_support=support,original_checkpoint_unchanged=True,
                        metric_reproduction=[dict(split=k,queries=n,channels=c,units=n*c,status='PASS',checkpoint_sha256=g.sha(ck)) for k,n in counts.items()]))
            done=dict(status='COMPLETE' if models else 'COMPLETE_WITH_MISSING_MODELS',channels=c,query_counts=counts,models=models,missing_models=[] if models else [dict(architecture=v,status='MISSING') for v in ['V0','V5']])
            put(src/'COMPLETE.json',done);h=g.sha(src/'COMPLETE.json')
            put(src/'PROVENANCE.json',dict(seed=0,all_queries=True,all_channels=True,data=dict(channels=c,counts=counts)))
            for split,n in counts.items():
                put(src/split/'SPLIT_GATE.json',dict(status='PASS',query_count=n,channels=c,units=n*c))
                put(src/'aggregate'/(split+'_schema.json'),dict(methods=['Random','Oracle']))
            table(dst/'MISSED_FALSE_SCORE_AND_POOLED_RANK.csv',[dict(split=split,method=m,queries=n,channels=c,pooled_median=50,pooled_p90=90,missed_mean_student_score=.1,false_selected_mean_student_score=.2) for split,n in counts.items() for m in ['Random','Oracle']])
            table(dst/'INPUT_STRATA_THRESHOLDS.csv',[dict(channel=k,feature=f) for k in range(c) for f in range(6)])
            for f in ['INPUT_STRATA_BY_CHANNEL.csv','INPUT_STRATA_EQUAL_CHANNEL_MACRO.csv','TOP10_METHOD_OVERLAP.csv','FULL_MODEL_FIXED_P100_RANKING.csv','val_MISSED_RANK_HISTOGRAM.npz','test_MISSED_RANK_HISTOGRAM.npz']:(dst/f).write_bytes(b'fixture data')
            put(src/'probe_train.json',dict(train_only=True))
            put(dst/'PROBE_SAMPLING_DISCLOSURE.json',dict(status='VERIFIED',all_channels=True,channels=c,test_selection=False,source=str(src/'probe_train.json'),sha256=g.sha(src/'probe_train.json')))
            table(dst/'CI_VALIDITY_REVIEW.csv',[
                dict(ci_status='INSUFFICIENT_TIME_BLOCKS',usable_for_significance=False,effective_nonoverlap_blocks=1.5,ci_low='',ci_high=''),
                dict(ci_status='EXPLORATORY_MULTIPLE_COMPARISONS_UNADJUSTED',usable_for_significance=False,effective_nonoverlap_blocks=3,ci_low=-1,ci_high=1)])
            put(dst/'CPU_GATE.json',dict(status='PASS',source_complete_sha256=h,all_query_channels=True))
            put(dst/'INFERENCE_GATE.json',dict(status='PASS',source_complete_sha256=h,splits=[dict(split=k,status='PASS',queries=n,channels=c,units=n*c,all_methods_ordered_top10_and_oracle_ids_exact=True) for k,n in counts.items()]))
            gate=dict(status='PASS',cell=cell,source_complete_sha256=h,cpu_gate_sha256=g.sha(dst/'CPU_GATE.json'),inference_gate_sha256=g.sha(dst/'INFERENCE_GATE.json'),deferred=[])
            put(dst/'DELIVERABLE_GATE.json',gate);phase.append(gate)
        put(s/(support+'_DELIVERABLE_GATE.json'),dict(status='PASS',cells=phase))
    put(d/'07_Summary/SUMMARY_GATE.json',dict(status='PASS'))
    put(d/'00_Audit/CONTROLLER_STATUS.json',dict(status='DIAGNOSTICS_COMPLETE_POST_TRAINING_GATE_PENDING'))
    put(s/'FINAL_DELIVERABLE_GATE.json',dict(status='PASS',cells=24,all_nonmissing_requested_details_complete=True,uncertainty_limitations_disclosed=True,no_significance_claims=True,missing_full_solar_checkpoints=8,
        original_summary_sha256=g.sha(d/'07_Summary/SUMMARY_GATE.json'),p100_gate_sha256=g.sha(s/'P100_DELIVERABLE_GATE.json'),full_gate_sha256=g.sha(s/'FULL_DELIVERABLE_GATE.json')))
    return d,s,g.expected_from_p100(root)

def test_full_artifacts_pass_without_release(tmp_path):
    d,s,e=fixture(tmp_path);proof=g.verify_all(tmp_path,e,prior.verify)
    assert proof['diagnostic_cells']==24 and len(proof['artifact_hashes'])==272
    assert proof['hold_detail_artifacts_checked'] and proof['checkpoint_hashes_checked']

@pytest.mark.parametrize('fault',['missing_cell','stale_source','stale_cpu','short_inf','summary_changed','final_status_only','missing_detail','method_missing','threshold_short','channel_reduced','short_split','bad_checkpoint','bad_probe_hash','ci_bad_bounds','ci_significance','model_split_missing','model_count_short','model_wrong_support','missing_model_hidden','provenance_changed','running','missing_training'])
def test_incomplete_or_tampered_never_releases(tmp_path,fault):
    d,s,e=fixture(tmp_path);src=d/'FULL/Solar_720';dst=s/'FULL/Solar_720'
    if fault=='missing_cell':(src/'COMPLETE.json').unlink()
    elif fault=='stale_source':put(src/'COMPLETE.json',dict(status='COMPLETE'))
    elif fault=='stale_cpu':put(dst/'CPU_GATE.json',dict(status='PASS',source_complete_sha256='wrong',all_query_channels=True))
    elif fault=='short_inf':
        p=dst/'INFERENCE_GATE.json';x=g.read(p);x['splits'][0]['queries']=1;put(p,x)
    elif fault=='summary_changed':put(d/'07_Summary/SUMMARY_GATE.json',dict(status='PASS',changed=True))
    elif fault=='final_status_only':put(s/'FINAL_DELIVERABLE_GATE.json',dict(status='PASS'))
    elif fault=='missing_detail':(dst/'MISSED_FALSE_SCORE_AND_POOLED_RANK.csv').unlink()
    elif fault=='method_missing':
        p=dst/'MISSED_FALSE_SCORE_AND_POOLED_RANK.csv';table(p,g.csvread(p)[:1])
    elif fault=='threshold_short':table(dst/'INPUT_STRATA_THRESHOLDS.csv',[dict(channel=0,feature=0)])
    elif fault=='channel_reduced':
        p=dst/'MISSED_FALSE_SCORE_AND_POOLED_RANK.csv';x=g.csvread(p);x[0]['channels']=1;table(p,x)
    elif fault=='short_split':
        p=src/'val/SPLIT_GATE.json';x=g.read(p);x['query_count']=1;put(p,x)
    elif fault=='bad_checkpoint':(tmp_path/'checkpoints/P100/ETTh1_96/V0.bin').write_bytes(b'changed')
    elif fault=='bad_probe_hash':put(src/'probe_train.json',dict(changed=True))
    elif fault in ('ci_bad_bounds','ci_significance'):
        p=dst/'CI_VALIDITY_REVIEW.csv';x=g.csvread(p)
        x[0]['ci_low']=0 if fault=='ci_bad_bounds' else ''
        if fault=='ci_significance':x[1]['usable_for_significance']=True
        table(p,x)
    elif fault in ('model_split_missing','model_count_short','model_wrong_support','missing_model_hidden'):
        p=src/'COMPLETE.json' if fault=='missing_model_hidden' else d/'FULL/ETTh1_96/COMPLETE.json';x=g.read(p)
        if fault=='missing_model_hidden':x['missing_models']=[]
        elif fault=='model_split_missing':x['models'][0]['metric_reproduction'].pop()
        elif fault=='model_count_short':x['models'][0]['metric_reproduction'][0]['queries']=1
        else:x['models'][0]['training_support']='P100'
        put(p,x)
    elif fault=='provenance_changed':put(d/'P100/Solar_720/PROVENANCE.json',dict(changed=True))
    elif fault=='running':put(d/'00_Audit/CONTROLLER_STATUS.json',dict(status='RUNNING'))
    else:(tmp_path/'training/47/COMPLETE.json').unlink()
    with pytest.raises((AssertionError,KeyError,FileNotFoundError)):g.verify_all(tmp_path,e,prior.verify)

def hold_fixture():
    h=dict(status=g.HELD,actual_training_started=False,controller_pid=4088382,reason='Required diagnostic detail gaps and degenerate H720 CI must be addressed before authorizing the final Solar Full8 predecessor gate')
    pin=dict(pid=4088382,start_ticks='123',command=['python','/experiments/solar_full_missing_checkpoints01/solar_full_controller.py'])
    actual=dict(pin,state='T');proof=dict(status='PASS',diagnostic_cells=24,expected_counts_checked=True,hold_detail_artifacts_checked=True,unusable_CI_excluded=True)
    return h,copy.deepcopy(h),pin,actual,[],proof

def test_verified_held_cpu_waiter_only():g.validate_held_target(*hold_fixture())

@pytest.mark.parametrize('fault',['children','training_started','hold_mismatch','wrong_pid','reused_pid','wrong_command','not_stopped','status_only','twentythree_cells','ci_not_closed'])
def test_reject_unsafe_held_retirement(fault):
    h,current,pin,actual,children,proof=hold_fixture()
    if fault=='children':children=['999']
    elif fault=='training_started':h['actual_training_started']=current['actual_training_started']=True
    elif fault=='hold_mismatch':current['reason']='different'
    elif fault=='wrong_pid':actual['pid']=99
    elif fault=='reused_pid':actual['start_ticks']='456'
    elif fault=='wrong_command':actual['command']=['python','training.py']
    elif fault=='not_stopped':actual['state']='S'
    elif fault=='status_only':proof=dict(status='PASS')
    elif fault=='twentythree_cells':proof['diagnostic_cells']=23
    else:proof['unusable_CI_excluded']=False
    with pytest.raises((AssertionError,KeyError)):g.validate_held_target(h,current,pin,actual,children,proof)

@pytest.mark.parametrize('worker_state',['R','S','D','T'])
def test_active_worker_never_boundary(worker_state):
    c=dict(pid=1,command=['controller.py'],start_ticks='10',state='T');w=dict(pid=2,start_ticks='20',state=worker_state);plan=dict(controller=c,worker=w)
    with pytest.raises(AssertionError):g.validate_boundary(plan,c,w,['2'])

def test_completed_worker_boundary_and_unrelated_child_rejection():
    c=dict(pid=1,command=['controller.py'],start_ticks='10',state='T');w=dict(pid=2,start_ticks='20',state='Z',exit_code=0);plan=dict(controller=c,worker=w)
    g.validate_boundary(plan,c,w,['2'])
    with pytest.raises(AssertionError):g.validate_boundary(plan,c,w,['2','3'])
    w['exit_code']=256
    with pytest.raises(AssertionError):g.validate_boundary(plan,c,w,['2'])
