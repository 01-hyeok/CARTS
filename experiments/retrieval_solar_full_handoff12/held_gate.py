"""A held state is not permission: verify all artifacts before retiring the old waiter."""
import csv,hashlib,importlib.util,json
from pathlib import Path

CELLS=[f'{d}_{h}' for d in ('ETTh1','Weather','Solar') for h in (96,192,336,720)]
HELD='HELD_CPU_WAITER_FOR_DELIVERABLE_COMPLETENESS'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def csvread(p):return list(csv.DictReader(Path(p).open()))
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def expected_from_p100(root):
    root=Path(root);diag=root/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01';out={}
    for cell in CELLS:
        p=diag/'P100'/cell/'PROVENANCE.json';v=read(p);done=read(p.with_name('COMPLETE.json'))
        assert v['seed']==0 and v['all_queries'] and v['all_channels']
        counts={s:v['data']['counts'][s] for s in ('val','test')};assert counts==done['query_counts']
        out[cell]=dict(channels=v['data']['channels'],counts=counts,p100_provenance_sha256=sha(p),
            model_names={'P100':sorted(m['model'] for m in done['models']),
                         'FULL':[] if cell.startswith('Solar_') else ['V0_FullTrained','V5_FullTrained']})
    return out
def verify_all(root,expected,base_verify=None):
    root=Path(root);diag=root/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01';supp=root/'results/TRACK-RETRIEVAL-DIAGNOSTIC-SUPPLEMENT01'
    if base_verify is None:base_verify=load_module(root/'experiments/solar_full_all_channel04/prerequisites.py','existing_artifact_gate').verify
    # Reuse the unchanged original 48-run/24-cell/hash/summary prerequisite gate.
    proof=base_verify(root,require_release=False)
    assert set(expected)==set(CELLS)
    artifacts={}
    for support in ('P100','FULL'):
        for cell in CELLS:artifacts.update(verify_cell(root,support,cell,expected[cell]))
    final=read(supp/'FINAL_DELIVERABLE_GATE.json')
    assert final['uncertainty_limitations_disclosed'] and final['no_significance_claims'] and final['missing_full_solar_checkpoints']==8
    return dict(proof,expected_counts_checked=True,hold_detail_artifacts_checked=True,unusable_CI_excluded=True,checkpoint_hashes_checked=True,artifact_hashes=artifacts)

def verify_cell(root,support,cell,e):
    root=Path(root);diag=root/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01';supp=root/'results/TRACK-RETRIEVAL-DIAGNOSTIC-SUPPLEMENT01'
    src=diag/support/cell;dst=supp/support/cell;done=read(src/'COMPLETE.json');p=read(src/'PROVENANCE.json')
    assert sha(diag/'P100'/cell/'PROVENANCE.json')==e['p100_provenance_sha256']
    assert done['channels']==p['data']['channels']==e['channels']
    assert done['query_counts']==e['counts']
    assert {s:p['data']['counts'][s] for s in ('val','test')}==e['counts']
    assert p['seed']==0 and p['all_queries'] and p['all_channels']
    assert sorted(m['model'] for m in done['models'])==e['model_names'][support]
    for m in done['models']:
        assert sha(m['checkpoint'])==m['sha256'],(cell,m['model'],'checkpoint changed')
        assert m['training_support']==m['evaluation_support']==support and m['original_checkpoint_unchanged']
        gates={g['split']:g for g in m['metric_reproduction']};assert set(gates)=={'val','test'}
        for split,n in e['counts'].items():
            g=gates[split]
            assert g['status']=='PASS' and g['queries']==n and g['channels']==e['channels'] and g['units']==n*e['channels'] and g['checkpoint_sha256']==m['sha256']
    if support=='FULL' and cell.startswith('Solar_'):
        assert {(m['architecture'],m['status']) for m in done['missing_models']}=={('V0','MISSING'),('V5','MISSING')}
    for split,n in e['counts'].items():
        g=read(src/split/'SPLIT_GATE.json')
        assert g['status']=='PASS' and g['query_count']==n and g['channels']==e['channels'] and g['units']==n*e['channels']
    # Close the actual hold reasons, not just a terminal status string.
    scores=csvread(dst/'MISSED_FALSE_SCORE_AND_POOLED_RANK.csv');assert scores
    for split,n in e['counts'].items():
        schema=read(src/'aggregate'/f'{split}_schema.json');rows=[r for r in scores if r['split']==split]
        assert {r['method'] for r in rows}==set(schema['methods'])
        assert all(int(r['queries'])==n and int(r['channels'])==e['channels'] for r in rows)
        assert all(k in r for r in rows for k in ['pooled_median','pooled_p90','missed_mean_student_score','false_selected_mean_student_score'])
    assert len(csvread(dst/'INPUT_STRATA_THRESHOLDS.csv'))==6*e['channels']
    required=['MISSED_FALSE_SCORE_AND_POOLED_RANK.csv','INPUT_STRATA_THRESHOLDS.csv','INPUT_STRATA_BY_CHANNEL.csv','INPUT_STRATA_EQUAL_CHANNEL_MACRO.csv','TOP10_METHOD_OVERLAP.csv','PROBE_SAMPLING_DISCLOSURE.json','CI_VALIDITY_REVIEW.csv','val_MISSED_RANK_HISTOGRAM.npz','test_MISSED_RANK_HISTOGRAM.npz']
    if support=='FULL' and done['models']:required.append('FULL_MODEL_FIXED_P100_RANKING.csv')
    for filename in required:
        assert (dst/filename).stat().st_size>0
    probe=read(dst/'PROBE_SAMPLING_DISCLOSURE.json')
    assert probe['status']=='VERIFIED' and probe['all_channels'] and probe['channels']==e['channels'] and not probe['test_selection']
    assert probe['sha256']==sha(probe['source'])
    ci=csvread(dst/'CI_VALIDITY_REVIEW.csv');assert ci
    for row in ci:
        # Adequate-block intervals remain exploratory, never significance
        # claims; insufficient-block bounds are explicitly withheld.
        assert row['ci_status'] in ('EXPLORATORY_MULTIPLE_COMPARISONS_UNADJUSTED','INSUFFICIENT_TIME_BLOCKS')
        assert row['usable_for_significance'].lower()=='false'
        if float(row.get('effective_nonoverlap_blocks') or 0)<2:
            assert row['ci_status']=='INSUFFICIENT_TIME_BLOCKS' and not row['ci_low'] and not row['ci_high']
    paths=[src/'COMPLETE.json',src/'PROVENANCE.json',*[dst/f for f in required]]
    return {str(p.relative_to(root)):sha(p) for p in paths}

def validate_held_target(hold,current,pin,actual,children,proof):
    assert proof['status']=='PASS' and proof['diagnostic_cells']==24 and proof['expected_counts_checked'] and proof['hold_detail_artifacts_checked'] and proof['unusable_CI_excluded']
    assert hold==current and hold['status']==HELD and not hold['actual_training_started']
    assert hold['controller_pid']==pin['pid']==actual['pid']==4088382
    assert pin['start_ticks']==actual['start_ticks'] and pin['command']==actual['command']
    assert actual['state']=='T' and not children
    assert any(x.endswith('/experiments/solar_full_missing_checkpoints01/solar_full_controller.py') for x in actual['command'])
    assert hold['reason']=='Required diagnostic detail gaps and degenerate H720 CI must be addressed before authorizing the final Solar Full8 predecessor gate'
def validate_boundary(plan,controller,worker,children):
    assert controller['pid']==plan['controller']['pid'] and controller['command']==plan['controller']['command'] and controller['start_ticks']==plan['controller']['start_ticks']
    assert controller['state']=='T'
    assert children==[str(plan['worker']['pid'])]
    assert worker['pid']==plan['worker']['pid'] and worker['start_ticks']==plan['worker']['start_ticks']
    assert worker['state']=='Z','Active worker must not be terminated or restarted'
    assert worker['exit_code']==0,'Failed GPU worker is not a successful boundary'
