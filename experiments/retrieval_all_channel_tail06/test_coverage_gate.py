import copy
import pytest
from coverage_gate import EXPECTED, require_weather720_release, require_safe_boundary


def fixture():
    rows = []
    for (model, split), (q, c, n, runtime, tf32) in EXPECTED.items():
        rows.append(dict(model=model, split=split, status='PASS', cell='Weather_720',
                         seed=0, evaluation_support='FULL', queries=q, units=q*c,
                         channels=c, candidates=n, torch=runtime, tf32=tf32,
                         checkpoint_sha256=model*32, ordered_mismatch_units=0,
                         set_mismatch_units=0, canonical_id_sha256='a'*64,
                         batched_id_sha256='a'*64, stored_prediction_pass=True,
                         metric_reproduction_pass=True, all_channels_gpu_batch=True,
                         serial_production_fallback=False, tolerance_changed=False,
                         reference_source='canonical_runtime_recomputed'))
    return rows, {'V0': 'V0'*32, 'V5': 'V5'*32}


def test_complete_scope():
    assert require_weather720_release(*fixture())


def test_prior_v5_only_pass_cannot_release_v0():
    rows, hashes = fixture()
    with pytest.raises(AssertionError):
        require_weather720_release([r for r in rows if r['model'] == 'V5'], hashes)


@pytest.mark.parametrize('field,value', [
    ('model', 'V5'), ('queries', 7), ('units', 147), ('channels', 1),
    ('candidates', 100), ('torch', '1.12'), ('tf32', True), ('seed', 1),
    ('evaluation_support', 'P100'), ('status', 'PARTIAL'),
    ('checkpoint_sha256', 'other'), ('ordered_mismatch_units', 1),
    ('set_mismatch_units', 1), ('batched_id_sha256', 'b'*64),
    ('stored_prediction_pass', False), ('metric_reproduction_pass', False),
    ('all_channels_gpu_batch', False), ('serial_production_fallback', True),
    ('tolerance_changed', True), ('reference_source', 'unverified_score_cache'),
])
def test_incomplete_or_changed_contract_rejected(field, value):
    rows, hashes = fixture()
    rows[0][field] = value
    with pytest.raises(AssertionError):
        require_weather720_release(rows, hashes)


def boundary_fixture():
    cells = ['ETTh1_96','ETTh1_192','ETTh1_336','ETTh1_720',
             'Weather_96','Weather_192','Weather_336']
    return dict(stop=dict(status='BLOCKED_CANONICAL_TAIL_PARITY',cell='Weather_720',
                          solar_training_release=False),
                delivered={c:dict(status='PASS',source_complete_sha256=c) for c in cells},
                source_hashes={c:c for c in cells}, active_controller=False, gpu_busy=False)


def test_safe_boundary():
    assert require_safe_boundary(**boundary_fixture())


@pytest.mark.parametrize('case', ['controller', 'gpu', 'missing336', 'bad_hash', 'failed', 'wrong_stop', 'solar'])
def test_premature_boundary_rejected(case):
    x = boundary_fixture()
    if case == 'controller': x['active_controller'] = True
    elif case == 'gpu': x['gpu_busy'] = True
    elif case == 'missing336': del x['delivered']['Weather_336']
    elif case == 'bad_hash': x['source_hashes']['Weather_336'] = 'changed'
    elif case == 'failed': x['delivered']['Weather_336']['status'] = 'FAILED'
    elif case == 'wrong_stop': x['stop']['cell'] = 'Weather_192'
    elif case == 'solar': x['stop']['solar_training_release'] = True
    with pytest.raises(AssertionError): require_safe_boundary(**x)
