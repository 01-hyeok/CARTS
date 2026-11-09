"""CPU-only release evidence checks. A V5 PASS is never a V0 PASS."""
import hashlib
from pathlib import Path

EXPECTED = {
    ('V0', 'val'): (4551, 21, 35448, '2.4.1+cu121', False),
    ('V0', 'test'): (9820, 21, 35448, '2.4.1+cu121', False),
    ('V5', 'val'): (4551, 21, 35448, '1.12', True),
    ('V5', 'test'): (9820, 21, 35448, '1.12', True),
}


def require_weather720_release(rows, checkpoint_hashes):
    """No defaults/inference for missing model, runtime, or full-split coverage."""
    assert len(rows) == len(EXPECTED), 'Exactly four model/split rows required'
    indexed = {(r['model'], r['split']): r for r in rows}
    assert set(indexed) == set(EXPECTED), 'Missing or duplicated model/split'
    for key, (queries, channels, candidates, runtime, tf32) in EXPECTED.items():
        r = indexed[key]
        assert r['status'] == 'PASS' and r['cell'] == 'Weather_720'
        assert r['seed'] == 0 and r['evaluation_support'] == 'FULL'
        assert r['queries'] == queries and r['units'] == queries * channels
        assert r['channels'] == channels and r['candidates'] == candidates
        assert r['torch'].startswith(runtime) and r['tf32'] is tf32
        assert r['checkpoint_sha256'] == checkpoint_hashes[key[0]]
        assert r['ordered_mismatch_units'] == r['set_mismatch_units'] == 0
        assert r['canonical_id_sha256'] == r['batched_id_sha256']
        assert len(r['canonical_id_sha256']) == 64
        assert r['stored_prediction_pass'] is True
        assert r['metric_reproduction_pass'] is True
        assert r['all_channels_gpu_batch'] is True
        assert r['serial_production_fallback'] is False
        assert r['tolerance_changed'] is False
        assert r['reference_source'] == 'canonical_runtime_recomputed'
    return True


def require_safe_boundary(stop, delivered, source_hashes, active_controller, gpu_busy):
    assert not active_controller, 'Bounded controller still alive'
    assert not gpu_busy, 'GPU1 has an existing compute process'
    assert stop['status'] == 'BLOCKED_CANONICAL_TAIL_PARITY'
    assert stop['cell'] == 'Weather_720' and stop['solar_training_release'] is False
    required = ['ETTh1_96', 'ETTh1_192', 'ETTh1_336', 'ETTh1_720',
                'Weather_96', 'Weather_192', 'Weather_336']
    assert set(delivered) == set(required)
    for cell in required:
        gate = delivered[cell]
        assert gate['status'] == 'PASS'
        assert gate['source_complete_sha256'] == source_hashes[cell]
    return True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
