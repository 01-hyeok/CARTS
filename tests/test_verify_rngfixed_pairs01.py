import json

from scripts.verify_rngfixed_pairs01 import verify_pair


def _write_pair(tmp_path, cell, method, noqa_cfg, noqa_epochs, qa_cfg, qa_epochs):
    for arm, cfg, epochs in (('NOQA', noqa_cfg, noqa_epochs), ('QA', qa_cfg, qa_epochs)):
        d = tmp_path / cell / method / arm
        d.mkdir(parents=True, exist_ok=True)
        (d / 'config.json').write_text(json.dumps(cfg))
        (d / 'epoch_metrics.json').write_text(json.dumps(epochs))


def test_identical_pair_passes(tmp_path):
    cfg = {'init_hash': 'abc', 'slot_heads_init_hash': 'def', 'router_init_hash': None}
    epochs = [{'epoch': 1, 'batch_order_sha256': 'h1'}, {'epoch': 2, 'batch_order_sha256': 'h2'}]
    _write_pair(tmp_path, 'Cell_96', 'V1', cfg, epochs, dict(cfg), [dict(e) for e in epochs])
    ok, report = verify_pair(tmp_path, 'Cell_96', 'V1')
    assert ok
    assert report['all_epoch_batch_orders_match']
    assert report['problems'] == []


def test_batch_order_mismatch_at_any_epoch_fails(tmp_path):
    """The original confound was a SYSTEMATIC per-epoch offset -- a
    checker that only looks at epoch 1 would miss later epochs still
    disagreeing if the offset pattern changed; this test mismatches
    epoch 2 specifically (epoch 1 matches) to prove the checker scans
    every epoch, not just the first."""
    cfg = {'init_hash': 'abc', 'slot_heads_init_hash': 'def', 'router_init_hash': None}
    noqa_epochs = [{'epoch': 1, 'batch_order_sha256': 'h1'}, {'epoch': 2, 'batch_order_sha256': 'h2'}]
    qa_epochs = [{'epoch': 1, 'batch_order_sha256': 'h1'}, {'epoch': 2, 'batch_order_sha256': 'DIFFERENT'}]
    _write_pair(tmp_path, 'Cell_96', 'V1', cfg, noqa_epochs, dict(cfg), qa_epochs)
    ok, report = verify_pair(tmp_path, 'Cell_96', 'V1')
    assert not ok
    assert not report['all_epoch_batch_orders_match']
    assert any('epoch 2' in p for p in report['problems'])


def test_init_hash_mismatch_fails(tmp_path):
    noqa_cfg = {'init_hash': 'abc', 'slot_heads_init_hash': 'def', 'router_init_hash': None}
    qa_cfg = {'init_hash': 'XYZ', 'slot_heads_init_hash': 'def', 'router_init_hash': None}
    epochs = [{'epoch': 1, 'batch_order_sha256': 'h1'}]
    _write_pair(tmp_path, 'Cell_96', 'V0', noqa_cfg, epochs, qa_cfg, [dict(epochs[0])])
    ok, report = verify_pair(tmp_path, 'Cell_96', 'V0')
    assert not ok
    assert not report['init_hash_match']


def test_slot_heads_hash_mismatch_fails(tmp_path):
    noqa_cfg = {'init_hash': 'abc', 'slot_heads_init_hash': 'def', 'router_init_hash': None}
    qa_cfg = {'init_hash': 'abc', 'slot_heads_init_hash': 'DIFFERENT', 'router_init_hash': None}
    epochs = [{'epoch': 1, 'batch_order_sha256': 'h1'}]
    _write_pair(tmp_path, 'Cell_96', 'V1', noqa_cfg, epochs, qa_cfg, [dict(epochs[0])])
    ok, report = verify_pair(tmp_path, 'Cell_96', 'V1')
    assert not ok
    assert not report['slot_heads_init_hash_match']


def test_v0_has_no_slot_heads_both_none_passes(tmp_path):
    """V0 has no query views -> slot_heads_init_hash is None for both
    arms by construction -- None == None must pass, not be flagged."""
    cfg = {'init_hash': 'abc', 'slot_heads_init_hash': None, 'router_init_hash': None}
    epochs = [{'epoch': 1, 'batch_order_sha256': 'h1'}]
    _write_pair(tmp_path, 'Cell_96', 'V0', cfg, epochs, dict(cfg), [dict(epochs[0])])
    ok, report = verify_pair(tmp_path, 'Cell_96', 'V0')
    assert ok
