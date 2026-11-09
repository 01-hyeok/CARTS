"""Dependency and runtime-interface gates; never creates GPU work."""
import json
from pathlib import Path
import pytest
import torch
import solar_full_train as t

def test_predecessor_gate_fails_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(t,'ROOT',tmp_path);monkeypatch.setattr(t,'DIAG',tmp_path/'diag')
    with pytest.raises(FileNotFoundError):t.prerequisites()
    oracle=tmp_path/'results/TRACK-P100-HEAD-RETRIEVAL-QUALITY01-ORACLE-K-SUPPLEMENT01/FINAL_GATE.json'
    oracle.parent.mkdir(parents=True);oracle.write_text(json.dumps({'status':'PASS'}))
    for support in ['P100','FULL']:
        for ds in ['ETTh1','Weather','Solar']:
            for h in [96,192,336,720]:
                p=t.DIAG/support/f'{ds}_{h}'/'COMPLETE.json';p.parent.mkdir(parents=True);p.write_text(json.dumps({'status':'COMPLETE'}))
    with pytest.raises(FileNotFoundError):t.prerequisites()
    p=t.DIAG/'07_Summary/SUMMARY_GATE.json';p.parent.mkdir();p.write_text(json.dumps({'status':'PASS'}))
    s=t.DIAG/'00_Audit/CONTROLLER_STATUS.json';s.parent.mkdir();s.write_text(json.dumps({'status':'RUNNING'}))
    with pytest.raises(AssertionError):t.prerequisites()
    s.write_text(json.dumps({'status':'DIAGNOSTICS_COMPLETE_POST_TRAINING_GATE_PENDING'}));assert t.prerequisites()
    p=t.DIAG/'FULL/Solar_720/COMPLETE.json';p.write_text(json.dumps({'status':'FAILED'}))
    with pytest.raises(AssertionError):t.prerequisites()

def test_loader_generator_checkpoint_contract():
    from preparation import make_training_loader
    loader=make_training_loader(torch.arange(71),32,0)
    state=loader.generator.get_state();first=[x.clone() for x in loader]
    loader.generator.set_state(state);again=list(loader)
    assert all(torch.equal(x,y) for x,y in zip(first,again))
    assert sum(len(x) for x in first)==71

def test_all_eight_order_is_fixed():
    import ast
    source=(Path(t.__file__).parent/'solar_full_controller.py').read_text()
    ast.parse(source)
    assert source.index('while not ready()')<source.index("if not (t.OUT/'SMOKE_GATE.json')")
    assert "for h in [96,192,336,720]" in source and "for n in [0,5]" in source
