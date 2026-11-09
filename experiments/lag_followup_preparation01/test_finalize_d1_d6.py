import json
import pytest
import finalize_d1_d6 as m

def test_missing_conditions_remain_missing(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'BASE',tmp_path/'base')
    monkeypatch.setattr(m,'POP',tmp_path/'pop');monkeypatch.setattr(m,'EXT',tmp_path/'ext');monkeypatch.setattr(m,'CELLS',['ETTh1_96'])
    gaps,gates,soft=m.requirements()
    assert len(gaps)==27 and not gates and str(soft) in gaps

def test_wrong_status_and_tampering_rejected(tmp_path):
    p=tmp_path/'gate.json';data=tmp_path/'values.csv';data.write_text('value\n1\n')
    p.write_text(json.dumps(dict(status='PARTIAL',files={'values.csv':m.sha(data)})))
    with pytest.raises(AssertionError):m.verified_gate(p,'PASS')
    p.write_text(json.dumps(dict(status='PASS',files={'values.csv':m.sha(data)})))
    assert m.verified_gate(p,'PASS')
    data.write_text('value\n2\n')
    with pytest.raises(AssertionError):m.verified_gate(p,'PASS')

def test_macro_does_not_mix_support_or_validation():
    rows=[dict(split='test',cell='E',variant='NATIVE',x='1'),dict(split='test',cell='E',variant='CONTROLLED',x='9'),dict(split='val',cell='E',variant='NATIVE',x='100')]
    assert m.macro(rows,['cell','variant'],'x')=={('E','NATIVE'):1.,('E','CONTROLLED'):9.}
