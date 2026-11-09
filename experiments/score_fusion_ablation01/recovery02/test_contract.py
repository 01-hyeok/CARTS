from pathlib import Path
HERE=Path(__file__).resolve().parent
def test_scope_and_strict_gate():
    s=(HERE/'run.py').read_text()
    assert "cell in CELLS and training=='FULL'" in s
    assert "len(g)<d.c" in s
    assert "r['head_id_differences']==0" in s
    assert "r['raw_exact'] and r['prob_exact']" in s
    assert "r['max_canonical_e0_score_delta']==0" in s
def test_only_normalization_patch():
    s=(HERE/'probe_norm.py').read_text()
    assert "src=src.replace(needle,'canonical_norm128(z,b)')" in s
    assert 'batch.short_query=types.MethodType' in s
    assert 'for c in range' not in s
def test_completed_inputs_unchanged():
    s=(HERE/'controller.py').read_text()
    assert "recovery02/CPU_READY.json" in s
    assert "original_controller" in s
    assert "cmd[:]=" in s
def test_qualified_import():
    s=(HERE/'run.py').read_text()
    assert "spec_from_file_location('score_fusion_weather_norm',HERE/'probe_norm.py')" in s
    assert 'from probe_norm import patch' not in s
