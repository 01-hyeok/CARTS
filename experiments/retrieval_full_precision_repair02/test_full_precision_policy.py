import pytest
from full_precision_policy import precision_policy,uses_score_cache,BRIDGE_CELLS,NATIVE_CELLS
def test_fixed_existing_cells():
    assert BRIDGE_CELLS|NATIVE_CELLS=={f'{d}_{h}' for d in ['ETTh1','Weather'] for h in [96,192,336,720]}
    for c in BRIDGE_CELLS:assert uses_score_cache(c) and not precision_policy('FULL','V0',c)['matmul_tf32']
    for c in NATIVE_CELLS:assert not uses_score_cache(c) and precision_policy('FULL','V0',c)['matmul_tf32']
def test_full_v5_unchanged():
    for c in BRIDGE_CELLS|NATIVE_CELLS:assert precision_policy('FULL','V5',c)==dict(matmul_tf32=True,cudnn_tf32=True)
def test_p100_unchanged_all_datasets():
    for c in ['ETTh1_96','Weather_192','Solar_720']:
        for a in ['V0','V5']:assert precision_policy('P100',a,c)==dict(matmul_tf32=False,cudnn_tf32=False)
def test_unknown_full_v0_fails_closed():
    with pytest.raises(AssertionError):precision_policy('FULL','V0','Solar_96')
