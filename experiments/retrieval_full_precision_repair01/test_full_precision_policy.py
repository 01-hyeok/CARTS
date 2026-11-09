from full_precision_policy import precision_policy
def test_historical_full_models_are_not_conflated():
    assert precision_policy('FULL','V0')==dict(matmul_tf32=False,cudnn_tf32=True)
    assert precision_policy('FULL','V5')==dict(matmul_tf32=True,cudnn_tf32=True)
def test_completed_p100_precision_unchanged():
    for a in ['V0','V5']:assert precision_policy('P100',a)==dict(matmul_tf32=False,cudnn_tf32=False)
