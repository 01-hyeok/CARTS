"""Historical precision policy; fixed by source/cache reproduction, never quality."""
BRIDGE_CELLS={'ETTh1_96','ETTh1_720','Weather_96','Weather_720'}
NATIVE_CELLS={'ETTh1_192','ETTh1_336','Weather_192','Weather_336'}

def uses_score_cache(cell):
    assert cell in BRIDGE_CELLS|NATIVE_CELLS, ('Full V0 runtime not verified',cell)
    return cell in BRIDGE_CELLS

def precision_policy(training_support, architecture, cell):
    assert architecture in ('V0','V5')
    # Original V0 cache and corrected V5 cache used different matmul policies.
    # Reproducing a cache is a numerical gate, not selecting on prediction MSE.
    return dict(matmul_tf32=training_support=='FULL' and (architecture=='V5' or not uses_score_cache(cell)),
                cudnn_tf32=training_support=='FULL')
