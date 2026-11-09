"""Historical precision policy; fixed by source/cache reproduction, never quality."""
def precision_policy(training_support, architecture):
    assert architecture in ('V0','V5')
    # Original V0 cache and corrected V5 cache used different matmul policies.
    # Reproducing a cache is a numerical gate, not selecting on prediction MSE.
    return dict(matmul_tf32=training_support=='FULL' and architecture=='V5',
                cudnn_tf32=training_support=='FULL')
