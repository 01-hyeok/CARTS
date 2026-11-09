from contracts import actual_batches, acceptable, select_backend, TARGETS
import pytest

def good(model='V0', batch=7, backend='cuda12_lt'):
    return dict(model=model,batch=batch,backend=backend,channels=21,bank_count=36216,
                query_exact=True,keys_exact=True,scores_exact=True,
                ordered_top500_exact=True,mask_exact=True)

def test_actual_tail_counts():
    assert actual_batches(4935)==[(0,32),(4928,4935)]
    assert actual_batches(10204)==[(0,32),(10176,10204)]

@pytest.mark.parametrize('key,value', [('channels',20),('bank_count',36215),
    ('query_exact',False),('keys_exact',False),('scores_exact',False),
    ('ordered_top500_exact',False),('mask_exact',False),('error','CUBLASLT status 15')])
def test_reject_incomplete_or_approximate(key,value):
    r=good();r[key]=value;assert not acceptable(r)

def test_selection_requires_both_models_both_actual_tails():
    rows=[good(m,b) for m in ['V0','V1'] for b in [7,28]]
    assert select_backend(rows) == {'V0:7':'cuda12_lt','V0:28':'cuda12_lt','V1:7':'cuda12_lt','V1:28':'cuda12_lt'}
    assert select_backend(rows[:-1]) is None

def test_backend_priority_is_predefined_and_not_best_future_metric():
    rows=[good(m,b,k) for m in ['V0','V1'] for b in [7,28] for k in ['split64','cuda12_lt']]
    assert set(select_backend(rows).values())=={'cuda12_lt'}
    assert all(t[2:] == ('P100','FULL') for t in TARGETS)
