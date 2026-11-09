import numpy as np
from d123_full_aggregate import spearman_rows,worst_share

def test_rank_ties_and_denominator_invariance():
    a=np.array([[1,1,3,4],[3,2,1,0]],float);b=a/7
    np.testing.assert_allclose(spearman_rows(a,b),[1,1],atol=1e-15)

def test_rank_reverse_and_undefined():
    a=np.array([[0,1,2,3],[1,1,1,1]],float)
    r=spearman_rows(a,-a)
    assert np.isclose(r[0],-1) and np.isnan(r[1])

def test_tail_contribution_timestamp_channel_mean():
    a=np.array([[4,0],[0,0],[-8,0],[2,0]])
    assert np.isclose(worst_share(a,.25),2/3) and worst_share(-np.ones((4,2)),.25) is None
