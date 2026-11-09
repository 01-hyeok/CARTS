import numpy as np
import pytest
from d56_cpu_index_cache import positions_cached

def reference(row,c,ids):
    lookup={int(v):i for i,v in enumerate(row['global_ids'][c]) if row['valid'][c,i]}
    assert all(int(v) in lookup for v in ids)
    return np.array([lookup[int(v)] for v in ids],np.int64)

@pytest.mark.parametrize('n',[10,100,500,36696])
def test_exact_original_indices_nonordered_ids_and_padding(n):
    rng=np.random.RandomState(0);v=rng.permutation(n)*3+7
    row=dict(global_ids=np.r_[v,[0,0,0]][None],valid=np.r_[np.ones(n,bool),[False]*3][None])
    for ids in [v[:10],v[-10:][::-1],v[rng.randint(n,size=200)],np.array([],np.int64)]:
        np.testing.assert_array_equal(positions_cached(row,0,ids),reference(row,0,ids))
    assert len(row['_position_cache'])==1

def test_invalid_candidate_is_rejected():
    row=dict(global_ids=np.array([[5,8,0]]),valid=np.array([[True,True,False]]))
    for bad in [0,6,9]:
        with pytest.raises(AssertionError):positions_cached(row,0,[bad])

def test_duplicate_valid_id_rejected():
    row=dict(global_ids=np.array([[5,5]]),valid=np.array([[True,True]]))
    with pytest.raises(AssertionError):positions_cached(row,0,[5])
