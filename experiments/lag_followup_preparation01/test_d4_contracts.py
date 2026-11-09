import sys
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('torch')
if not Path('/data/pjh_workspace/CARTS').exists():pytest.skip('server adapter required',allow_module_level=True)
sys.path.insert(0,str(Path(__file__).resolve().parent))
from d4_run import rerank

def fixture():
    rng=np.random.RandomState(0);rows=[]
    for i in range(3):
        ids=np.arange(110)[None,None].repeat(7,1)
        rows.append(dict(base_score=np.broadcast_to(np.arange(110,0,-1),(1,7,110)).copy().astype(float),
             gain=rng.rand(1,7,110),recent_difference=rng.rand(1,7,110),mse=rng.rand(1,7,110),
             global_ids=ids,rerank_count=np.array([100])))
    return rows

def test_alpha_beta_zero_preserves_all_channel_order():
    rows=fixture();_,_,ids=rerank(rows,100,0,0,np.ones(7),np.ones(7))
    assert ids.shape==(3,7,10)
    np.testing.assert_array_equal(ids,np.broadcast_to(np.arange(10),(3,7,10)))

def test_future_labels_cannot_change_rerank_selection():
    rows=fixture();_,_,expected=rerank(rows,100,.3,.1,np.ones(7),np.ones(7))
    for row in rows:row['mse']=row['mse'][...,::-1].copy()*100
    _,_,actual=rerank(rows,100,.3,.1,np.ones(7),np.ones(7))
    np.testing.assert_array_equal(actual,expected)

def test_candidate_pool_is_not_replaced_by_oracle_appendix():
    rows=fixture()
    for row in rows:
        row['gain'][...,100:]=1e6;row['base_score'][...,100:]=1e6
    _,_,actual=rerank(rows,100,1,0,np.ones(7),np.ones(7))
    assert np.all(actual<100)
