import copy
import numpy as np
import pytest
import torch
from test_batch_core import data_fixture
import batch_core as f
from evaluation import FrozenFull
from future_p100 import pooled_scores,pooled_probability

@pytest.mark.parametrize('nh',[0,5])
def test_p100_complete_channel_axis_and_same_id_order(nh):
    m,h,_,_=f.rt.model_init('Solar_96',0,nh,'cpu');m.eval()
    if h is not None:h.eval()
    x=torch.randn(2,137,96);history=torch.randn(2,137,100,96)
    with torch.no_grad():
        score=pooled_scores(m,h,x,history);p=pooled_probability(score)
        assert p.shape==(2,137,100);torch.testing.assert_close(p.sum(-1),torch.ones(2,137))
        # Independent channel-index checks only in this test, not production.
        for c in (0,61,136):
            ref=pooled_probability(pooled_scores(m,h,x[:,c:c+1],history[:,c:c+1]))
            torch.testing.assert_close(p[:,c:c+1],ref,atol=2e-6,rtol=2e-5)
            assert torch.equal(f.rt.stable_topk_indices(p[:,c],10,largest=True),f.rt.stable_topk_indices(ref[:,0],10,largest=True))

def test_all_channel_selected_value_mapping():
    d=data_fixture(192,137,41);ids=torch.stack([torch.randperm(41)[:10] for _ in range(2*137)]).reshape(2,137,10);last=torch.randn(2,137)
    out=f.selected_values(d,ids,last);assert out.shape==(2,137,10,192)
    for c in (0,67,136):
        _,values=f.original.bank_views(d,[c]);ref=values[0][ids[:,c]]+last[:,c,None,None]
        assert torch.equal(ref,out[:,c])

@pytest.mark.parametrize('nheads',[0,5])
def test_all_channel_eval_equal_predictions_and_metrics(tmp_path,nheads):
    from types import SimpleNamespace
    from evaluation import evaluate
    d=data_fixture(96,137,41);d.cell='Solar_96'
    data=d.series.numpy();d.datasets={'val':SimpleNamespace(data_x=data,data_y=data,starts=np.arange(5))}
    class Dataset:
        data_x=data;data_y=data;starts=np.arange(5)
        def __len__(self):return 5
    d.datasets={'val':Dataset()}
    m,h,_,_=f.rt.model_init('Solar_96',0,nheads,'cpu');m.eval()
    if h is not None:h.eval()
    result=evaluate(d,m,h,'val',2,11,tmp_path);assert result['units']==5*137
    pred=np.load(tmp_path/'val_prediction.npy');ids=np.load(tmp_path/'val_ids.npy')
    frozen=FrozenFull(d,m,h,2,11)
    with torch.no_grad():
        for a,b in [(0,2),(2,4),(4,5)]:
            p=frozen.probability('val',a,b);pick=f.rt.stable_topk_indices(p.reshape(-1,41),10,largest=True).reshape(b-a,137,10)
            assert np.array_equal(ids[a:b],pick.numpy())
            x,y,_=f.query(d,'val',a,b);v=f.selected_values(d,pick,x[:,:,-1]);assert np.array_equal(pred[a:b],v.mean(2).permute(0,2,1).numpy())
