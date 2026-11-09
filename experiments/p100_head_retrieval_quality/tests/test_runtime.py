import torch
import numpy as np
from runtime import kl, metric_rows, configurations, FIELDS
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_k_multislot_predictive_retrieval01 import kl_loss_from_prob
from utils.mean_mixture_selection import mean_mixture_topk_selection

def test_actual_runtime_loss_reuses_canonical_zscore_teacher_and_kl():
    g=torch.Generator().manual_seed(71)
    scores=torch.randn(3,5,100,generator=g,requires_grad=True)
    values=torch.randn(3,100,7,generator=g)
    last=torch.randn(3,generator=g);target=torch.randn(3,7,generator=g)
    loss,d,p=kl(scores,values,last,target)
    mask=torch.ones(3,100,dtype=torch.bool)
    assert torch.equal(p,normalized_teacher_prob(d,mask,.1))
    expected=kl_loss_from_prob(p,torch.softmax(scores/.1,-1).mean(1),mask)
    assert torch.equal(loss,expected)
    assert torch.isfinite(torch.autograd.grad(loss,scores)[0]).all()
    assert not torch.allclose(p,torch.softmax(-d/.1,-1))

def test_subset_inventory_contains_all_31_and_three_secondary_k():
    c=configurations(5,True)
    assert len([x for x in c if x[2]==10])==31
    assert len({x[1] for x in c if x[2]==10})==31
    assert sorted(x[2] for x in c if x[2]!=10)==[1,5,20]

def test_metric_arithmetic_and_oracle_regret():
    values=torch.arange(100.).reshape(1,100,1).expand(1,100,3)
    target=torch.zeros(1,3);last=torch.zeros(1)
    d=values.square().mean(-1);ids=torch.arange(10).reshape(1,10)
    m=metric_rows(d,values,last,target,ids,ids,torch.zeros(1))
    assert float(m[0,FIELDS.index('retmse')])==28.5
    assert float(m[0,FIELDS.index('agg_mse')])==20.25
    assert float(m[0,FIELDS.index('agg_mae')])==4.5
    assert float(m[0,FIELDS.index('recall')])==1
    assert float(m[0,FIELDS.index('regret')])==0

def test_random_draw_stream_is_chunk_invariant():
    def draw(chunks):
        g=torch.Generator().manual_seed(990000)
        return torch.cat([torch.stack([torch.randperm(100,generator=g)[:10] for _ in range(n)]) for n in chunks])
    assert torch.equal(draw([37]),draw([16,16,5]))
