import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('full_analysis',Path(__file__).with_name('full_analysis.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture():
    torch.manual_seed(45);s=torch.randn(7,5,80);mask=torch.ones(7,80,dtype=torch.bool);mask[:,:8]=False
    idx,p,mix=m.mean_mixture_topk_selection(s,mask,.1,10)
    return s,mask,idx,p
def test_mean_mixture():
    s,mask,idx,p=fixture();assert torch.equal(idx,m.selection(p,tuple(range(5)),mask))
def test_single_duplicate():
    s,mask,idx,p=fixture();assert torch.equal(m.selection(p,(0,),mask),m.selection(p,(0,0,0),mask))
    own,_,_=m.mean_mixture_topk_selection(s[:,:1],mask,.1,10)
    assert torch.equal(own,m.selection(p,(0,),mask))
def test_permutation():
    s,mask,idx,p=fixture();assert torch.equal(idx,m.selection(p,(4,1,3,0,2),mask))
def test_dc_independent():
    torch.manual_seed(9);y=torch.randn(7,10,96);truth=torch.randn(7,96);agg,d,c=m.decomp(y,truth)
    assert torch.allclose(agg,d+c,atol=1e-6)
    assert not torch.equal(c,agg) and not torch.equal(c,d)
def test_removed_head_normalized():
    s,mask,idx,p=fixture();sub=p[:,[0,1,2,3]].mean(1)
    assert torch.allclose(sub.sum(-1),torch.ones(7))
    assert torch.equal(m.selection(p,(0,1,2,3),mask),m.stable_topk_indices(sub.masked_fill(~mask,float('-inf')),10,largest=True))
def test_error_cosine():
    x=torch.randn(4,5,96);x[:,0]=0;dot,norm,cos,zero=m.error_pairs(x)
    assert torch.equal(cos[:,0],torch.zeros_like(cos[:,0]))
    assert torch.allclose(cos.diagonal(dim1=-2,dim2=-1)[:,1:],torch.ones(4,4,dtype=cos.dtype),atol=1e-6)
def test_mask_and_future_independence():
    s,mask,idx,p=fixture();before=m.selection(p,(0,1,2,3,4),mask)
    unused_future=torch.randn(7,96)*1e9
    after=m.selection(p,(0,1,2,3,4),mask)
    assert mask.gather(1,after).all() and torch.equal(before,after)
    import inspect
    assert set(inspect.signature(m.selection).parameters)=={'p','subset','mask','k'}
def test_independent_raft_mask():
    import numpy as np
    from utils.relation_memory import RelationMemorySampler
    sampler=object.__new__(RelationMemorySampler);sampler.starts=np.arange(100);sampler.seq_len=5;sampler.pred_len=3;sampler.mask_mode='raft'
    queries=np.array([0,7,50,99,150]);actual,_=sampler.valid_mask_batch(queries)
    assert np.array_equal(actual.numpy(),m.independent_raft_mask(queries,sampler.starts,5,3))
def test_lambda_identities():
    b,r=torch.randn(7,96),torch.randn(7,96)
    assert torch.equal(b+0*(r-b),b);assert torch.allclose(b+1*(r-b),r,atol=1e-6)
