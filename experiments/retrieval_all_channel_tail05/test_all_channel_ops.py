import torch
from all_channel_ops import channel_linear,score_all,global_mixture,distance_all

def test_linear_all_channels():
    torch.manual_seed(0);x=torch.randn(7,3,9);m=torch.nn.Linear(9,5)
    assert torch.allclose(channel_linear(x,m),torch.stack([m(a) for a in x]),atol=2e-6,rtol=2e-6)

def test_score_axes():
    torch.manual_seed(0);q=torch.randn(3,7,5,8);k=torch.randn(7,100,8)
    out=score_all(q,k)
    ref=torch.stack([torch.einsum('bsd,nd->bsn',q[:,c],k[c]) for c in range(7)],1)
    assert out.shape==(3,7,5,100) and torch.allclose(out,ref,atol=2e-6,rtol=2e-6)

def test_global_normalization():
    s=torch.arange(3*7*5*100,dtype=torch.float32).reshape(3,7,5,100)%13
    mask=torch.ones(3,7,100,dtype=torch.bool);mask[...,-9:]=False
    p=global_mixture(s,mask)
    assert torch.allclose(p.sum(-1),torch.ones(3,7)) and (p[...,-9:]==0).all()
    assert torch.equal(p,global_mixture(s[:,:,[3,2,1,0,4]],mask))

def test_distance_all_channels():
    torch.manual_seed(0);v=torch.randn(4,7,128,96);q=torch.randn(4,7);t=torch.randn(4,7,96)
    from utils.candidate_pool import pooled_future_mse
    ref=torch.stack([pooled_future_mse(v[:,c].contiguous(),q[:,c].contiguous(),t[:,c].contiguous()) for c in range(7)],1)
    assert torch.equal(distance_all(v,q,t),ref)

def test_full137_h720_shape():
    v=torch.zeros(1,137,16,720);q=torch.ones(1,137);t=torch.zeros(1,137,720)
    assert torch.equal(distance_all(v,q,t),torch.ones(1,137,16))
