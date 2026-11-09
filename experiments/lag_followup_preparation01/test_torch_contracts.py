"""Torch CPU equivalence is distinct from actual checkpoint/GPU reproduction."""
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from .past_lag import alignment,torch_alignment
from .softlag import objective_gradient,torch_parameters,torch_forward

@pytest.mark.parametrize('length',[96,192,336,720])
def test_torch_patch_cpu(length):
    rng=np.random.RandomState(0);q=rng.randn(2,7,length);x=rng.randn(2,7,4,length)
    expected,lag=alignment(q,x,candidate_chunk=3)
    actual,lags=torch_alignment(torch.tensor(q),torch.tensor(x),candidate_chunk=3)
    np.testing.assert_array_equal(lag,lags)
    np.testing.assert_allclose(actual.numpy(),expected,rtol=1e-12,atol=1e-12)

@pytest.mark.parametrize('headwise',[False,True])
def test_torch_analytic_gradient(headwise):
    rng=np.random.RandomState(1);g=rng.randn(2,3,5,11);a=rng.randn(2,3,11,5);c=rng.rand(2,3,11)
    teacher=rng.rand(2,3,11);teacher/=teacher.sum(-1,keepdims=True)
    bias=rng.randn(5 if headwise else 1,5)*.01;alpha=.2
    ref,ga,gb=objective_gradient(g,a,c,teacher,alpha,bias)
    p1,p2=torch_parameters(headwise,5);p1.data.fill_(alpha);p2.data.copy_(torch.tensor(bias))
    u,p=torch_forward(torch.tensor(g),torch.tensor(a),torch.tensor(c),p1,p2,torch.ones(2,3,11,dtype=torch.bool))
    t=torch.tensor(teacher);loss=(t*(t.log()-p.log())).sum(-1).mean();loss.backward()
    np.testing.assert_allclose(loss.detach().numpy(),ref,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(p1.grad.numpy(),ga,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(p2.grad.numpy(),gb,rtol=1e-10,atol=1e-12)
