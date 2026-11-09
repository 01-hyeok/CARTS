import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from score_fusion import fuse,numpy_fusion,GAMMAS

@pytest.mark.parametrize('method',['E1','E2','E3','E4'])
def test_torch_reference_and_axis(method):
    z=np.random.default_rng(3).uniform(-1,1,(3,7,5,103)).astype('float32')
    t=torch.from_numpy(z);s=fuse(t,method,.1)
    np.testing.assert_allclose(s.numpy(),numpy_fusion(z,method,.1),atol=1e-14,rtol=1e-14)
    idx=torch.tensor([0,5,2,55,90,102])
    assert torch.equal(s[...,idx],fuse(t[...,idx],method,.1))
    assert torch.equal(s,torch.cat([fuse(t[i:i+1],method,.1) for i in range(3)]))

@pytest.mark.parametrize('gamma',GAMMAS)
def test_lse_bounds(gamma):
    g=torch.linspace(-1,1,5*101,dtype=torch.float32).reshape(1,1,5,101)
    a=fuse(g,'E4',gamma);mean=fuse(g,'E1');maximum=fuse(g,'E2')
    assert (a>=mean-1e-14).all() and (a<=maximum+1e-14).all()
    assert torch.isfinite(a).all()
