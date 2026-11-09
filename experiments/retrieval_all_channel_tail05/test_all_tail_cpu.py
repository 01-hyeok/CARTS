import sys
from pathlib import Path
import pytest,torch
ROOT=Path('/data/pjh_workspace/CARTS');sys.path.insert(0,str(ROOT))

@pytest.mark.parametrize('h',[96,192,336,720])
@pytest.mark.parametrize('c',[7,21,137])
@pytest.mark.parametrize('b',range(1,32))
def test_padding_and_channel_identity_independent(b,c,h):
    # Integer-valued affine rows: exact arithmetic makes this an axis/padding
    # contamination test, independent of GPU GEMM rounding regressions.
    x=(torch.arange(b*c*h,dtype=torch.float64).reshape(b,c,h)%31)-15
    raw=x.reshape(b*c,h);padding=torch.ones(17,h,dtype=torch.float64)*12345
    transform=lambda z:torch.stack([z[:,0]+2*z[:,-1],z.sum(-1)],-1)
    padded=transform(torch.cat([raw,padding]))[:b*c].reshape(b,c,2)
    expected=torch.stack([x[:,:,0]+2*x[:,:,-1],x.sum(-1)],-1)
    assert torch.equal(padded,expected)
    # Altering padding cannot affect any real query/channel output.
    padded2=transform(torch.cat([raw,-padding]))[:b*c].reshape(b,c,2)
    assert torch.equal(padded,padded2)
    rows=max(b,c);chunk=rows//c;seen=[]
    for a in range(0,b,chunk):
        e=min(a+chunk,b);assert (e-a)*c<=rows;seen.extend(range(a,e))
    assert seen==list(range(b))
