import sys,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import runtime as rt
import torch,pytest
from types import SimpleNamespace
from vector_ops import *

class Encoder(torch.nn.Module):
    def __init__(self,h):
        super().__init__();self.net=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(h,256),torch.nn.GELU(),torch.nn.Dropout(0),torch.nn.Linear(256,128),torch.nn.LayerNorm(128),torch.nn.Linear(128,128),torch.nn.GELU(),torch.nn.Linear(128,128))
    def forward(self,x):return F.normalize(self.net(x),dim=-1)
class Heads(torch.nn.Module):
    def __init__(self,n):super().__init__();self.W=torch.nn.Parameter(torch.randn(n,128,128)*.01)
    def forward(self,z):return F.normalize(torch.einsum('bd,sde->bse',z,self.W),dim=-1)
class Model(torch.nn.Module):
    def __init__(self,h):super().__init__();self.encoder=Encoder(h)

@pytest.mark.parametrize('h',[8,16])
def test_strided_gather(h):
    torch.manual_seed(4);data=SimpleNamespace(h=h,c=3,device='cpu',series=torch.randn(300,3));data.future_series=data.series
    ids=torch.stack([torch.randperm(200)[:100].expand(2,100) for _ in range(3)])
    x=torch.randn(2,h,3);y=torch.randn_like(x);q,t,hi,v=flat_inputs(data,x,y,ids,0,3)
    for c in range(3):
        assert torch.equal(q[c*2:(c+1)*2],x[:,:,c])
        assert torch.equal(t[c*2:(c+1)*2],y[:,:,c])
        expected=data.series[:,c][ids[c][...,None]+torch.arange(h)]
        assert torch.equal(hi.reshape(3,2,100,h)[c],expected)
        values=data.series[:,c][ids[c][...,None]+h+torch.arange(h)]-data.series[:,c][ids[c]+h-1][...,None]
        assert torch.equal(v.reshape(3,2,100,h)[c],values)

@pytest.mark.parametrize('nheads',[0,1,2,5])
def test_weighted_gradient_and_forward(nheads):
    torch.set_num_threads(2);torch.manual_seed(42)
    data=SimpleNamespace(h=8,c=3,device='cpu',series=torch.randn(300,3));data.future_series=data.series
    ids=torch.stack([torch.randperm(200)[:100].expand(2,100) for _ in range(3)])
    x=torch.randn(2,8,3);y=torch.randn_like(x)
    m=Model(8);heads=Heads(nheads) if nheads else None
    v=copy.deepcopy(m);vh=copy.deepcopy(heads);dr=install_dropout(v)
    losses=[];scores=[]
    for c in range(3):
        q,t,hi,values=flat_inputs(data,x,y,ids,c,c+1);s=vector_scores(m,heads,q,hi)
        loss=rt.kl(s,values,q[:,-1],t)[0];(loss/3).backward();losses.append(float(loss));scores.append(s)
    newloss=batch_backward(rt,data,v,vh,dr,x,y,ids,chunk=2)
    assert abs(newloss-sum(losses)/3)<2e-6
    p=list(m.parameters())+(list(heads.parameters()) if heads else [])
    vp=list(v.parameters())+(list(vh.parameters()) if vh else [])
    for a,b in zip(p,vp):assert torch.allclose(a.grad,b.grad,rtol=2e-4,atol=2e-7)
    q,t,hi,_=flat_inputs(data,x,y,ids,0,3)
    assert torch.allclose(torch.cat(scores),vector_scores(v,vh,q,hi,dr),rtol=2e-5,atol=2e-6)

def test_head_duplicate_mean():
    s=torch.randn(5,1,100);p=torch.softmax(s/.1,-1)
    assert torch.allclose(p.mean(1),p.expand(-1,5,-1).mean(1),atol=1e-7)

def test_manifest_seed0_stop48():
    import json
    from controller_vector import original
    rows=json.loads((rt.HERE/'configs/run_manifest.json').read_text())
    assert len(original.validate_manifest(rows))==48 and all(x['seed']==0 for x in rows)

def test_controller_routes_solar_only(monkeypatch):
    import controller_vector as c
    called=[];monkeypatch.setattr(c,'execute_original',lambda command,label,status:called.append(command))
    c.execute(['python','-u','old.py','train','Solar_96_V2_seed0'],'Solar_96_V2_seed0','STAGE1_RUNNING')
    assert called[-1][2]==str(c.HERE/'runtime_vector.py')
    c.execute(['python','-u','old.py','pool','Solar_192'],'pool_Solar_192','POOL_GENERATION')
    assert called[-1][2]=='old.py'
    with pytest.raises(AssertionError):
        c.execute(['python','-u','old.py','train','Solar_96_V2_seed1'],'Solar_96_V2_seed1','STAGE1_RUNNING')
