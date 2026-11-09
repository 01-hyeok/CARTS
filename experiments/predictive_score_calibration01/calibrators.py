"""Past-only calibration models. Teacher/futures are never accepted here."""
import torch
from torch import nn

class TrainNormalizer(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.register_buffer('mean',torch.zeros(dim,dtype=torch.float64))
        self.register_buffer('scale',torch.ones(dim,dtype=torch.float64))
        self.register_buffer('fitted',torch.tensor(False))
    def fit(self,x,split):
        if split!='train' or self.fitted.item():raise ValueError('Fit once on train')
        x=x.detach().double().reshape(-1,self.mean.numel())
        if not torch.isfinite(x).all():raise ValueError('Nonfinite features')
        self.mean.copy_(x.mean(0));self.scale.copy_(x.std(0,unbiased=False).clamp_min(1e-8));self.fitted.fill_(True)
    def forward(self,x):
        if not self.fitted.item():raise RuntimeError('Unfitted train normalizer')
        return (x.double()-self.mean)/self.scale

def base_score(p,mask):
    assert (p[mask]>0).all() and (p[~mask]==0).all()
    return .1*p.double().masked_fill(~mask,1).log().masked_fill(~mask,-torch.inf)

class ResidualCalibrator(nn.Module):
    def __init__(self,dim=5,active=None,width=32):
        super().__init__();self.normalizer=TrainNormalizer(dim)
        self.register_buffer('active',torch.ones(dim,dtype=torch.float64) if active is None else torch.as_tensor(active,dtype=torch.float64))
        self.layers=nn.Sequential(nn.Linear(dim,width),nn.SiLU(),nn.Linear(width,1)).double()
        self.beta=nn.Parameter(torch.tensor(0.,dtype=torch.float64))
    def score(self,p,x,mask,ranking=False):
        # Never use the zero-beta ranking shortcut in the differentiable loss.
        if ranking and self.beta.detach().item()==0:return p.masked_fill(~mask,-torch.inf)
        z=self.normalizer(x)*self.active
        return (base_score(p,mask)+self.beta*self.layers(z).squeeze(-1)).masked_fill(~mask,-torch.inf)
    def logprob(self,p,x,mask):return (self.score(p,x,mask)/.1).log_softmax(-1)

class FixedHeadWeights(nn.Module):
    def __init__(self):
        super().__init__();self.logits=nn.Parameter(torch.zeros(5,dtype=torch.float64))
    def logprob(self,p,x,mask):
        head=(x.double().transpose(-1,-2)/.1).masked_fill(~mask.unsqueeze(-2),-torch.inf).log_softmax(-1)
        return torch.logsumexp(head+self.logits.log_softmax(0).view(*([1]*(head.ndim-2)),5,1),dim=-2)
    def score(self,p,x,mask,ranking=False):
        if ranking and torch.equal(self.logits.detach(),torch.zeros_like(self.logits)):return p.masked_fill(~mask,-torch.inf)
        return .1*self.logprob(p,x,mask)

def teacher_kl(teacher,logp,mask):
    t=teacher.detach().double()
    assert torch.isfinite(t).all() and (t>=0).all() and (t[~mask]==0).all()
    assert torch.allclose(t.sum(-1),torch.ones_like(t.sum(-1)),atol=2e-6,rtol=0)
    return (t*(t.clamp_min(1e-300).log()-logp.masked_fill(~mask,0))).sum(-1).mean()
