"""Channel-major Solar encoder batching; unchanged candidates and KL definition."""
import torch
import torch.nn.functional as F
import numpy as np


class ReplayDropout(torch.nn.Module):
    def __init__(self, p):
        super().__init__(); self.p=p; self.mask=None
    def forward(self, x):
        if not self.training or self.p == 0: return x
        assert self.mask is not None and self.mask.shape == x.shape
        return torch.ops.aten.native_dropout_backward(x, self.mask, 1/(1-self.p))


def install_dropout(model):
    found=[]
    for parent in model.encoder.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Dropout):
                replacement=ReplayDropout(child.p); setattr(parent,name,replacement); found.append(replacement)
    assert len(found)==1
    return found[0]


def masks_in_legacy_order(batch, channels, p, device):
    # Same RNG-consuming native-dropout shapes/call order as each legacy channel.
    q=[];k=[]
    for _ in range(channels):
        q.append(F.dropout(torch.ones(batch,256,device=device),p,True).ne(0))
        k.append(F.dropout(torch.ones(batch*100,256,device=device),p,True).ne(0))
    return torch.cat(q),torch.cat(k)


def pool_all(data,split,starts):
    if split not in data.pools:
        data.pools[split]=np.load(data.pool_dir/f'{split}.npy',mmap_mode='r')
    rows=np.asarray(starts,dtype=np.int64)-int(data.datasets[split].starts[0])
    return torch.from_numpy(np.array(data.pools[split][rows],dtype=np.int64,copy=True)).to(data.device).permute(1,0,2).contiguous()


def flat_inputs(data,x,y,ids,c0,c1):
    # Strided compact-series views avoid [C,B,P,L] int64 position tensors.
    channels=torch.arange(c0,c1,device=data.device)[:,None,None]
    part=ids[c0:c1]
    hist=data.series.T.unfold(1,data.h,1)[channels,part]
    future=data.future_series.T.unfold(1,data.h,1)[channels,part+data.h]
    last=data.series.T[channels,part+data.h-1]
    return (x[:,:,c0:c1].permute(2,0,1).reshape(-1,data.h),
            y[:,:,c0:c1].permute(2,0,1).reshape(-1,data.h),
            hist.reshape(-1,data.h), (future-last[...,None]).reshape(-1,100,data.h))


def vector_scores(model,heads,query,hist,dropout=None,masks=None):
    if dropout is not None: dropout.mask=None if masks is None else masks[0]
    zq=model.encoder((query-query[:,-1:]).unsqueeze(1))
    if dropout is not None: dropout.mask=None if masks is None else masks[1]
    zk=model.encoder((hist-hist[:,-1:]).unsqueeze(1)).reshape(len(query),100,-1)
    zk=F.normalize(zk,dim=-1)
    if heads is None:return torch.einsum('bd,bpd->bp',F.normalize(zq,dim=-1),zk).unsqueeze(1)
    return torch.einsum('bsd,bpd->bsp',heads(zq),zk)


def batch_backward(rt,data,model,heads,dropout,x,y,ids,chunk=137):
    value=0.
    for c0 in range(0,data.c,chunk):
        c1=min(c0+chunk,data.c)
        q,target,hist,values=flat_inputs(data,x,y,ids,c0,c1)
        masks=masks_in_legacy_order(len(x),c1-c0,dropout.p,data.device) if model.training and dropout.p else None
        sc=vector_scores(model,heads,q,hist,dropout,masks)
        loss,_,_=rt.kl(sc,values,q[:,-1],target)
        assert torch.isfinite(loss)
        weight=(c1-c0)/data.c
        (loss*weight).backward();value+=float(loss.detach())*weight
        dropout.mask=None
        del sc,loss,masks,q,target,hist,values
    return value
