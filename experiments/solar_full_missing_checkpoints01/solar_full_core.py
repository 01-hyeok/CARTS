"""Full-support Solar gradient core. No pruning, stale key bank or new objective."""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path('/data/pjh_workspace/CARTS')
OLD=ROOT/'experiments/p100_head_retrieval_quality_legacy_pool02'
sys.path[:0]=[str(OLD),str(ROOT)]
import runtime as rt
from utils.full_candidate_bank import encode_raw_channel_first
from scripts.train_horizon_retrieval_expert01 import kl_loss

def encode_group(model,x,bank):
    """x [B,C,L], bank [C,N,L]; live query AND key encoder gradients."""
    b,c,h=x.shape;n=bank.shape[1]
    zq=encode_raw_channel_first(model,x.reshape(b*c,h,1),0).reshape(b,c,-1)
    zk=encode_raw_channel_first(model,bank.reshape(c*n,h,1),0).reshape(c,n,-1)
    return zq,F.normalize(zk,dim=-1)

def scores_group(model,heads,x,bank):
    zq,key=encode_group(model,x,bank);b,c,d=zq.shape
    query=F.normalize(zq,dim=-1)[:,:,None] if heads is None else heads(zq.reshape(b*c,d)).reshape(b,c,-1,d)
    return torch.einsum('bcsd,cnd->bcsn',query,key)

@torch.no_grad()
def teacher_distances(values,last,target,chunk=128):
    """Exact existing alignment/addition/subtraction, streaming candidate futures.

    values [C,N,H], last [B,C], target [B,C,H]. Only [B,C,chunk,H]
    is materialized, retaining every candidate and the complete target horizon.
    """
    b,c=last.shape;n=values.shape[1]
    distance=last.new_empty(b,c,n)
    for a in range(0,n,chunk):
        aligned=values[None,:,a:a+chunk]+last[:,:,None,None]
        distance[:,:,a:a+chunk]=((aligned-target[:,:,None])**2).mean(-1)
    return distance

def objective(score,distance,mask,heads):
    b,c,s,n=score.shape
    valid=mask[:,None].expand(b,c,n).reshape(b*c,n)
    teacher=rt.normalized_teacher_prob(distance.reshape(b*c,n),valid,.1)
    if heads is None:
        # Historical Full V0 uses log_softmax KL, not the V5 probability clamp.
        return kl_loss(teacher,score[:,:,0].reshape(b*c,n),valid,.1)
    probabilities=torch.softmax((score/.1).masked_fill(~mask[:,None,None],-float('inf')),-1).mean(2)
    return rt.kl_loss_from_prob(teacher,probabilities.reshape(b*c,n),valid)

def bank_views(data,channels):
    """Strided channel windows over compact train data; all bank starts retained."""
    n=len(data.starts);h=data.h
    series=data.series[:,channels].T.contiguous()
    future=data.future_series[:,channels].T.contiguous()
    history=series.unfold(-1,h,1)[:,:n]
    values=future.unfold(-1,h,1)[:,h:h+n]-series[:,h-1:h-1+n,None]
    return history,values

def chunk_backward(model,heads,x,y,history,values,mask,total_channels,candidate_chunk=128):
    d=teacher_distances(values,x[:,:,-1],y,candidate_chunk)
    loss=objective(scores_group(model,heads,x,history),d,mask,heads)
    weighted=loss*(x.shape[1]/total_channels)
    weighted.backward()
    return float(weighted.detach())

def rng_snapshot():
    import random,numpy as np
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy=np.random.get_state(),python=random.getstate())

def rng_restore(state):
    import random,numpy as np
    torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy']);random.setstate(state['python'])
