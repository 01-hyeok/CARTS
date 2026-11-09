"""All-channel Full training/evaluation; only query/candidate axes are chunked."""
import sys
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
ROOT=Path('/data/pjh_workspace/CARTS')
_initial_path=list(sys.path)
sys.path.insert(0,str(ROOT/'experiments/solar_full_missing_checkpoints01'))
import solar_full_core as original
# Import the verified core without letting its old directory shadow the new
# train/controller/diagnostic modules when they are imported later.
sys.path[:]=_initial_path+[p for p in sys.path if p not in _initial_path]
rt=original.rt
encode_raw_channel_first=original.encode_raw_channel_first
objective=original.objective
rng_snapshot=original.rng_snapshot
rng_restore=original.rng_restore

def query(data,split,start,end):
    ds=data.datasets[split];idx=np.arange(start,end)[:,None]+np.arange(data.h)[None]
    x=torch.as_tensor(np.asarray(ds.data_x[idx],np.float32),device=data.device).permute(0,2,1).contiguous()
    y=torch.as_tensor(np.asarray(ds.data_y[idx+data.h],np.float32),device=data.device).permute(0,2,1).contiguous()
    return x,y,ds.starts[start:end]

def history_chunk(data,start,end):
    idx=torch.arange(start,end,device=data.device)[:,None]+data.offsets
    return data.series[idx].permute(2,0,1).contiguous() # C,P,H: complete C

def values_chunk(data,start,end):
    ids=torch.arange(start,end,device=data.device)
    value=(data.future_series[ids[:,None]+data.h+data.offsets]-data.series[ids+data.h-1,None,:]).permute(2,0,1)
    # Match original bank_views' C,H,N physical layout. Merely making C,N,H
    # contiguous changes FP32 horizon-reduction order (observed max 3.8e-6).
    # Candidate streaming changes the stride magnitude, not the axis order.
    return value.transpose(-1,-2).contiguous().transpose(-1,-2)

def selected_values(data,ids,last):
    c=torch.arange(data.c,device=data.device)[None,:,None,None]
    value=data.future_series[ids[...,None]+data.h+data.offsets,c]
    offset=data.series[ids+data.h-1,torch.arange(data.c,device=data.device)[None,:,None]]
    return value-offset[...,None]+last[:,:,None,None]

def dropout_module(model):
    drops=[m for m in model.encoder.modules() if isinstance(m,torch.nn.Dropout)]
    assert model.encoder.encoder_type=='mlp' and len(drops)==1,'Unreviewed architecture; stop'
    return drops[0]

@torch.no_grad()
def dropout_plan(model,b,c,n,device):
    drop=dropout_module(model)
    if not model.training or drop.p==0:return None
    width=model.encoder.encoder[0].out_features
    # Exactly the canonical grouped RNG order/shapes: query B*C, then key C*N.
    # Store boolean masks only. Chunking never changes logical dropout draws.
    qm=F.dropout(torch.ones(b*c,width,device=device),p=drop.p,training=True).ne(0).reshape(b,c,width)
    km=F.dropout(torch.ones(c*n,width,device=device),p=drop.p,training=True).ne(0).reshape(c,n,width)
    return qm,km

@contextmanager
def fixed_dropout(model,mask):
    if mask is None:
        yield;return
    drop=dropout_module(model);old=drop.forward
    def apply(x):
        assert x.shape==mask.shape and drop.training
        return x*mask.to(x.dtype)*(1.0/(1.0-drop.p))
    drop.forward=apply
    try:yield
    finally:drop.forward=old

def encode(model,x,mask=None):
    shape=x.shape[:-1];h=x.shape[-1]
    m=None if mask is None else mask.reshape(-1,mask.shape[-1])
    with fixed_dropout(model,m):z=encode_raw_channel_first(model,x.reshape(-1,h,1),0)
    return z.reshape(*shape,-1)

def keys(model,data,candidate_chunk,masks=None,recompute=False):
    assert candidate_chunk>0;parts=[];n=len(data.starts)
    for a in range(0,n,candidate_chunk):
        b=min(a+candidate_chunk,n)
        # Raw window chunks are created inside the checkpoint closure, not
        # retained as large checkpoint inputs across the full bank.
        def block(a=a,b=b):
            h=history_chunk(data,a,b);assert h.shape[0]==data.c
            mask=None if masks is None else masks[:,a:b]
            return F.normalize(encode(model,h,mask),dim=-1)
        parts.append(checkpoint(block,use_reentrant=False,preserve_rng_state=False) if recompute else block())
    result=torch.cat(parts,1);assert result.shape[:2]==(data.c,n)
    return result

def scores(model,heads,x,key,qmask=None):
    b,c,_=x.shape;assert c==key.shape[0]
    z=encode(model,x,qmask);d=z.shape[-1]
    q=F.normalize(z,dim=-1)[:,:,None] if heads is None else heads(z.reshape(b*c,d)).reshape(b,c,-1,d)
    return torch.einsum('bcsd,cnd->bcsn',q,key)

@torch.no_grad()
def distances(data,x,y,candidate_chunk):
    b,c,_=x.shape;assert c==data.c;n=len(data.starts);out=x.new_empty(b,c,n)
    for a in range(0,n,candidate_chunk):
        v=values_chunk(data,a,min(a+candidate_chunk,n))
        out[...,a:a+candidate_chunk]=((v[None]+x[:,:,-1,None,None]-y[:,:,None])**2).mean(-1)
    return out

def backward_batch(data,model,heads,x,y,starts,query_chunk,candidate_chunk,plan=None,distance_chunk=128):
    # Caller owns zero_grad and exactly one optimizer step for original batch32.
    b,c,_=x.shape;assert c==data.c and query_chunk>0 and candidate_chunk>0
    plan=dropout_plan(model,b,c,len(data.starts),x.device) if plan is None else plan
    mask=data.mask(starts);total=0.
    for a in range(0,b,query_chunk):
        stop=min(a+query_chunk,b);qm=None if plan is None else plan[0][a:stop];km=None if plan is None else plan[1]
        key=keys(model,data,candidate_chunk,km,recompute=True)
        score=scores(model,heads,x[a:stop],key,qm)
        d=distances(data,x[a:stop],y[a:stop],distance_chunk)
        loss=objective(score,d,mask[a:stop],heads)*((stop-a)/b)
        loss.backward();total+=float(loss.detach())
        del key,score,d,loss
    return total

def validate_training_shape(data,query_chunk,candidate_chunk):
    # Chunked GEMM/gradient accumulation passes gradient tolerances but failed
    # the unchanged Adam-update gate. Never silently deploy that fallback.
    assert query_chunk==32 and candidate_chunk>=len(data.starts), 'BLOCKED: chunked training Adam parity is unverified; full grouped encoder shape required'

def probability(score,mask,heads):
    if heads is None:return score[:,:,0].masked_fill(~mask[:,None],-float('inf'))
    return torch.softmax((score/.1).masked_fill(~mask[:,None,None],-float('inf')),-1).mean(2)
