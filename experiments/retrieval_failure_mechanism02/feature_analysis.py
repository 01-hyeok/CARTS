"""Reuse existing past-only feature definitions; all channels remain batched."""
import numpy as np
import torch
import torch.nn.functional as F
from diagnostic_metrics import features,FEATURES,temporal
from adapter import topk
from geometry import describe,FIELDS as GEOMETRY_FIELDS

@torch.no_grad()
def past_delta_scores(data,x,chunk=128):
    B,C,H=x.shape;N=len(data.starts);score=torch.empty(B,C,N,device=data.device)
    q=F.normalize(x-x[:,:,-1,None],dim=-1)
    for a in range(0,N,chunk):
        ids=torch.arange(a,min(a+chunk,N),device=data.device).expand(B,C,-1)
        h=data.histories_all(ids);key=F.normalize(h-h[:,:,:,-1,None],dim=-1)
        score[...,a:a+chunk]=(key*q[:,:,None]).sum(-1)
    return score

def random_scores(starts,C,N,device):
    a=np.stack([np.stack([np.random.default_rng(np.random.SeedSequence([0,int(q),c,20261102])).random(N) for c in range(C)]) for q in starts])
    return torch.as_tensor(a,device=device,dtype=torch.float32)

@torch.no_grad()
def groups(data,x,ev,random_ids):
    B,C,H=x.shape;ids=torch.cat([ev['oracle_ids'],ev['selected_ids'],random_ids],-1)
    history=data.histories_all(ids);f=features(x.reshape(B*C,H),history.reshape(B*C,30,H)).reshape(B,C,30,-1)
    oracle=ev['oracle_ids'];selected=ev['selected_ids']
    found=(oracle[...,None]==selected[:,:,None]).any(-1);hit=(selected[...,None]==oracle[:,:,None]).any(-1)
    masks=[found,~found,~hit,torch.ones_like(found)]
    blocks=[f[...,:10,:],f[...,:10,:],f[...,10:20,:],f[...,20:,:]]
    sums=[];counts=[]
    for block,mask in zip(blocks,masks):
        finite=torch.isfinite(block)&mask[...,None]
        sums.append(block.masked_fill(~finite,0).double().sum(-2));counts.append(finite.sum(-2))
    return torch.stack(sums,2),torch.stack(counts,2)

@torch.no_grad()
def geometry_sets(f,x,dist,mask,ids,past_score,ev,random_score,raw_spaces):
    data=f.data;B,C,H=x.shape
    if f.batch.keys is None:f.batch.build_keys()
    output={}
    for size in (10,30):
        oi=ev['oracle100_ids'][...,:size]
        ri=ids.gather(-1,topk(random_score.masked_fill(~mask,-float('inf')),size))
        pi=ids.gather(-1,topk(past_score.masked_fill(~mask,-float('inf')),size))
        for group,chosen in [('oracle',oi),('random',ri),('past_nearest',pi)]:
            hist=data.histories_all(chosen);embed=f.batch.keys[torch.arange(C,device=data.device)[None,:,None],chosen]
            spaces={'candidate_embedding':embed}
            if raw_spaces:
                spaces.update(raw=hist,delta_last=hist-hist[:,:,:,-1,None],pearson=F.normalize(hist-hist.mean(-1,keepdim=True),dim=-1))
            for space,z in spaces.items():
                stats,labels=describe(z)
                output[f'{size}_{group}_{space}_stats']=stats
                output[f'{size}_{group}_{space}_labels']=labels
            output[f'{size}_{group}_ids']=chosen
            values=data.values_all(chosen)+x[:,:,-1,None,None]
            output[f'{size}_{group}_future_dispersion']=values.double().var(-2,unbiased=False).mean(-1)
        output[f'{size}_oracle_random_overlap']=(oi[...,None]==ri[:,:,None]).any(-1).double().mean(-1)
        output[f'{size}_oracle_past_overlap']=(oi[...,None]==pi[:,:,None]).any(-1).double().mean(-1)
    return output
