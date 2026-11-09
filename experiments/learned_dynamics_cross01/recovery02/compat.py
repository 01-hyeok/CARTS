"""Versioned evaluation-only stable ordering compatibility; training unchanged."""
import sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'core'))
import final_evaluation as original

def stable_order(v):
    # mergesort preserves the input position of equal keys, including infinities.
    a=v.detach().cpu().numpy()
    return torch.from_numpy(np.argsort(a,axis=-1,kind='mergesort').copy()).to(v.device)

def ranks(value,mask):
    v=value.double().masked_fill(~mask,torch.inf)
    order=stable_order(v);s=v.gather(-1,order);N=v.shape[-1]
    pos=torch.arange(N,device=v.device).expand_as(order)
    start=torch.cat([torch.ones_like(s[...,:1],dtype=torch.bool),s[...,1:]!=s[...,:-1]],-1)
    end=torch.cat([s[...,:-1]!=s[...,1:],torch.ones_like(s[...,:1],dtype=torch.bool)],-1)
    first=torch.where(start,pos,torch.zeros_like(pos)).cummax(-1).values
    last=torch.where(end,pos,torch.full_like(pos,N)).flip(-1).cummin(-1).values.flip(-1)
    result=torch.empty_like(v);result.scatter_(-1,order,(first+last).double()/2)
    return result

def install():
    original.ranks=ranks

