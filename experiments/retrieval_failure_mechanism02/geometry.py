"""Small-positive-set geometry. K=1/null is a first-class outcome.

All leading dimensions (query AND channel) are batched. K-means here is only a
diagnostic on frozen vectors, never an encoder/head parameter update.
"""
import torch

FIELDS=['pair_distance_mean','pair_distance_p90','effective_rank',
        'k2_silhouette','k2_min_size','k2_stability_ari','k3_silhouette','k3_min_size','k3_stability_ari',
        'k2_within_distance','k2_between_distance','k3_within_distance','k3_between_distance']

def squared(x):
    d=(x.square().sum(-1,keepdim=True)+x.square().sum(-1).unsqueeze(-2)-2*torch.matmul(x,x.transpose(-1,-2)))
    return d.clamp_min(0)/x.shape[-1] # roundoff in squared-distance identity only

def lloyd(x,k,iterations=15):
    n=x.shape[-2];flat=x.reshape(-1,n,x.shape[-1]);B=len(flat);r=torch.arange(B,device=x.device)
    centers=[flat[:,0]]
    for _ in range(1,k):
        z=torch.stack(centers,1);d=(flat[:,:,None]-z[:,None]).square().mean(-1).min(-1).values
        centers.append(flat[r,d.argmax(-1)])
    z=torch.stack(centers,1)
    for _ in range(iterations):
        dist=(flat[:,:,None]-z[:,None]).square().mean(-1);label=dist.argmin(-1)
        one=torch.nn.functional.one_hot(label,k).to(x.dtype);size=one.sum(1)
        means=torch.einsum('bnk,bnd->bkd',one,flat)/size.clamp_min(1)[...,None]
        z=torch.where((size>0)[...,None],means,z)
    return label.reshape(*x.shape[:-2],n),size.reshape(*x.shape[:-2],k)

def adjusted_rand(a,b,k):
    assert a.shape==b.shape
    joint=torch.nn.functional.one_hot(a*k+b,k*k).sum(-2).double()
    tabs=joint.reshape(*a.shape[:-1],k,k);row=tabs.sum(-1);col=tabs.sum(-2)
    comb=lambda v:v*(v-1)/2
    n=a.shape[-1];s=comb(tabs).sum((-1,-2));ra=comb(row).sum(-1);cb=comb(col).sum(-1)
    exp=ra*cb/(n*(n-1)/2);den=(ra+cb)/2-exp
    return torch.where(den.abs()<1e-15,torch.ones_like(den),(s-exp)/den)

def silhouette(pair,label,k):
    n=label.shape[-1];one=torch.nn.functional.one_hot(label,k).to(pair.dtype);counts=one.sum(-2)
    sums=torch.matmul(pair.sqrt(),one);own=sums.gather(-1,label[...,None])[...,0]
    own_n=counts.gather(-1,label)-1;a=own/own_n.clamp_min(1)
    means=sums/counts.clamp_min(1).unsqueeze(-2)
    means=means.masked_fill(one.bool(),float('inf')).masked_fill((counts==0).unsqueeze(-2),float('inf'))
    b=means.min(-1).values;s=(b-a)/torch.maximum(a,b).clamp_min(1e-15)
    s=torch.where((own_n<=0)|~torch.isfinite(s),torch.zeros_like(s),s)
    return s.mean(-1)

@torch.no_grad()
def describe(x):
    assert x.ndim==4,'B,C,P,D required; channels may not be looped'
    n=x.shape[-2];pair=squared(x);tri=torch.triu_indices(n,n,1,device=x.device)
    values=pair[...,tri[0],tri[1]].sqrt()
    centered=x-x.mean(-2,keepdim=True);gram=torch.matmul(centered,centered.transpose(-1,-2)).double()
    eig=torch.linalg.eigvalsh(gram).clamp_min(0);p=eig/eig.sum(-1,keepdim=True).clamp_min(1e-30)
    erank=torch.exp(-(p*torch.log(p.clamp_min(1e-30))).sum(-1));erank=torch.where(eig.sum(-1)>1e-20,erank,torch.zeros_like(erank))
    vals=[values.mean(-1),torch.quantile(values,.9,dim=-1),erank];labels=[]
    gen=torch.Generator(device='cpu').manual_seed(0)
    subs=[torch.randperm(n,generator=gen)[:max(2,int(.8*n))].sort().values.to(x.device) for _ in range(20)]
    sep=[]
    for k in (2,3):
        label,size=lloyd(x,k);stability=[]
        for sub in subs:
            fit,_=lloyd(x[...,sub,:],k)
            stability.append(adjusted_rand(label[...,sub],fit,k))
        vals.extend([silhouette(pair,label,k),size.min(-1).values,torch.stack(stability,-1).median(-1).values]);labels.append(label)
        same=(label[...,tri[0]]==label[...,tri[1]])
        for mask in (same,~same):
            den=mask.sum(-1);sep.append((values.masked_fill(~mask,0).sum(-1)/den.clamp_min(1)).masked_fill(den==0,float('nan')))
    return torch.stack([v.double() for v in vals+sep],-1),torch.stack(labels,-2)

def validation_choice(positive,random_control):
    """CPU/NumPy arrays, aggregate validation only. No forced cluster choice."""
    import numpy as np
    pos=np.asarray(positive);rnd=np.asarray(random_control)
    assert pos.shape==rnd.shape and pos.shape[-1]==len(FIELDS)
    choices=[]
    for k,offset in [(2,3),(3,6)]:
        silhouette=float(np.median(pos[...,offset]));stability=float(np.median(pos[...,offset+2]));size=float(np.median(pos[...,offset+1]))
        advantage=silhouette-float(np.median(rnd[...,offset]))
        if silhouette>=.25 and stability>=.8 and size>=3 and advantage>=.10:choices.append((silhouette,k))
    return max(choices,key=lambda x:(x[0],-x[1]))[1] if choices else 1
