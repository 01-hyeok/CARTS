"""CPU diagnostics; no CUDA environment mutation or future-driven fitting."""
import numpy as np

def tie_rank(x):
    order=np.argsort(x,axis=-1,kind='stable');s=np.take_along_axis(x,order,-1);n=x.shape[-1];pos=np.broadcast_to(np.arange(n),x.shape)
    start=np.concatenate([np.ones_like(s[...,:1],bool),s[...,1:]!=s[...,:-1]],-1)
    end=np.concatenate([s[...,:-1]!=s[...,1:],np.ones_like(s[...,:1],bool)],-1)
    lo=np.maximum.accumulate(np.where(start,pos,0),axis=-1);hi=np.minimum.accumulate(np.where(end,pos,n)[...,::-1],axis=-1)[...,::-1]
    ranks=np.empty(x.shape,float);np.put_along_axis(ranks,order,(lo+hi)/2,axis=-1);return ranks

def pairs(n,budget=512):
    if n*(n-1)//2<=budget:return np.triu_indices(n,1)
    rng=np.random.RandomState(1000+n);chosen=set()
    while len(chosen)<budget:
        a,b=rng.randint(n,size=2)
        if a!=b:chosen.add((min(a,b),max(a,b)))
    arr=np.array(sorted(chosen));return arr[:,0],arr[:,1]

def metrics(score,distance,pair_indices=None):
    assert score.shape==distance.shape and score.ndim==2 and score.shape[-1]>=10
    sr=tie_rank(score);dr=tie_rank(-distance);sr-=sr.mean(-1,keepdims=True);dr-=dr.mean(-1,keepdims=True)
    den=np.sqrt((sr*sr).sum(-1)*(dr*dr).sum(-1));defined=den>0;rho=np.divide((sr*dr).sum(-1),den,out=np.zeros(len(score)),where=defined)
    i,j=pairs(score.shape[-1]) if pair_indices is None else pair_indices;truth=distance[:,j]-distance[:,i];pred=score[:,i]-score[:,j];valid=np.abs(truth)>1e-10
    correct=np.where(pred==0,.5,(pred*truth>0).astype(float));weight=np.abs(truth)*valid
    chosen=np.argsort(-score,axis=-1,kind='stable')[:,:10];oracle=np.argsort(distance,axis=-1,kind='stable')[:,:10]
    sm=np.take_along_axis(distance,chosen,-1);om=np.take_along_axis(distance,oracle,-1);discount=1/np.log2(np.arange(2,12))
    return dict(rho_sum=float(rho.sum()),rho_n=int(defined.sum()),pair_correct=float((correct*valid).sum()),pair_n=int(valid.sum()),pair_ties=int((~valid).sum()),margin_correct=float((correct*weight).sum()),margin_sum=float(weight.sum()),ndcg_sum=float((((1/(1+sm))*discount).sum(-1)/((1/(1+om))*discount).sum(-1)).sum()),recall_sum=float((chosen[:,:,None]==oracle[:,None,:]).any(-1).mean(-1).sum()),rows=len(score))

def aggregate(values):
    return dict(spearman=values['rho_sum']/values['rho_n'] if values['rho_n'] else None,spearman_defined=values['rho_n'],pair_accuracy=values['pair_correct']/values['pair_n'] if values['pair_n'] else None,pair_denominator=values['pair_n'],future_ties=values['pair_ties'],margin_pair_accuracy=values['margin_correct']/values['margin_sum'] if values['margin_sum'] else None,ndcg10=values['ndcg_sum']/values['rows'],local_oracle_recall10=values['recall_sum']/values['rows'],query_channels=values['rows'])
