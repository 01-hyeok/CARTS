"""Predefined paired time-block bootstrap; all channels move together."""
import numpy as np

def paired_blocks(delta,length,repeats=2000,seed=0):
    x=np.asarray(delta,np.float64);assert x.ndim>=2
    # Channel dependence is retained through an equal-channel timestamp mean.
    t=np.nanmean(x,axis=1);q=len(t);block=min(int(length),q)
    sums=np.concatenate([np.zeros_like(t[:1]),np.nancumsum(t,axis=0)])
    counts=np.concatenate([np.zeros_like(t[:1]),np.cumsum(np.isfinite(t),axis=0)])
    # Non-overlapping temporal blocks (last shorter block retains exact weight).
    starts=np.arange(0,q,block);ends=np.minimum(starts+block,q)
    bs=sums[ends]-sums[starts];bc=counts[ends]-counts[starts]
    rng=np.random.default_rng(seed);est=[]
    for _ in range(repeats):
        pick=rng.integers(0,len(starts),len(starts));den=bc[pick].sum(0)
        est.append(np.divide(bs[pick].sum(0),den,out=np.full_like(den,np.nan,dtype=float),where=den>0))
    a=np.asarray(est);lo,hi=np.nanquantile(a,[.025,.975],axis=0)
    return dict(mean=np.nanmean(t,axis=0),ci_low=lo,ci_high=hi,block_length=block,effective_blocks=len(starts),
                sufficient=len(starts)>=4,repeats=repeats,seed=seed,queries=q,channels=x.shape[1],unit='timestamp blocks, all channels jointly',
                interpretation='exploratory; does not estimate training-seed uncertainty')

def sensitivity(delta,horizon):
    return [(label,paired_blocks(delta,length)) for label,length in [('primary',2*horizon),('half',horizon),('double',4*horizon)]]
