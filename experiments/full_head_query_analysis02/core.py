import json,hashlib,itertools
from pathlib import Path
import numpy as np
NAMES=['1','2','3','4','5','12345','2345','1345','1245','1235','1234']
SUBSETS=[''.join(map(str,s)) for n in range(1,6) for s in itertools.combinations(range(1,6),n)]
FEATURES=['normalized_trend','difference_volatility','spectral_peak_fraction','normalized_max_jump']
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def features(x):
 x=np.asarray(x,dtype=np.float64);n=x.shape[1];t=np.arange(n)-(n-1)/2;t=t[None,:,None]
 centered=x-x.mean(1,keepdims=True);slope=(centered*t).sum(1)/(t*t).sum();scale=x.std(1);diff=np.diff(x,axis=1);vol=diff.std(1)
 detrended=centered-slope[:,None,:]*t;power=abs(np.fft.rfft(detrended,axis=1))**2;power[:,0]=0
 peak=power.max(1)/np.maximum(power.sum(1),1e-20)
 return np.stack([slope*n/np.maximum(scale,1e-10),vol,peak,abs(diff).max(1)/np.maximum(vol,1e-10)],-1).astype('float32')
def bins(val,test):
 cuts=np.quantile(val,[1/3,2/3],axis=0).transpose(1,2,0)
 return cuts,(test[...,None]>cuts[None]).sum(-1).astype('int8')
def block_ci(delta,mask=None,block=192,reps=5000,family=20):
 d=np.asarray(delta,dtype='float64');assert d.ndim==2
 m=np.ones_like(d,dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
 assert m.shape==d.shape
 n,c=d.shape;st=list(range(0,n,block));s=np.stack([np.where(m[i:i+block],d[i:i+block],0).sum(0) for i in st]);cnt=np.stack([m[i:i+block].sum(0) for i in st]);den=cnt.sum(0);valid=den>0
 point=float(np.mean(s.sum(0)[valid]/den[valid])) if valid.any() else float('nan')
 if not valid.any():return dict(mean=point,n_query_channels=0,n_channels=0,n_blocks=len(st),ci_low=None,ci_high=None,family_low=None,family_high=None,status='EMPTY')
 rng=np.random.default_rng(20261029);draw=rng.integers(0,len(st),size=(reps,len(st)));bs=s[draw].sum(1);bc=cnt[draw].sum(1);v=bc>0
 means=np.divide(bs,bc,out=np.zeros_like(bs),where=v);eligible=(bc[:,valid]>0).all(1);boot=means[eligible][:,valid].mean(1)
 if not len(boot):return dict(mean=point,n_query_channels=int(m.sum()),n_channels=int(valid.sum()),n_blocks=len(st),ci_low=None,ci_high=None,family_low=None,family_high=None,status='NO_FULL_CHANNEL_BOOTSTRAP_DRAWS',bootstrap_valid_draws=0,bootstrap_requested_draws=reps)
 lo,hi=np.quantile(boot,[.025,.975]);flo,fhi=np.quantile(boot,[.025/family,1-.025/family]);enough=len(st)>=8
 return dict(mean=point,n_query_channels=int(m.sum()),n_queries=int(m.any(1).sum()),n_channels=int(valid.sum()),n_blocks=len(st),full_blocks=n//block,block_length=block,bootstrap_valid_draws=int(eligible.sum()),bootstrap_requested_draws=reps,ci_low=float(lo),ci_high=float(hi),family_low=float(flo) if enough else None,family_high=float(fhi) if enough else None,status='EXPLORATORY' if enough else 'LOW_BLOCK_COUNT')
def swap_masks(old,new):
 return ~(new[...,None]==old[...,None,:]).any(-1),~(old[...,None]==new[...,None,:]).any(-1)
def delta_decomposition(error,change,lam):
 return 2*lam*np.mean(error*change,axis=-1),lam*lam*np.mean(change*change,axis=-1)

def comparison_events(delta,selected_ids,reference_ids):
 changed=(np.sort(selected_ids,axis=-1)!=np.sort(reference_ids,axis=-1)).any(-1)
 return changed,(delta>0)&changed,(delta<0)&changed
