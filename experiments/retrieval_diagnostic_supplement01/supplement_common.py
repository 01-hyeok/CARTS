"""Additive diagnostic completion; does not modify original artifacts."""
import csv, hashlib, json, math, os
from pathlib import Path
import numpy as np

ROOT = Path('/data/pjh_workspace/CARTS')
ORIGINAL = ROOT / 'experiments/retrieval_failure_attribution01'
SOURCE = ROOT / 'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01'
HERE = ROOT / 'experiments/retrieval_diagnostic_supplement01'
OUT = ROOT / 'results/TRACK-RETRIEVAL-DIAGNOSTIC-SUPPLEMENT01'
CELLS = [f'{d}_{h}' for d in ['ETTh1', 'Weather', 'Solar'] for h in [96,192,336,720]]

def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def clean(x):
    if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [clean(v) for v in x]
    if isinstance(x,np.generic): x=x.item()
    if isinstance(x,float) and not math.isfinite(x): return None
    return x
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(clean(x),indent=2,ensure_ascii=False)+'\n',encoding='utf8');os.replace(tmp,p)
def csvwrite(p,rows):
    assert rows, str(p)
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows([clean(r) for r in rows])
def csvread(p): return list(csv.DictReader(Path(p).open(encoding='utf8')))
def source_hashes(): return {p.name:sha(p) for p in HERE.glob('*.py')}

def thresholds(validation):
    """Fixed validation-only per-channel terciles. Undefined inputs stay undefined."""
    assert validation.ndim==2
    out=np.full((validation.shape[1],2),np.nan)
    for c in range(validation.shape[1]):
        a=validation[:,c];a=a[np.isfinite(a)]
        if len(a):out[c]=np.quantile(a,[1/3,2/3])
    return out
def strata(values,cuts):
    assert cuts.shape==(values.shape[1],2)
    # Ties use low first, then middle. Empty strata are retained in tables.
    return np.where(~np.isfinite(values),-1,np.where(values<=cuts[None,:,0],0,np.where(values<=cuts[None,:,1],1,2)))
def interval_status(q,block):
    effective=q/min(q,block)
    return ('INSUFFICIENT_TIME_BLOCKS' if effective<2 else 'EXPLORATORY_MULTIPLE_COMPARISONS_UNADJUSTED',effective)
def pooled_rank_stats(hist):
    """Exact integer-rank empirical quantiles, numpy linear interpolation convention."""
    hist=np.asarray(hist,np.int64);n=int(hist.sum())
    if n==0:return dict(missed_candidates=0,pooled_median=None,pooled_p90=None,pooled_mean=None)
    cumulative=hist.cumsum()
    def quantile(p):
        v=(n-1)*p;lo=math.floor(v);hi=math.ceil(v)
        a=np.searchsorted(cumulative,lo+1);b=np.searchsorted(cumulative,hi+1)
        return float(a+(b-a)*(v-lo))
    return dict(missed_candidates=n,pooled_median=quantile(.5),pooled_p90=quantile(.9),pooled_mean=float(np.dot(np.arange(len(hist),dtype=np.float64),hist)/n))
