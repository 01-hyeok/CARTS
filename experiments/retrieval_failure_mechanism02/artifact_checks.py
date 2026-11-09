"""CPU checks for complete populations and unchanged diagnostic block files."""
from pathlib import Path
import numpy as np
from io_utils import read,sha

def validated_blocks(base,split,smoke=False):
    base=Path(base);gate=read(base/split/'SPLIT_GATE.json');proof=read(base/'REPRODUCTION_GATE.json')['identity']
    assert gate['status']==('SMOKE_PASS' if smoke else 'PASS') and gate['identity']==proof
    paths=sorted((base/split).glob('batch*.npz'));assert paths
    q=[];units=0
    for p in paths:
        side=read(p.with_suffix('.json'))
        assert side['status']=='PASS' and side['identity']==proof and side['sha256']==sha(p)
        assert side['channels']==gate['channels']
        with np.load(p) as z:
            starts=z['query_start'];assert len(starts)==side['queries']
            assert all(z[k].shape[:2]==(len(starts),gate['channels']) for k in z.files if k.endswith('__metrics'))
            q.extend(starts.tolist());units+=len(starts)*gate['channels']
    q=np.asarray(q,np.int64);assert len(np.unique(q))==len(q) and np.all(np.diff(q)>0)
    assert units==gate['units'] and len(q)==gate['queries']
    if not smoke:
        d=proof['data'];assert len(q)==d['counts'][split]
        np.testing.assert_array_equal(q,np.arange(d['starts'][split][0],d['starts'][split][1]+1))
    return paths,q,gate

def concatenate(paths,key):
    arrays=[]
    for p in paths:
        with np.load(p) as z:arrays.append(z[key].copy())
    return np.concatenate(arrays)

def strict_head_choice(validation_head_metrics):
    x=np.asarray(validation_head_metrics)
    assert x.ndim==4 and x.shape[2]==5 and np.isfinite(x[...,0]).all()
    return int(x[...,0].mean((0,1)).argmin())

def geometry_choice(p,r):
    choices=[]
    for k,o in [(2,3),(3,6)]:
        sil=float(np.median(p[...,o]));size=float(np.median(p[...,o+1]));stability=float(np.median(p[...,o+2]))
        if sil>=.25 and size>=3 and stability>=.8 and sil-float(np.median(r[...,o]))>=.10:choices.append((sil,k))
    return max(choices,key=lambda z:(z[0],-z[1]))[1] if choices else 1

def finite_float(value):
    v=float(value);return v if np.isfinite(v) else None
