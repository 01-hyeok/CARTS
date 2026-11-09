"""Isolated track locations and immutable artifact utilities."""
import os, json, hashlib, datetime, tempfile
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS')
HERE=ROOT/'experiments/learned_dynamics_cross01'
OUT=ROOT/'results/TRACK-LEARNED-DYNAMICS-CROSS01'
GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
ARMS=['D1_L2','D1_MLP','D2_L2','D2_MLP','C1_MLP','C2_MLP']
CELLS=['ETTh1_96','ETTh1_720','Weather_96','Weather_336','Weather_720','ETTh1_192','ETTh1_336','Weather_192']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=str(p.parent),prefix=p.name+'.')
    with os.fdopen(fd,'w') as f:json.dump(x,f,indent=2,allow_nan=False)
    os.replace(tmp,p)
def torch_save(p,x):
    import torch
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');torch.save(x,tmp);os.replace(tmp,p)
def load(p):
    import torch
    try:return torch.load(p,map_location='cpu',weights_only=False)
    except TypeError:return torch.load(p,map_location='cpu')
def sources():return {p.name:sha(p) for p in HERE.glob('*.py')}
