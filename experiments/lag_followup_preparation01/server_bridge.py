"""Dedicated frozen adapter for the two lag tracks; never mutates old sources."""
import hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
import numpy as np

ROOT=Path('/data/pjh_workspace/CARTS')
HERE=Path(__file__).resolve().parent
GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
sys.path.insert(0,str(ROOT/'experiments/retrieval_failure_mechanism02'))
from adapter import Data,Frozen,inventory,normalize,topk,distances,rt
import torch

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def dump(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=str(p.parent),prefix=p.name+'.')
    with os.fdopen(fd,'w') as f:json.dump(obj,f,indent=2,allow_nan=False)
    os.replace(tmp,p)

def snapshot():
    raw=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=index,uuid,memory.free,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],universal_newlines=True)
    i,u,free,used,total,util=[x.strip() for x in raw.strip().split(',')]
    assert i=='1' and u==GPU
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],universal_newlines=True)
    processes=[]
    for line in apps.splitlines():
        parts=[x.strip() for x in line.split(',')]
        if parts and parts[0]==GPU:processes.append(parts[1:])
    return dict(uuid=u,free_mib=int(free),used_mib=int(used),total_mib=int(total),utilization_percent=int(util),processes=processes)

def start_gpu():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
    before=snapshot()
    assert len(before['processes'])<=1,'At most the two explicitly authorized lag tracks may coexist'
    if before['processes']:
        owners=[]
        for track in ['TRACK-SOFTLAG-HEAD-PILOT01','TRACK-HEAD-SHIFT-DIAGNOSTIC01']:
            p=ROOT/'results'/track/'00_Audit/GPU_OWNER.json'
            if not p.exists():continue
            owner=json.loads(p.read_text());proc=Path('/proc')/str(owner['pid'])
            if proc.exists():
                cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
                if owner['pid']!=os.getpid() and str(HERE) in cmd and any(x in cmd for x in ['softlag_run.py','d4_run.py','gpu_gate.py']):owners.append(owner)
        assert len(owners)==1,'Existing GPU process is not an authenticated other lag-track worker'
        before['authorized_other_owner']=owners[0]
    assert before['free_mib']>=24576
    rt.gpu_guard();torch.set_num_threads(1)
    torch.manual_seed(0);np.random.seed(0)
    torch.cuda.set_per_process_memory_fraction(.40,0)
    return before

def register_owner(track):
    dump(ROOT/'results'/track/'00_Audit/GPU_OWNER.json',dict(pid=os.getpid(),command=sys.argv,gpu_uuid=GPU))

def frozen(cell,support):
    os.chdir(ROOT)
    data=Data(cell)
    row=next(r for r in inventory(cell) if r['architecture']=='V5' and r['training_support']==support)
    prior=ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02/checkpoint_runs'/cell/('V5_'+support)/'REPRODUCTION_GATE.json'
    gate=json.loads(prior.read_text())
    assert gate['status']=='PASS' and gate['identity']['checkpoint_sha256']==row['checkpoint_sha256']
    assert gate['identity']['pool_hashes']==data.pool_hashes
    assert {r['split'] for r in gate['rows']}=={'val','test'}
    assert all(r['ordered_mismatch_count']==0 and r['units']==r['queries']*r['channels'] for r in gate['rows'])
    f=Frozen(data,row)
    return data,f,dict(checkpoint=row,reproduction_gate_sha256=sha(prior),pool_hashes=data.pool_hashes,
                       data=data.data_info,source_hashes={name:sha(HERE/name) for name in ['server_bridge.py','past_lag.py','softlag.py']},
                       original_runtime='UNKNOWN',current_torch=torch.__version__)

def native(f,split,a,b):
    d=f.data
    if f.full_trained:
        g,mask=f.full_logits(split,a,b)
        ids=torch.arange(len(d.starts),device=d.device).expand(b-a,d.c,-1)
    else:
        g,ids=f.native_p100_logits(split,a,b);mask=torch.ones_like(ids,dtype=torch.bool)
    return g,mask,ids,normalize(g,mask)

def phased_mixture(g,mask,phase,support):
    """Preserve original C-first CUDA softmax row alignment for sampled queries.

    Each sampled row owns four query slots; its original query index modulo four
    determines its slot. Channel axis remains intact in one softmax call.
    """
    if support=='P100':return normalize(g,mask)
    B,C,S,N=g.shape
    z=g.new_zeros(B,4,C,S,N)
    m=torch.zeros(B,4,C,N,device=g.device,dtype=torch.bool)
    m[:]=True
    ar=torch.arange(B,device=g.device)
    z[ar,phase]=g;m[ar,phase]=mask
    p=normalize(z.reshape(B*4,C,S,N),m.reshape(B*4,C,N))
    return p.reshape(B,4,C,N)[ar,phase]

def sample_indices(n,k):return np.unique(np.linspace(0,n-1,min(n,k),dtype=np.int64))
