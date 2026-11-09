"""Reuse immutable, previously validated all-channel kernels and source artifacts."""
import sys,os,json,time,datetime,hashlib,subprocess,fcntl
from pathlib import Path
from fusion_config import ROOT,HERE,OUT,GPU,POP,PROTOCOL
LAG=ROOT/'experiments/lag_followup_preparation01'
sys.path.insert(0,str(LAG))
from server_bridge import Data,Frozen,inventory,normalize,topk,rt,sha,dump,snapshot
from d123_full_population import candidate_distances,OracleSource,head_probability
from d56_complete_run import save,read_npz,canonical_cross_ids,reproduction
import torch
import numpy as np
sys.path.insert(0,str(LAG/'population_recovery06'))
from exact_patch import install
sys.path.insert(0,str(HERE))
from score_fusion import fuse,GAMMAS,gamma_select

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def guard():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
    before=snapshot();assert not before['processes'],('GPU1 is occupied; no existing process is changed',before)
    assert before['free_mib']>=24576
    rt.gpu_guard();torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
    torch.cuda.set_per_process_memory_fraction(.65,0)
    return before

def source_hashes():
    paths=set(HERE.glob('*.py'))
    paths.add(HERE/'SOURCE_REQUEST.md')
    for module in list(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if p and str(p).startswith(str(ROOT)) and str(p).endswith('.py'):paths.add(Path(p))
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}

class Prior:
    def __init__(self,d,row):
        self.d=d;self.row=row;self.base=POP/d.cell/('V5_'+row['training_support'])
        p=self.base/'GATE.json';self.g=json.loads(p.read_text());i=self.g['identity']
        assert self.g['status']=='PASS_FULL_QUERY_D123_RAW'
        assert i['checkpoint']==row and i['pool_hashes']==d.pool_hashes and i['data']==d.data_info
        assert i['current_torch']==torch.__version__,('runtime mismatch',i['current_torch'],torch.__version__)
        assert self.g['queries']=={s:len(d.datasets[s]) for s in ('val','test')} and self.g['channels']==d.c
        assert i['adapter_sha256']==sha(ROOT/'experiments/retrieval_failure_mechanism02/adapter.py')
        assert i['source_sha256']==sha(LAG/'d123_full_population.py')
        assert self.g['canonical_native_ordered_ids_checked']
        self.identity=dict(gate_path=str(p),gate_sha256=sha(p))
        self.oracles={}
    def batch(self,s,a,b):
        p=self.base/s/('batch%05d.npz'%(a//32));assert sha(p)==self.g['files'][str(p.relative_to(self.base))]
        z=read_npz(p);assert np.array_equal(z['query_start'],self.d.datasets[s].starts[a:b])
        return z
    def baseline(self,support,split,a,b):
        key=(support,split)
        if key not in self.oracles:self.oracles[key]=OracleSource(self.d,support,split)
        src=self.oracles[key];oids,oret,proof=src.batch(a,b)
        p=src.path/split/('batch%05d.npz'%(a//32));z=read_npz(p)
        mi=list(z['method_names']).index('Random');ki=list(z['metric_names']).index('retmse')
        ids=torch.as_tensor(z['selected_ids'][:,:,mi],device=self.d.device)
        return oids,oret,ids,z['metrics'][:,:,mi,ki],proof

def setup(cell,training):
    os.chdir(ROOT);before=guard();patch=install();d=Data(cell)
    row=next(r for r in inventory(cell) if r['architecture']=='V5' and r['training_support']==training)
    assert reproduction(d,row) is not None
    f=Frozen(d,row);f.precision();prior=Prior(d,row)
    identity=dict(checkpoint=row,pool_hashes=d.pool_hashes,data=d.data_info,protocol=PROTOCOL,
        source_hashes=source_hashes(),torch_version=torch.__version__,cuda_version=torch.version.cuda,
        original_runtime='UNKNOWN',patch=patch,prior=prior.identity,gpu_uuid=GPU)
    return d,f,prior,identity,before

def batch_logits(f,split,a,b):
    d=f.data;g,m=f.full_logits(split,a,b);pool=d.pool_all(split,a,b)
    assert m.gather(-1,pool).all() and (pool.sort(-1).values[...,1:]!=pool.sort(-1).values[...,:-1]).all()
    ids=torch.arange(len(d.starts),device=d.device).expand(b-a,d.c,-1)
    controlled=g.gather(-1,pool[:,:,None].expand(-1,-1,5,-1))
    pg,pids=f.native_p100_logits(split,a,b) if not f.full_trained else (controlled,pool)
    assert torch.equal(pool,pids)
    return dict(FULL=(g,m,ids),P100=(pg,torch.ones_like(pool,dtype=torch.bool),pool)),controlled

def variant_name(training,support):
    return 'FULL' if support=='FULL' else ('P100_NATIVE' if training=='P100' else 'P100_CONTROLLED')

def score(g,mask,method,gamma=None):
    return normalize(g,mask) if method=='E0' else fuse(g,method,gamma).masked_fill(~mask,-float('inf'))

def e0_gate(g,m,ids,prior,variant,d,x,y):
    p=normalize(g,m);chosen=ids.gather(-1,topk(p,10))
    assert np.array_equal(chosen.cpu().numpy(),prior[variant+'__selected_ids']),('E0 ordered ID mismatch',variant)
    headids=prior[variant+'__head_ids'].reshape(*ids.shape[:2],50)
    # Map stored global IDs back through the immutable support, including ordered P100.
    if ids.shape[-1]>100:loc=torch.as_tensor(headids,device=ids.device)
    else:
        target=torch.as_tensor(headids,device=ids.device)
        found=ids[...,None,:]==target[...,None]
        assert found.sum(-1).eq(1).all();loc=found.long().argmax(-1)
    actual=g.gather(-1,loc[:,:,None].expand(-1,-1,5,-1))
    assert np.array_equal(actual.cpu().numpy(),prior[variant+'__head_candidate_all_head_logits']),('head cosine mismatch',variant)
    saved=prior[variant+'__head_candidate_mixture_score'].reshape(*ids.shape[:2],50)
    assert np.array_equal(p.gather(-1,loc).cpu().numpy(),saved),('E0 score mismatch',variant)
    ds=candidate_distances(d,x,y,chosen)
    assert np.array_equal(ds.cpu().numpy(),prior[variant+'__selected_mse']),('E0 candidate MSE mismatch',variant)
    return p,chosen

def rank100(v):
    order=topk(v,100);rank=torch.empty_like(order)
    rank.scatter_(-1,order,torch.arange(100,device=v.device).expand_as(order))
    return rank.double()

def stability(full_score,p_score,pool):
    restricted=full_score.gather(-1,pool);a=rank100(restricted);b=rank100(p_score)
    rho=1-6*(a-b).square().sum(-1)/(100*(100**2-1))
    return rho,(topk(restricted,10)[...,None]==topk(p_score,10)[...,None,:]).any(-1).double().mean(-1)

def metric_arrays(d,x,y,ids,selected,oracle,oid_mse,random_ids,head_ids,e0):
    sm=candidate_distances(d,x,y,selected).double();ret=sm.mean(-1)
    om=oid_mse.double();oret=om.mean(-1)
    assert (ret-oret>=-2e-6).all(),'Negative Oracle gap beyond existing MSE tolerance'
    discount=1/torch.log2(torch.arange(2,12,device=ids.device,dtype=torch.float64))
    ndcg=((1/(1+sm))*discount).sum(-1)/((1/(1+om))*discount).sum(-1)
    recall=(selected[...,None]==oracle[...,None,:]).any(-1).double().mean(-1)
    overlap=(selected[...,None]==e0[...,None,:]).any(-1).double().mean(-1)
    # Oracle candidates give a deduplicated 10-ID universe, avoiding union duplicates.
    B,C=ret.shape;h=head_ids.reshape(B,C,5,10)
    membership=(oracle[:,:,None,:,None]==h[:,:,:,None,:]).any(-1).sum(2)
    useful=membership>0;exclusive=membership==1
    found=(oracle[...,None]==selected[...,None,:]).any(-1)
    return dict(selected_ids=selected.cpu().numpy(),selected_mse=sm.cpu().numpy(),retmse=ret.cpu().numpy(),
        oracle_gap=(ret-oret).cpu().numpy(),recall=recall.cpu().numpy(),ndcg10=ndcg.cpu().numpy(),overlap_e0=overlap.cpu().numpy(),
        useful_available=useful.sum(-1).cpu().numpy(),useful_retained=(useful&found).sum(-1).cpu().numpy(),
        exclusive_available=exclusive.sum(-1).cpu().numpy(),exclusive_retained=(exclusive&found).sum(-1).cpu().numpy())
