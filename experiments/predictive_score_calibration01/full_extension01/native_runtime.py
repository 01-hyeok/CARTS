"""Read-only canonical runtime for native Full and same-checkpoint transfer."""
import os,sys,json,importlib.util
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
sys.path[:0]=[str(PARENT),str(ROOT/'experiments/score_fusion_ablation01'),str(ROOT/'experiments/learned_dynamics_cross01')]
import evaluate as ev
from train_models import Features
from learned_scores import dynamics_pairs,cross_pairs
from experiment_common import sha,load,dump,GPU
from calibrators import ResidualCalibrator,FixedHeadWeights,teacher_kl
from streaming import assemble,stable_topk,global_logprob
from train_pair_permutation import TrainPairPermutation
import numpy as np,torch
OUT=ROOT/'results/TRACK-PREDICTIVE-SCORE-CALIBRATION01';OLD=ROOT/'results/TRACK-LEARNED-DYNAMICS-CROSS01'

class NativeContext:
    def __init__(self,cell,training_support,evaluation_support):
        if training_support not in ('P100','FULL') or evaluation_support not in ('P100','FULL'):raise ValueError('Bad support')
        self.cell=cell;self.training_support=training_support;self.support=evaluation_support
        self.d,self.f,self.prior,self.identity,self.before=ev.setup(cell,training_support)
        if training_support=='FULL' and cell.startswith('Weather'):
            path=ROOT/'experiments/score_fusion_ablation01/recovery02/probe_norm.py'
            spec=importlib.util.spec_from_file_location('full_extension_norm',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            proof=json.loads((ev.OUT/'recovery02'/(cell+'_NORM_PROBE.json')).read_text())
            assert all(r['raw_exact'] and r['prob_exact'] and r['ids_match_old'] for r in proof['rows']);module.patch(self.f.batch)
        self.features=Features(self.d,OLD/'features'/cell)
        self.permutation=TrainPairPermutation(len(self.d.starts),self.d.h,self.d.device)
        self.p100_donors=np.random.RandomState(0).permutation(len(self.d.datasets['train']))
        self.p100_pool=np.load(self.d.pool_dir/'train.npy',mmap_mode='r')
        self.slot_permutation=np.random.RandomState(0).permutation(100)
        # Validate the closed-form eligible-pair indexing against repository RAFT.
        q=np.unique(np.linspace(0,len(self.d.starts)-1,17,dtype=int))
        actual=self.d.mask(q).sum(-1)
        assert torch.equal(actual,self.permutation.counts[torch.as_tensor(q,device=self.d.device)])
        self.checked=set()
    @torch.no_grad()
    def batch(self,split,a,need_distance=True):
        d=self.d;b=min(a+32,len(d.datasets[split]));assert a%32==0
        if self.training_support=='P100' and self.support=='P100':
            g,ids=self.f.native_p100_logits(split,a,b);mask=torch.ones_like(ids,dtype=torch.bool)
        else:
            g,fullmask=self.f.full_logits(split,a,b)
            if self.support=='FULL':
                ids=torch.arange(len(d.starts),device=d.device).expand(b-a,d.c,-1);mask=fullmask
            else:
                ids=d.pool_all(split,a,b);assert fullmask.gather(-1,ids).all()
                g=g.gather(-1,ids[:,:,None].expand(-1,-1,5,-1));mask=torch.ones_like(ids,dtype=torch.bool)
        p=ev.normalize(g,mask)
        assert d.mask(d.datasets[split].starts[a:b])[:,None].expand(b-a,d.c,-1).gather(-1,ids).eq(mask).all()
        if self.support=='FULL':assert ids.shape[-1]==len(d.starts)
        dist=None
        if need_distance:
            x,y,_=d.query(split,a,b);dist=ev.candidate_distances(d,x,y,ids)
        if split!='train':
            raw=self.prior.batch(split,a,b);variant=ev.variant_name(self.training_support,self.support);pos=stable_topk(p.masked_fill(~mask,-torch.inf))
            assert np.array_equal(ids.gather(-1,pos).cpu().numpy(),raw[variant+'__selected_ids']),'Canonical baseline ordered IDs differ'
            if dist is not None:assert np.array_equal(dist.gather(-1,pos).cpu().numpy(),raw[variant+'__selected_mse']),'Canonical baseline candidate MSE differs'
        if a in {0,(len(d.datasets[split])-1)//32*32}:
            ref=ev.canonical_logits(self.f,split,a,b,self.support);rp=ev.normalize(ref,mask)
            assert torch.equal(stable_topk(p.masked_fill(~mask,-torch.inf)),stable_topk(rp.masked_fill(~mask,-torch.inf)))
            assert torch.allclose(p,rp,atol=2e-8,rtol=2e-6)
            self.checked.add((split,a))
        return dict(g=g,p=p,mask=mask,ids=ids,distance=dist,a=a,b=b)
    def teacher(self,batch,split):
        if split!='train':raise ValueError('Optimizer teacher is Train-only')
        d,m=batch['distance'],batch['mask'];n=d.shape[-1]
        return ev.rt.normalized_teacher_prob(d.reshape(-1,n),m.reshape(-1,n),.1).reshape_as(d).double()
    def auxiliary(self,split,rows,ids,shuffle=False,offset=0):
        if not shuffle:return torch.cat([self.features.pair(split,rows,ids,'DYN'),self.features.pair(split,rows,ids,'CROSS')],-1)
        if self.training_support=='P100':
            donors=self.p100_donors[np.asarray(rows)%len(self.p100_donors)]
            slots=self.slot_permutation[np.arange(offset,offset+ids.shape[-1])%100]
            donorids=torch.from_numpy(np.array(self.p100_pool[donors],copy=True)).to(device=self.d.device,dtype=torch.long)[...,torch.as_tensor(slots,device=self.d.device,dtype=torch.long)]
            return torch.cat([self.features.pair('train',donors,donorids,'DYN'),self.features.pair('train',donors,donorids,'CROSS')],-1)
        starts=self.d.datasets[split].starts[np.asarray(rows)]
        q,c,_=self.permutation.map(rows,ids,split,starts)
        channels=torch.arange(self.d.c,device=self.d.device)[None,:,None]
        train=self.features.data['train']
        qfeatures={k:v[q,channels] for k,v in train.items()}
        cfeatures={k:train[k][c,channels] for k in ('phi','coef','recent','residual')}
        return torch.cat([dynamics_pairs(qfeatures,cfeatures,self.features.norm),cross_pairs(qfeatures,cfeatures,self.features.norm)],-1)
    def score(self,model,name,phase,batch,split,local_rows,chunk=256):
        start,stop=local_rows;rows=np.arange(batch['a']+start,batch['a']+stop)
        p=batch['p'][start:stop];mask=batch['mask'][start:stop];ids=batch['ids'][start:stop];g=batch['g'][start:stop]
        if name=='fixed':return model.score(p,g.transpose(-1,-2).double(),mask,ranking=False)
        def piece(i,j):
            x=g[...,i:j].transpose(-1,-2).double()
            if phase=='P2':x=torch.cat([x,self.auxiliary(split,rows,ids[...,i:j],name=='shuffled',i)],-1)
            return model.score(p[...,i:j],x,mask[...,i:j],ranking=False)
        return assemble(p.shape[-1],chunk,piece)
    def ranking(self,model,name,phase,batch,split,local_rows):
        start,stop=local_rows
        if name=='fixed':return model.score(batch['p'][start:stop],batch['g'][start:stop].transpose(-1,-2).double(),batch['mask'][start:stop],ranking=True)
        if name!='fixed' and float(model.beta.detach())==0:return batch['p'][start:stop].masked_fill(~batch['mask'][start:stop],-torch.inf)
        return self.score(model,name,phase,batch,split,local_rows)
    @torch.no_grad()
    def prediction_metrics(self,split,rows,chosen):
        # Only selected K=10 futures are materialized; no candidate reduction.
        d=self.d;ds=d.datasets[split];idx=np.asarray(rows)[:,None]+np.arange(d.h)
        x=torch.from_numpy(np.asarray(ds.data_x[idx],np.float32)).to(d.device).permute(0,2,1)
        y=torch.from_numpy(np.asarray(ds.data_y[idx+d.h],np.float32)).to(d.device).permute(0,2,1)
        future=d.values_all(chosen).double()+x[:,:,-1,None,None].double();error=future.mean(-2)-y.double()
        return error.square().mean(-1),error.abs().mean(-1)

def new_model(name,phase,device='cpu'):
    torch.manual_seed(0)
    if name=='fixed':return FixedHeadWeights().to(device)
    active=None
    if phase=='P2':active=[1]*5+([1]*15 if name in ('dynamics','both','shuffled') else [0]*15)+([1]*18 if name in ('cross','both','shuffled') else [0]*18)
    return ResidualCalibrator(5 if phase=='P1' else 38,active).to(device)
