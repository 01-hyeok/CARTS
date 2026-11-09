"""Canonical cached P100 training; separate P1/P2 gates; GPU1 only."""
import os,sys,json,time,copy,io
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments/p100_head_retrieval_quality_legacy_pool02'),str(ROOT/'experiments/learned_dynamics_cross01')]
import runtime as rt
from train_models import Features
from experiment_common import sha,load,dump,torch_save,GPU
from calibrators import ResidualCalibrator,FixedHeadWeights,teacher_kl
import numpy as np
import torch
OUT=ROOT/'results/TRACK-PREDICTIVE-SCORE-CALIBRATION01';OLD=ROOT/'results/TRACK-LEARNED-DYNAMICS-CROSS01'

class Run:
    def __init__(self,cell,phase):
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU;rt.gpu_guard();torch.set_num_threads(1)
        self.cell=cell;self.phase=phase;self.dest=OUT/'runs/P100'/cell/phase
        assert not self.dest.exists(),'No overwrite/resume without explicit state validation'
        self.dest.mkdir(parents=True);self.d=rt.CellData(cell,'cuda:0');self.cache=OUT/'canonical_cache/P100'/cell
        self.proof=json.loads((self.cache/'COMPLETE.json').read_text());assert self.proof['status']=='PASS_ALL_QUERY_P100_HEAD_CACHE'
        assert self.d.data_info==self.proof['identity']['data']
        self.features=Features(self.d,OLD/'features'/cell) if phase=='P2' else None
        self.pool={s:np.load(self.d.pool_dir/(s+'.npy'),mmap_mode='r') for s in ('train','val','test')}
        self.donor_permutation=np.random.RandomState(0).permutation(len(self.d.datasets['train']))
        self.names=['fixed','cosine'] if phase=='P1' else ['cosine','dynamics','cross','both','shuffled']
        self.arms={};self.opts={};self.history=[];self.started=time.time()
    def batch(self,split,a,teacher=False):
        path=self.cache/split/('batch%05d.npz'%(a//32));assert sha(path)==self.proof['files'][str(path.relative_to(self.cache))]
        with np.load(path) as z:
            p=torch.from_numpy(z['p'].copy()).to('cuda');g=torch.from_numpy(z['g'].copy()).to('cuda');ids=torch.from_numpy(z['ids'].copy()).to('cuda');starts=z['starts'].copy()
        b=a+len(p);assert np.array_equal(starts,self.d.datasets[split].starts[a:b]);assert np.array_equal(ids.cpu().numpy(),self.pool[split][a:b])
        side=json.loads(path.with_suffix('.json').read_text());distpath=Path(side['old_scalar_cache']);assert sha(distpath)==side['old_scalar_sha256']
        with np.load(distpath) as z:dist=torch.from_numpy(z['distance'].copy()).to('cuda')
        mask=torch.ones_like(p,dtype=torch.bool);x=g.transpose(-1,-2).double()
        aux=shuf=None
        if self.phase=='P2':
            rows=np.arange(a,b);aux=torch.cat([self.features.pair(split,rows,ids,'DYN'),self.features.pair(split,rows,ids,'CROSS')],-1)
            # Fixed Train donor policy, independent of future labels and epoch.
            N=len(self.d.datasets['train']);donor=self.donor_permutation[np.arange(a,b)%N]
            donorids=torch.from_numpy(np.array(self.pool['train'][donor],copy=True)).to('cuda')
            shuf=torch.cat([self.features.pair('train',donor,donorids,'DYN'),self.features.pair('train',donor,donorids,'CROSS')],-1)
            # Candidate position permutation breaks association with actual candidate IDs.
            permutation=torch.as_tensor(np.random.RandomState(0).permutation(100),device='cuda');shuf=shuf[:,:,permutation]
        t=None
        if teacher:
            assert split=='train','Teacher creation forbidden outside Train in optimizer path'
            t=rt.normalized_teacher_prob(dist.reshape(-1,100),mask.reshape(-1,100),.1).reshape_as(dist).double()
        return p,x,mask,dist,t,aux,shuf,ids
    def inputs(self,name,x,aux,shuf):
        if self.phase=='P1':return x
        return torch.cat([x,shuf if name=='shuffled' else aux],-1)
    def initialize(self):
        # Identical deterministic Train calibration population for all models.
        N=len(self.d.datasets['train']);blocks=np.unique((np.linspace(0,N-1,256,dtype=int)//32)*32);parts=[]
        for a in blocks:
            p,x,m,d,t,aux,shuf,ids=self.batch('train',int(a));parts.append(self.inputs('both',x,aux,shuf).detach().cpu().reshape(-1,5 if self.phase=='P1' else 38))
        fit=torch.cat(parts);normalizer=None
        for name in self.names:
            torch.manual_seed(0)
            if name=='fixed':arm=FixedHeadWeights()
            else:
                active=None
                if self.phase=='P2':active=[1]*5+([1]*15 if name in ('dynamics','both','shuffled') else [0]*15)+([1]*18 if name in ('cross','both','shuffled') else [0]*18)
                arm=ResidualCalibrator(5 if self.phase=='P1' else 38,active=active);arm.normalizer.fit(fit,'train')
            arm.to('cuda');self.arms[name]=arm;self.opts[name]=torch.optim.AdamW(arm.parameters(),lr=.001)
        dump(self.dest/'PROTOCOL.json',dict(phase=self.phase,cell=self.cell,epochs=30,early_stopping=False,selection='minimum ALL validation RetMSE; earliest tie',teacher='Train-only valid-support zscore negative FutureMSE tau=.1',student_tau=.1,rank_loss_weight=0,seed=0,all_train_queries=True,all_channels=True,calibration_blocks=blocks.tolist(),calibration_is_not_training_subsample=True,train_parameter_counts={n:sum(p.numel() for p in a.parameters() if p.requires_grad) for n,a in self.arms.items()},shuffle='seed0 Train timestamp permutation indexed by row%N; seed0 candidate permutation; validation/test donors exclusively Train',identity=self.proof['identity'],old_dynamics_comparison='Different architecture and fixed budget; not an exact rerun',test_selection=False))
    def smoke(self):
        p,x,m,d,t,aux,shuf,ids=self.batch('train',0,teacher=True);rows=[]
        for name,base in self.arms.items():
            z=self.inputs(name,x,aux,shuf);arm=copy.deepcopy(base);opt=torch.optim.AdamW(arm.parameters(),lr=.001)
            assert torch.equal(torch.argsort(arm.score(p,z,m,ranking=True),descending=True,stable=True)[...,:10],torch.argsort(p,descending=True,stable=True)[...,:10])
            assert torch.allclose(arm.logprob(p,z,m).exp(),p.double(),rtol=2e-6,atol=2e-8)
            for step in range(2):
                opt.zero_grad();loss=teacher_kl(t,arm.logprob(p,z,m),m);assert torch.isfinite(loss);loss.backward()
                gradients={n:float(v.grad.norm()) if v.grad is not None else 0. for n,v in arm.named_parameters()};assert all(np.isfinite(v) for v in gradients.values())
                assert sum(gradients.values())>0
                if name!='fixed' and step==1:assert gradients['layers.0.weight']>0
                opt.step()
            clone=copy.deepcopy(arm);buf=io.BytesIO();torch.save(arm.state_dict(),buf);buf.seek(0);clone.load_state_dict(torch.load(buf,weights_only=False));assert torch.equal(arm.score(p,z,m),clone.score(p,z,m))
            rows.append(dict(name=name,final_gradients=gradients,initial_ids_exact=True,save_load_exact=True))
        dump(self.dest/'INTEGRATION_GATE.json',dict(status='PASS',rows=rows,official_training=False))
    @torch.no_grad()
    def evaluate(self,split,save=False):
        totals={n:0. for n in self.names};units=0
        for a in range(0,len(self.d.datasets[split]),32):
            p,x,m,d,t,aux,shuf,ids=self.batch(split,a);units+=d.shape[0]*d.shape[1];out={}
            for n,arm in self.arms.items():
                s=arm.score(p,self.inputs(n,x,aux,shuf),m,ranking=True);pos=torch.argsort(s,descending=True,stable=True)[...,:10];ret=d.gather(-1,pos).double().mean(-1);totals[n]+=float(ret.sum())
                if save:
                    chosen=ids.gather(-1,pos);out[n+'__selected_ids']=chosen.cpu().numpy();out[n+'__retmse']=ret.cpu().numpy();out[n+'__selected_mse']=d.gather(-1,pos).cpu().numpy()
                    ds=self.d.datasets[split];ix=np.arange(a,a+len(p))[:,None]+np.arange(self.d.h);xp=torch.from_numpy(np.asarray(ds.data_x[ix],np.float32)).to('cuda').permute(0,2,1);yy=torch.from_numpy(np.asarray(ds.data_y[ix+self.d.h],np.float32)).to('cuda').permute(0,2,1)
                    preds=[]
                    for c in range(self.d.c):preds.append(self.d.values(chosen[:,c],c).double().mean(-2)+xp[:,c,-1,None])
                    err=torch.stack(preds,1)-yy.double();out[n+'__agg_mse']=err.square().mean(-1).cpu().numpy();out[n+'__agg_mae']=err.abs().mean(-1).cpu().numpy()
            if save:
                folder=self.dest/'final_raw'/split;folder.mkdir(parents=True,exist_ok=True)
                with (folder/('batch%05d.npz'%(a//32))).open('xb') as f:np.savez(f,query_start=self.d.datasets[split].starts[a:a+len(p)],**out)
        return {n:v/units for n,v in totals.items()}
    def run(self):
        self.initialize();self.smoke();best={};val=self.evaluate('val')
        reference=json.loads((OLD/'P100'/self.cell/'FULL_RESULTS.json').read_text())
        original=next(r['retmse'] for r in reference if r['method']=='S0' and r['split']=='val')
        assert all(np.isclose(value,original,atol=1e-12,rtol=0) for value in val.values()),'Epoch0 full-validation baseline RetMSE mismatch'
        for n in self.names:
            best[n]=dict(epoch=0,retmse=val[n]);torch_save(self.dest/'checkpoints'/n/'epoch00.pt',dict(model=self.arms[n].state_dict(),optimizer=self.opts[n].state_dict(),epoch=0));self.history.append(dict(name=n,epoch=0,val_retmse=val[n]))
        dump(self.dest/'TRAINING_STARTED.json',dict(at=time.time(),gpu=GPU,all_train_queries=True,all_channels=True))
        for epoch in range(1,31):
            blocks=np.random.RandomState(epoch).permutation(np.arange(0,len(self.d.datasets['train']),32));losses={n:0. for n in self.names};units=0
            for i,a in enumerate(blocks):
                p,x,m,d,t,aux,shuf,ids=self.batch('train',int(a),teacher=True);nunits=d.shape[0]*d.shape[1];units+=nunits
                for n,arm in self.arms.items():
                    self.opts[n].zero_grad();loss=teacher_kl(t,arm.logprob(p,self.inputs(n,x,aux,shuf),m),m);assert torch.isfinite(loss);loss.backward()
                    assert all(v.grad is None or torch.isfinite(v.grad).all() for v in arm.parameters());self.opts[n].step();losses[n]+=float(loss.detach())*nunits
                if i%25==0:dump(self.dest/'CURRENT.json',dict(phase='TRAIN',epoch=epoch,block=i,total=len(blocks),pid=os.getpid(),at=time.time()));print('TRAIN',self.cell,self.phase,epoch,i,len(blocks),flush=True)
            val=self.evaluate('val')
            for n in self.names:
                if val[n]<best[n]['retmse']:best[n]=dict(epoch=epoch,retmse=val[n])
                torch_save(self.dest/'checkpoints'/n/('epoch%02d.pt'%epoch),dict(model=self.arms[n].state_dict(),optimizer=self.opts[n].state_dict(),epoch=epoch))
                self.history.append(dict(name=n,epoch=epoch,val_retmse=val[n],train_kl=losses[n]/units))
            dump(self.dest/'HISTORY.json',self.history)
        dump(self.dest/'SELECTION.json',dict(status='FROZEN_BEFORE_TEST',best=best))
        for n in self.names:self.arms[n].load_state_dict(load(self.dest/'checkpoints'/n/('epoch%02d.pt'%best[n]['epoch']))['model'])
        validation=self.evaluate('val',save=True);test=self.evaluate('test',save=True)
        dump(self.dest/'COMPLETE.json',dict(status='PASS_ALL_QUERY_P100',validation=validation,test=test,selection=best,seconds=time.time()-self.started,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),primary='RetMSE@10',exploratory_test=True))
if __name__=='__main__':
    os.chdir(ROOT);Run(sys.argv[1],sys.argv[2]).run()
