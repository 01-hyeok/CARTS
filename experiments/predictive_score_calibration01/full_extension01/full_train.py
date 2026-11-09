"""Native Full P1/P2 implementation, gated before official training."""
import argparse,copy,io,time,json,os
from native_runtime import *
from experiment_common import torch_save

class FullRun:
    def __init__(self,cell,phase,gate_only=False):
        self.cell=cell;self.phase=phase;self.gate_only=gate_only
        suffix='gate_runs' if gate_only else 'runs'
        self.dest=OUT/suffix/'FULL'/cell/phase
        assert not self.dest.exists(),'Preserve previous Full attempt; use reviewed recovery'
        if not gate_only:
            gate=OUT/'full_extension01/P0'/cell/'COMPLETE.json'
            assert json.loads(gate.read_text())['status']=='PASS_FULL_P0_ALL_QUERY'
            for c in ('ETTh1_96','ETTh1_192','ETTh1_336','ETTh1_720','Weather_96','Weather_192','Weather_336','Weather_720'):
                assert json.loads((OUT/'runs/P100'/c/'P2/COMPLETE.json').read_text())['status']=='PASS_ALL_QUERY_P100'
            if phase=='P2':assert json.loads((OUT/'runs/FULL'/cell/'P1/COMPLETE.json').read_text())['status']=='PASS_ALL_QUERY_FULL'
        self.ctx=NativeContext(cell,'FULL','FULL');self.d=self.ctx.d;self.dest.mkdir(parents=True)
        self.names=['fixed','cosine'] if phase=='P1' else ['cosine','dynamics','cross','both','shuffled']
        self.arms={};self.opts={};self.history=[];self.started=time.time();self.times=dict(calibration=0.,smoke=0.,train=0.,validation=0.,test=0.)
        self.gamma=json.loads((ev.OUT/'checkpoints'/cell/'V5_FULL/GAMMA_SELECTION.json').read_text())['chosen_gamma']
    def initialize(self):
        begin=time.time();rows=np.unique(np.linspace(0,len(self.d.datasets['train'])-1,256,dtype=int));parts=[];population=[]
        for block in np.unique(rows//32):
            batch=self.ctx.batch('train',int(block*32),need_distance=False)
            for row in rows[rows//32==block]:
                local=int(row-block*32);valid=torch.nonzero(batch['mask'][local,0],as_tuple=False).flatten();assert len(valid)>=128
                pos=valid[torch.as_tensor(np.linspace(0,len(valid)-1,128,dtype=int),device=self.d.device)]
                ids=batch['ids'][local:local+1,:,pos];g=batch['g'][local:local+1,:,:,pos].transpose(-1,-2).double()
                x=g if self.phase=='P1' else torch.cat([g,self.ctx.auxiliary('train',np.array([row]),ids)],-1)
                parts.append(x.detach().cpu().reshape(-1,5 if self.phase=='P1' else 38));population.append(dict(row=int(row),candidate_ids=pos.cpu().tolist()))
        fitted=torch.cat(parts)
        for name in self.names:
            arm=new_model(name,self.phase)
            if name!='fixed':arm.normalizer.fit(fitted,'train')
            self.arms[name]=arm.to(self.d.device);self.opts[name]=torch.optim.AdamW(arm.parameters(),lr=.001)
        self.times['calibration']=time.time()-begin
        dump(self.dest/'PROTOCOL.json',dict(cell=self.cell,phase=self.phase,support='FULL',source_identity=self.ctx.identity,seed=0,epochs=30,early_stopping=False,all_train_queries=True,all_channels=True,candidate_bank_count=len(self.d.starts),candidate_reduction=False,query_encoder_batch=32,scorer_microbatch=1,candidate_feature_chunk=256,global_candidate_normalization=True,teacher='canonical valid-support zscore negative FutureMSE; tau=.1; Train queries only',student_tau=.1,rank_loss_weight=0,calibration=population,shuffle=dict(policy='TrainPairPermutation; exact bijection of all valid Train pairs per channel; held-out uses fixed timestamp/candidate-ID extension with Train-only donors',total=self.ctx.permutation.total,multiplier=self.ctx.permutation.multiplier,offset=self.ctx.permutation.offset),parameters={n:sum(p.numel() for p in a.parameters()) for n,a in self.arms.items()},selection='minimum ALL validation RetMSE, earliest exact tie; frozen before final Test',test_selection=False))
    def smoke(self):
        began=time.time();batch=self.ctx.batch('train',0);teacher=self.ctx.teacher(batch,'train')[:1];mask=batch['mask'][:1];rows=[]
        for name,original in self.arms.items():
            actual=self.ctx.ranking(original,name,self.phase,batch,'train',(0,1));expected=batch['p'][:1].masked_fill(~mask,-torch.inf)
            assert torch.equal(stable_topk(actual),stable_topk(expected))
            initial=global_logprob(self.ctx.score(original,name,self.phase,batch,'train',(0,1)),mask)
            assert torch.allclose(initial.exp(),batch['p'][:1].double(),rtol=2e-6,atol=2e-8)
            a=copy.deepcopy(original);b=copy.deepcopy(original)
            if name=='fixed':a.logits.data[0]=.1;b.logits.data[0]=.1
            else:a.beta.data.fill_(.1);b.beta.data.fill_(.1)
            s1=self.ctx.score(a,name,self.phase,batch,'train',(0,1),137)
            s2=self.ctx.score(b,name,self.phase,batch,'train',(0,1),batch['p'].shape[-1])
            assert torch.allclose(s1[mask],s2[mask],atol=1e-9,rtol=1e-9)
            l1=teacher_kl(teacher,global_logprob(s1,mask),mask);l2=teacher_kl(teacher,global_logprob(s2,mask),mask);l1.backward();l2.backward()
            assert torch.allclose(l1,l2,atol=1e-9,rtol=1e-9)
            for p,q in zip(a.parameters(),b.parameters()):assert torch.allclose(p.grad,q.grad,atol=1e-8,rtol=1e-8)
            del a,b,s1,s2,l1,l2
            arm=copy.deepcopy(original);opt=torch.optim.AdamW(arm.parameters(),lr=.001);grad=[]
            for step in range(2):
                opt.zero_grad();score=self.ctx.score(arm,name,self.phase,batch,'train',(0,1));loss=teacher_kl(teacher,global_logprob(score,mask),mask);assert torch.isfinite(loss);loss.backward()
                norms={n:float(p.grad.norm()) if p.grad is not None else 0. for n,p in arm.named_parameters()}
                assert all(np.isfinite(v) for v in norms.values()) and sum(norms.values())>0
                if name!='fixed' and step==1:assert norms['layers.0.weight']>0
                grad.append(norms);opt.step()
            clone=copy.deepcopy(arm);buffer=io.BytesIO();torch.save(arm.state_dict(),buffer);buffer.seek(0);clone.load_state_dict(torch.load(buffer,map_location=self.d.device))
            with torch.no_grad():assert torch.equal(self.ctx.score(arm,name,self.phase,batch,'train',(0,1)),self.ctx.score(clone,name,self.phase,batch,'train',(0,1)))
            rows.append(dict(name=name,initial_ordered_ids_exact=True,full_monolithic_chunk_loss_gradient_pass=True,gradient=grad,save_load_exact=True))
        assert all(p.grad is None for p in list(self.ctx.f.model.parameters())+list(self.ctx.f.heads.parameters()))
        self.times['smoke']=time.time()-began
        dump(self.dest/'INTEGRATION_GATE.json',dict(status='PASS_ACTUAL_FULL_SUPPORT',full_candidates=batch['p'].shape[-1],rows=rows,official_training=False,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
    @torch.no_grad()
    def evaluate(self,split,save=False):
        began=time.time();sums={n:0. for n in self.names};units=0;baseline=0.;secondary={}
        for a in range(0,len(self.d.datasets[split]),32):
            batch=self.ctx.batch(split,a);parts={};units+=(batch['b']-a)*self.d.c
            for j in range(batch['b']-a):
                mask=batch['mask'][j:j+1];dist=batch['distance'][j:j+1].double();pp=batch['p'][j:j+1];ids=batch['ids'][j:j+1];basepos=stable_topk(pp.masked_fill(~mask,-torch.inf));baseret=dist.gather(-1,basepos).mean(-1);baseline+=float(baseret.sum())
                oraclepos=stable_topk((-dist).masked_fill(~mask,-torch.inf));oracle=ids.gather(-1,oraclepos);om=dist.gather(-1,oraclepos)
                scores={n:self.ctx.ranking(arm,n,self.phase,batch,split,(j,j+1)) for n,arm in self.arms.items()}
                if save:
                    scores.update(S0=pp.masked_fill(~mask,-torch.inf),E1=ev.score(batch['g'][j:j+1],mask,'E1',None),E4=ev.score(batch['g'][j:j+1],mask,'E4',self.gamma),Oracle10=(-dist).masked_fill(~mask,-torch.inf))
                for name,s in scores.items():
                    pos=stable_topk(s);selected=ids.gather(-1,pos);sm=dist.gather(-1,pos);ret=sm.mean(-1)
                    if name in sums:sums[name]+=float(ret.sum())
                    if save:
                        agg,mae=self.ctx.prediction_metrics(split,[a+j],selected);discount=1/torch.log2(torch.arange(2,12,device=self.d.device,dtype=torch.float64))
                        metrics=dict(retmse=ret,agg_mse=agg,agg_mae=mae,oracle_gap=ret-om.mean(-1),oracle_recall=(selected[...,None]==oracle[...,None,:]).any(-1).double().mean(-1),ndcg10=((1/(1+sm))*discount).sum(-1)/((1/(1+om))*discount).sum(-1),best_of10=sm.min(-1).values,delta_retmse_vs_s0=ret-baseret)
                        parts.setdefault(name+'__selected_ids',[]).append(selected.cpu().numpy());parts.setdefault(name+'__selected_mse',[]).append(sm.cpu().numpy())
                        for metric,value in metrics.items():parts.setdefault(name+'__'+metric,[]).append(value.cpu().numpy());secondary[(name,metric)]=secondary.get((name,metric),0.)+float(value.sum())
            if save:
                folder=self.dest/'final_raw'/split;folder.mkdir(parents=True,exist_ok=True)
                with (folder/('batch%05d.npz'%(a//32))).open('xb') as f:np.savez(f,query_start=self.d.datasets[split].starts[a:batch['b']],**{k:np.concatenate(v) for k,v in parts.items()})
            if a%1024==0:dump(self.dest/'CURRENT.json',dict(phase='EVALUATION',split=split,end=batch['b'],total=len(self.d.datasets[split]),at=time.time()))
        original=json.loads((OLD/'FULL'/self.cell/'FULL_RESULTS.json').read_text());old=next(r['retmse'] for r in original if r['split']==split and r['method']=='S0')
        assert np.isclose(baseline/units,old,rtol=2e-6,atol=2e-6)
        self.times['test' if split=='test' else 'validation']+=time.time()-began
        if save:dump(self.dest/(split.upper()+'_METRICS.json'),[dict(method=n,metric=m,value=v/units) for (n,m),v in secondary.items()])
        return {n:v/units for n,v in sums.items()},baseline/units
    def run(self):
        self.initialize();self.smoke()
        if self.gate_only:
            dump(self.dest/'COMPLETE.json',dict(status='PASS_FULL_ACTUAL_INTEGRATION_ONLY',official_training=False,timings=self.times));return
        val,base=self.evaluate('val');assert all(np.isclose(v,base,rtol=0,atol=1e-12) for v in val.values());best={n:dict(epoch=0,retmse=val[n]) for n in self.names}
        for n in self.names:torch_save(self.dest/'checkpoints'/n/'epoch00.pt',dict(model=self.arms[n].state_dict(),optimizer=self.opts[n].state_dict(),epoch=0))
        dump(self.dest/'TRAINING_STARTED.json',dict(at=time.time(),gpu=GPU,all_train_queries=True,all_channels=True,full_candidates=len(self.d.starts)))
        for epoch in range(1,31):
            begin=time.time();blocks=np.random.RandomState(epoch).permutation(np.arange(0,len(self.d.datasets['train']),32));sums={n:0. for n in self.names};units=0
            for block,a in enumerate(blocks):
                batch=self.ctx.batch('train',int(a));teacher=self.ctx.teacher(batch,'train');B=batch['b']-int(a);units+=B*self.d.c
                for n,arm in self.arms.items():
                    self.opts[n].zero_grad();lossvalue=0.
                    for j in range(B):
                        score=self.ctx.score(arm,n,self.phase,batch,'train',(j,j+1));loss=teacher_kl(teacher[j:j+1],global_logprob(score,batch['mask'][j:j+1]),batch['mask'][j:j+1]);assert torch.isfinite(loss)
                        (loss/B).backward();lossvalue+=float(loss.detach())/B
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in arm.parameters());self.opts[n].step();sums[n]+=lossvalue*B*self.d.c
                if block%10==0:dump(self.dest/'CURRENT.json',dict(phase='TRAIN',epoch=epoch,block=block,total=len(blocks),pid=os.getpid(),at=time.time()));print('FULL_TRAIN',self.cell,self.phase,epoch,block,len(blocks),flush=True)
            self.times['train']+=time.time()-begin;val,_=self.evaluate('val')
            for n in self.names:
                if val[n]<best[n]['retmse']:best[n]=dict(epoch=epoch,retmse=val[n])
                torch_save(self.dest/'checkpoints'/n/('epoch%02d.pt'%epoch),dict(model=self.arms[n].state_dict(),optimizer=self.opts[n].state_dict(),epoch=epoch));self.history.append(dict(name=n,epoch=epoch,val_retmse=val[n],train_kl=sums[n]/units))
            dump(self.dest/'HISTORY.json',self.history)
        dump(self.dest/'SELECTION.json',dict(status='FROZEN_BEFORE_TEST',best=best))
        for n in self.names:self.arms[n].load_state_dict(load(self.dest/'checkpoints'/n/('epoch%02d.pt'%best[n]['epoch']))['model'])
        val,_=self.evaluate('val',True);test,_=self.evaluate('test',True)
        dump(self.dest/'COMPLETE.json',dict(status='PASS_ALL_QUERY_FULL',validation=val,test=test,selection=best,timings=self.times,seconds=time.time()-self.started,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),full_candidates=len(self.d.starts),exploratory_test=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('cell');parser.add_argument('phase',choices=['P1','P2']);parser.add_argument('--gate-only',action='store_true');args=parser.parse_args();os.chdir(ROOT)
    try:FullRun(args.cell,args.phase,args.gate_only).run()
    except BaseException as e:
        import traceback
        failure=OUT/'full_extension01'/('FAILURE_%s_%s_%d.json'%(args.cell,args.phase,int(time.time())))
        dump(failure,dict(error=repr(e),traceback=traceback.format_exc(),no_candidate_reduction=True,preserved=True));raise
