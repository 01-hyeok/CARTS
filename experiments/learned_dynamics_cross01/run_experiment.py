"""Native-support full-query Phase A. Isolated artifacts and mandatory gates."""
import sys,os,json,time,copy,io,traceback,argparse,csv,fcntl
from pathlib import Path
from experiment_common import *
import numpy as np
import torch
from learned_scores import *
from train_models import Features,Arm
sys.path.insert(0,str(ROOT/'experiments/score_fusion_ablation01'))
import evaluate as ev
import importlib.util
sp=importlib.util.spec_from_file_location('dyn_weather_norm',ROOT/'experiments/score_fusion_ablation01/recovery02/probe_norm.py')
norm_patch=importlib.util.module_from_spec(sp);sp.loader.exec_module(norm_patch)

def npz_save(path,values):
    path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists(),'Immutable completed block already exists'
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as stream:np.savez(stream,**values)
    os.replace(tmp,path)

class Experiment:
    def __init__(self,cell,support):
        ready=json.loads((OUT/'CPU_IMPLEMENTATION_GATE.json').read_text())
        assert ready['status']=='PASS' and ready['source_hashes']==sources(),'CPU gate/source mismatch'
        self.cell=cell;self.support=support;self.dest=OUT/support/cell
        assert not (self.dest/'COMPLETE.json').exists(),'Completed experiment is immutable'
        if (self.dest/'TRAINING_STARTED.json').exists():
            raise RuntimeError('Prior training exists: explicit checkpoint recovery required, no overwrite')
        self.dest.mkdir(parents=True,exist_ok=True)
        os.chdir(ROOT)
        self.d,self.f,self.prior,self.identity,self.before=ev.setup(cell,support)
        if support=='FULL' and cell.startswith('Weather'):
            proof=json.loads((ev.OUT/'recovery02'/(cell+'_NORM_PROBE.json')).read_text())
            assert all(r['raw_exact'] and r['prob_exact'] and r['ids_match_old'] for r in proof['rows'])
            norm_patch.patch(self.f.batch)
        self.features=Features(self.d,OUT/'features'/cell)
        self.identity.update(track='learned_dynamics_cross01',new_sources=sources(),feature_manifest=self.features.provenance,
            train_mask='original offline nonoverlap RAFT',test_selection=False,
            block_shuffle='seed0+epoch canonical B32 blocks, all channels; same order across arms',
            architecture_dtype='new scorers float64; frozen V5 retains canonical runtime/precision')
        dump(self.dest/'IDENTITY.json',self.identity)
        self.cache={};self.arms={};self.opts={};self.history=[];self.steps=0
        self.started=time.perf_counter();self.raw_time=0;self.scoring_time=0

    def ids_mask(self,split,a,b):
        d=self.d
        if self.support=='P100':
            ids=d.pool_all(split,a,b);return ids,torch.ones_like(ids,dtype=torch.bool)
        ids=torch.arange(len(d.starts),device=d.device).expand(b-a,d.c,-1)
        return ids,d.mask(d.datasets[split].starts[a:b])[:,None].expand_as(ids)

    @torch.no_grad()
    def native(self,split,a,b):
        if self.support=='P100':
            g,ids=self.f.native_p100_logits(split,a,b);mask=torch.ones_like(ids,dtype=torch.bool)
        else:
            g,mask=self.f.full_logits(split,a,b);ids=torch.arange(len(self.d.starts),device=self.d.device).expand(b-a,self.d.c,-1)
        p=ev.normalize(g,mask)
        return g,p,mask,ids

    def gpu_gate(self):
        rows=[]
        with torch.no_grad():
            for split in ('val','test'):
                N=len(self.d.datasets[split])
                for a in sorted({0,(N-1)//32*32}):
                    b=min(a+32,N);g,p,m,ids=self.native(split,a,b)
                    ref=ev.canonical_logits(self.f,split,a,b,self.support);rp=ev.normalize(ref,m)
                    actual=ids.gather(-1,ev.topk(p.masked_fill(~m,-torch.inf),10))
                    expected=ids.gather(-1,ev.topk(rp.masked_fill(~m,-torch.inf),10))
                    assert torch.equal(actual,expected),'Canonical ordered Top10 mismatch'
                    assert torch.allclose(p,rp,rtol=2e-6,atol=2e-8),'Canonical probability mismatch'
                    old=self.prior.batch(split,a,b);vn=ev.variant_name(self.support,self.support)
                    assert np.array_equal(actual.cpu().numpy(),old[vn+'__selected_ids'])
                    x,y,_=self.d.query(split,a,b);dist=ev.candidate_distances(self.d,x,y,actual)
                    assert np.array_equal(dist.cpu().numpy(),old[vn+'__selected_mse'])
                    reconstructed=masked_log_probability(log_v5_score(p,m)/.1,m).exp()
                    assert torch.allclose(reconstructed,p.double(),rtol=2e-6,atol=2e-8)
                    rows.append(dict(split=split,start=a,end=b,channels=self.d.c,ordered_ids_exact=True,
                        candidate_mse_exact=True,max_p_delta=float((p-rp).abs().max()),max_log_roundtrip_delta=float((reconstructed-p).abs().max())))
        dump(self.dest/'GPU_BASELINE_GATE.json',dict(status='PASS_NATIVE_FIRST_TAIL_CANONICAL',rows=rows,at=now(),before=self.before))
        print('BASELINE_GATE_PASS',self.cell,self.support,flush=True)

    @torch.no_grad()
    def batch(self,split,a):
        d=self.d;b=min(a+32,len(d.datasets[split]));ids,mask=self.ids_mask(split,a,b)
        path=self.dest/'frozen_scalar_cache'/split/('batch%05d.npz'%(a//32))
        if self.support=='P100' and path.exists():
            meta=json.loads(path.with_suffix('.json').read_text());assert ev.sha(path)==meta['sha256']
            assert meta['identity_checkpoint']==self.f.row['checkpoint_sha256'] and meta['pool_hash']==d.pool_hashes[split]
            with np.load(path) as z:
                assert np.array_equal(z['starts'],d.datasets[split].starts[a:b])
                p=torch.from_numpy(z['p'].copy()).to(d.device);dist=torch.from_numpy(z['distance'].copy()).to(d.device)
        else:
            t=time.perf_counter();_,p,mask,ids=self.native(split,a,b)
            x,y,_=d.query(split,a,b);dist=ev.candidate_distances(d,x,y,ids)
            self.raw_time+=time.perf_counter()-t
            if split!='train':
                prior=self.prior.batch(split,a,b);vn=ev.variant_name(self.support,self.support)
                pick=ids.gather(-1,ev.topk(p.masked_fill(~mask,-torch.inf),10))
                assert np.array_equal(pick.cpu().numpy(),prior[vn+'__selected_ids'])
                assert np.array_equal(dist.gather(-1,ev.topk(p.masked_fill(~mask,-torch.inf),10)).cpu().numpy(),prior[vn+'__selected_mse'])
            if self.support=='P100':
                npz_save(path,dict(starts=d.datasets[split].starts[a:b],p=p.cpu().numpy(),distance=dist.cpu().numpy()))
                dump(path.with_suffix('.json'),dict(sha256=ev.sha(path),identity_checkpoint=self.f.row['checkpoint_sha256'],pool_hash=self.d.pool_hashes[split]))
        teacher=ev.rt.normalized_teacher_prob(dist.reshape(-1,dist.shape[-1]),mask.reshape(-1,mask.shape[-1]),.1).reshape_as(dist).double()
        return b,ids,mask,p,dist,teacher

    def calibrate(self):
        d=self.d;count=min(256,len(d.datasets['train']));rows=np.unique(np.linspace(0,len(d.datasets['train'])-1,count,dtype=np.int64))
        candidates=[]
        for r in rows:
            if self.support=='P100':z=np.array(d.cpu_pool['train'][r],copy=True)
            else:
                valid=d.sampler.valid_indices(d.datasets['train'].starts[r]);assert len(valid)>=128
                z=np.broadcast_to(valid[np.linspace(0,len(valid)-1,128,dtype=np.int64)],(d.c,128)).copy()
            candidates.append(z)
        ids=torch.tensor(np.stack(candidates),device=d.device);kinds={}
        for kind in ('L2','DYN','CROSS'):kinds[kind]=self.features.pair('train',rows,ids,kind)
        for name in ARMS:
            torch.manual_seed(0);arm=Arm(name).to(d.device);arm.calibrate(kinds[arm.kind]);self.arms[name]=arm
            self.opts[name]=torch.optim.AdamW([p for p in arm.parameters() if p.requires_grad],lr=.001)
        calibration=dict(train_row=rows,query_start=d.datasets['train'].starts[rows],candidate_ids=ids.cpu().numpy())
        if (self.dest/'CALIBRATION_POPULATION.npz').exists():
            with np.load(self.dest/'CALIBRATION_POPULATION.npz') as prior:
                assert all(np.array_equal(prior[k],v) for k,v in calibration.items())
        else:npz_save(self.dest/'CALIBRATION_POPULATION.npz',calibration)
        dump(self.dest/'CALIBRATION.json',dict(status='TRAIN_ONLY_FIXED',at=now(),timestamps=len(rows),channels=d.c,candidates_per_query_channel=ids.shape[-1],
            feature_fit='all train windows',pair_and_score_fit='deterministic 256 evenly spaced Train timestamps; all P100 IDs or 128 evenly spaced eligible Full IDs',
            learning_population='ALL TRAIN queries; calibration sample is NOT training subsampling',
            population_sha256=sha(self.dest/'CALIBRATION_POPULATION.npz')))
        del kinds,ids

    def scorer_gate(self):
        b,ids,mask,p,dist,t=self.batch('train',0);rows=np.arange(min(b,2));ii=ids[:len(rows)];mm=mask[:len(rows)];pp=p[:len(rows)];tt=t[:len(rows)]
        results=[]
        for name,base in self.arms.items():
            arm=copy.deepcopy(base);opt=torch.optim.AdamW([v for v in arm.parameters() if v.requires_grad],lr=.001)
            for step in range(2):
                opt.zero_grad();raw=arm.raw_chunks(self.features,'train',rows,ii)
                loss=kl_from_log_probability(tt,arm.logprob(pp,raw,mm),mm);assert torch.isfinite(loss);loss.backward()
                grad=sum(float(v.grad.square().sum()) for v in arm.scorer.parameters() if v.grad is not None)**.5
                if step==1 or not arm.fused:assert grad>0,(name,'scorer gradient missing')
                if arm.fused:assert arm.fusion.beta.grad is not None and torch.isfinite(arm.fusion.beta.grad)
                opt.step()
            with torch.no_grad():
                s1=arm.raw_chunks(self.features,'train',rows,ii,chunk=31);s2=arm.raw_chunks(self.features,'train',rows,ii,chunk=128)
                assert torch.allclose(s1,s2,rtol=1e-10,atol=1e-10),(name,'chunk mismatch')
                buf=io.BytesIO();torch.save(arm.state_dict(),buf);buf.seek(0)
                clone=copy.deepcopy(base);clone.load_state_dict(torch.load(buf,map_location=self.d.device))
                assert torch.equal(s2,clone.raw_chunks(self.features,'train',rows,ii,chunk=128))
                results.append(dict(arm=name,gradient_norm=grad,chunk_max_delta=float((s1-s2).abs().max()),save_load_exact=True))
        assert all(v.grad is None for v in self.f.model.parameters())
        assert all(v.grad is None for v in self.f.heads.parameters())
        dump(self.dest/'SCORER_GATE.json',dict(status='PASS_ACTUAL_TRAIN_GRADIENT_CHUNK_SAVELOAD',rows=results,smoke_updates_not_official_training=True,at=now()))

    @torch.no_grad()
    def validate(self,names):
        sums={n:[0.,0.,0] for n in names};N=len(self.d.datasets['val']);micro=32 if self.support=='P100' else 2
        for a in range(0,N,32):
            b,ids,mask,p,dist,t=self.batch('val',a)
            for j in range(0,b-a,micro):
                sl=slice(j,min(j+micro,b-a));rows=np.arange(a+sl.start,a+sl.stop)
                for name in names:
                    arm=self.arms[name];raw=arm.raw_chunks(self.features,'val',rows,ids[sl])
                    s=arm.score(p[sl],raw,mask[sl],ranking=True);idx=ev.topk(s,10)
                    ret=dist[sl].gather(-1,idx).double().mean(-1)
                    loss=kl_from_log_probability(t[sl],arm.logprob(p[sl],raw,mask[sl]),mask[sl])
                    assert torch.isfinite(loss) and torch.isfinite(ret).all()
                    sums[name][0]+=float(ret.sum());sums[name][1]+=float(loss)*ret.numel();sums[name][2]+=ret.numel()
            if a%1024==0:dump(self.dest/'CURRENT.json',dict(phase='VALIDATION',query_end=b,total=N,at=now()))
        return {n:dict(retmse=v[0]/v[2],kl=v[1]/v[2],units=v[2]) for n,v in sums.items()}

    def train(self):
        val=self.validate(ARMS);state={n:dict(best=val[n]['retmse'],best_epoch=0,bad=0,active=True) for n in ARMS}
        for n in ARMS:
            row=dict(arm=n,epoch=0,train_kl=None,val_retmse=val[n]['retmse'],val_kl=val[n]['kl'],steps=0,beta=float(self.arms[n].fusion.beta.detach()))
            self.history.append(row);torch_save(self.dest/'checkpoints'/n/'epoch00.pt',dict(model=self.arms[n].state_dict(),optimizer=self.opts[n].state_dict(),epoch=0,state=state[n],identity=self.identity))
        dump(self.dest/'TRAINING_STARTED.json',dict(at=now(),official_population='all Train queries and channels',arms=ARMS,seed=0,support=self.support))
        print('OFFICIAL_TRAINING_STARTED',self.cell,self.support,flush=True)
        micro=32 if self.support=='P100' else 2;N=len(self.d.datasets['train'])
        for epoch in range(1,31):
            active=[n for n in ARMS if state[n]['active']]
            if not active:break
            blocks=np.random.RandomState(epoch).permutation(np.arange(0,N,32));tot={n:[0.,0,0.] for n in active}
            order_hash=__import__('hashlib').sha256(blocks.tobytes()).hexdigest()
            for block,a in enumerate(blocks):
                a=int(a);b,ids,mask,p,dist,t=self.batch('train',a)
                for name in active:
                    arm=self.arms[name];opt=self.opts[name];opt.zero_grad();lvalue=0.
                    for j in range(0,b-a,micro):
                        sl=slice(j,min(j+micro,b-a));rows=np.arange(a+sl.start,a+sl.stop)
                        raw=arm.raw_chunks(self.features,'train',rows,ids[sl])
                        loss=kl_from_log_probability(t[sl],arm.logprob(p[sl],raw,mask[sl]),mask[sl])
                        assert torch.isfinite(loss),(name,epoch,a,'nonfinite loss')
                        weight=(sl.stop-sl.start)/(b-a);(loss*weight).backward();lvalue+=float(loss.detach())*weight
                    grads=[v.grad for v in arm.parameters() if v.grad is not None]
                    gn=sum(float(v.double().square().sum()) for v in grads)**.5
                    assert np.isfinite(gn),(name,epoch,a,'nonfinite gradient');opt.step()
                    units=(b-a)*self.d.c;tot[name][0]+=lvalue*units;tot[name][1]+=units;tot[name][2]+=gn
                    with (self.dest/'STEP_HISTORY.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(arm=name,epoch=epoch,block=block,start=a,loss=lvalue,gradient_norm=gn,lr=.001))+'\n')
                self.steps+=1
                if block%25==0:
                    dump(self.dest/'CURRENT.json',dict(phase='TRAIN',epoch=epoch,block=block,total_blocks=len(blocks),active=active,at=now(),pid=os.getpid()))
                    print('TRAIN',self.cell,self.support,epoch,block,len(blocks),flush=True)
            val=self.validate(active)
            for n in active:
                st=state[n];improved=val[n]['retmse']<st['best']
                if improved:st.update(best=val[n]['retmse'],best_epoch=epoch,bad=0)
                else:st['bad']+=1
                if st['bad']>=5:st['active']=False
                row=dict(arm=n,epoch=epoch,train_kl=tot[n][0]/tot[n][1],val_retmse=val[n]['retmse'],val_kl=val[n]['kl'],
                    gradient_norm_mean=tot[n][2]/len(blocks),steps=self.steps,beta=float(self.arms[n].fusion.beta.detach()),batch_order_sha256=order_hash,
                    seconds=time.perf_counter()-self.started)
                self.history.append(row)
                torch_save(self.dest/'checkpoints'/n/('epoch%02d.pt'%epoch),dict(model=self.arms[n].state_dict(),optimizer=self.opts[n].state_dict(),epoch=epoch,state=st,identity=self.identity))
                print('EPOCH_COMPLETE',n,epoch,val[n],flush=True)
            dump(self.dest/'TRAIN_HISTORY.json',self.history);dump(self.dest/'EPOCH_STATE.json',dict(epoch=epoch,arms=state,at=now()))
        dump(self.dest/'SELECTION.json',dict(status='FROZEN_BEFORE_FINAL_TEST',criterion='minimum full validation RetMSE; earliest exact tie',arms=state,at=now()))
        for n in ARMS:self.arms[n].load_state_dict(load(self.dest/'checkpoints'/n/('epoch%02d.pt'%state[n]['best_epoch']))['model'])
        return state

    def run(self,gate_only=False):
        self.gpu_gate();self.calibrate();self.scorer_gate()
        if gate_only:
            dump(self.dest/'READY_TO_TRAIN.json',dict(status='PASS',at=now()));return
        state=self.train()
        from final_evaluation import finish
        finish(self,state)
        dump(self.dest/'COMPLETE.json',dict(status='PASS_ALL_QUERY_TRAIN_VAL_TEST',at=now(),cell=self.cell,support=self.support,
            selection=state,source_hashes=sources(),identity=self.identity,queries={s:len(d) for s,d in self.d.datasets.items()},channels=self.d.c,
            elapsed_seconds=time.perf_counter()-self.started,raw_encoding_distance_seconds=self.raw_time,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved()))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cell',choices=CELLS);ap.add_argument('support',choices=['P100','FULL']);ap.add_argument('--gate-only',action='store_true');args=ap.parse_args()
    try:
        lockpath=OUT/'GPU.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
        lock=lockpath.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        Experiment(args.cell,args.support).run(args.gate_only)
    except BaseException as e:
        dump(OUT/args.support/args.cell/('FAILURE_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S')+'.json'),dict(error=repr(e),traceback=traceback.format_exc(),at=now(),preserved=True));raise
