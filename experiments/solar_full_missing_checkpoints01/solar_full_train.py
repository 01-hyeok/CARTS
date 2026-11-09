"""Eight explicitly authorized Solar Full runs, blocked until all prior diagnostics.

Fresh seed0 encoder initialization; existing z-score teacher and canonical V0/V5
losses. Fixed ten epochs, batch32, global Full-candidate normalization. No Stage2.
"""
import argparse,copy,gc,json,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
import solar_full_core as core
rt=core.rt;ROOT=core.ROOT;HERE=Path(__file__).resolve().parent
DIAG=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01'
OUT=ROOT/'results/TRACK-SOLAR-FULL-MISSING-CHECKPOINTS01'
CKPT=ROOT/'checkpoints/track_solar_full_missing_checkpoints01'
GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'

def read(p):return json.loads(Path(p).read_text())
def dump(p,x):rt.write_json(p,x)
def source_hashes():return {p.name:rt.sha256_file(p) for p in HERE.glob('*.py')}
def prerequisites():
    cells=[f'{d}_{h}' for d in ['ETTh1','Weather','Solar'] for h in [96,192,336,720]]
    assert read(ROOT/'results/TRACK-P100-HEAD-RETRIEVAL-QUALITY01-ORACLE-K-SUPPLEMENT01/FINAL_GATE.json')['status']=='PASS'
    for support in ['P100','FULL']:
        for cell in cells:assert read(DIAG/support/cell/'COMPLETE.json')['status'] in ['COMPLETE','COMPLETE_WITH_MISSING_MODELS']
    assert read(DIAG/'07_Summary/SUMMARY_GATE.json')['status']=='PASS'
    state=read(DIAG/'00_Audit/CONTROLLER_STATUS.json')
    assert state['status']=='DIAGNOSTICS_COMPLETE_POST_TRAINING_GATE_PENDING',state
    return True

def guard():
    prerequisites()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    assert GPU not in apps,'Another GPU1 process is present; never compete or stop it'
    rt.gpu_guard();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def batch_update(data,model,heads,opt,x,y,starts,channel_chunk):
    opt.zero_grad(set_to_none=True);mask=data.mask(starts);loss=0.
    for a in range(0,data.c,channel_chunk):
        channels=list(range(a,min(a+channel_chunk,data.c)))
        history,values=core.bank_views(data,channels)
        loss+=core.chunk_backward(model,heads,x[:,:,channels].permute(0,2,1),y[:,:,channels].permute(0,2,1),history,values,mask,data.c)
        del history,values
    for p in model.encoder.parameters():assert p.grad is not None and torch.isfinite(p.grad).all()
    if heads is not None:
        for p in heads.parameters():assert p.grad is not None and torch.isfinite(p.grad).all()
    assert np.isfinite(loss);opt.step();return loss

@torch.no_grad()
def evaluate(data,model,heads,split,dest=None):
    """Canonical per-channel eval; full key bank refreshed once per channel."""
    model.eval()
    if heads is not None:heads.eval()
    n=len(data.datasets[split]);count=0;sums=np.zeros(7,np.float64);begin=time.perf_counter()
    if dest is not None:
        dest.mkdir(parents=True,exist_ok=True)
        preds=np.lib.format.open_memmap(dest/f'{split}_prediction.npy',mode='w+',dtype=np.float32,shape=(n,data.h,data.c))
        picks=np.lib.format.open_memmap(dest/f'{split}_ids.npy',mode='w+',dtype=np.int32,shape=(n,data.c,10))
        measures=np.lib.format.open_memmap(dest/f'{split}_query_metrics.npy',mode='w+',dtype=np.float32,shape=(n,data.c,7))
    for c in range(data.c):
        history=data.bank_history(c)
        keys=core.F.normalize(core.encode_raw_channel_first(model,history,0),dim=-1)
        values=data.values(torch.arange(len(data.starts),device=data.device),c)
        for a in range(0,n,32):
            b=min(a+32,n);x,y,starts=data.channel_batch(split,a,b,c);mask=data.mask(starts)
            q=core.encode_raw_channel_first(model,x,0)
            if heads is None:
                score=core.F.normalize(q,dim=-1)@keys.T;prob=score.masked_fill(~mask,-float('inf'))
            else:
                score=torch.einsum('bsd,nd->bsn',heads(q),keys)
                _,_,prob=rt.mean_mixture_topk_selection(score,mask,.1,10)
            selected=rt.stable_topk_indices(prob,10,largest=True)
            aligned=values[selected]+x[:,-1,0,None,None];truth=y[:,:,0]
            error=aligned.mean(1)-truth
            ret=((aligned-truth[:,None])**2).mean(-1).mean(-1)
            mse=error.square().mean(-1);mae=error.abs().mean(-1)
            met=torch.full((b-a,7),float('nan'),device=data.device)
            met[:,:3]=torch.stack([ret,mse,mae],-1)
            if dest is not None:
                from scripts.train_factorial_e2e01 import individual_utility_memsafe
                from scripts.train_patch_retrieval_expert01 import ndcg_at_k
                d=-individual_utility_memsafe(values,x[:,-1,0],truth,1024)
                oracle=rt.stable_topk_indices(d.masked_fill(~mask,float('inf')),10,largest=False)
                met[:,3]=(selected[:,:,None]==oracle[:,None]).any(-1).float().mean(-1)
                met[:,4]=d.gather(1,selected).mean(-1)-d.gather(1,oracle).mean(-1)
                met[:,5]=ndcg_at_k(selected,d,mask,10)
                met[:,6]=(mse<(x[:,-1,0,None]-truth).square().mean(-1)).float()
                preds[a:b,:,c]=aligned.mean(1).cpu().numpy();picks[a:b,c]=selected.cpu().numpy();measures[a:b,c]=met.cpu().numpy()
            sums+=met.double().sum(0).cpu().numpy();count+=b-a
        del history,keys,values
        if c%16==0:print('FULL EVAL',data.cell,split,c+1,'/',data.c,flush=True)
    assert count==n*data.c
    if dest is not None:
        preds.flush();picks.flush();measures.flush();np.save(dest/f'{split}_query_start.npy',data.datasets[split].starts)
    names=['retmse10','agg_mse10','agg_mae10','recall10','oracle_regret','ndcg10','last_win_rate']
    result=dict(split=split,queries=n,channels=data.c,units=count,seconds=time.perf_counter()-begin,
        **{k:float(v) for k,v in zip(names,sums/count) if np.isfinite(v)})
    if dest is not None:dump(dest/f'{split}_metrics.json',result)
    return result

def gpu_equivalence_gate():
    """Small real-architecture GPU check; only called after predecessor gates."""
    results=[]
    for nheads in [0,5]:
        model,heads,_,_=rt.model_init('Solar_720',0,nheads,'cuda:0');model.eval()
        if heads is not None:heads.eval()
        other=copy.deepcopy(model);oh=copy.deepcopy(heads)
        torch.manual_seed(143);x=torch.randn(2,3,720,device='cuda:0');bank=torch.randn(3,257,720,device='cuda:0')
        values=torch.randn_like(bank);y=torch.randn_like(x);mask=torch.ones(2,257,device='cuda:0',dtype=torch.bool);mask[:,:7]=False
        distance=core.teacher_distances(values,x[:,:,-1],y)
        score=core.scores_group(model,heads,x,bank);loss=core.objective(score,distance,mask,heads);loss.backward()
        ref_loss=0.;max_score=0.;max_grad=0.
        for c in range(3):
            ref=core.scores_group(other,oh,x[:,c:c+1],bank[c:c+1]);max_score=max(max_score,float((score[:,c:c+1]-ref).abs().max()))
            torch.testing.assert_close(score[:,c:c+1],ref,atol=2e-6,rtol=2e-5)
            part=core.objective(ref,distance[:,c:c+1],mask,oh)/3;part.backward();ref_loss+=float(part)
        assert abs(float(loss)-ref_loss)<2e-5
        a=list(model.parameters())+([] if heads is None else list(heads.parameters()))
        b=list(other.parameters())+([] if oh is None else list(oh.parameters()))
        for p,q in zip(a,b):
            assert (p.grad is None)==(q.grad is None)
            if p.grad is not None:
                max_grad=max(max_grad,float((p.grad-q.grad).abs().max()));torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=2e-4)
        # Same SGD update is an equivalence diagnostic, not the training optimizer.
        torch.optim.SGD(a,lr=.001).step();torch.optim.SGD(b,lr=.001).step()
        for p,q in zip(a,b):torch.testing.assert_close(p,q,atol=2e-6,rtol=2e-5)
        model.train()
        if heads is not None:heads.train()
        state=core.rng_snapshot();first=core.scores_group(model,heads,x,bank);core.rng_restore(state);second=core.scores_group(model,heads,x,bank)
        assert torch.equal(first,second)
        results.append(dict(model=f'V{nheads}',horizon=720,status='PASS',score_max_abs=max_score,gradient_max_abs=max_grad,loss_abs=abs(float(loss)-ref_loss),dropout_replay_exact=True))
        del model,heads,other,oh,a,b,score,loss,first,second;gc.collect();torch.cuda.empty_cache()
    return results

def smoke():
    guard();equivalence=gpu_equivalence_gate();results=[];chosen=None
    # Worst-horizon V5 determines one common channel chunk for all eight runs.
    for chunk in [137,64,32,16,8]:
        data=model=heads=opt=x=y=None
        try:
            data=rt.CellData('Solar_720','cuda:0');model,heads,_,_=rt.model_init('Solar_720',0,5,'cuda:0')
            model.train();heads.train();x,y,starts=data.batch('train',np.arange(32))
            opt=torch.optim.Adam(list(model.parameters())+list(heads.parameters()),lr=.001)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
            loss=batch_update(data,model,heads,opt,x,y,starts,chunk);torch.cuda.synchronize()
            results.append(dict(cell='Solar_720',model='V5',channel_chunk=chunk,status='PASS',loss=loss,seconds=time.perf_counter()-t,
                full_candidates=len(data.starts),channels=data.c,batch=32,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
            chosen=chunk;break
        except torch.cuda.OutOfMemoryError as e:
            results.append(dict(channel_chunk=chunk,status='OOM',error=str(e)));dump(OUT/'SMOKE_ATTEMPTS.json',results)
        finally:
            del data,model,heads,opt,x,y;gc.collect();torch.cuda.empty_cache()
    assert chosen is not None,'H720 real Full gradient smoke OOM at all safe chunks'
    data=rt.CellData('Solar_96','cuda:0');model,heads,_,_=rt.model_init('Solar_96',0,5,'cuda:0')
    model.train();heads.train();x,y,starts=data.batch('train',np.arange(32));opt=torch.optim.Adam(list(model.parameters())+list(heads.parameters()),lr=.001)
    loss=batch_update(data,model,heads,opt,x,y,starts,chosen)
    results.append(dict(cell='Solar_96',model='V5',channel_chunk=chosen,status='PASS',loss=loss,full_candidates=len(data.starts),channels=data.c,batch=32))
    dump(OUT/'SMOKE_GATE.json',dict(status='PASS',at=rt.now(),source_hashes=source_hashes(),canonical_source_hashes=rt.source_hashes(),channel_chunk=chosen,results=results,
        gpu_equivalence=equivalence,dropout_policy='seed0 deterministic grouped path; not bit-identical to historical per-channel dropout order',
        full_candidate_global_normalization=True,query_and_key_gradients=True))
    print('FULL SOLAR SMOKE PASS',json.dumps(results),flush=True)

def train(cell,nheads):
    guard();gate=read(OUT/'SMOKE_GATE.json');assert gate['status']=='PASS' and gate['source_hashes']==source_hashes()
    assert gate['canonical_source_hashes']==rt.source_hashes()
    assert cell in [f'Solar_{h}' for h in [96,192,336,720]] and nheads in [0,5]
    dest=OUT/cell/f'V{nheads}'/'seed0';ck=CKPT/cell/f'V{nheads}'/'seed0'
    if (dest/'COMPLETE.json').exists():
        done=read(dest/'COMPLETE.json');assert done['source_hashes']==source_hashes();assert rt.sha256_file(done['checkpoint'])==done['checkpoint_sha256'];return
    dest.mkdir(parents=True,exist_ok=True);ck.mkdir(parents=True,exist_ok=True)
    begin=time.perf_counter();data=rt.CellData(cell,'cuda:0');assert data.c==137
    model,heads,init_hash,head_hash=rt.model_init(cell,0,nheads,'cuda:0')
    params=list(model.parameters())+([] if heads is None else list(heads.parameters()))
    opt=torch.optim.Adam(params,lr=.001);loader=data.loader('train',batch=32,seed=0);history=[];start_epoch=1
    if (ck/'latest.pth').exists():
        saved=torch.load(ck/'latest.pth',map_location='cpu',weights_only=False);assert saved['source_hashes']==source_hashes()
        model.load_state_dict(saved['model_state_dict']);opt.load_state_dict(saved['optimizer'])
        if heads is not None:heads.load_state_dict(saved['slot_heads_state_dict'])
        loader.generator.set_state(saved['loader_rng']);core.rng_restore(saved['rng']);history=saved['history'];start_epoch=saved['epoch']+1
    dump(dest/'PROVENANCE.json',dict(cell=cell,model=f'V{nheads}',seed=0,source_hashes=source_hashes(),data=data.data_info,
        full_support=True,teacher='canonical per-query valid FULL z-score FutureMSE, tau=.1',student='global full softmax, tau=.1; V5 mean probabilities',
        objective='historical V0 log-softmax KL' if nheads==0 else 'historical V5 kl_loss_from_prob',batch=32,epochs=10,adam_lr=.001,
        initialization='seed0 fresh Model; saved reference args only, no pretrained weights',init_hash=init_hash,head_hash=head_hash,
        query_key_gradients=True,channel_chunk=gate['channel_chunk'],dropout=gate['dropout_policy'],selection='minimum validation Mean-Mixture RetMSE after 10 fixed epochs',stage2=False))
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch,11):
        model.train()
        if heads is not None:heads.train()
        t=time.perf_counter();loss_sum=0.;count=0;steps=0
        for x,y,starts in loader:
            x=x.to(data.device).float();y=y.to(data.device).float()
            value=batch_update(data,model,heads,opt,x,y,starts,gate['channel_chunk'])
            loss_sum+=value*len(x);count+=len(x);steps+=1
            if steps%25==0:
                dump(dest/'CURRENT.json',dict(epoch=epoch,steps=steps,queries=count,total_queries=len(data.datasets['train']),at=rt.now(),pid=os.getpid()))
                print('FULL TRAIN',cell,nheads,'epoch',epoch,'step',steps,'/',len(loader),'loss',value,flush=True)
        assert count==len(data.datasets['train'])
        val=evaluate(data,model,heads,'val')
        history.append(dict(epoch=epoch,train_kl=loss_sum/count,train_queries=count,steps=steps,val=val,seconds=time.perf_counter()-t))
        saved=dict(epoch=epoch,model_state_dict=model.state_dict(),slot_heads_state_dict=None if heads is None else heads.state_dict(),optimizer=opt.state_dict(),
            rng=core.rng_snapshot(),loader_rng=loader.generator.get_state(),history=history,source_hashes=source_hashes(),args=vars(data.args))
        rt.atomic_torch(ck/f'epoch{epoch:02d}.pth',saved);rt.atomic_torch(ck/'latest.pth',saved)
        dump(dest/'history.json',history);print('FULL EPOCH COMPLETE',cell,nheads,epoch,json.dumps(val),flush=True)
    assert len(history)==10
    selected=min(history,key=lambda r:(r['val']['retmse10'],r['epoch']));path=ck/f'epoch{selected["epoch"]:02d}.pth'
    payload=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(payload['model_state_dict'])
    if heads is not None:heads.load_state_dict(payload['slot_heads_state_dict'])
    selection=dict(epoch=selected['epoch'],validation_retmse=selected['val']['retmse10'],checkpoint=str(path),sha256=rt.sha256_file(path),test_used_for_selection=False)
    dump(dest/'checkpoint_selection.json',selection)
    val=evaluate(data,model,heads,'val',dest/'cache');test=evaluate(data,model,heads,'test',dest/'cache')
    assert val['retmse10']==selected['val']['retmse10'],'Selected validation metric did not reproduce'
    dump(dest/'COMPLETE.json',dict(status='COMPLETE',at=rt.now(),cell=cell,model=f'V{nheads}',seed=0,checkpoint=str(path),checkpoint_sha256=selection['sha256'],
        selection=selection,validation=val,test=test,source_hashes=source_hashes(),canonical_source_hashes=rt.source_hashes(),wall_seconds_current_attempt=time.perf_counter()-begin,
        completed_epoch_train_and_validation_wall_seconds=sum(r['seconds'] for r in history),final_evaluation_wall_seconds=val['seconds']+test['seconds'],peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),full_candidates=len(data.starts),channels=137,stage2=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--cell');ap.add_argument('--heads',type=int,choices=[0,5]);args=ap.parse_args()
    try:
        if args.smoke:smoke()
        else:train(args.cell,args.heads)
    except BaseException as exc:
        dump(OUT/'IMMEDIATE_ISSUE.json',dict(status='FAILED',at=rt.now(),cell=args.cell,heads=args.heads,error=repr(exc),traceback=traceback.format_exc()))
        raise
