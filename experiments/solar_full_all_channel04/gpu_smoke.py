"""Deferred GPU gates. Imported only after complete predecessor artifact checks."""
import copy,gc,time
import numpy as np
import torch
import batch_core as core
import evaluation

def equivalence():
    records=[]
    for h in (96,192,336,720):
      for nheads in (0,5):
        from types import SimpleNamespace
        # All 137 channels even for the synthetic arithmetic comparison.
        c=137;n=37;b=2;torch.manual_seed(143)
        series=torch.randn(n+2*h-1,c,device='cuda:0')
        data=SimpleNamespace(c=c,h=h,starts=np.arange(n),series=series,future_series=series,offsets=torch.arange(h,device='cuda:0'),device='cuda:0',mask=lambda s:torch.ones(len(s),n,device='cuda:0',dtype=torch.bool))
        model,heads,_,_=core.rt.model_init('Solar_'+str(h),0,nheads,'cuda:0');model.eval()
        if heads is not None:heads.eval()
        other=copy.deepcopy(model);oh=copy.deepcopy(heads);x=torch.randn(b,c,h,device='cuda:0');y=torch.randn_like(x)
        hist,val=core.original.bank_views(data,list(range(c)))
        direct=core.objective(core.original.scores_group(other,oh,x,hist),core.original.teacher_distances(val,x[:,:,-1],y),data.mask(range(b)),oh);direct.backward()
        got=core.backward_batch(data,model,heads,x,y,np.arange(b),b,n)
        assert abs(float(direct)-got)<2e-5
        pp=list(model.parameters())+([] if heads is None else list(heads.parameters()));qq=list(other.parameters())+([] if oh is None else list(oh.parameters()))
        maxgrad=0.
        for p,q in zip(pp,qq):
            assert (p.grad is None)==(q.grad is None)
            if p.grad is not None:
                maxgrad=max(maxgrad,float((p.grad-q.grad).abs().max()));torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=2e-4)
        torch.optim.Adam(pp,lr=.001).step();torch.optim.Adam(qq,lr=.001).step()
        for p,q in zip(pp,qq):torch.testing.assert_close(p,q,atol=2e-6,rtol=2e-5)
        # Training dropout replay and chunk-invariant RNG consumption, using
        # identical source weights and masks. No random draw on recomputation.
        model.train();other=copy.deepcopy(model)
        if heads is not None:heads.train()
        oh=copy.deepcopy(heads);model.zero_grad();other.zero_grad()
        if heads is not None:heads.zero_grad();oh.zero_grad()
        state=core.rng_snapshot()
        direct=core.objective(core.original.scores_group(other,oh,x,hist),core.original.teacher_distances(val,x[:,:,-1],y),data.mask(range(b)),oh);direct.backward()
        after=core.rng_snapshot();core.rng_restore(state)
        got=core.backward_batch(data,model,heads,x,y,np.arange(b),b,n)
        end=core.rng_snapshot()
        assert torch.equal(after['torch'],end['torch'])
        assert all(torch.equal(a,b) for a,b in zip(after['cuda'],end['cuda']))
        assert abs(float(direct)-got)<2e-5
        for p,q in zip(list(model.parameters())+([] if heads is None else list(heads.parameters())),list(other.parameters())+([] if oh is None else list(oh.parameters()))):
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=2e-4)
        records.append(dict(horizon=h,heads=nheads,channels=c,status='PASS',gradient_max_abs=maxgrad,dropout_rng_exact=True))
        del data,model,heads,other,oh,pp,qq,hist,val,direct,x,y;gc.collect();torch.cuda.empty_cache()
    return records

def real_attempt(cell,query_chunk,candidate_chunk):
    data=core.rt.CellData(cell,'cuda:0');assert data.c==137
    model,heads,_,_=core.rt.model_init(cell,0,5,'cuda:0');model.train();heads.train()
    x,y,starts=data.batch('train',np.arange(32));x=x.permute(0,2,1).contiguous();y=y.permute(0,2,1).contiguous()
    core.validate_training_shape(data,query_chunk,candidate_chunk)
    opt=torch.optim.Adam(list(model.parameters())+list(heads.parameters()),lr=.001)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.perf_counter();opt.zero_grad(set_to_none=True)
    loss=core.backward_batch(data,model,heads,x,y,starts,query_chunk,candidate_chunk)
    for p in list(model.encoder.parameters())+list(heads.parameters()):assert p.grad is not None and torch.isfinite(p.grad).all()
    opt.step();assert all(int(v['step'])==1 for v in opt.state.values());torch.cuda.synchronize()
    train_seconds=time.perf_counter()-begin;model.eval();heads.eval()
    frozen=evaluation.FrozenFull(data,model,heads,query_chunk,candidate_chunk)
    with torch.no_grad():
        p=frozen.probability('val',0,32);assert p.shape==(32,137,len(data.starts)) and torch.isfinite(p).all()
        # Saved config must replay exact ordered IDs for validation/followup.
        other=frozen.probability('val',0,32)
        a=core.rt.stable_topk_indices(p.reshape(-1,p.shape[-1]),10,largest=True)
        b=core.rt.stable_topk_indices(other.reshape(-1,p.shape[-1]),10,largest=True);assert torch.equal(a,b)
        assert torch.equal(p,other)
    torch.cuda.synchronize()
    return dict(status='PASS',cell=cell,channels=137,batch=32,query_chunk=query_chunk,candidate_chunk=candidate_chunk,
        full_candidates=len(data.starts),query_encoder_shape=[min(query_chunk,32)*137,data.h,1],candidate_encoder_shape=[min(candidate_chunk,len(data.starts))*137,data.h,1],
        loss=loss,train_seconds=train_seconds,total_seconds=time.perf_counter()-begin,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),ordered_id_replay_exact=True)

def run(train):
    train.guard();eq=equivalence();attempts=[];chosen=None
    # CPU Adam gate rejected smaller encoder GEMM/gradient shapes. Keep the
    # original grouped shape; future-distance arrays still stream P128. If
    # this cannot fit, STOP rather than use a numerically unverified fallback.
    # 10**9 is capped by actual bank size in keys(): every candidate retained.
    for q,p in ((32,10**9),):
        try:
            row=real_attempt('Solar_720',q,p);attempts.append(row);chosen=(q,p);break
        except torch.cuda.OutOfMemoryError as e:
            attempts.append(dict(status='OOM',query_chunk=q,candidate_chunk=p,channels=137,error=str(e)))
            train.dump(train.OUT/'SMOKE_ATTEMPTS.json',attempts)
        finally:gc.collect();torch.cuda.empty_cache()
    assert chosen is not None,'All-channel full-shape H720 OOM; smaller encoder chunks failed CPU Adam gate. Stop/report; no channel reduction'
    for h in (96,192,336):attempts.append(real_attempt('Solar_'+str(h),*chosen));gc.collect();torch.cuda.empty_cache()
    train.dump(train.OUT/'SMOKE_GATE.json',dict(status='PASS',at=train.rt.now(),source_hashes=train.source_hashes(),canonical_source_hashes=train.rt.source_hashes(),
        query_chunk=chosen[0],candidate_chunk=chosen[1],channels=137,all_channels_together=True,results=attempts,gpu_equivalence=eq,
        dropout_policy='canonical grouped query B*C then key C*N masks; fixed across Q/P chunks and recomputation; differs from historical channel-serial RNG order',
        full_candidate_global_normalization=True,query_and_key_gradients=True))
