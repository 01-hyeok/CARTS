"""Isolated Full-only tensor channel-batch adaptation; historical files untouched."""
import sys,time,json,hashlib
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments/retrieval_full_precision_repair02'),str(ROOT)]
import full_precision_runtime as previous
import diagnostic_adapter as adapter
import diagnostic_run as run
import torch
import numpy as np
from packed_batch import PackedChannelFull
from all_channel_ops import distance_all

previous_probability=adapter.Frozen.probability
previous_immutable=adapter.Frozen.immutable_check
previous_snapshot=adapter.snapshot_hashes


def probability(self,split,start,end):
    assert self.support=='FULL','Completed P100 diagnostics are preserved, not rerun'
    if self.heads is None and previous.uses_score_cache(self.data.cell):
        # Already verified immutable scores have all channel/candidate axes.
        return previous_probability(self,split,start,end)
    if not hasattr(self,'all_channel_accelerator'):
        self.all_channel_accelerator=PackedChannelFull(self)
    return self.all_channel_accelerator.probability(split,start,end)


def immutable(self):
    result=previous_immutable(self)
    result['all_channel_execution']='true channel-batched tensor inputs; only query/candidate chunking; padding preserves canonical GEMM row counts; batched scores/global softmax; no streams or per-channel encoder calls'
    result['original_generation_runtime']='UNKNOWN'
    if hasattr(self,'all_channel_accelerator'):
        result['channel_execution_shapes']=self.all_channel_accelerator.shapes[:2]
    else:result['channel_execution_shapes']='immutable all-channel canonical score cache reused'
    return result


def snapshot():
    result=previous_snapshot();result.update({str(p.relative_to(ROOT)):adapter.rt.sha256_file(p) for p in HERE.iterdir() if p.suffix in ['.py','.cu','.so']});return result


@torch.no_grad()
def prepare_block(data,split,a,b,support,calendar,ridge,chunk=128):
    assert support=='FULL'
    x,y,starts=data.query(split,a,b);n=b-a;c=data.c;N=len(data.starts)
    ids_all=torch.arange(N,device=data.device).expand(n,c,N)
    mask=data.mask(starts)[:,None].expand(n,c,N)
    d=torch.empty(n,c,N,device=data.device);base=torch.empty(n,c,N,4,device=data.device);probe=torch.empty(n,c,N,3,device=data.device)
    cols=run.probe_columns(calendar);oracle_seconds=feature_seconds=0.
    for k in range(0,N,chunk):
        ids=ids_all[...,k:k+chunk]
        run.sync();t=time.perf_counter();values=data.values_all(ids)
        d[...,k:k+chunk]=distance_all(values,x[:,:,-1],y)
        run.sync();oracle_seconds+=time.perf_counter()-t;t=time.perf_counter()
        f=run.feature_block(data,x,starts,ids,calendar)
        base[...,k:k+chunk,:]=f[...,:4];probe[...,k:k+chunk,:]=ridge.predict(f[...,cols])
        run.sync();feature_seconds+=time.perf_counter()-t
    # These CPU RNG loops do not serialize GPU channel computation.
    rnd=np.stack([run.fixed_random_scores(starts,j,N) for j in range(c)],1)
    scores={'Random':torch.as_tensor(rnd,device=data.device,dtype=torch.float32)}
    for j,name in enumerate(run.BASE_NAMES[1:]):scores[name]=base[...,j]
    return dict(x=x,y=y,starts=starts,ids=ids_all,mask=mask,d=d,scores=scores,probe=probe,oracle_seconds=oracle_seconds,feature_seconds=feature_seconds)


@torch.no_grad()
def reproduce(data,models,support,out):
    from scripts.train_patch_retrieval_expert01 import ndcg_at_k
    assert support=='FULL';begin=time.perf_counter();gates=[];torch.cuda.reset_peak_memory_stats()
    if not models:
        run.dump(out/'CANONICAL_REPRODUCTION.json',dict(status='PASS_DATA_ONLY_MODELS_MISSING',seconds=0,rows=[],missing_models_not_imputed=True));return
    for split in ('val','test'):
        sums={m.name:np.zeros(4) for m in models};count=0;hashes={m.name:hashlib.sha256() for m in models};max_pred={m.name:0. for m in models};canonical_hashes={m.name:hashlib.sha256() for m in models}
        for a in range(0,len(data.datasets[split]),32):
            b=min(a+32,len(data.datasets[split]));B=b-a;C=data.c;N=len(data.starts)
            probs={m.name:m.probability(split,a,b) for m in models};x,y,starts=data.query(split,a,b)
            mask=data.mask(starts)[:,None].expand(B,C,N);d=torch.empty(B,C,N,device=data.device)
            for k in range(0,N,128):
                ids=torch.arange(k,min(k+128,N),device=data.device).expand(B,C,-1)
                d[...,k:k+128]=distance_all(data.values_all(ids),x[:,:,-1],y)
            flatmask=mask.reshape(B*C,N);flatd=d.reshape(B*C,N)
            oracle=adapter.rt.stable_topk_indices(flatd.masked_fill(~flatmask,float('inf')),10,largest=False)
            for m in models:
                pick=adapter.rt.stable_topk_indices(probs[m.name].reshape(B*C,N),10,largest=True)
                # Reference-only comparison: canonical original channel calls
                # are never used as the production inference/fallback path.
                canonical_probability=previous_probability(m,split,a,b)
                canonical_pick=adapter.rt.stable_topk_indices(canonical_probability.reshape(B*C,N),10,largest=True)
                assert torch.equal(pick,canonical_pick),(m.name,split,a,'canonical ordered TopK ID mismatch')
                canonical_hashes[m.name].update(canonical_pick.reshape(B,C,10).cpu().numpy().transpose(1,0,2).copy().tobytes())
                ids=pick.reshape(B,C,10);hashes[m.name].update(ids.cpu().numpy().transpose(1,0,2).copy().tobytes())
                vals=data.values_all(ids)+x[:,:,-1,None,None]
                pred=vals.reshape(B*C,10,data.h).mean(1);truth=y.reshape(B*C,data.h)
                ret=flatd.gather(1,pick).mean(1);agg=(pred-truth).square().mean(1)
                recall=(pick[:,:,None]==oracle[:,None,:]).any(-1).float().mean(1)
                ndcg=ndcg_at_k(pick,flatd,flatmask,10)
                sums[m.name]+=torch.stack([ret,agg,recall,ndcg],1).double().sum(0).cpu().numpy()
                expected=m.reference(split)[0][a:b].transpose(0,2,1);new=pred.reshape(B,C,data.h).cpu().numpy()
                max_pred[m.name]=max(max_pred[m.name],float(abs(new-expected).max()))
                assert np.allclose(new,expected,atol=1e-5,rtol=1e-5),(m.name,split,a,'unchanged canonical prediction tolerance')
            count+=B*C
            if a==0:run.dump(out/f'REAL_BATCH_GATE_{split}.json',dict(status='PASS',queries=B,channels=C,support=support,all_valid_candidates=True,all_channel_gpu=True,models=[m.name for m in models]))
            if a%512==0:print('ALL_CHANNEL_REPRODUCTION',data.cell,split,b,'/',len(data.datasets[split]),flush=True)
        assert count==len(data.datasets[split])*data.c
        for m in models:
            got=sums[m.name]/count
            if split=='test':
                expected=m.reference(split)[1]
                for j,k in enumerate(['retmse10','agg_mse10','recall10','ndcg10']):assert np.isclose(got[j],expected[k],atol=1e-5,rtol=1e-4),(m.name,k,got[j],expected[k])
            assert hashes[m.name].hexdigest()==canonical_hashes[m.name].hexdigest()
            gates.append(dict(status='PASS',model=m.name,split=split,units=count,metric_names=['retmse','agg_mse','recall','legacy_ndcg10'],metrics=got.tolist(),ordered_candidate_id_sha256=hashes[m.name].hexdigest(),canonical_ordered_id_sha256=canonical_hashes[m.name].hexdigest(),ordered_mismatches=0,max_stored_prediction_difference=max_pred[m.name]))
    run.dump(out/'CANONICAL_REPRODUCTION.json',dict(status='PASS',seconds=time.perf_counter()-begin,rows=gates,missing_models_not_imputed=True,all_channel_gpu=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),reference_only_channel_loop=True,production_serial_fallback=False))


adapter.Frozen.probability=probability
adapter.Frozen.immutable_check=immutable
adapter.snapshot_hashes=snapshot
run.snapshot_hashes=snapshot
run.prepare_block=prepare_block
run.reproduce=reproduce
# Reuse the original metric implementation, substituting only its three
# channel-wise reductions. Exact source assertions fail closed on upstream edit.
import inspect,textwrap
src=textwrap.dedent(inspect.getsource(run.evaluate_micro))
changes={
 'pred=torch.stack([val[:,ci].contiguous().mean(1) for ci in range(C)],1)':
 'pred=val.contiguous().reshape(B*C,val.shape[2],data.h).mean(1).reshape(B,C,data.h)',
 "torch.stack([(pred[:,ci].contiguous()-y[:,ci].contiguous()).square().mean(-1) for ci in range(C)],1).reshape(-1)":
 "(pred.reshape(B*C,data.h)-y.reshape(B*C,data.h)).square().mean(-1)",
 "torch.stack([(pred[:,ci].contiguous()-y[:,ci].contiguous()).abs().mean(-1) for ci in range(C)],1).reshape(-1)":
 "(pred.reshape(B*C,data.h)-y.reshape(B*C,data.h)).abs().mean(-1)"}
for old,new in changes.items():
    assert src.count(old)==1,old
    src=src.replace(old,new)
exec(compile(src,str(HERE/'runtime_all_channels.py')+'::evaluate_micro','exec'),run.__dict__)
