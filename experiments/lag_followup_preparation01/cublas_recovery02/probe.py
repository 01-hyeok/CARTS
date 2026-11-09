"""Boundary-only, past-input kernel investigation. Never modifies production.

Channel loops are canonical reference checks only. Every alternative production
kernel consumes all 21 channels in one batch. No future values select a backend.
"""
import sys, json, time, traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import server_bridge as bridge
import adapter
import torch
import torch.nn.functional as F
from cublaslt_batch import CublasLtBatch
from cublas_batch import CublasBatch
from exact_short import Split64
from contracts import BACKENDS,actual_batches,select_backend
ROOT=bridge.ROOT
OUT=ROOT/'results/TRACK-HEAD-SHIFT-DIAGNOSTIC01/extensions_d5_d6/cublas_recovery02'

@torch.no_grad()
def canonical_keys(f):
    parts=[];d=f.data
    for c in range(d.c):
        history=d.bank_history(c);chunks=[]
        for i in range(0,len(history),3200):
            x=history[i:i+3200];n=len(x)
            if n<3200:x=torch.cat([x,x.new_zeros(3200-n,d.h,1)])
            chunks.append(bridge.rt.encode_raw(f.model,x,0)[:n])
        parts.append(F.normalize(torch.cat(chunks),dim=-1))
    return torch.stack(parts)

@torch.no_grad()
def canonical_query(f,split,a,b):
    parts=[]
    for c in range(f.data.c):
        x,_,_=f.data.channel_batch(split,a,b,c,future=False)
        z=bridge.rt.encode_raw(f.model,x,0)
        parts.append(F.normalize(z,dim=-1)[:,None] if f.heads is None else f.heads(z))
    return torch.stack(parts,1)

@torch.no_grad()
def batched_query(f,split,a,b):
    d=f.data;x,_,_=d.query(split,a,b);n=b-a
    if n<d.c:
        return adapter.p100_v0_short_query(f.batch,x) if f.heads is None else f.batch.short_query(x)
    step=max(1,n//d.c)
    return torch.cat([f.batch.encode(x[i:i+step],n,project=True) for i in range(0,n,step)])

@torch.no_grad()
def reference_scores(f,q,k):
    # Diagnostic only; these scores never substitute for a failed batch kernel.
    return torch.stack([(q[:,c,0]@k[c].T)[:,None] if f.heads is None
                       else torch.einsum('bsd,nd->bsn',q[:,c],k[c])
                       for c in range(f.data.c)],1)

@torch.no_grad()
def main():
    before=bridge.start_gpu();OUT.mkdir(parents=True,exist_ok=True)
    data=bridge.Data('Weather_336');assert data.c==21 and len(data.starts)==36216
    rows=[];regr=[];baseline=[];begin=time.perf_counter()
    kernels={'cuda12_lt':CublasLtBatch('libcublasLt.so.12'),
             'split64':Split64(),'cuda11_blas':CublasBatch()}
    for model in ['V0','V1']:
        row=next(r for r in bridge.inventory(data.cell) if r['architecture']==model and r['training_support']=='P100')
        f=bridge.Frozen(data,row);f.precision();assert not f.tf32
        f.batch.build_keys();reference_key=canonical_keys(f)
        key_exact=torch.equal(reference_key,f.batch.keys)
        assert key_exact,'candidate encoder mismatch; no score recovery may hide it'
        for split in ['val','test']:
            for a,b in actual_batches(len(data.datasets[split])):
                qref=canonical_query(f,split,a,b);q=batched_query(f,split,a,b)
                ref=reference_scores(f,qref,reference_key)
                mask=data.mask(data.datasets[split].starts[a:b])[:,None].expand(b-a,data.c,-1)
                rid=bridge.topk(ref[:,:,0].masked_fill(~mask,-float('inf')),500)
                base=dict(model=model,split=split,start=a,end=b,batch=b-a,channels=data.c,
                          bank_count=len(data.starts),query_exact=torch.equal(q,qref),keys_exact=key_exact,
                          mask_exact=True,query_max_delta=float((q-qref).abs().max()),checkpoint_sha256=row['checkpoint_sha256'])
                if b-a==32:
                    original,om=f.full_logits(split,a,b)
                    baseline.append(dict(**base,scores_exact=torch.equal(original,ref),mask_exact_original=torch.equal(om,mask),
                                         ordered_top500_exact=torch.equal(bridge.topk(original[:,:,0].masked_fill(~om,-float('inf')),500),rid)))
                    continue
                qq=q.permute(1,0,2,3).contiguous().reshape(data.c,b-a,128)
                for name in BACKENDS:
                    result=dict(base,backend=name)
                    try:
                        torch.cuda.synchronize();tick=time.perf_counter()
                        if name=='split64':value=kernels[name](qq,f.batch.keys)
                        else:value=kernels[name].score(qq,f.batch.keys,68,1,0) if name=='cuda12_lt' else kernels[name].score(qq,f.batch.keys,68,-1)
                        torch.cuda.synchronize();value=value.permute(1,0,2)[:,:,None]
                        result.update(seconds=time.perf_counter()-tick,scores_exact=torch.equal(value,ref),
                                      score_max_delta=float((value-ref).abs().max()),
                                      ordered_top500_exact=torch.equal(bridge.topk(value[:,:,0].masked_fill(~mask,-float('inf')),500),rid),
                                      input_shape=list(qq.shape),keys_shape=list(f.batch.keys.shape),channel_batch=data.c)
                    except Exception as error:result['error']=repr(error)
                    rows.append(result);bridge.dump(OUT/'PROBE_PROGRESS.json',dict(rows=rows,baseline=baseline))
                    print('TAIL_KERNEL',json.dumps(result),flush=True)
                del ref,q,qref
        # Real past-only score tensors, all tail sizes 1..31. Separate regression
        # evidence from the actual dataset tails required for a scoped patch.
        q=canonical_query(f,'val',0,32)
        for n in range(1,32):
            ref=reference_scores(f,q[:n],reference_key)
            qq=q[:n].permute(1,0,2,3).contiguous().reshape(data.c,n,128)
            for name in BACKENDS:
                r=dict(model=model,batch=n,backend=name,channel_batch=data.c)
                try:
                    if name=='split64':value=kernels[name](qq,f.batch.keys)
                    else:value=kernels[name].score(qq,f.batch.keys,68,1,0) if name=='cuda12_lt' else kernels[name].score(qq,f.batch.keys,68,-1)
                    value=value.permute(1,0,2)[:,:,None]
                    r.update(scores_exact=torch.equal(value,ref),score_max_delta=float((value-ref).abs().max()))
                except Exception as error:r['error']=repr(error)
                regr.append(r)
        del f,reference_key,q;torch.cuda.empty_cache()
    plan=select_backend(rows)
    baseline_pass=all(r['query_exact'] and r['scores_exact'] and r['mask_exact_original'] and r['ordered_top500_exact'] for r in baseline)
    result=dict(status='PASS_ACTUAL_TAIL_EXACT_KERNELS_PRODUCTION_NOT_ACTIVATED' if plan and baseline_pass else 'BLOCKED_EXACT_KERNEL_NOT_FOUND',
                plan=plan,baseline=baseline,actual_tail_rows=rows,score_tail_1_to_31=regr,
                whole_sample_replay_pending=True,production_activated=False,all_channels_single_batch=True,
                original_result_or_checkpoint_modified=False,seconds=time.perf_counter()-begin,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                gpu_before=before,gpu_after=bridge.snapshot(),source_hashes={p.name:bridge.sha(p) for p in HERE.glob('*.py')})
    bridge.dump(OUT/'PROBE_GATE.json',result);print(result['status'],flush=True)
    if not plan or not baseline_pass:raise RuntimeError('No exact scoped solution; production remains unchanged')

if __name__=='__main__':
    try:main()
    except BaseException as error:
        bridge.dump(OUT/'PROBE_ERROR.json',dict(error=repr(error),traceback=traceback.format_exc(),production_activated=False));raise
