import sys,json,os
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');sys.path[:0]=[str(ROOT/'experiments/retrieval_full_precision_repair02'),str(ROOT)]
import full_precision_runtime
import diagnostic_adapter as a
import torch
import torch.nn.functional as F
from packed_batch import PackedChannelFull
from all_channel_ops import score_all
from cublas_batch import CublasBatch
OUT=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/TAIL05'
@torch.no_grad()
def main():
    a.rt.gpu_guard();d=a.Data('Weather_192');f=a.Frozen(d,next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V0'),'FULL');f.precision()
    end=len(d.datasets['test']);start=end//32*32;x,_,starts=d.query('test',start,end);B,C,_=x.shape
    ref=f.probability('test',start,end);keys=torch.stack([f.keys[c] for c in range(C)])
    fast=PackedChannelFull(f);qfast=fast.short_query(x)
    q=torch.stack([F.normalize(a.encode_raw_channel_first(f.model,x[:,c,:,None].contiguous(),0),dim=-1) for c in range(C)],1)[:,:,None]
    mask=d.mask(starts)[:,None];top=lambda v:a.rt.stable_topk_indices(v.reshape(B*C,-1),10,largest=True)
    pick=top(ref);rows=[]
    for name,qq in [('canonical',q),('batched',qfast)]:
      for tf32 in [True,False]:
        torch.backends.cuda.matmul.allow_tf32=tf32
        with torch.autograd.profiler.profile(use_cuda=True) as profile:score=score_all(qq,keys)[:,:,0].masked_fill(~mask,-float('inf'))
        delta=(score-ref).abs();delta=delta[torch.isfinite(delta)];ids=top(score)
        kernels=sorted(set(e.name for e in profile.function_events if any(k in e.name.lower() for k in ['gemm','gemv','ampere','volta'])))
        rows.append(dict(query=name,tf32=tf32,score_maxdiff=float(delta.max()),ordered_mismatch=int((ids!=pick).any(-1).sum()),set_mismatch=int((ids.sort(-1).values!=pick.sort(-1).values).any(-1).sum()),kernels=kernels))
    f.precision()
    with torch.autograd.profiler.profile(use_cuda=True) as profile:q[:,0,0].contiguous()@keys[0].T
    canonical_kernels=[dict(name=e.name,kernels=[str(k) for k in e.kernels]) for e in profile.function_events if e.kernels]
    backend=CublasBatch()
    for name,qq in [('canonical',q),('batched',qfast)]:
      for compute in (77,68):
       for algo in (-1,99):
        try:
            score=backend.score(qq[:,:,0].permute(1,0,2),keys,compute,algo).permute(1,0,2).masked_fill(~mask,-float('inf'))
            delta=(score-ref).abs();delta=delta[torch.isfinite(delta)];ids=top(score)
            rows.append(dict(query=name,backend='cublas_strided_batch',compute=compute,algo=algo,score_maxdiff=float(delta.max()),ordered_mismatch=int((ids!=pick).any(-1).sum()),set_mismatch=int((ids.sort(-1).values!=pick.sort(-1).values).any(-1).sum())))
        except RuntimeError as e:rows.append(dict(compute=compute,algo=algo,error=str(e)))
    result=dict(rows=rows,canonical_score_kernels=canonical_kernels);a.rt.write_json(OUT/'WEATHER192_V0_SCORE_KERNELS.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
