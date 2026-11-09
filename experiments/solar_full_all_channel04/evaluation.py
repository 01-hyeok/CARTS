"""All-channel evaluation. Query/candidate chunking only; no channel fallback."""
import json,time
from pathlib import Path
import numpy as np
import torch
import batch_core as f

class FrozenFull:
    def __init__(self,data,model,heads,query_chunk,candidate_chunk):
        self.data=data;self.model=model;self.heads=heads
        self.query_chunk=query_chunk;self.candidate_chunk=candidate_chunk
        model.eval()
        if heads is not None:heads.eval()
        with torch.no_grad():self.key=f.keys(model,data,candidate_chunk)
    @torch.no_grad()
    def probability(self,split,a,b):
        x,_,starts=f.query(self.data,split,a,b);out=[]
        for q in range(0,b-a,self.query_chunk):
            e=min(q+self.query_chunk,b-a)
            out.append(f.probability(f.scores(self.model,self.heads,x[q:e],self.key),self.data.mask(starts[q:e]),self.heads))
        return torch.cat(out)

@torch.no_grad()
def evaluate(data,model,heads,split,query_chunk,candidate_chunk,dest=None):
    from scripts.train_patch_retrieval_expert01 import ndcg_at_k
    frozen=FrozenFull(data,model,heads,query_chunk,candidate_chunk)
    n=len(data.datasets[split]);count=0;sums=np.zeros(7,np.float64);begin=time.perf_counter()
    if dest is not None:
        dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
        predmap=np.lib.format.open_memmap(dest/f'{split}_prediction.npy',mode='w+',dtype=np.float32,shape=(n,data.h,data.c))
        idmap=np.lib.format.open_memmap(dest/f'{split}_ids.npy',mode='w+',dtype=np.int32,shape=(n,data.c,10))
        metrics=np.lib.format.open_memmap(dest/f'{split}_query_metrics.npy',mode='w+',dtype=np.float32,shape=(n,data.c,7))
    for a in range(0,n,query_chunk):
        b=min(a+query_chunk,n);B=b-a;C=data.c;N=len(data.starts)
        x,y,starts=f.query(data,split,a,b);mask=data.mask(starts)
        score=f.scores(model,heads,x,frozen.key);p=f.probability(score,mask,heads)
        pick=f.rt.stable_topk_indices(p.reshape(B*C,N),10,largest=True).reshape(B,C,10)
        assert bool(mask[:,None].expand(B,C,N).gather(-1,pick).all())
        assert bool((pick.sort(-1).values[...,1:]!=pick.sort(-1).values[...,:-1]).all())
        aligned=f.selected_values(data,pick,x[:,:,-1]);pred=aligned.mean(2);err=pred-y
        ret=(aligned-y[:,:,None]).square().mean(-1).mean(-1);mse=err.square().mean(-1);mae=err.abs().mean(-1)
        met=torch.full((B,C,7),float('nan'),device=data.device);met[...,:3]=torch.stack([ret,mse,mae],-1)
        if dest is not None:
            d=f.distances(data,x,y,min(candidate_chunk,128)).reshape(B*C,N);valid=mask[:,None].expand(B,C,N).reshape(B*C,N);sel=pick.reshape(B*C,10)
            oracle=f.rt.stable_topk_indices(d.masked_fill(~valid,float('inf')),10,largest=False)
            met[...,3]=(sel[:,:,None]==oracle[:,None]).any(-1).float().mean(-1).reshape(B,C)
            met[...,4]=(d.gather(1,sel).mean(-1)-d.gather(1,oracle).mean(-1)).reshape(B,C)
            met[...,5]=ndcg_at_k(sel,d,valid,10).reshape(B,C)
            met[...,6]=(mse<(x[:,:,-1,None]-y).square().mean(-1)).float()
            predmap[a:b]=pred.permute(0,2,1).cpu().numpy();idmap[a:b]=pick.cpu().numpy();metrics[a:b]=met.cpu().numpy()
        sums+=met.reshape(B*C,7).double().sum(0).cpu().numpy();count+=B*C
        if a%(query_chunk*128)==0:print('ALL_CHANNEL_FULL_EVAL',data.cell,split,b,'/',n,'channels',C,flush=True)
    assert count==n*data.c
    names=['retmse10','agg_mse10','agg_mae10','recall10','oracle_regret','ndcg10','last_win_rate']
    result=dict(split=split,queries=n,channels=data.c,units=count,seconds=time.perf_counter()-begin,all_channel_batch=True,
                **{k:float(v) for k,v in zip(names,sums/count) if np.isfinite(v)})
    if dest is not None:
        predmap.flush();idmap.flush();metrics.flush();np.save(dest/f'{split}_query_start.npy',data.datasets[split].starts);f.rt.write_json(dest/f'{split}_metrics.json',result)
    return result
