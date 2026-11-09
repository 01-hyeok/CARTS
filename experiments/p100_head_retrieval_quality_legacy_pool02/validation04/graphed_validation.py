"""Canonical B32 operations bundled in a CUDA graph; full candidate-cache reuse."""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import runtime as rt
import torch,numpy as np
from cached_validation import CandidateCache


@torch.no_grad()
def evaluate(data,model,heads,split,detailed=False,dest=None,baselines=False,trace=None):
    assert split=='val' and not detailed and dest is None and not baselines
    model.eval()
    if heads is not None:heads.eval()
    ds=data.datasets[split];cache=CandidateCache(rt.scores)
    sums=np.zeros(7,dtype=np.float64);last_sum=np.zeros(2,dtype=np.float64);count=0;gap_zero=0
    stats=dict(graph_captures=0,graph_replays=0,graph_short_batches_canonical=0)
    static_bank=torch.empty((len(data.starts),data.args.d_model),device=data.device)
    static_past=torch.empty(len(data.series),device=data.device)
    static_future=torch.empty(len(data.future_series),device=data.device)
    sub_index=torch.arange(max(1,0 if heads is None else heads.W.shape[0]),device=data.device)
    graph=None
    for c in range(data.c):
        first_x,first_y,first_starts=data.channel_batch(split,0,32,c)
        first_ids=data.pool(split,first_starts,c)
        cache(model,heads,first_x,data,c,first_ids,query_channel=0)
        fallback=cache.fallback
        if not fallback:static_bank.copy_(cache.bank)
        static_past.copy_(data.series[:,c]);static_future.copy_(data.future_series[:,c])
        def forward(x,y,ids,force_legacy=False):
            if force_legacy or len(x)!=32:sc=rt.scores(model,heads,x,data,c,ids,query_channel=0)
            else:
                zq=rt.encode_raw(model,x,0);zk=rt.F.normalize(static_bank[ids],dim=-1)
                if heads is None:sc=torch.einsum('bd,bpd->bp',rt.F.normalize(zq,dim=-1),zk).unsqueeze(1)
                else:sc=torch.einsum('bsd,bpd->bsp',heads(zq),zk)
            mask=torch.ones_like(ids,dtype=torch.bool)
            _,ph,pm=rt.mean_mixture_topk_selection(sc,mask,.1,10)
            values=static_future[ids[...,None]+data.h+data.offsets]-static_past[ids+data.h-1][...,None]
            qlast=x[:,-1,0];target=y[:,:,0]
            d=rt.pooled_future_mse(values,qlast,target);oracle=rt.stable_topk_indices(d,100,largest=False)
            sd=d.gather(1,oracle);gap=sd[:,10]-sd[:,9]
            le=qlast[:,None]-target;lm=le.square().mean(1);la=le.abs().mean(1)
            # Match canonical subset indexing/reduction, even for All-heads.
            pick=rt.stable_topk_indices(ph[:,sub_index].mean(1),10,largest=True)
            metric=rt.metric_rows(d,values,qlast,target,pick,oracle[:,:10],lm)
            packed=torch.cat([metric,torch.stack([lm,la,(gap==0).float()],1)],1)
            return sc,packed
        if graph is None and not fallback:
            sx=first_x.clone();sy=first_y.clone();si=first_ids.clone()
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):forward(sx,sy,si)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):score_static,packed_static=forward(sx,sy,si)
            stats['graph_captures']+=1
        for start in range(0,len(ds),32):
            end=min(start+32,len(ds));x,y,starts=data.channel_batch(split,start,end,c);ids=data.pool(split,starts,c)
            if len(x)==32 and not fallback:
                sx.copy_(x);sy.copy_(y);si.copy_(ids);graph.replay();score,packed=score_static,packed_static;stats['graph_replays']+=1
            else:score,packed=forward(x,y,ids,force_legacy=True);stats['graph_short_batches_canonical']+=1
            if trace is not None:trace(model,heads,x,data,c,ids,score=score,query_channel=0)
            arr=packed.cpu().numpy();assert np.isfinite(arr).all() and (arr[:,4]>=-1e-5).all()
            sums+=arr[:,:7].astype(np.float64).sum(0);last_sum+=arr[:,7:9].astype(np.float64).sum(0)
            gap_zero+=int(arr[:,9].sum());count+=len(x)
    assert count==len(ds)*data.c
    means=last_sum/count;nheads=0 if heads is None else heads.W.shape[0]
    row=dict(name='All',heads=list(range(1,max(1,nheads)+1)),k=10,**dict(zip(rt.FIELDS,(sums/count).tolist())))
    row['last_skill']=None if means[0]<=1e-12 else 1-row['agg_mse']/means[0]
    return dict(cell=data.cell,split=split,query_count=len(ds),channels=data.c,unit_count=count,rows=[row],
        last_mse=float(means[0]),last_mae=float(means[1]),oracle_k10_boundary_tie_fraction=gap_zero/count,
        validation_acceleration=dict(method='canonical B32 CUDA graph + exact-shape candidate cache',**cache.stats,**stats))
