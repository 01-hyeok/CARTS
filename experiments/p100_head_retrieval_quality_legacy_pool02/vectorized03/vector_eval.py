"""Same per-query metrics, with channel-grouped inference and bounded output buffers."""
import itertools,os,time
from pathlib import Path
import numpy as np
import torch
from vector_ops import pool_all,flat_inputs,vector_scores


@torch.no_grad()
def evaluate(rt,data,model,heads,split,detailed=False,dest=None,baselines=False,chunk=32,max_queries=None):
    model.eval()
    if heads is not None: heads.eval()
    nheads=0 if heads is None else heads.W.shape[0]
    cfgs=rt.configurations(nheads,detailed);ds=data.datasets[split]
    nq=len(ds) if max_queries is None else min(max_queries,len(ds))
    sums=np.zeros((len(cfgs),len(rt.FIELDS)),dtype=np.float64);last_sum=np.zeros(2,dtype=np.float64)
    count=0;pair_sums={};union_sum=0.;gap_zero=0
    baseline_sums=np.zeros((22,len(rt.FIELDS)),dtype=np.float64) if baselines else None
    for c0 in range(0,data.c,chunk):
        c1=min(data.c,c0+chunk);nc=c1-c0
        storage=[dict(metrics=[],candidate_ids=[],last=[],input_features=[],oracle_k10_gap=[],baselines=[]) for _ in range(nc)]
        gens=[[torch.Generator(device='cpu').manual_seed(990000+1000*c+j) for j in range(20)] for c in range(c0,c1)]
        for start in range(0,nq,32):
            end=min(start+32,nq);nb=end-start
            x,y,starts=data.batch(split,range(start,end));ids_all=pool_all(data,split,starts)
            q,target,hist,values=flat_inputs(data,x,y,ids_all,c0,c1)
            ids=ids_all[c0:c1].reshape(-1,100)
            sc=vector_scores(model,heads,q,hist)
            mask=torch.ones_like(ids,dtype=torch.bool)
            _,ph,pm=rt.mean_mixture_topk_selection(sc,mask,.1,10)
            qlast=q[:,-1];d=rt.pooled_future_mse(values,qlast,target)
            oracle_order=rt.stable_topk_indices(d,100,largest=False)
            sd=d.gather(1,oracle_order);gap=sd[:,10]-sd[:,9]
            gap_zero+=int((gap==0).sum());le=qlast[:,None]-target
            # Preserve canonical reduction launch shapes for deterministic
            # Last-repeat and input strata, including the final short batch.
            lm=torch.cat([part.square().mean(1) for part in le.split(nb)])
            la=torch.cat([part.abs().mean(1) for part in le.split(nb)])
            last=torch.stack([lm,la],1);last_sum+=last.double().sum(0).cpu().numpy()
            this=[];picked=[]
            for name,subset,k in cfgs:
                prob=ph[:,list(subset)].mean(1);pick=rt.stable_topk_indices(prob,k,largest=True)
                met=rt.metric_rows(d,values,qlast,target,pick,oracle_order[:,:k],lm)
                assert bool((met[:,4]>=-1e-5).all())
                this.append(met)
                padded=torch.full((len(q),20),-1,dtype=torch.int32,device=data.device)
                padded[:,:k]=ids.gather(1,pick).int();picked.append(padded)
            mat=torch.stack(this,1).cpu().numpy();sums+=mat.astype(np.float64).sum(0);count+=len(q)
            if nheads>1 and detailed:
                hs=rt.stable_topk_indices(ph,10,largest=True)
                for a,b in itertools.combinations(range(nheads),2):
                    overlap=(hs[:,a,:,None]==hs[:,b,None,:]).any(-1).float().mean(-1)
                    pair_sums[(a,b)]=pair_sums.get((a,b),0.)+float(overlap.double().sum())
                ordered=hs.reshape(len(q),-1).sort(-1).values
                union_sum+=float((1+(ordered[:,1:]!=ordered[:,:-1]).sum(-1)).double().sum())
            if baselines:
                bm=[]
                for j in range(22):
                    if j==0:pick=oracle_order[:,:10]
                    elif j==1:pick=torch.arange(10,device=data.device).expand(len(q),10)
                    else:
                        pick=torch.stack([torch.randperm(100,generator=gens[c][j-2])[:10] for c in range(nc) for _ in range(nb)]).to(data.device)
                    bm.append(rt.metric_rows(d,values,qlast,target,pick,oracle_order[:,:10],lm))
                ba=torch.stack(bm,1).cpu().numpy();baseline_sums+=ba.astype(np.float64).sum(0)
            if dest is not None:
                arrays=dict(metrics=mat.reshape(nc,nb,len(cfgs),len(rt.FIELDS)),
                    candidate_ids=torch.stack(picked,1).cpu().numpy().reshape(nc,nb,len(cfgs),20),
                    last=last.cpu().numpy().reshape(nc,nb,2),
                    input_features=torch.cat([torch.stack([part.std(1,unbiased=False),(part[:,1:]-part[:,:-1]).abs().mean(1)],1) for part in q.split(nb)]).cpu().numpy().reshape(nc,nb,2),
                    oracle_k10_gap=gap.cpu().numpy().reshape(nc,nb))
                if baselines:arrays['baselines']=ba.reshape(nc,nb,22,len(rt.FIELDS))
                for ci in range(nc):
                    for key,value in arrays.items():storage[ci][key].append(value[ci].copy())
            del sc,values,hist,ph,pm,q,target,x,y
        if dest is not None:
            dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
            for ci,c in enumerate(range(c0,c1)):
                payload={k:np.concatenate(v) for k,v in storage[ci].items() if v}
                payload.update(query_start=ds.starts[:nq],channel=np.int32(c),config_names=np.array([v[0] for v in cfgs]),metric_names=np.array(rt.FIELDS))
                tmp=dest/f'channel{c:03d}.tmp.npz';np.savez_compressed(tmp,**payload);os.replace(tmp,dest/f'channel{c:03d}.npz')
        if detailed:print(f'EVAL VECTORIZED {data.cell} {split} channels={c1}/{data.c}',flush=True)
    assert count==nq*data.c
    last_mean=last_sum/count;rows=[]
    for i,(name,sub,k) in enumerate(cfgs):
        row=dict(name=name,heads=[x+1 for x in sub],k=k,**dict(zip(rt.FIELDS,(sums[i]/count).tolist())))
        row['last_skill']=None if last_mean[0]<=1e-12 else 1-row['agg_mse']/last_mean[0];rows.append(row)
    result=dict(cell=data.cell,split=split,query_count=nq,channels=data.c,unit_count=count,rows=rows,
        last_mse=float(last_mean[0]),last_mae=float(last_mean[1]),oracle_k10_boundary_tie_fraction=gap_zero/count)
    if nheads>1 and detailed:
        result['diversity']=dict(pair_overlap={f'H{a+1}_H{b+1}':v/count for (a,b),v in pair_sums.items()},
            mean_overlap=float(np.mean(list(pair_sums.values())))/count,union_size=union_sum/count,normalized_union=union_sum/count/(10*nheads))
    if baselines:
        bm=baseline_sums/count
        result['baselines']={'P100 Individual-MSE Oracle@10':dict(zip(rt.FIELDS,bm[0].tolist())),
            'Raw cosine Top-10':dict(zip(rt.FIELDS,bm[1].tolist())),'Random Top-10':dict(zip(rt.FIELDS,bm[2:].mean(0).tolist())),
            'Random MC draw standard deviation':dict(zip(rt.FIELDS,bm[2:].std(0,ddof=1).tolist()))}
        for key,row in result['baselines'].items():
            row['last_skill']=None if last_mean[0]<=1e-12 else (row['agg_mse']/last_mean[0] if key=='Random MC draw standard deviation' else 1-row['agg_mse']/last_mean[0])
        result['random_draw_metrics']=bm[2:].tolist()
    return result
