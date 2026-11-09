"""All-channel GPU diagnostics, with explicit metric axes and no clipped regret."""
import torch
from adapter import topk

METRICS=['retmse','agg_mse','agg_mae','recall10','oracle_retmse','ranking_regret','exchange_regret',
 'ndcg10_inverse_mse','ndcg50_inverse_mse','ndcg100_inverse_mse','recall50_vs_oracle50',
 'recall100_vs_oracle100','oracle10coverage50','oracle10coverage100','last_win','last_tie',
 'oracle_boundary_gap','student_boundary_gap','student_top1_top10_gap',
 'missed_count','missed_mean_rank','missed_mean_score','false_mean_score','missed_mean_mse','false_mean_mse']

def take(a,ids):return a.gather(-1,ids)

@torch.no_grad()
def evaluate(data,x,y,d,score,mask,global_ids):
    B,C,N=d.shape;assert score.shape==mask.shape==global_ids.shape==d.shape
    assert bool((mask.sum(-1)>=100).all()),'This K100 auxiliary requires 100 valid candidates'
    assert bool(torch.isfinite(d[mask]).all()) and bool((d[mask]>=0).all())
    ordered=topk(score.masked_fill(~mask,-float('inf')),100)
    oracle=topk(-d.masked_fill(~mask,float('inf')),100)
    pick=ordered[...,:10];o10=oracle[...,:10]
    selected=take(global_ids,pick);oid=take(global_ids,o10)
    sd=take(d,pick).double();od=take(d,o10).double()
    ret=sd.mean(-1);oret=od.mean(-1)
    hit=(pick[...,None]==o10[:,:,None]).any(-1);found=(o10[...,None]==pick[:,:,None]).any(-1)
    assert torch.equal((~hit).sum(-1),(~found).sum(-1))
    exchange=(sd.masked_fill(hit,0).sum(-1)-od.masked_fill(found,0).sum(-1))/10
    assert torch.allclose(exchange,ret-oret,atol=1e-10,rtol=1e-10)
    assert bool((ret>=oret-2e-6).all()),'Negative regret is invalid; never clamp'
    vals=data.values_all(selected)+x[:,:,-1,None,None]
    pred=vals.reshape(B*C,10,data.h).mean(1).reshape(B,C,data.h)
    err=pred-y;agg=err.square().mean(-1);mae=err.abs().mean(-1)
    last=(x[:,:,-1,None]-y).square().mean(-1)
    v={'retmse':ret,'agg_mse':agg,'agg_mae':mae,'recall10':hit.double().mean(-1),
       'oracle_retmse':oret,'ranking_regret':ret-oret,'exchange_regret':exchange,
       'last_win':(agg<last).double(),'last_tie':(agg==last).double(),
       'oracle_boundary_gap':take(d,oracle[...,10:11])[...,0]-od[...,-1],
       'student_boundary_gap':take(score,ordered[...,9:10])[...,0]-take(score,ordered[...,10:11])[...,0],
       'student_top1_top10_gap':take(score,ordered[...,:1])[...,0]-take(score,ordered[...,9:10])[...,0]}
    for k in (10,50,100):
        sk=ordered[...,:k];ok=oracle[...,:k]
        disc=1/torch.log2(torch.arange(2,k+2,device=d.device,dtype=torch.float64))
        v[f'ndcg{k}_inverse_mse']=((1/(1+take(d,sk).double()))*disc).sum(-1)/((1/(1+take(d,ok).double()))*disc).sum(-1)
        if k>10:
            v[f'recall{k}_vs_oracle{k}']=(sk[...,None]==ok[:,:,None]).any(-1).double().mean(-1)
            v[f'oracle10coverage{k}']=(o10[...,None]==sk[:,:,None]).any(-1).double().mean(-1)
    # Exact candidate ranks without materializing all candidate-pair comparisons.
    ranks=[];oscore=take(score,o10);index=torch.arange(N,device=d.device)
    for k in range(10):
        z=oscore[...,k,None];oi=o10[...,k,None]
        ranks.append(1+(((score>z)|((score==z)&(index<oi)))&mask).sum(-1))
    ranks=torch.stack(ranks,-1);miss=~found;false=~hit;count=miss.sum(-1)
    def masked_mean(a,m):
        den=m.sum(-1);result=(a.double().masked_fill(~m,0).sum(-1)/den.clamp_min(1))
        return result.masked_fill(den==0,float('nan'))
    v.update(missed_count=count,missed_mean_rank=masked_mean(ranks,miss),missed_mean_score=masked_mean(oscore,miss),
             false_mean_score=masked_mean(take(score,pick),false),missed_mean_mse=masked_mean(od,miss),false_mean_mse=masked_mean(sd,false))
    return dict(metrics=torch.stack([v[k].double() for k in METRICS],-1),selected_ids=selected,
                oracle_ids=oid,oracle100_ids=take(global_ids,oracle),oracle100_mse=take(d,oracle),
                missed_ranks=ranks.masked_fill(~miss,0),false_ids=selected.masked_fill(~false,-1),
                missed_ids=oid.masked_fill(~miss,-1),prediction=pred,order100=ordered,
                selected_mse=sd,selected_scores=take(score,pick),last=last)

@torch.no_grad()
def normalization_control(logits,mask,pool_ids):
    z=(logits/.1).double().masked_fill(~mask[:,:,None],-float('inf'))
    pooled=z.gather(-1,pool_ids[:,:,None].expand(-1,-1,z.shape[2],-1))
    zf=torch.logsumexp(z,-1);zp=torch.logsumexp(pooled,-1)
    mass=torch.exp(zp-zf);p=torch.exp(pooled-zp[...,None]);restricted=torch.exp(pooled-zf[...,None]).mean(2)
    reconstructed=(mass[...,None]*p).mean(2)
    assert torch.allclose(restricted,reconstructed,atol=2e-15,rtol=2e-12)
    if z.shape[2]==1:
        # Hold logits exactly: subtracting one scalar cannot be allowed to change
        # a one-head ranking through floating-point probability underflow.
        same=topk(pooled[:,:,0],100)
        assert torch.equal(same,topk(pooled[:,:,0]-zf[:,:,0,None],100))
    return dict(local=p.mean(2),restricted=restricted,mass=mass,weights=mass/mass.sum(-1,keepdim=True),
                logZ_full=zf,logZ_pool=zp,pool_logits=pooled*.1)

@torch.no_grad()
def head_union(head_score,mixture,mask):
    B,C,S,N=head_score.shape;assert S==5
    hp=topk(head_score.masked_fill(~mask[:,:,None],-float('inf')),10)
    union=torch.zeros(B,C,N,dtype=torch.bool,device=mixture.device)
    union.scatter_(-1,hp.reshape(B,C,-1),True)
    deployable=topk(mixture.masked_fill(~union,-float('inf')),10)
    overlap=[]
    for a in range(5):
        for b in range(a+1,5):overlap.append((hp[:,:,a,:,None]==hp[:,:,b,None,:]).any(-1).float().mean(-1))
    return dict(head_ids=hp,union_mask=union,deployable=deployable,
                pair_overlap=torch.stack(overlap,-1),union_size=union.sum(-1))
