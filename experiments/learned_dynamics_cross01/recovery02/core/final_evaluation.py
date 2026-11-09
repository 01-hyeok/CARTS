"""All-query final native-support metrics and per-query evidence, after selection."""
import numpy as np
import torch
import time,json,csv
from experiment_common import *

def ranks(value,mask):
    """Average tie rank, zero based, valid candidates only; invalid at end."""
    v=value.double().masked_fill(~mask,torch.inf)
    order=torch.argsort(v,dim=-1,stable=True);s=v.gather(-1,order);N=v.shape[-1]
    pos=torch.arange(N,device=v.device).expand_as(order)
    start=torch.cat([torch.ones_like(s[...,:1],dtype=torch.bool),s[...,1:]!=s[...,:-1]],-1)
    end=torch.cat([s[...,:-1]!=s[...,1:],torch.ones_like(s[...,:1],dtype=torch.bool)],-1)
    first=torch.where(start,pos,torch.zeros_like(pos)).cummax(-1).values
    last=torch.where(end,pos,torch.full_like(pos,N)).flip(-1).cummin(-1).values.flip(-1)
    result=torch.empty_like(v);result.scatter_(-1,order,(first+last).double()/2)
    return result

def corr_ranks(a,b,mask):
    count=mask.sum(-1);mean=(count-1).double()/2
    aa=(a-mean[...,None]).masked_fill(~mask,0);bb=(b-mean[...,None]).masked_fill(~mask,0)
    den=(aa.square().sum(-1)*bb.square().sum(-1)).sqrt()
    return (aa*bb).sum(-1)/den.clamp_min(1e-30),den>0

@torch.no_grad()
def finish(ex,state):
    import run_experiment as runner
    ev=runner.ev;d=ex.d;dest=ex.dest;summary=[];micro=32 if ex.support=='P100' else 2
    names=['S0','D0','C0']+ARMS+['Oracle10','Random','LastRepeat']
    final_diag={}
    for name,arm in ex.arms.items():
        info=dict(beta=float(arm.fusion.beta),epoch=state[name]['best_epoch'],new_score_mean=float(arm.fusion.calibration.mean),new_score_scale=float(arm.fusion.calibration.scale))
        if arm.kind=='L2':info['positive_weights']=torch.nn.functional.softplus(arm.scorer.raw_weight).cpu().tolist()
        final_diag[name]=info
    dump(dest/'LEARNED_PARAMETERS.json',final_diag)
    for split in ('val','test'):
        sums={n:{} for n in names};units={n:0 for n in names};N=len(d.datasets[split]);beg=time.perf_counter()
        for a in range(0,N,32):
            b,ids,mask,p,dist,teacher=ex.batch(split,a);x,y,_=d.query(split,a,b)
            output={'query_start':d.datasets[split].starts[a:b]};lists={}
            official_oracle,_,random_ids,_,_=ex.prior.baseline(ex.support,split,a,b)
            for j in range(0,b-a,micro):
                sl=slice(j,min(j+micro,b-a));rows=np.arange(a+sl.start,a+sl.stop)
                ii,mm,pp,dd=ids[sl],mask[sl],p[sl],dist[sl]
                oracle_pos=ev.topk((-dd).masked_fill(~mm,-torch.inf),10);oracle=ii.gather(-1,oracle_pos)
                assert torch.equal(oracle,official_oracle[sl]),'Oracle ordered IDs differ from independently verified source'
                base_pos=ev.topk(pp.masked_fill(~mm,-torch.inf),10);base=ii.gather(-1,base_pos)
                dr=ranks(dd,mm);br=ranks(-pp,mm)
                scores={'S0':pp.masked_fill(~mm,-torch.inf)}
                qphi,cphi=ex.features.pair(split,rows,ii,'L2');scores['D0']=-(qphi-cphi).square().sum(-1)
                del qphi,cphi
                # Cross/D0 feature creation is bounded by consumer query chunks.
                scores['C0']=ex.features.pair(split,rows,ii,'C0')
                for name,arm in ex.arms.items():
                    raw=arm.raw_chunks(ex.features,split,rows,ii)
                    scores[name]=arm.score(pp,raw,mm,ranking=True)
                for name in names:
                    if name=='LastRepeat':
                        err=x[sl,:,-1,None]-y[sl]
                        vals={'retmse':err.square().mean(-1),'agg_mse':err.square().mean(-1),'agg_mae':err.abs().mean(-1),
                              'delta_retmse_vs_s0':err.square().mean(-1).double()-dd.gather(-1,base_pos).double().mean(-1)}
                        chosen=None
                    else:
                        if name=='Oracle10':pos=oracle_pos;chosen=oracle
                        elif name=='Random':
                            chosen=random_ids[sl]
                            if ex.support=='FULL':pos=chosen
                            else:
                                match=ii[...,None,:]==chosen[...,None];assert match.sum(-1).eq(1).all();pos=match.long().argmax(-1)
                        else:
                            score=scores[name].masked_fill(~mm,-torch.inf);pos=ev.topk(score,10);chosen=ii.gather(-1,pos)
                        assert mm.gather(-1,pos).all()
                        sm=dd.gather(-1,pos).double();om=dd.gather(-1,oracle_pos).double()
                        future=d.values_all(chosen)+x[sl,:,-1,None,None];err=future.double().mean(-2)-y[sl].double()
                        ret=sm.mean(-1);gap=ret-om.mean(-1);assert (gap>=-2e-6).all()
                        found=(chosen[...,None]==oracle[...,None,:]).any(-1)
                        base_oracle=(oracle[...,None]==base[...,None,:]).any(-1)
                        chosen_oracle=(oracle[...,None]==chosen[...,None,:]).any(-1)
                        vals=dict(retmse=ret,agg_mse=err.square().mean(-1),agg_mae=err.abs().mean(-1),
                            oracle_recall=found.double().mean(-1),oracle_gap=gap,best_of10=sm.min(-1).values,
                            overlap_s0=(chosen[...,None]==base[...,None,:]).any(-1).double().mean(-1),
                            recovered_missed_oracle=(chosen_oracle&~base_oracle).sum(-1).double(),
                            lost_baseline_oracle=(~chosen_oracle&base_oracle).sum(-1).double(),
                            delta_retmse_vs_s0=ret-dd.gather(-1,base_pos).double().mean(-1))
                        if name in scores:
                            sr=ranks(-scores[name],mm);rho,defined=corr_ranks(sr,dr,mm);rhob,definedb=corr_ranks(sr,br,mm)
                            vals['score_future_spearman']=rho;vals['score_future_spearman_defined']=defined.double()
                            vals['score_v5_spearman']=rhob;vals['score_v5_spearman_defined']=definedb.double()
                            lists.setdefault(name+'__oracle_rank',[]).append(sr.gather(-1,oracle_pos).cpu().numpy())
                            valid=scores[name].masked_fill(~mm,0).double();count=mm.sum(-1)
                            vals['score_mean']=valid.sum(-1)/count
                            vals['score_std']=((valid-vals['score_mean'][...,None]).square().masked_fill(~mm,0).sum(-1)/count).sqrt()
                        lists.setdefault(name+'__selected_ids',[]).append(chosen.cpu().numpy())
                        lists.setdefault(name+'__selected_mse',[]).append(sm.cpu().numpy())
                    for metric,value in vals.items():
                        lists.setdefault(name+'__'+metric,[]).append(value.cpu().numpy())
                        sums[name][metric]=sums[name].get(metric,0.)+float(value.double().sum())
                    units[name]+=vals['retmse'].numel()
            for key,values in lists.items():output[key]=np.concatenate(values)
            path=dest/'final_raw'/split/('batch%05d.npz'%(a//32));runner.npz_save(path,output)
            dump(path.with_suffix('.json'),dict(sha256=sha(path),queries=b-a,channels=d.c,at=now()))
            if a%1024==0:
                dump(dest/'CURRENT.json',dict(phase='FINAL_EVALUATION',split=split,query_end=b,total=N,at=now()))
                print('FINAL',ex.cell,ex.support,split,b,N,flush=True)
        for name in names:
            row=dict(dataset=ex.cell.rsplit('_',1)[0],horizon=d.h,support=ex.support,seed=0,split=split,method=name,queries=N,channels=d.c,
                checkpoint=ex.f.row['checkpoint'],scorer_epoch=state[name]['best_epoch'] if name in state else None,
                **{k:v/units[name] for k,v in sums[name].items()})
            # Spearman is undefined for constant rows, excluded from denominator.
            for k in ('score_future_spearman','score_v5_spearman'):
                if k in row:
                    defined=sums[name][k+'_defined'];row[k]=sums[name][k]/defined if defined else None
            summary.append(row)
        official=(float(ex.f.row['canonical_retmse']) if ex.support=='P100' else float(ex.f.reference('test')['metrics']['retmse10']))
        if split=='test':
            baseline=next(r for r in summary if r['split']=='test' and r['method']=='S0')
            actual=baseline['retmse']
            assert np.isclose(actual,official,rtol=2e-6,atol=2e-6),(actual,official,'Final official S0 mismatch')
            official_agg=float(ex.f.row['canonical_agg_mse']) if ex.support=='P100' else float(ex.f.reference('test')['metrics']['agg_mse10'])
            assert np.isclose(baseline['agg_mse'],official_agg,rtol=2e-6,atol=2e-6),'Final S0 AggMSE mismatch'
        dump(dest/('FINAL_'+split.upper()+'_GATE.json'),dict(status='PASS_ALL_QUERY',queries=N,channels=d.c,seconds=time.perf_counter()-beg,at=now()))
    dump(dest/'FULL_RESULTS.json',summary)
    fields=sorted({k for row in summary for k in row})
    with (dest/'FULL_RESULTS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(summary)
