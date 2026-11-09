"""Frozen, sequential detail inference; no fitting and no original file writes."""
import argparse, gc, json, os, resource, sys, time
from pathlib import Path
import numpy as np
from supplement_common import *
sys.path.insert(0,str(ORIGINAL))
import torch
from diagnostic_adapter import Data,Frozen,inventory,rt,GPU,snapshot_hashes
from diagnostic_run import Calendar,prepare_block,save_npz
from diagnostic_metrics import MomentRidge,spearman
from diagnostic_core import stable_order,support_metrics

FIELDS=['missed_score_sum','false_selected_score_sum','missed_count','false_selected_count']

def rank_detail(score,ids,mask,oracle_ids):
    """Oracle is an evaluation label only; ranking is determined before it is used."""
    order=stable_order(score,mask);picked=ids.gather(-1,order[...,:10])
    assert bool((mask.gather(-1,order[...,:10])).all())
    assert bool((picked.sort(-1).values[...,1:]!=picked.sort(-1).values[...,:-1]).all())
    # Full IDs are sorted bank indices, P100 order is inherited, never resorted.
    matches=oracle_ids[...,None]==ids[...,None,:]
    assert bool((matches.sum(-1)==1).all())
    op=matches.long().argmax(-1)
    found=(oracle_ids[...,None]==picked[...,None,:]).any(-1)
    false=~(picked[...,None]==oracle_ids[...,None,:]).any(-1)
    missed=~found;assert torch.equal(missed.sum(-1),false.sum(-1))
    ranks=torch.empty_like(order);ranks.scatter_(-1,order,torch.arange(1,score.shape[-1]+1,device=score.device).expand_as(order))
    mr=ranks.gather(-1,op).masked_fill(found,0)
    ms=score.gather(-1,op).double().masked_fill(found,0).sum(-1)
    fs=score.gather(-1,order[...,:10]).double().masked_fill(~false,0).sum(-1)
    values=torch.stack([ms,fs,missed.sum(-1).double(),false.sum(-1).double()],-1)
    return picked,mr,values

def restore_probe(path):
    state=read(path)['state'];r=MomentRidge(len(state['mean']))
    for k in ['mean','scale','weights']:setattr(r,k,np.array(state[k],np.float64))
    for k in ['ymean','n','selected']:setattr(r,k,state[k])
    r.alphas=tuple(state['alphas']);return r

def run(support,cell):
    import subprocess
    assert GPU not in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True),'GPU1 busy; no competing inference'
    src=SOURCE/support/cell;dest=OUT/support/cell
    original_done=read(src/'COMPLETE.json');assert original_done['status'].startswith('COMPLETE')
    assert read(dest/'CPU_GATE.json')['status']=='PASS'
    if support=='FULL':assert read(OUT/'P100_DELIVERABLE_GATE.json')['status']=='PASS'
    if (dest/'INFERENCE_GATE.json').exists():
        g=read(dest/'INFERENCE_GATE.json');assert g['status']=='PASS' and g['source_complete_sha256']==sha(src/'COMPLETE.json')
        print('SUPPLEMENT VERIFIED RESUME',support,cell,flush=True);return
    provenance=dict(source_complete_sha256=sha(src/'COMPLETE.json'),original_source_hashes=snapshot_hashes(),supplement_source_hashes=source_hashes(),support=support,cell=cell)
    if (dest/'INFERENCE_PROVENANCE.json').exists():assert read(dest/'INFERENCE_PROVENANCE.json')==provenance,'source changed during supplement; preserve partial outputs'
    else:dump(dest/'INFERENCE_PROVENANCE.json',provenance)
    rt.gpu_guard();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter();data=Data(cell)
    rows=inventory(cell,support);models=[Frozen(data,r,support) for r in rows if r['status']=='VALID']
    calendar=Calendar(data);ridge=restore_probe(src/'probe_selection.json')
    summary=[];restricted_summary=[];split_gates=[];inference_seconds=0.;prepare_seconds=0.;rank_seconds=0.
    for split in ['val','test']:
        schema=read(src/'aggregate'/f'{split}_schema.json');names=schema['methods'];N=100 if support=='P100' else len(data.starts)
        hist=np.zeros((len(names),N+1),np.int64);sums=np.zeros((len(names),4));qmean_sum=np.zeros((len(names),2));qmean_count=np.zeros((len(names),2),np.int64)
        total_queries=0;restricted_sums={};restricted_count=0
        for a in range(0,len(data.datasets[split]),32):
            b=min(a+32,len(data.datasets[split]));target=dest/'raw_details'/split/f'batch{a//32:05d}.npz';oldpath=src/split/f'batch{a//32:05d}.npz'
            if target.exists():
                with np.load(target) as z:
                    assert z['source_batch_sha256'].item()==sha(oldpath)
                    mr=z['missed_ranks'].copy();sc=z['score_sums_counts'].copy();restricted=z['restricted_metrics'].copy();rnames=z['restricted_names'].tolist()
            else:
                with np.load(oldpath) as z:
                    expected=z['selected_ids'].copy();oracle=z['oracle_ids'].copy();query=z['query_start'].copy();assert z['method_names'].tolist()==names
                assert np.array_equal(query,data.datasets[split].starts[a:b])
                torch.cuda.synchronize();t=time.perf_counter();learned={m.name:m.probability(split,a,b) for m in models};torch.cuda.synchronize();inference_seconds+=time.perf_counter()-t
                rankparts=[];scoreparts=[];restricted_parts=[];rnames=[m.name for m in models] if support=='FULL' else []
                for aa in range(a,b,4 if support=='FULL' else 32):
                    bb=min(aa+(4 if support=='FULL' else 32),b)
                    torch.cuda.synchronize();t=time.perf_counter()
                    block=prepare_block(data,split,aa,bb,support,calendar,ridge,128)
                    torch.cuda.synchronize();prepare_seconds+=time.perf_counter()-t;t=time.perf_counter()
                    B,C,_=block['d'].shape;ids=block['ids'];mask=block['mask'];scores=dict(block['scores'])
                    scores.update({k:p[aa-a:bb-a] for k,p in learned.items()})
                    if split=='val':
                        for j,alpha in enumerate(ridge.alphas):scores['Ridge_alpha'+str(alpha)]=-block['probe'][...,j]
                    else:scores['Ridge_val_selected']=-block['probe'][...,ridge.selected]
                    scores['IndividualMSE_Oracle']=-block['d'];assert list(scores)==names
                    ori=torch.as_tensor(oracle[aa-a:bb-a],device=data.device)
                    computed=ids.gather(-1,stable_order(block['d'],mask,False)[...,:10]);assert torch.equal(computed,ori),'oracle ordered IDs changed'
                    ranks=[];scs=[]
                    for i,name in enumerate(names):
                        picked,mrr,ss=rank_detail(scores[name],ids,mask,ori)
                        assert np.array_equal(picked.cpu().numpy(),expected[aa-a:bb-a,:,i]),(cell,split,a,name,'selected ordered IDs changed')
                        ranks.append(mrr.cpu().numpy().astype(np.int32));scs.append(ss.cpu().numpy())
                    rankparts.append(np.stack(ranks,-2));scoreparts.append(np.stack(scs,-2))
                    rr=[]
                    if support=='FULL':
                        pool=data.pool_all(split,aa,bb);pd=block['d'].gather(-1,pool).reshape(B*C,100);valid=torch.ones_like(pd,dtype=torch.bool)
                        for name in rnames:
                            ps=scores[name].gather(-1,pool).reshape(B*C,100)
                            m=support_metrics(ps,pd,valid)
                            rr.append(torch.stack([m['retmse'],m['oracle_retmse'],m['ranking_regret'],m['recall'],m['ndcg_inverse_mse'],spearman(ps,pd,valid)],-1).reshape(B,C,6).cpu().numpy())
                    restricted_parts.append(np.stack(rr,-2) if rr else np.empty((B,C,0,6)))
                    torch.cuda.synchronize();rank_seconds+=time.perf_counter()-t
                    del block,scores,ranks,scs;gc.collect()
                mr=np.concatenate(rankparts);sc=np.concatenate(scoreparts);restricted=np.concatenate(restricted_parts)
                save_npz(target,dict(missed_ranks=mr,score_sums_counts=sc,score_fields=np.array(FIELDS),method_names=np.array(names),query_start=query,source_batch_sha256=np.array(sha(oldpath)),restricted_metrics=restricted,restricted_names=np.array(rnames),restricted_fields=np.array(['retmse','oracle_retmse','ranking_regret','recall','ndcg_inverse_mse','spearman'])))
                if a==0:print('REAL GPU GATE PASS',support,cell,split,'all method Top10 and oracle ordered IDs exact; channels',data.c,flush=True)
                del learned;gc.collect()
            assert mr.shape==(b-a,data.c,len(names),10) and sc.shape==(b-a,data.c,len(names),4)
            for mi,name in enumerate(names):
                vals=mr[:,:,mi].ravel();hist[mi]+=np.bincount(vals[vals>0],minlength=N+1)
                sums[mi]+=sc[:,:,mi].sum(axis=(0,1),dtype=np.float64)
                for gi in range(2):
                    count=sc[:,:,mi,gi+2];valid=count>0
                    qmean_sum[mi,gi]+=(sc[:,:,mi,gi][valid]/count[valid]).sum(dtype=np.float64);qmean_count[mi,gi]+=valid.sum()
            for ri,name in enumerate(rnames):restricted_sums[name]=restricted_sums.get(name,np.zeros(6))+np.nansum(restricted[:,:,ri],axis=(0,1))
            total_queries+=b-a;restricted_count+=(b-a)*data.c
            dump(dest/'CURRENT.json',dict(status='RUNNING',phase='DETAIL_INFERENCE',support=support,cell=cell,split=split,completed_queries=b,total_queries=len(data.datasets[split]),channels=data.c,pid=os.getpid()))
            if a%512==0:print('SUPPLEMENT',support,cell,split,b,'/',len(data.datasets[split]),'channels',data.c,flush=True)
        assert total_queries==original_done['query_counts'][split] and data.c==original_done['channels']
        for mi,name in enumerate(names):
            assert sums[mi,2]==sums[mi,3]==hist[mi].sum()
            summary.append(dict(cell=cell,support=support,split=split,method=name,queries=total_queries,channels=data.c,**pooled_rank_stats(hist[mi]),missed_mean_student_score=sums[mi,0]/sums[mi,2] if sums[mi,2] else None,false_selected_mean_student_score=sums[mi,1]/sums[mi,3] if sums[mi,3] else None,mean_query_missed_score=qmean_sum[mi,0]/qmean_count[mi,0] if qmean_count[mi,0] else None,mean_query_false_selected_score=qmean_sum[mi,1]/qmean_count[mi,1] if qmean_count[mi,1] else None,score_definition='actual inference ranking score; P100 learned=mean-head probability; Full V0=cosine; Full V5=global mean-head probability; baseline=its defined ranking score; Oracle=-futureMSE evaluation only',score_comparison_across_different_methods='not calibrated; do not compare magnitudes across score types',quantile_definition='pooled candidate ranks, linear empirical quantiles, not mean of query quantiles'))
        np.savez_compressed(dest/f'{split}_MISSED_RANK_HISTOGRAM.npz',histogram=hist,methods=np.array(names))
        for name,total in restricted_sums.items():restricted_summary.append(dict(cell=cell,support=support,split=split,method=name,definition='same Full-trained model global score restricted to frozen P100 IDs; no softmax refit, no parameter change',queries=total_queries,channels=data.c,**dict(zip(['retmse','oracle_retmse','ranking_regret','recall','ndcg_inverse_mse','spearman'],(total/restricted_count).tolist()))))
        split_gates.append(dict(split=split,status='PASS',queries=total_queries,channels=data.c,units=total_queries*data.c,all_methods_ordered_top10_and_oracle_ids_exact=True))
    csvwrite(dest/'MISSED_FALSE_SCORE_AND_POOLED_RANK.csv',summary)
    if restricted_summary:csvwrite(dest/'FULL_MODEL_FIXED_P100_RANKING.csv',restricted_summary)
    dump(dest/'INFERENCE_GATE.json',dict(status='PASS',source_complete_sha256=sha(src/'COMPLETE.json'),splits=split_gates,checkpoint_integrity=[m.immutable_check() for m in models],missing_models=[r for r in rows if r['status']!='VALID'],all_methods=True,stage1_training=False,stage2=False,gpu_uuid=GPU,total_wall_seconds=time.perf_counter()-begin,retrieval_seconds=inference_seconds,feature_preparation_seconds=prepare_seconds,ranking_seconds=rank_seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),cpu_ram_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    print('SUPPLEMENT INFERENCE COMPLETE',support,cell,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('support',choices=['P100','FULL']);p.add_argument('cell');a=p.parse_args()
    try:run(a.support,a.cell)
    except BaseException as e:
        import traceback
        dump(OUT/'IMMEDIATE_ISSUE.json',dict(status='FAILED',support=a.support,cell=a.cell,error=repr(e),traceback=traceback.format_exc()));raise
