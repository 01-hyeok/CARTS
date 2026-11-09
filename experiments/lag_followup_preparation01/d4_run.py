"""Past-only lag diagnostics and validation-selected non-trained reranking.

Model scores search the entire unchanged support. Only the reranking pool is
Top100/200 (future-blind); Oracle-conditioned features are separate diagnostics.
"""
import argparse,csv,datetime,fcntl,time,traceback
from pathlib import Path
import numpy as np
from server_bridge import *
from past_lag import torch_alignment,shift_features

BASE=ROOT/'results/TRACK-HEAD-SHIFT-DIAGNOSTIC01'
ALPHAS=[0.,.01,.03,.1,.3,1.];BETAS=[0.,.01,.03,.1]

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def state(out,phase,**kw):dump(out/'CURRENT.json',dict(phase=phase,at=now(),pid=os.getpid(),**kw))
def save(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez(f,**payload)
    os.replace(tmp,path)
def csvout(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

@torch.no_grad()
def build_split(data,f,split,out,proof):
    rows=sample_indices(len(data.datasets[split]),32);result=[]
    for i in rows:
        i=int(i);path=out/split/('query%05d.npz'%i)
        if path.exists():
            m=json.loads(path.with_suffix('.json').read_text());assert m['identity']==proof and m['sha256']==sha(path)
            with np.load(path) as z:result.append({k:z[k].copy() for k in z.files})
            continue
        a=i//32*32;b=min(a+32,len(data.datasets[split]));j=i-a
        logits,mask,ids,score=native(f,split,a,b);x,y,starts,d=distances(data,split,a,b)
        if f.support=='P100':d=d.gather(-1,ids)
        logits=logits[j:j+1];mask=mask[j:j+1];ids=ids[j:j+1];score=score[j:j+1];d=d[j:j+1];x=x[j:j+1]
        r=min(200,d.shape[-1]);ordered=topk(score.masked_fill(~mask,-float('inf')),r)
        oracle=topk(-d.masked_fill(~mask,float('inf')),10)
        heads=topk(logits.masked_fill(~mask[:,:,None],-float('inf')),10)
        rng=np.random.RandomState(int(starts[j])%4294967295)
        random=topk(torch.as_tensor(rng.random_sample(mask.shape),device=data.device).masked_fill(~mask,-float('inf')),10)
        local=torch.cat([ordered,oracle,heads.reshape(1,data.c,50),random],-1)
        global_ids=ids.gather(-1,local);mse=d.gather(-1,local)
        assert mask.gather(-1,local).all()
        picked=global_ids[...,:10];ref=f.reference(split)
        if f.support=='P100':assert np.array_equal(picked.cpu().numpy(),ref['ids'][i:i+1])
        else:
            pred=(data.values_all(picked)+x[:,:,-1,None,None]).mean(2)
            assert np.allclose(pred.cpu().numpy(),ref['prediction'][i:i+1],atol=1e-5,rtol=1e-5)
        hist=data.histories_all(global_ids);lagscore,lags=torch_alignment(x,hist,candidate_chunk=64)
        feat=shift_features(lagscore.cpu().numpy(),lags)
        recent=(hist[...,-16:]-x[:,:,None,-16:]).square().mean(-1)
        cp=global_ids.cpu().numpy();o=ids.gather(-1,oracle).cpu().numpy();s=picked.cpu().numpy()
        isoracle=(cp[...,None]==o[:,:,None]).any(-1);isselected=(cp[...,None]==s[:,:,None]).any(-1)
        unique=np.ones(cp.shape,bool)
        for k in range(cp.shape[-1]):unique[...,k]=~(cp[...,:k]==cp[...,k,None]).any(-1)
        randommask=np.zeros(cp.shape,bool);randommask[...,-10:]=True
        threshold=d.gather(-1,oracle)[...,-1].cpu().numpy()
        groups=dict(correct_oracle=isoracle&isselected&unique,missed_oracle=isoracle&~isselected&unique,
                    false_selected=isselected&~isoracle&unique,random_control=randommask,
                    near_oracle_5pct=(mse.cpu().numpy()<=threshold[...,None]+np.maximum(1e-8,.05*threshold[...,None]))&unique)
        arr=dict(query_index=np.array([i]),query_start=np.array([starts[j]]),global_ids=cp,mse=mse.cpu().numpy(),
                 base_score=score.gather(-1,local).cpu().numpy(),recent_difference=recent.cpu().numpy(),
                 query_std=x.std(-1,unbiased=False).cpu().numpy(),candidate_std=hist.std(-1,unbiased=False).cpu().numpy(),
                 oracle_mse=d.gather(-1,oracle).cpu().numpy(),rerank_count=np.array([r]),full_bank_count=np.array([len(data.starts)]),
                 evaluation_candidate_count=np.array([d.shape[-1]]),**feat)
        arr.update({'group_'+k:v for k,v in groups.items()})
        save(path,arr);dump(path.with_suffix('.json'),dict(identity=proof,sha256=sha(path),at=now(),query_index=i,channels=data.c,base_retrieval_entire_support=True,
                    rerank_pool_future_blind=True,oracle_group_non_deployable=True,canonical_native_top10_pass=True))
        result.append(arr);state(out,'D4_FEATURES',split=split,completed=len(result),total=len(rows),query_index=i,all_channels=data.c)
        print('D4 FEATURES',data.cell,f.support,split,len(result),'/',len(rows),'query',i,flush=True)
    assert len(result)==len(rows)
    return result

def concat(rows,key):return np.concatenate([r[key] for r in rows],axis=0)
def rerank(rows,k,alpha,beta,score_scale,recent_scale):
    score=concat(rows,'base_score')[...,:k];gain=concat(rows,'gain')[...,:k]
    recent=concat(rows,'recent_difference')[...,:k];mse=concat(rows,'mse')[...,:k]
    adjusted=score+score_scale[None,:,None]*(alpha*gain-beta*recent/recent_scale[None,:,None])
    # Original TopR is already in canonical score/global-ID order; stable sort
    # keeps exact base ordering when alpha=beta=0 and preserves fixed ties.
    ix=np.argsort(-adjusted,axis=-1,kind='stable')[...,:10]
    ret=np.take_along_axis(mse,ix,-1).astype(np.float64).mean(-1)
    ids=np.take_along_axis(concat(rows,'global_ids')[...,:k],ix,-1)
    oracle=concat(rows,'global_ids')[:,:,int(rows[0]['rerank_count'][0]):int(rows[0]['rerank_count'][0])+10]
    recall=(ids[...,None]==oracle[:,:,None]).any(-1).mean(-1)
    if alpha==beta==0:assert np.array_equal(ids,concat(rows,'global_ids')[...,:10])
    return ret,recall,ids

def aggregate(out,val,test,cell,support):
    rows=[];probe=[];selection=[]
    for split,arr in [('val',val),('test',test)]:
        for group in ['correct_oracle','missed_oracle','false_selected','random_control','near_oracle_5pct']:
            use=concat(arr,'group_'+group)
            for feature in ['zero','best','gain','best_lag','margin','entropy','recent_difference','candidate_std','mse']:
                v=concat(arr,feature)[use]
                rows.append(dict(cell=cell,support=support,split=split,group=group,feature=feature,count=len(v),
                    mean=float(v.mean()) if len(v) else '',median=float(np.median(v)) if len(v) else '',
                    p90=float(np.quantile(v,.9)) if len(v) else '',p95=float(np.quantile(v,.95)) if len(v) else ''))
    r=int(val[0]['rerank_count'][0]);ks=[100,200] if r==200 else [100]
    for k in ks:
        score_scale=np.maximum(1e-8,concat(val,'base_score')[...,:k].std((0,2)))
        recent_scale=np.maximum(1e-8,np.median(concat(val,'recent_difference')[...,:k],axis=(0,2)))
        allval=[]
        for a in ALPHAS:
            for b in BETAS:
                ret,rec,_=rerank(val,k,a,b,score_scale,recent_scale)
                allval.append(dict(alpha=a,beta=b,val_retmse=float(ret.mean()),val_recall=float(rec.mean())))
        selected=min(allval,key=lambda z:(z['val_retmse'],z['alpha'],z['beta']))
        selection.append(dict(pool=k,score_scale=score_scale.tolist(),recent_scale=recent_scale.tolist(),grid=allval,selected=selected,test_used=False))
        for name,a,b in [('B0',0.,0.),('ValidationSelected',selected['alpha'],selected['beta'])]:
            ret,rec,ids=rerank(test,k,a,b,score_scale,recent_scale)
            save(out/('%s_R%d_test.npz'%(name,k)),dict(retmse=ret,recall=rec,selected_ids=ids,query_start=concat(test,'query_start')))
            probe.append(dict(cell=cell,support=support,pool=k,method=name,alpha=a,beta=b,retmse=float(ret.mean()),recall=float(rec.mean()),queries=len(test),channels=ret.shape[1],sampling='32 fixed timestamps/all channels',candidate_pool_source='frozen model TopR on full unchanged support'))
    csvout(out/'TEMPORAL_SHIFT_GROUP_SUMMARY.csv',rows);csvout(out/'FUTURE_BLIND_RERANK_RESULTS.csv',probe);dump(out/'VALIDATION_SELECTION.json',selection)

def main(cell,support):
    out=BASE/'D4'/('V5_'+cell+'_'+support);out.mkdir(parents=True,exist_ok=True)
    lock=(BASE/'00_Audit/d4_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=start_gpu();register_owner('TRACK-HEAD-SHIFT-DIAGNOSTIC01');begin=time.perf_counter();data,f,identity=frozen(cell,support)
    proof=dict(identity=identity,runner_sha256=sha(HERE/'d4_run.py'),protocol_sha256=sha(HERE/'protocol.json'),
               sampling='32 fixed evenly spaced timestamps per split/all channels',model='V5',training_support=support,evaluation_support=support,
               lag_definition='query[t] vs candidate[t+delta]; past-only common patches',future_blind_rerank=True,
               candidate_limit='Top100/200 rerank only; base retrieval retains entire support',alpha=ALPHAS,beta=BETAS,
               score_scaling='per-channel std of validation TopR mixture scores',recent_scaling='per-channel median validation TopR last16 unshifted MSE')
    if (out/'PROVENANCE.json').exists():assert json.loads((out/'PROVENANCE.json').read_text())==proof
    else:dump(out/'PROVENANCE.json',proof)
    torch.cuda.reset_peak_memory_stats()
    val=build_split(data,f,'val',out,proof)
    # Grid and scales are fixed using validation; test features do not select them.
    test=build_split(data,f,'test',out,proof);aggregate(out,val,test,cell,support)
    torch.cuda.synchronize();assert sha(f.row['checkpoint'])==identity['checkpoint']['checkpoint_sha256']
    dump(out/'COMPONENT_GATE.json',dict(status='D4_FEATURES_AND_RERANK_COMPLETE_STATISTICS_PENDING',at=now(),queries={'val':len(val),'test':len(test)},channels=data.c,
       full_bank_count=len(data.starts),all_channels_single_batch=True,original_checkpoint_unchanged=True,
       seconds=time.perf_counter()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_before=before,gpu_after=snapshot(),
       pending=['paired time-block CI','past-feature/relevance correlation','V0/V1 and cross-support supplemental comparisons','final all-condition report']))
    state(out,'D4_FEATURES_AND_RERANK_COMPLETE_STATISTICS_PENDING')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cell');p.add_argument('support');a=p.parse_args()
    try:main(a.cell,a.support)
    except BaseException as e:
        state(BASE/'D4'/('V5_'+a.cell+'_'+a.support),'BLOCKED',error=repr(e),traceback=traceback.format_exc(),preserved=True);raise
