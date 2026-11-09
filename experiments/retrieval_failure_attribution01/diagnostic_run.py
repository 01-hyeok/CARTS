#!/usr/bin/env python3
"""Frozen retrieval-only diagnostics; one cell per resumable process.

No optimizer is constructed for any CARTS model. The only fitted object is a
small train-only ridge relevance probe. Full support is blocked until every
P100 cell is complete. Main GPU exclusivity is enforced by the controller.
"""
import argparse,csv,datetime,gc,json,math,os,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from diagnostic_adapter import ROOT,HERE,OUT,OLD,GPU,rt,Data,Frozen,inventory,snapshot_hashes
from diagnostic_metrics import FEATURES,PROBE_FEATURES,METRICS,features,temporal,query_metrics,MomentRidge
from diagnostic_core import stable_order,fixed_random_scores,pair_mse_from_dispersion,regret_decomposition

BASE_NAMES=['Random','Raw Cosine','Delta-last Cosine','Pearson','PAA-16 Cosine']
NEIGHBOR_NAMES=['PastNearest1','PastNearest10','PastNearest20','PastNearest50','Random10','FutureNearest10_DIAGNOSTIC']
NEIGHBOR_FIELDS=['past_distance','mean_query_future_mse','future_variance','mean_pairwise_future_mse','past_future_correlation']

def dump(p,x):rt.write_json(p,x)
def sync():torch.cuda.synchronize()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load_json(p):return json.loads(Path(p).read_text())

class Calendar:
    def __init__(self,data):
        self.data=data;self.values=None
        self.names=['hour_similarity','weekday_similarity','weekend_similarity','month_similarity']
        if data.cell.startswith('Solar_'):return
        import pandas as pd
        path=rt.config()['data'][data.cell.rsplit('_',1)[0]]['path']
        dates=pd.to_datetime(pd.read_csv(path,usecols=['date'])['date'],errors='raise')
        assert dates.notna().all() and dates.is_monotonic_increasing
        a=np.stack([dates.dt.hour.to_numpy()+dates.dt.minute.to_numpy()/60,dates.dt.dayofweek.to_numpy(),(dates.dt.dayofweek>=5).to_numpy(),dates.dt.month.to_numpy()-1],-1)
        self.values=torch.as_tensor(a,device=data.device,dtype=torch.float32)
    def pair(self,qstarts,ids):
        if self.values is None:return None
        q=self.values[torch.as_tensor(qstarts,device=self.data.device)+self.data.h-1]
        c=self.values[ids+self.data.h-1]
        delta=(q[:,None,None,:]-c).abs()
        out=[]
        for j,period in [(0,24),(1,7),(2,None),(3,12)]:
            out.append((delta[...,j]==0).float() if period is None else 1-torch.minimum(delta[...,j],period-delta[...,j])/(period/2))
        return torch.stack(out,-1)


def feature_block(data,x,starts,ids,calendar):
    b,c,p=ids.shape;history=data.histories_all(ids)
    f=features(x.reshape(b*c,data.h),history.reshape(b*c,p,data.h)).reshape(b,c,p,-1)
    ca=calendar.pair(starts,ids)
    return f if ca is None else torch.cat([f,ca],-1)


def probe_columns(calendar):return PROBE_FEATURES+([] if calendar.values is None else list(range(len(FEATURES),len(FEATURES)+4)))


@torch.no_grad()
def reproduce(data,models,support,out):
    """Phase0: canonical values/IDs/metrics before probe fitting or attribution."""
    from scripts.train_patch_retrieval_expert01 import ndcg_at_k
    from scripts.train_factorial_e2e01 import individual_utility_memsafe
    begin=time.perf_counter();gates=[]
    if not models:
        dump(out/'CANONICAL_REPRODUCTION.json',dict(status='PASS_DATA_ONLY_MODELS_MISSING',seconds=0,rows=[],missing_models_not_imputed=True,
            note='No learned-model reproduction is claimed; independent baseline/Oracle evaluation remains permitted'))
        return
    for split in ('val','test'):
        sums={m.name:np.zeros(4) for m in models};count=0;hashes={m.name:__import__('hashlib').sha256() for m in models}
        max_pred={m.name:0. for m in models}
        for a in range(0,len(data.datasets[split]),32):
            b=min(a+32,len(data.datasets[split]));probs={m.name:m.probability(split,a,b) for m in models}
            for c in range(data.c):
                x,y,starts=data.channel_batch(split,a,b,c);qlast=x[:,-1,0];truth=y[:,:,0]
                mask=data.mask(starts) if support=='FULL' else torch.ones(b-a,100,device=data.device,dtype=torch.bool)
                ids=torch.arange(len(data.starts),device=data.device).expand(b-a,-1) if support=='FULL' else data.pool(split,starts,c)
                if support=='FULL':
                    bankids=torch.arange(len(data.starts),device=data.device);values=data.values(bankids,c)
                    d=-individual_utility_memsafe(values,qlast,truth,4096)
                else:
                    values=data.values(ids,c);d=rt.pooled_future_mse(values,qlast,truth)
                oracle=rt.stable_topk_indices(d.masked_fill(~mask,float('inf')),10,largest=False)
                for m in models:
                    pick=rt.stable_topk_indices(probs[m.name][:,c],10,largest=True)
                    actual=ids.gather(1,pick);hashes[m.name].update(actual.cpu().numpy().tobytes())
                    vals=values[pick] if support=='FULL' else values.gather(1,pick[:,:,None].expand(-1,-1,data.h))
                    pred=vals.mean(1)+qlast[:,None]
                    # Legacy averages aligned values, preserving the addition order.
                    pred=(vals+qlast[:,None,None]).mean(1)
                    ret=d.gather(1,pick).mean(1);agg=(pred-truth).square().mean(1)
                    recall=(pick[:,:,None]==oracle[:,None,:]).any(-1).float().mean(1)
                    ndcg=ndcg_at_k(pick,d,mask,10)
                    sums[m.name]+=torch.stack([ret,agg,recall,ndcg],1).double().sum(0).cpu().numpy()
                    if support=='P100':
                        oldids,oldmetric=m.reference(split)
                        assert np.array_equal(actual.cpu().numpy(),oldids[a:b,c])
                        for tensor,j in [(ret,0),(agg,1),(recall,3)]:
                            assert np.allclose(tensor.cpu().numpy(),oldmetric[a:b,c,j],atol=2e-6,rtol=2e-6),(m.name,split,c,a,'P100 canonical metric gate')
                    else:
                        expected=m.reference(split)[0][a:b,:,c];new=pred.cpu().numpy()
                        max_pred[m.name]=max(max_pred[m.name],float(np.abs(new-expected).max()))
                        assert np.allclose(new,expected,atol=1e-5,rtol=1e-5),(m.name,split,c,a,'Full canonical prediction gate')
                count+=b-a
            if a==0:dump(out/f'REAL_BATCH_GATE_{split}.json',dict(status='PASS',queries=b-a,channels=data.c,support=support,all_valid_candidates=True,models=[m.name for m in models]))
            if a%512==0:print('REPRODUCTION',support,data.cell,split,b,'/',len(data.datasets[split]),flush=True)
        assert count==len(data.datasets[split])*data.c
        for m in models:
            got=sums[m.name]/count
            if support=='FULL' and split=='test':
                expected=m.reference(split)[1]
                for j,k in enumerate(['retmse10','agg_mse10','recall10','ndcg10']):assert np.isclose(got[j],expected[k],atol=1e-5,rtol=1e-4),(m.name,k,got[j],expected[k])
            gates.append(dict(status='PASS',model=m.name,split=split,units=count,metric_names=['retmse','agg_mse','recall','legacy_ndcg10'],metrics=got.tolist(),ordered_candidate_id_sha256=hashes[m.name].hexdigest(),max_stored_prediction_difference=max_pred[m.name]))
    dump(out/'CANONICAL_REPRODUCTION.json',dict(status='PASS',seconds=time.perf_counter()-begin,rows=gates,missing_models_not_imputed=True))


@torch.no_grad()
def fit_probe(data,support,calendar,out):
    """Future-free stratified training sample; primary val/test are never subsampled."""
    columns=probe_columns(calendar);ridge=MomentRidge(len(columns));begin=time.perf_counter()
    qrows=np.unique(np.linspace(0,len(data.datasets['train'])-1,min(256,len(data.datasets['train']))).round().astype(np.int64))
    checksum=__import__('hashlib').sha256()
    for a in range(0,len(qrows),4):
        rows=qrows[a:a+4];ds=data.datasets['train'];pos=rows[:,None]+np.arange(data.h)[None]
        x=torch.as_tensor(np.asarray(ds.data_x[pos],np.float32),device=data.device).permute(0,2,1).contiguous()
        y=torch.as_tensor(np.asarray(ds.data_y[pos+data.h],np.float32),device=data.device).permute(0,2,1).contiguous()
        starts=ds.starts[rows]
        if support=='P100':ids=torch.as_tensor(np.array(data.cpu_pool['train'][rows],np.int64,copy=True),device=data.device)
        else:
            mask=data.mask(starts).cpu().numpy();chosen=[]
            for i,q in enumerate(starts):
                eligible=np.flatnonzero(mask[i]);assert len(eligible)>=100
                chosen.append(np.stack([np.random.default_rng(np.random.SeedSequence([0,int(q),c,8821])).choice(eligible,100,replace=False) for c in range(data.c)]))
            ids=torch.as_tensor(np.stack(chosen),device=data.device)
        checksum.update(ids.cpu().numpy().tobytes());f=feature_block(data,x,starts,ids,calendar)
        values=data.values_all(ids);d=(values+x[...,-1,None,None]-y[:,:,None]).square().mean(-1)
        ridge.update(f[...,columns].reshape(-1,len(columns)).cpu().numpy(),d.reshape(-1).cpu().numpy(),'train')
    ridge.finish();sync()
    dump(out/'probe_train.json',dict(status='FITTED_TRAIN_ONLY',seconds=time.perf_counter()-begin,train_query_rows=qrows.tolist(),
        train_query_count=len(qrows),channels=data.c,pairs=ridge.n,train_candidate_policy='stored identical P100' if support=='P100' else 'uniform 100 eligible Full IDs per training query/channel, seed0, future-free',
        validation_test_policy='ALL queries, ALL channels, ALL eligible evaluation-support candidates',
        budget_reason='small diagnostic probe fit; primary retrieval evaluation is not subsampled',candidate_id_sha256=checksum.hexdigest(),
        feature_names=[(FEATURES+calendar.names)[i] for i in columns],state=ridge.state(),test_used_for_fit=False))
    return ridge


@torch.no_grad()
def prepare_block(data,split,a,b,support,calendar,ridge,chunk=128):
    """All channels together; candidate future/history are bounded by chunk."""
    x,y,starts=data.query(split,a,b);n=b-a;c=data.c;N=100 if support=='P100' else len(data.starts)
    ids_all=data.pool_all(split,a,b) if support=='P100' else torch.arange(N,device=data.device).expand(n,c,N)
    mask=torch.ones(n,c,N,device=data.device,dtype=torch.bool) if support=='P100' else data.mask(starts)[:,None].expand(n,c,N)
    d=torch.empty(n,c,N,device=data.device);base=torch.empty(n,c,N,4,device=data.device);probe=torch.empty(n,c,N,3,device=data.device)
    cols=probe_columns(calendar);oracle_seconds=feature_seconds=0.
    for k in range(0,N,chunk):
        ids=ids_all[...,k:k+chunk]
        sync();t=time.perf_counter();values=data.values_all(ids)
        # Preserve canonical per-channel future-distance kernel shape. The large
        # histories/features remain grouped; only this inexpensive reduction is
        # canonicalized, because Weather ties are sensitive to FP32 reduction.
        for ci in range(c):
            d[:,ci,k:k+chunk]=rt.pooled_future_mse(values[:,ci].contiguous(),x[:,ci,-1].contiguous(),y[:,ci].contiguous())
        sync();oracle_seconds+=time.perf_counter()-t;t=time.perf_counter()
        f=feature_block(data,x,starts,ids,calendar)
        base[...,k:k+chunk,:]=f[...,:4]
        probe[...,k:k+chunk,:]=ridge.predict(f[...,cols])
        sync();feature_seconds+=time.perf_counter()-t
    rnd=np.stack([fixed_random_scores(starts,j,N) for j in range(c)],1)
    scores={'Random':torch.as_tensor(rnd,device=data.device,dtype=torch.float32)}
    for j,name in enumerate(BASE_NAMES[1:]):scores[name]=base[...,j]
    return dict(x=x,y=y,starts=starts,ids=ids_all,mask=mask,d=d,scores=scores,probe=probe,
                oracle_seconds=oracle_seconds,feature_seconds=feature_seconds)


@torch.no_grad()
def selected_values(data,ids,x):return data.values_all(ids)+x[...,-1,None,None]


@torch.no_grad()
def neighborhood(data,block,oracle_order):
    x,y=block['x'],block['y'];b,c,_=x.shape;d=block['d'];mask=block['mask'];ids=block['ids']
    past=block['scores']['Delta-last Cosine'];order=stable_order(past.reshape(b*c,-1),mask.reshape(b*c,-1)).reshape(b,c,-1)
    random=stable_order(block['scores']['Random'].reshape(b*c,-1),mask.reshape(b*c,-1)).reshape(b,c,-1)
    result=[]
    from diagnostic_metrics import pearson_masked
    for name in NEIGHBOR_NAMES:
        if name.startswith('PastNearest'):sel=order[...,:int(name.replace('PastNearest',''))]
        elif name=='Random10':sel=random[...,:10]
        else:sel=oracle_order[...,:10]
        vals=selected_values(data,ids.gather(-1,sel),x);pd=1-past.gather(-1,sel);fd=d.gather(-1,sel)
        var=(vals-vals.mean(-2,keepdim=True)).square().mean(dim=(-2,-1))
        pair=pair_mse_from_dispersion(vals)
        corr=pearson_masked(pd.reshape(b*c,-1),fd.reshape(b*c,-1),torch.ones(b*c,sel.shape[-1],device=data.device,dtype=torch.bool)).reshape(b,c)
        result.append(torch.stack([pd.mean(-1),fd.mean(-1),var,pair,corr],-1).cpu().numpy())
    return np.stack(result,-2)


@torch.no_grad()
def evaluate_micro(data,split,a,b,support,calendar,ridge,learned,probe_index,chunk):
    block=prepare_block(data,split,a,b,support,calendar,ridge,chunk);x,y,d,mask,ids=[block[k] for k in ['x','y','d','mask','ids']]
    B,C,N=d.shape;names=BASE_NAMES+list(learned)
    scores=dict(block['scores']);scores.update(learned)
    if probe_index is None:
        for j,alpha in enumerate(ridge.alphas):scores['Ridge_alpha'+str(alpha)]=-block['probe'][...,j]
    else:scores['Ridge_val_selected']=-block['probe'][...,probe_index]
    scores['IndividualMSE_Oracle']=-d;names=list(scores)
    oracle_order=stable_order(d.reshape(B*C,N),mask.reshape(B*C,N),False).reshape(B,C,N)
    neighbor=neighborhood(data,block,oracle_order)
    lm=(x[...,-1,None]-y).square().mean(-1);la=(x[...,-1,None]-y).abs().mean(-1)
    mats=[];selected=[];group_arrays=[];prediction={};group_names=[];ranking_seconds=0.
    feature_names=FEATURES+([] if calendar.values is None else calendar.names)
    random_pick=stable_order(scores['Random'].reshape(B*C,N),mask.reshape(B*C,N)).reshape(B,C,N)[...,:10]
    for name in names:
        sync();t_rank=time.perf_counter()
        m=query_metrics(scores[name].reshape(B*C,N),d.reshape(B*C,N),mask.reshape(B*C,N),full=support=='FULL')
        sync();ranking_seconds+=time.perf_counter()-t_rank
        pick=m['selected'].reshape(B,C,10);chosen=ids.gather(-1,pick);val=selected_values(data,chosen,x)
        pred=torch.stack([val[:,ci].contiguous().mean(1) for ci in range(C)],1);prediction[name]=pred
        m['agg_mse']=torch.stack([(pred[:,ci].contiguous()-y[:,ci].contiguous()).square().mean(-1) for ci in range(C)],1).reshape(-1)
        m['agg_mae']=torch.stack([(pred[:,ci].contiguous()-y[:,ci].contiguous()).abs().mean(-1) for ci in range(C)],1).reshape(-1)
        m['last_mse']=lm.reshape(-1);m['last_mae']=la.reshape(-1)
        m['past10_distance']=torch.as_tensor(neighbor[...,1,0],device=data.device).reshape(-1)
        m['past10_future_dispersion']=torch.as_tensor(neighbor[...,1,3],device=data.device).reshape(-1)
        mats.append(torch.stack([m[k].double() for k in METRICS],-1).reshape(B,C,-1).cpu().numpy());selected.append(chosen.cpu().numpy())
        if name in learned or name.startswith('Ridge'):
            oracle=m['oracle'].reshape(B,C,10);hit=(pick[...,None]==oracle[...,None,:]).any(-1)
            found=(oracle[...,None]==pick[...,None,:]).any(-1)
            gp=[(oracle,found),(oracle,~found),(pick,~hit),(random_pick,torch.ones_like(hit))]
            ga=[]
            for gi,gm in gp:
                ff=feature_block(data,x,block['starts'],ids.gather(-1,gi),calendar)
                finite=torch.isfinite(ff)&gm[...,None]
                sums=torch.where(finite,ff,torch.zeros_like(ff)).sum(-2);counts=finite.sum(-2)
                mean=torch.where(counts>0,sums/counts.clamp_min(1),torch.full_like(sums,float('nan')))
                gd=d.gather(-1,gi);gcount=gm.sum(-1)
                gf=torch.where(gcount>0,(gd*gm).sum(-1)/gcount.clamp_min(1),torch.full_like(gd[...,0],float('nan')))
                ga.append(torch.cat([mean,gf[...,None],gcount[...,None]],-1).cpu().numpy())
            group_arrays.append(np.stack(ga,-2));group_names.append(name)
    # MSE prediction loss for fixed alphas, used only for validation choice.
    losses=((block['probe']-d[...,None]).double().square()*mask[...,None]).sum(dim=(0,1,2)).cpu().numpy()
    # Full decomposition uses exactly the previously frozen P100 membership.
    p100_oracle=None;coverage=None;p100_ids=data.pool_all(split,a,b)
    if support=='FULL':
        pd=d.gather(-1,p100_ids);pi=stable_order(pd.reshape(B*C,100),descending=False).reshape(B,C,100)[...,:10]
        p100_oracle=pd.gather(-1,pi).double().mean(-1).cpu().numpy()
        oi=ids.gather(-1,oracle_order[...,:10]);coverage=(oi[...,None]==p100_ids[...,None,:]).any(-1).float().mean(-1).cpu().numpy()
    return dict(names=names,metrics=np.stack(mats,-2),selected_ids=np.stack(selected,-2),group_names=group_names,
        group_features=np.stack(group_arrays,-3),feature_names=feature_names+['evaluation_future_mse','group_size'],neighborhood=neighbor,
        oracle_ids=ids.gather(-1,oracle_order[...,:10]).cpu().numpy(),probe_validation_loss_sums=losses,
        probe_pairs=int(mask.sum()),p100_oracle=p100_oracle,coverage=coverage,query_start=block['starts'],
        oracle_seconds=block['oracle_seconds'],feature_seconds=block['feature_seconds'],ranking_seconds=ranking_seconds,predictions=prediction)


def save_npz(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.stem+'.tmp.npz')
    np.savez_compressed(tmp,**payload);os.replace(tmp,path)


@torch.no_grad()
def evaluate_split(data,models,support,split,calendar,ridge,out,probe_index=None):
    dest=out/split;dest.mkdir(exist_ok=True);begin=time.perf_counter();sum_loss=np.zeros(3);pairs=0;retrieval_time=0.;oracle_time=0.;feature_time=0.;ranking_time=0.;gpu_samples=[]
    model_sums={m.name:np.zeros(4) for m in models};max_errors={m.name:0. for m in models};units=0
    for a in range(0,len(data.datasets[split]),32):
        b=min(a+32,len(data.datasets[split]));target=dest/f'batch{a//32:05d}.npz'
        # Resume only after a complete split gate; partial split is recomputed.
        sync();t=time.perf_counter();prob={m.name:m.probability(split,a,b) for m in models};sync();retrieval_time+=time.perf_counter()-t
        pieces=[]
        for aa in range(a,b,4 if support=='FULL' else 32):
            bb=min(aa+(4 if support=='FULL' else 32),b);chunk=128
            while True:
                error=None
                try:part=evaluate_micro(data,split,aa,bb,support,calendar,ridge,{n:p[aa-a:bb-a] for n,p in prob.items()},probe_index,chunk)
                except RuntimeError as e:
                    if 'out of memory' not in str(e).lower():raise
                    error=str(e)
                if error is None:break
                dump(out/f'OOM_{split}_{aa}_{chunk}.json',dict(at=now(),error=error,query_start=aa,chunk=chunk,candidate_support_unchanged=True))
                assert chunk>16,'OOM at minimum candidate chunk; stop and report'
                chunk//=2;gc.collect();torch.cuda.empty_cache()
            for model in models:
                mi=part['names'].index(model.name);metric=part['metrics'][:,:,mi]
                selected=part['selected_ids'][:,:,mi]
                model_sums[model.name]+=metric[..., [METRICS.index(k) for k in ['retmse','agg_mse','recall','ndcg_inverse_mse']]].astype(np.float64).sum(axis=(0,1))
                if support=='P100':
                    oldids,oldmetric=model.reference(split)
                    assert np.array_equal(selected,oldids[aa:bb]),(model.name,'ordered ID gate')
                    for newcol,oldcol in [('retmse_fp32_compat',0),('agg_mse',1),('agg_mae',2),('recall',3)]:
                        # Check the old FP32 reduction against saved metrics.
                        # Primary RetMSE/regret independently accumulate the same
                        # selected canonical distances in FP64 to avoid cancellation.
                        diff=np.max(np.abs(metric[...,METRICS.index(newcol)]-oldmetric[aa:bb,:,oldcol]))
                        max_errors[model.name]=max(max_errors[model.name],float(diff))
                        assert diff<=2e-6,(model.name,split,newcol,float(diff))
                elif model.full_trained:
                    oldpred,_=model.reference(split);new=part['predictions'][model.name].permute(0,2,1).cpu().numpy()
                    expected=oldpred[aa:bb];diff=float(np.max(np.abs(new-expected)));max_errors[model.name]=max(max_errors[model.name],diff)
                    assert np.allclose(new,expected,atol=1e-5,rtol=1e-5),(model.name,split,'stored Full retrieval prediction mismatch',diff)
            part.pop('predictions');oracle_time+=part.pop('oracle_seconds');feature_time+=part.pop('feature_seconds');ranking_time+=part.pop('ranking_seconds')
            sum_loss+=part['probe_validation_loss_sums'];pairs+=part['probe_pairs'];pieces.append(part)
        keys=['metrics','selected_ids','group_features','neighborhood','oracle_ids','query_start']
        payload={k:np.concatenate([p[k] for p in pieces],axis=0) for k in keys}
        payload.update(method_names=np.array(pieces[0]['names']),metric_names=np.array(METRICS),group_method_names=np.array(pieces[0]['group_names']),feature_names=np.array(pieces[0]['feature_names']),neighborhood_names=np.array(NEIGHBOR_NAMES),neighborhood_fields=np.array(NEIGHBOR_FIELDS))
        if support=='FULL':payload.update(p100_oracle=np.concatenate([p['p100_oracle'] for p in pieces]),full_oracle_coverage=np.concatenate([p['coverage'] for p in pieces]))
        save_npz(target,payload);units+=(b-a)*data.c
        dump(out/'CURRENT.json',dict(status='RUNNING',support=support,cell=data.cell,split=split,completed_queries=b,total_queries=len(data.datasets[split]),channels=data.c,at=now(),pid=os.getpid()))
        if a%256==0:
            sample=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
            gpu_samples.append(dict(at=now(),query_end=b,observed=sample))
            print('PROGRESS',support,data.cell,split,b,'/',len(data.datasets[split]),'channels',data.c,'GPU',sample,flush=True)
        del prob,pieces,payload;gc.collect()
    assert units==len(data.datasets[split])*data.c
    for m in models:
        gate=dict(split=split,queries=len(data.datasets[split]),channels=data.c,units=units,max_canonical_difference=max_errors[m.name],status='PASS',checkpoint_sha256=m.row['checkpoint_sha256'])
        if support=='FULL' and m.full_trained and split=='test':
            expected=m.reference(split)[1];got=model_sums[m.name]/units
            for i,k in enumerate(['retmse10','agg_mse10','recall10']):
                assert np.isclose(got[i],expected[k],atol=1e-5,rtol=1e-4),(m.name,k,got[i],expected[k])
            gate['test_retmse_recall_agg_reproduced']=True
        m.reproduction.append(gate)
    result=dict(status='PASS',split=split,query_count=len(data.datasets[split]),channels=data.c,units=units,
        seconds=time.perf_counter()-begin,retrieval_seconds=retrieval_time,oracle_seconds=oracle_time,feature_seconds=feature_time,ranking_seconds=ranking_time,observed_gpu_samples=gpu_samples,
        probe_validation_loss=(sum_loss/pairs).tolist(),probe_pairs=pairs,model_gates=[m.reproduction[-1] for m in models])
    dump(dest/'SPLIT_GATE.json',result);return result


def run_cell(support,cell):
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    assert GPU not in apps,'GPU1 acquired by another process before diagnostic start; stop without contention'
    rt.gpu_guard();protocol=load_json(HERE/'PROTOCOL.json');assert cell in protocol['cell_order']
    if support=='FULL':
        assert all((OUT/'P100'/c/'COMPLETE.json').exists() for c in protocol['cell_order']),'Full blocked until all P100 cells complete'
    out=OUT/support/cell;out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():
        old=load_json(out/'COMPLETE.json');current_hashes=snapshot_hashes()
        if old['source_hashes']!=current_hashes:
            migration=load_json(OUT/'00_Audit/NUMERICAL_REPAIR01/COMPLETED_REUSE_GATE.json')
            record=migration['completed'][support+'/'+cell]
            assert migration['status']=='PASS' and migration['current_source_hashes']==current_hashes
            assert record['source_hashes']==old['source_hashes'] and record['complete_sha256']==rt.sha256_file(out/'COMPLETE.json')
            for model in old['models']:assert rt.sha256_file(model['checkpoint'])==model['sha256']
        print('VERIFIED RESUME',support,cell);return
    begin=time.perf_counter();torch.cuda.reset_peak_memory_stats();data=Data(cell)
    calendar=Calendar(data);rows=inventory(cell,support);missing=[r for r in rows if r['status']!='VALID']
    models=[Frozen(data,r,support) for r in rows if r['status']=='VALID']
    # Optional transfers are excluded from this first execution rather than
    # silently treating a runtime-changing transfer as a Full-trained model.
    dump(out/'MODEL_STATUS.json',dict(models=[m.row for m in models],missing=missing,support_transfer='NOT_RUN; optional separate numerical reproduction gate required'))
    dump(out/'PROVENANCE.json',dict(cell=cell,support=support,seed=0,all_channels=True,all_queries=True,data=data.data_info,
        pool_hashes=data.pool_hashes,source_hashes=snapshot_hashes(),torch_version=torch.__version__,gpu=rt.gpu_guard(),
        checkpoint_policy='fixed prior validation choice; never test reselect',full_dependencies='DEFERRED_FULL_DEPENDENCY' if support=='P100' else 'COMPUTED_THIS_PHASE'))
    reproduce(data,models,support,out)
    ridge=fit_probe(data,support,calendar,out)
    val=evaluate_split(data,models,support,'val',calendar,ridge,out,None)
    ridge.choose(val['probe_validation_loss'],'val');dump(out/'probe_selection.json',dict(status='VALIDATION_SELECTED',state=ridge.state(),test_used_for_selection=False))
    test=evaluate_split(data,models,support,'test',calendar,ridge,out,ridge.selected)
    # Reporting is CPU-only and never changes selection or checkpoints.
    from diagnostic_report import summarize_cell
    summarize_cell(support,cell)
    complete=dict(status='COMPLETE_WITH_MISSING_MODELS' if missing else 'COMPLETE',support=support,cell=cell,completed_at=now(),
        query_counts={s:len(data.datasets[s]) for s in ('val','test')},channels=data.c,models=[m.immutable_check() for m in models],missing_models=missing,
        total_wall_seconds=time.perf_counter()-begin,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        oracle_computation_seconds=val['oracle_seconds']+test['oracle_seconds'],retrieval_inference_seconds=val['retrieval_seconds']+test['retrieval_seconds'],ranking_analysis_seconds=val['ranking_seconds']+test['ranking_seconds'],feature_extraction_seconds=val['feature_seconds']+test['feature_seconds'],
        probe_training_seconds=load_json(out/'probe_train.json')['seconds'],canonical_reproduction_seconds=load_json(out/'CANONICAL_REPRODUCTION.json')['seconds'],
        full_candidate_count=len(data.starts),p100_candidate_count=100,
        cpu_ram_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,source_hashes=snapshot_hashes(),stage1_retraining=False,stage2=False)
    dump(out/'COMPLETE.json',complete);print('CELL COMPLETE',support,cell,json.dumps(complete),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('support',choices=['P100','FULL']);ap.add_argument('cell');args=ap.parse_args()
    try:run_cell(args.support,args.cell)
    except BaseException as e:
        out=OUT/args.support/args.cell;dump(out/'FAILED.json',dict(status='FAILED',at=now(),error=repr(e),traceback=traceback.format_exc(),support=args.support,cell=args.cell))
        dump(OUT/'00_Audit/IMMEDIATE_ISSUE.json',dict(status='BLOCKED',at=now(),support=args.support,cell=args.cell,error=repr(e),log=str(out/'FAILED.json')))
        raise
