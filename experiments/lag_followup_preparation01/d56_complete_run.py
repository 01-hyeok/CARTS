"""Only missing D5/D6 candidates/features. Full support; all-channel GPU calls.

Immutable shared distance and past-feature shards avoid recomputing them for
different models. No [query,candidate,horizon,channel] population tensor exists.
"""
import argparse,datetime,fcntl,os,time,traceback
from pathlib import Path
import numpy as np
from server_bridge import *
from utils.full_candidate_bank import encode_raw_channel_first
from past_lag import torch_alignment,shift_features
from d56_metrics import LAG_MARGIN_EPS

BASE=ROOT/'results/TRACK-HEAD-SHIFT-DIAGNOSTIC01';OUT=BASE/'extensions_d5_d6'
FIELDS=['zero','best','gain','best_lag','margin','entropy','recent_difference','candidate_std','near_tie_lag_fraction']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def state(path,phase,**kw):dump(path/'CURRENT.json',dict(phase=phase,at=now(),pid=os.getpid(),**kw))
def save(path,values):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as s:np.savez_compressed(s,**values)
    os.replace(tmp,path)
def read_npz(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}

def exact_head_probabilities(g,mask,query_index):
    if g.shape[-1]<=100:return torch.softmax(g.masked_fill(~mask[:,:,None],-float('inf'))/.1,-1)
    _,C,H,N=g.shape;phase=int(query_index)%4
    z=g.new_zeros(C,4,H,N);z[:,phase]=g[0].masked_fill(~mask[0,:,None],-float('inf'))
    return torch.softmax(z/.1,-1)[:,phase][None]

@torch.no_grad()
def canonical_cross_ids(f,split,a,b,evaluation):
    """REFERENCE GATE ONLY. Never supplies or replaces production selections."""
    data=f.data;f.precision();parts=[]
    if not hasattr(f,'extension_reference_keys'):f.extension_reference_keys={}
    for c in range(data.c):
        x,_,starts=data.channel_batch(split,a,b,c,future=False)
        if c not in f.extension_reference_keys:
            hist=data.bank_history(c)
            if f.full_trained:key=encode_raw_channel_first(f.model,hist,0)
            else:
                keys=[]
                for s in range(0,len(hist),3200):
                    xx=hist[s:s+3200];n=len(xx)
                    if n<3200:xx=torch.cat([xx,xx.new_zeros(3200-n,data.h,1)])
                    keys.append(rt.encode_raw(f.model,xx,0)[:n])
                key=torch.cat(keys)
            f.extension_reference_keys[c]=torch.nn.functional.normalize(key,dim=-1)
        z=encode_raw_channel_first(f.model,x,0) if f.full_trained else rt.encode_raw(f.model,x,0)
        q=torch.nn.functional.normalize(z,dim=-1)[:,None] if f.heads is None else f.heads(z)
        key=f.extension_reference_keys[c]
        logits=torch.matmul(q[:,0],key.T)[:,None] if f.heads is None else torch.einsum('bsd,nd->bsn',q,key)
        ids=data.pool_all(split,a,b)[:,c] if evaluation=='P100' else torch.arange(len(data.starts),device=data.device).expand(b-a,-1)
        mask=torch.ones_like(ids,dtype=torch.bool) if evaluation=='P100' else data.mask(starts)
        if evaluation=='P100':logits=logits.gather(-1,ids[:,None].expand(-1,logits.shape[1],-1))
        score=logits[:,0].masked_fill(~mask,-float('inf')) if f.n<=1 else torch.softmax(logits.masked_fill(~mask[:,None],-float('inf'))/.1,-1).mean(1)
        parts.append(ids.gather(-1,topk(score,10)))
    return torch.stack(parts,1)

def reproduction(data,row):
    candidates=[ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02/checkpoint_runs'/data.cell/(row['architecture']+'_'+row['training_support'])/'REPRODUCTION_GATE.json',
        BASE/'00_Audit/extended_native_reproduction'/data.cell/(row['architecture']+'_'+row['training_support'])/'REPRODUCTION_GATE.json',
        OUT/'native_reproduction'/data.cell/(row['architecture']+'_'+row['training_support'])/'REPRODUCTION_GATE.json']
    for p in candidates:
        if not p.exists():continue
        g=json.loads(p.read_text());assert g['status']=='PASS'
        assert g['identity']['checkpoint_sha256']==row['checkpoint_sha256'] and g['identity']['pool_hashes']==data.pool_hashes
        assert {x['split'] for x in g['rows']}=={'val','test'}
        assert all(x['queries']==len(data.datasets[x['split']]) and x['channels']==data.c and x['ordered_mismatch_count']==0 and x['units']==x['queries']*x['channels'] for x in g['rows'])
        return p
    return None

def common_identity(data):
    return dict(cell=data.cell,data=data.data_info,pool_hashes=data.pool_hashes,lag_source_sha256=sha(HERE/'past_lag.py'),
        definition='existing canonical B32 distances; train scaler; delta_last values; mask unchanged',
        near_tie_lag_margin=LAG_MARGIN_EPS,mask_source_sha256=sha(ROOT/'experiments/retrieval_failure_attribution01/diagnostic_adapter.py'))

def context(data,split,i):
    """Immutable QxCxN scalar distances only, canonical original B32 reduction."""
    path=OUT/'shared'/data.cell/split/('query%05d_context.npz'%i);side=path.with_suffix('.json');identity=common_identity(data)
    if path.exists():
        m=json.loads(side.read_text());assert m['identity']==identity and sha(path)==m['sha256'];return read_npz(path),path
    a=i//32*32;b=min(a+32,len(data.datasets[split]));j=i-a
    x,y,starts,d=distances(data,split,a,b)
    mask=data.mask(starts)[j:j+1,None].expand(1,data.c,-1)
    pool=data.pool_all(split,i,i+1)
    assert bool(mask.gather(-1,pool).all()) and bool((pool.sort(-1).values[...,1:]!=pool.sort(-1).values[...,:-1]).all())
    values=dict(distance=d[j].cpu().numpy(),mask=mask[0].cpu().numpy(),pool_ids=pool[0].cpu().numpy(),
       x=x[j].cpu().numpy(),y=y[j].cpu().numpy(),query_index=np.array([i]),query_start=np.array([starts[j]]))
    save(path,values);dump(side,dict(identity=identity,sha256=sha(path),at=now(),canonical_batch_start=a,canonical_batch_end=b))
    return values,path

def lag_features(data,split,i,ctx,ids,valid):
    """Append-only per-query feature shards shared across checkpoint/support.

    D4 cached values are retained exactly. Missing lag-alternative count requires
    alignment recomputation; it is audited against all overlapping D4 values.
    """
    folder=OUT/'shared'/data.cell/split/('query%05d_lag'%i);folder.mkdir(parents=True,exist_ok=True)
    identity=common_identity(data);known={};shard_hashes={}
    for path in sorted(folder.glob('part*.npz')):
        meta=json.loads(path.with_suffix('.json').read_text());assert meta['identity']==identity and sha(path)==meta['sha256']
        z=read_npz(path);shard_hashes[str(path.relative_to(OUT))]=meta['sha256']
        for c,j in zip(*np.nonzero(z['valid'])):
            key=(int(c),int(z['ids'][c,j]));v=np.array([z[k][c,j] for k in FIELDS])
            if key in known:assert np.array_equal(known[key],v)
            known[key]=v
    missing=[]
    for c in range(data.c):missing.append([int(v) for v in ids[c,valid[c]] if (c,int(v)) not in known])
    amount=max(map(len,missing));reused_d4=0;max_d4_delta=0.
    if amount:
        mi=np.zeros((data.c,amount),np.int64);mv=np.zeros(mi.shape,bool)
        for c,v in enumerate(missing):mi[c,:len(v)]=v;mv[c,:len(v)]=True
        arrays={k:np.zeros(mi.shape,np.float64) for k in FIELDS};qx=torch.as_tensor(ctx['x'][None],device=data.device)
        for a in range(0,amount,64):
            cid=torch.as_tensor(mi[:,a:a+64][None],device=data.device);hist=data.histories_all(cid)
            aa,lags=torch_alignment(qx,hist,candidate_chunk=64);aa=aa.cpu().numpy();feat=shift_features(aa,lags)
            for k,v in feat.items():arrays[k][:,a:a+64]=v[0]
            arrays['near_tie_lag_fraction'][:,a:a+64]=((aa>=aa.max(-1,keepdims=True)-LAG_MARGIN_EPS).sum(-1)/aa.shape[-1])[0]
            arrays['recent_difference'][:,a:a+64]=(hist[...,-16:]-qx[:,:,None,-16:]).square().mean(-1).cpu().numpy()[0]
            arrays['candidate_std'][:,a:a+64]=hist.std(-1,unbiased=False).cpu().numpy()[0]
        # Reuse any exact-provenance D4 feature values for this same data/ID.
        for support in ['FULL','P100']:
            src=BASE/'D4'/('V5_'+data.cell+'_'+support);p=src/split/('query%05d.npz'%i)
            if not p.exists() or not (src/'COMPONENT_GATE.json').exists():continue
            prov=json.loads((src/'PROVENANCE.json').read_text());meta=json.loads(p.with_suffix('.json').read_text())
            assert meta['identity']==prov and sha(p)==meta['sha256']
            assert prov['identity']['pool_hashes']==data.pool_hashes and prov['identity']['data']==data.data_info
            assert prov['identity']['source_hashes']['past_lag.py']==identity['lag_source_sha256']
            old=read_npz(p)
            for c in range(data.c):
                lookup={int(v):k for k,v in enumerate(old['global_ids'][0,c])}
                for j,v in enumerate(missing[c]):
                    if v not in lookup:continue
                    k=lookup[v]
                    for name in FIELDS[:-1]:
                        value=float(old[name][0,c,k]);delta=abs(arrays[name][c,j]-value);max_d4_delta=max(max_d4_delta,delta)
                        np.testing.assert_allclose(arrays[name][c,j],value,atol=1e-11,rtol=1e-11)
                        arrays[name][c,j]=value
                    reused_d4+=1
        path=folder/('part%04d.npz'%len(list(folder.glob('part*.npz'))));save(path,dict(ids=mi,valid=mv,**arrays))
        dump(path.with_suffix('.json'),dict(identity=identity,sha256=sha(path),at=now(),d4_values_reused=reused_d4,max_d4_delta_before_reuse=max_d4_delta,
             new_lag_alternative_count=True,all_channels_single_batch=True,candidate_chunk=64,current_torch=torch.__version__))
        shard_hashes[str(path.relative_to(OUT))]=sha(path)
        for c,j in zip(*np.nonzero(mv)):known[(int(c),int(mi[c,j]))]=np.array([arrays[k][c,j] for k in FIELDS])
    result={k:np.zeros(ids.shape,np.float64) for k in FIELDS}
    for c,j in zip(*np.nonzero(valid)):
        vals=known[(int(c),int(ids[c,j]))]
        for k,v in zip(FIELDS,vals):result[k][c,j]=v
    return result,shard_hashes

@torch.no_grad()
def case(data,f,training,evaluation,out,proof,smoke=False):
    checkpoints={};stat=[]
    for split in ['val','test']:
        rows=sample_indices(len(data.datasets[split]),32)
        if smoke:rows=rows[[0,-1]]
        for q,i in enumerate(rows):
            i=int(i);dest=out/split/('query%05d.npz'%i);side=dest.with_suffix('.json')
            if dest.exists():
                meta=json.loads(side.read_text());assert meta['identity']==proof and sha(dest)==meta['sha256'];checkpoints[str(dest.relative_to(out))]=meta['sha256'];continue
            tick=time.perf_counter();ctx,context_path=context(data,split,i)
            a=i//32*32;b=min(a+32,len(data.datasets[split]));j=i-a
            pool=torch.as_tensor(ctx['pool_ids'][None],device=data.device);d=torch.as_tensor(ctx['distance'][None],device=data.device)
            fullmask=torch.as_tensor(ctx['mask'][None],device=data.device)
            if training==evaluation=='P100':
                g,allids=f.native_p100_logits(split,a,b);g=g[j:j+1];allids=allids[j:j+1];mask=torch.ones_like(allids,dtype=torch.bool)
                score=normalize(g,mask);variant='P100_NATIVE';sd=d.gather(-1,allids)
            else:
                g,mask=f.full_logits(split,a,b)
                if evaluation=='FULL':
                    score=normalize(g,mask,raw_single=f.n<=1)[j:j+1];g=g[j:j+1];mask=mask[j:j+1]
                    allids=torch.arange(d.shape[-1],device=data.device).expand(1,data.c,-1);sd=d;variant='FULL'
                else:
                    p=data.pool_all(split,a,b);gg=g.gather(-1,p[:,:,None].expand(-1,-1,g.shape[2],-1))
                    mask=torch.ones_like(p,dtype=torch.bool);score=normalize(gg,mask,raw_single=f.n<=1)[j:j+1]
                    g=gg[j:j+1];mask=mask[j:j+1];allids=p[j:j+1];sd=d.gather(-1,allids);variant='P100_CONTROLLED'
            assert bool(fullmask.gather(-1,allids).all())
            R=min(500 if evaluation=='FULL' else 100,int(mask.sum(-1).min()))
            ordered=topk(score.masked_fill(~mask,-float('inf')),R)
            oracle=topk(-sd.masked_fill(~mask,float('inf')),30)
            heads=topk(g.masked_fill(~mask[:,:,None],-float('inf')),10)
            oid=allids.gather(-1,oracle)[0].cpu().numpy();ordered_ids=allids.gather(-1,ordered)[0].cpu().numpy()
            om=sd.gather(-1,oracle)[0].cpu().numpy();threshold=sd.gather(-1,oracle)[...,:10].amax(-1)
            near=(sd<=threshold[...,None]+(.05*threshold).clamp_min(1e-8)[...,None])&mask
            near_threshold=(threshold+(.05*threshold).clamp_min(1e-8))[0].cpu().numpy()
            near_count=near.sum(-1)[0].cpu().numpy();maxrand=max(30,int(near_count.max()))
            rng=np.random.RandomState(int(ctx['query_start'][0])%4294967295)
            rs=torch.as_tensor(rng.random_sample(mask.shape),device=data.device).masked_fill(~mask,-float('inf'))
            rand=topk(rs,min(maxrand,int(mask.sum(-1).min())))
            rand_ids=allids.gather(-1,rand)[0].cpu().numpy();head_ids=allids[:,:,None].expand(-1,-1,g.shape[2],-1).gather(-1,heads)[0].cpu().numpy()
            # Exact initial rank in original score order, including deterministic ties.
            ranked=topk(score.masked_fill(~mask,-float('inf')),score.shape[-1])
            inv=torch.empty_like(ranked);inv.scatter_(-1,ranked,torch.arange(1,score.shape[-1]+1,device=data.device).expand_as(ranked))
            oracle_rank=inv.gather(-1,oracle)[0].cpu().numpy()
            full_oracle=topk(-d.masked_fill(~fullmask,float('inf')),10)[0].cpu().numpy()
            near_ids=[];union=[]
            for c in range(data.c):
                ni=allids[0,c,near[0,c]].cpu().numpy();near_ids.append(ni)
                # No future-dependent candidate is appended to deployable TopR.
                union.append(np.unique(np.concatenate([ordered_ids[c],oid[c],head_ids[c].ravel(),rand_ids[c],ni,full_oracle[c] if evaluation=='P100' else np.empty(0,dtype=np.int64)])))
            width=max(map(len,union));ids=np.zeros((data.c,width),np.int64);valid=np.zeros_like(ids,bool);eligible=np.zeros_like(ids,bool)
            for c,v in enumerate(union):ids[c,:len(v)]=v;valid[c,:len(v)]=True
            gid=torch.as_tensor(ids[None],device=data.device)
            # Full-oracle-out-of-P100 appended solely for separate diagnosis.
            eligible=np.ones_like(valid)
            if evaluation=='P100':eligible=(ids[:,:,None]==ctx['pool_ids'][:,None]).any(-1)
            eligible &= valid
            feat,shards=lag_features(data,split,i,ctx,ids,valid)
            mse=d.gather(-1,gid)[0].cpu().numpy()
            new=dict(global_ids=ids,valid=valid,eligible=eligible,mse=mse,ordered_ids=ordered_ids,ordered_scores=score.gather(-1,ordered)[0].cpu().numpy(),
                oracle_ids=oid,oracle_mse=om,oracle_initial_rank=oracle_rank,head_ids=head_ids,random_ids=rand_ids,pool_ids=ctx['pool_ids'],full_oracle_ids=full_oracle,
                near_support_count=near_count,query_index=np.array([i]),query_start=ctx['query_start'],query_std=ctx['x'].std(-1),
                near_threshold=near_threshold,near_mask=(mse<=near_threshold[:,None])&eligible,
                evaluation_candidate_count=mask.sum(-1)[0].cpu().numpy(),full_bank_count=np.array([len(data.starts)]),**feat)
            if f.n==5:
                ph=exact_head_probabilities(g,mask,i)
                assert torch.equal(ph.mean(2),score),'per-head probability replay mismatch'
                loc=torch.zeros(1,data.c,len(data.starts),device=data.device,dtype=torch.long)
                loc.scatter_(-1,allids,torch.arange(allids.shape[-1],device=data.device).expand_as(allids))
                ix=loc.gather(-1,gid);ixh=ix[:,:,None].expand(-1,-1,5,-1)
                new['candidate_head_logits']=g.gather(-1,ixh)[0].cpu().numpy()
                new['candidate_head_probabilities']=ph.gather(-1,ixh)[0].cpu().numpy()
                new['candidate_mixture_scores']=score.gather(-1,ix)[0].cpu().numpy()
                new['candidate_mixture_rank']=inv.gather(-1,ix)[0].cpu().numpy()
                loo=[];loom=[]
                for h in range(5):
                    p=ph[:,:,[k for k in range(5) if k!=h]].mean(2)
                    pick=topk(p.masked_fill(~mask,-float('inf')),10)
                    loo.append(allids.gather(-1,pick)[0].cpu().numpy());loom.append(sd.gather(-1,pick)[0].cpu().numpy())
                new['loo_ids']=np.stack(loo,1);new['loo_mse']=np.stack(loom,1)
            # Preserve native canonical model output; cross support is separately labeled.
            if training==evaluation:
                ref=f.reference(split)
                if training=='P100':assert np.array_equal(ordered_ids[:,:10],ref['ids'][i])
                else:
                    expected=f.canonical_full_ids(split,a,b)[j].cpu().numpy();assert np.array_equal(ordered_ids[:,:10],expected)
                    pred=(data.values_all(torch.as_tensor(ordered_ids[:,:10][None],device=data.device))+torch.as_tensor(ctx['x'][None,:,-1,None,None],device=data.device)).mean(2)
                    assert np.allclose(pred.cpu().numpy()[0],ref['prediction'][i],atol=1e-5,rtol=1e-5)
            # Reuse old cross-support output only after exact file/identity checks.
            # Absence never makes a cross-support result a reproduced native row.
            previous=ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02/checkpoint_runs'/data.cell/f.name/split/('batch%05d.npz'%(a//32))
            prior_cross=False
            if training!=evaluation and previous.exists():
                pm=json.loads(previous.with_suffix('.json').read_text());assert pm['sha256']==sha(previous)
                assert pm['identity']['checkpoint_sha256']==f.row['checkpoint_sha256'] and pm['identity']['pool_hashes']==data.pool_hashes
                with np.load(previous) as z:
                    assert np.array_equal(ordered_ids[:,:10],z[variant+'__selected_ids'][j]);prior_cross=True
            elif training!=evaluation:
                expected=canonical_cross_ids(f,split,a,b,evaluation)[j].cpu().numpy()
                assert np.array_equal(ordered_ids[:,:10],expected),'new canonical cross-support ordered Top10 mismatch; no fallback'
                prior_cross='NEW_REFERENCE_GATE_ONLY_CHANNEL_LOOP'
            d4=BASE/'D4'/('V5_'+data.cell+'_'+training)/split/('query%05d.npz'%i)
            reused=False
            if f.n==5 and training==evaluation and d4.exists():
                meta=json.loads(d4.with_suffix('.json').read_text());assert sha(d4)==meta['sha256'];old=read_npz(d4);r=int(old['rerank_count'][0])
                assert np.array_equal(ordered_ids[:,:r],old['global_ids'][0,:,:r])
                assert np.array_equal(oid[:,:10],old['global_ids'][0,:,r:r+10])
                np.testing.assert_array_equal(score.gather(-1,ordered)[0,:,:r].cpu().numpy(),old['base_score'][0,:,:r])
                reused=True
            torch.cuda.synchronize();save(dest,new)
            dump(side,dict(identity=proof,sha256=sha(dest),context_sha256=sha(context_path),lag_shard_hashes=shards,
                at=now(),seconds=time.perf_counter()-tick,query_index=i,channels=data.c,variant=variant,R=R,
                past_only_features=True,future_blind_ordered_pool=True,D4_topR_ordered_exact=reused,
                outside_P100_full_oracle_diagnostic_only=evaluation=='P100',native_canonical_checked=training==evaluation,
                existing_cross_support_ordered_top10_reproduced=prior_cross,
                all_channels_single_batch=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            checkpoints[str(dest.relative_to(out))]=sha(dest)
            state(out,'FEATURE_EXTENSION',split=split,queries=q+1,total=len(rows),actual_query=i,model=f.name,training_support=training,evaluation_support=evaluation)
            print('D56',data.cell,f.name,evaluation,split,q+1,'/',len(rows),'query',i,'all channels',data.c,'bank',len(data.starts),flush=True)
    return checkpoints

def main(cell,model,training,evaluation,smoke=False):
    out=OUT/('feature_smoke' if smoke else 'complete_features')/cell/(model+'_'+training+'_to_'+evaluation);out.mkdir(parents=True,exist_ok=True)
    lock=(BASE/'00_Audit/d4_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=start_gpu();begin=time.perf_counter();os.chdir(ROOT);data=Data(cell)
    row=next(r for r in inventory(cell) if r['architecture']==model and r['training_support']==training)
    f=Frozen(data,row);prior=reproduction(data,row)
    if prior is None:
        from worker import reproduce,identity
        gd=OUT/'native_reproduction'/cell/(model+'_'+training);gd.mkdir(parents=True,exist_ok=True)
        reproduce(f,gd,identity(row,data));prior=reproduction(data,row);assert prior
    proof=dict(cell=cell,checkpoint=row,training_support=training,evaluation_support=evaluation,seed=0,
        native_reproduction_gate=str(prior),native_reproduction_sha256=sha(prior),pool_hashes=data.pool_hashes,data=data.data_info,
        sample='32 linspace validation and test timestamps/all channels',lag_definition=common_identity(data),
        source_sha256=sha(__file__),protocol_sha256=sha(OUT/'PROTOCOL.json'),original_runtime='UNKNOWN',current_torch=torch.__version__)
    pp=out/'PROVENANCE.json'
    if pp.exists():assert json.loads(pp.read_text())==proof
    else:dump(pp,proof)
    torch.cuda.reset_peak_memory_stats();files=case(data,f,training,evaluation,out,proof,smoke)
    assert sha(row['checkpoint'])==row['checkpoint_sha256'] and len(files)==(4 if smoke else 64)
    dump(out/'FEATURE_GATE.json',dict(status='PASS_FIRST_TAIL_SMOKE_NOT_OFFICIAL' if smoke else 'PASS_COMPLETE_D5_D6_FEATURES_AGGREGATION_PENDING',at=now(),identity=proof,
        files=files,queries={'val':2 if smoke else 32,'test':2 if smoke else 32},channels=data.c,seconds=time.perf_counter()-begin,gpu_before=before,gpu_after=snapshot(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        all_channels_single_batch=True,full_bank_count=len(data.starts),full_support_topR=True,source_checkpoint_unchanged=True))
    state(out,'COMPLETE_FEATURES_AGGREGATION_PENDING')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cell');p.add_argument('model',choices=['V0','V1','V5']);p.add_argument('training',choices=['FULL','P100']);p.add_argument('evaluation',choices=['FULL','P100']);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:main(a.cell,a.model,a.training,a.evaluation,a.smoke)
    except BaseException as e:
        out=OUT/'complete_features'/a.cell/(a.model+'_'+a.training+'_to_'+a.evaluation)
        state(out,'BLOCKED',error=repr(e),traceback=traceback.format_exc());raise
