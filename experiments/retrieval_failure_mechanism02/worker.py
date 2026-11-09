"""Frozen single-checkpoint sequential worker; full populations, both supports.

Reproduction must finish on both full splits before this checkpoint contributes
any new diagnostics. Failures preserve all files and require an explicit repair.
"""
import argparse,gc,os,resource,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from protocol import ROOT,GPU_UUID,PROTOCOL
from io_utils import read,dump,save,sha,now,source_hashes,asnumpy,id_bytes
from adapter import Data,Frozen,inventory,normalize,topk,distances,rt
from gpu_metrics import evaluate,normalization_control,head_union,METRICS
import ranking
import resources
from feature_analysis import past_delta_scores,random_scores,groups,geometry_sets,FEATURES,GEOMETRY_FIELDS

OUT=ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02'

def guard():
    # New explicit parallel authorization supersedes the previous global wait.
    s=resources.before_context();rt.gpu_guard();resources.bound_context();return s

def identity(row,data):
    return dict(checkpoint_sha256=row['checkpoint_sha256'],pool_hashes=data.pool_hashes,source_hashes=source_hashes(),
                protocol_sha256=sha(OUT/'00_Audit/PROTOCOL.json'),data=data.data_info)

@torch.no_grad()
def native_batch(f,split,a,b,d,x,y):
    B,C,_=x.shape
    if not f.full_trained:
        logits,ids=f.native_p100_logits(split,a,b);mask=torch.ones_like(ids,dtype=torch.bool)
        p=normalize(logits,mask);dist=d.gather(-1,ids)
    else:
        logits,mask=f.full_logits(split,a,b);ids=torch.arange(d.shape[-1],device=d.device).expand(B,C,-1)
        p=normalize(logits,mask,raw_single=f.n<=1);dist=d
    ev=evaluate(f.data,x,y,dist,p,mask,ids)
    return ev

def canonical_p100_metrics(ev):
    """Compare stored FP32 metrics in their original arithmetic, all channels.

    FP64 diagnostics remain unmodified for strict regret/exchange identities.
    """
    shape=ev['selected_ids'].shape[:2]
    ret=ev['selected_mse'].float().reshape(-1,10).mean(-1).reshape(shape)
    hit=(ev['selected_ids'][...,None]==ev['oracle_ids'][:,:,None,:]).any(-1)
    recall=hit.float().mean(-1)
    return torch.stack([ret,ev['metrics'][...,1].float(),ev['metrics'][...,2].float(),recall],-1)


def reproduce(f,out,proof):
    path=out/'REPRODUCTION_GATE.json'
    if path.exists():
        prior=read(path);assert prior['status']=='PASS' and prior['identity']==proof;return
    begin=time.perf_counter();rows=[]
    for split in ('val','test'):
        total=len(f.data.datasets[split]);sums=np.zeros(4);maxpred=0.;maxmetric=0.;maxdiagnostic=0.;units=0
        import hashlib
        ids_hash=hashlib.sha256();ref_hash=hashlib.sha256()
        for a in range(0,total,32):
            if a%256==0:resources.during()
            b=min(a+32,total);x,y,starts,d=distances(f.data,split,a,b)
            ev=native_batch(f,split,a,b,d,x,y);ids=asnumpy(ev['selected_ids']);metrics=asnumpy(ev['metrics'])
            ref=f.reference(split);ids_hash.update(id_bytes(ids))
            if not f.full_trained:
                expected=ref['ids'][a:b]
                assert np.array_equal(ids,expected),(f.name,split,a,'ordered P100 IDs mismatch; no fallback')
                old=ref['metrics'][a:b,:,:4];new=asnumpy(canonical_p100_metrics(ev))
                maxdiagnostic=max(maxdiagnostic,float(abs(old-metrics[...,:4]).max()))
                delta=float(abs(old-new).max());maxmetric=max(delta,maxmetric)
                assert delta<=2e-6,(f.name,split,a,'stored P100 metric mismatch',delta)
            else:
                expected=asnumpy(f.canonical_full_ids(split,a,b))
                assert np.array_equal(ids,expected),(f.name,split,a,'canonical Full ordered IDs mismatch')
                old=ref['prediction'][a:b];new=asnumpy(ev['prediction']);delta=float(abs(old-new).max());maxpred=max(maxpred,delta)
                assert np.allclose(old,new,atol=1e-5,rtol=1e-5),(f.name,split,a,'stored Full prediction mismatch',delta)
            ref_hash.update(id_bytes(expected));sums+=metrics[...,:4].sum((0,1));units+=(b-a)*f.data.c
            dump(out/'CURRENT.json',dict(phase='REPRODUCTION',model=f.name,split=split,queries=b,total=total,channels=f.data.c,pid=os.getpid(),at=now()))
            if a%256==0:print('MECHANISM02 REPRODUCTION',f.data.cell,f.name,split,b,'/',total,flush=True)
        assert units==total*f.data.c and ids_hash.hexdigest()==ref_hash.hexdigest()
        mean=sums/units
        if f.full_trained and split=='test':
            ref=f.reference(split)['metrics']
            for i,k in [(0,'retmse10'),(1,'agg_mse10'),(3,'recall10')]:
                assert np.isclose(mean[i],ref[k],atol=1e-5,rtol=1e-4),(f.name,k,mean[i],ref[k])
        rows.append(dict(model=f.name,split=split,status='PASS',queries=total,channels=f.data.c,units=units,
                         ordered_id_sha256=ids_hash.hexdigest(),canonical_ordered_id_sha256=ref_hash.hexdigest(),
                         ordered_mismatch_count=0,max_prediction_difference=maxpred,max_query_metric_difference=maxmetric,
                         max_diagnostic_fp64_vs_stored_fp32_difference=maxdiagnostic,
                         query_metric_comparison='canonical FP32 vs stored FP32; diagnostic FP64 retained' if not f.full_trained else 'stored Full prediction',
                         metrics=mean.tolist(),source_checkpoint_sha256=f.row['checkpoint_sha256']))
    dump(path,dict(status='PASS',identity=proof,rows=rows,seconds=time.perf_counter()-begin,
                   reference_only_channel_loop=f.full_trained,production_channel_loop=False,source=f.identity()))

@torch.no_grad()
def analyze(f,out,proof,smoke=False):
    g=read(out/'REPRODUCTION_GATE.json');assert g['status']=='PASS' and g['identity']==proof
    begin=time.perf_counter();records=[];files={};observations=[]
    phase_wall=dict(oracle_distance=0.,retrieval_scores=0.,ranking=0.,features_geometry=0.,multihead=0.)
    for split in ('val','test'):
        total=len(f.data.datasets[split]);sample=set(ranking.rows(total).tolist());starttime=time.perf_counter()
        epsilon=read(out/'NEAROPTIMAL_VALIDATION.json') if split=='test' else None
        offsets=range(0,total,32) if not smoke else sorted({0,((total-1)//32)*32})
        seen_queries=0
        for a in offsets:
            if a%256==0:resources.during()
            b=min(a+32,total);target=out/split/f'batch{a//32:05d}.npz';side=target.with_suffix('.json')
            if target.exists():
                done=read(side);assert done['status']=='PASS' and done['identity']==proof and sha(target)==done['sha256']
                files[str(target.relative_to(out))]=done['sha256'];seen_queries+=done['queries'];continue
            torch.cuda.synchronize();tick=time.perf_counter()
            x,y,starts,d=distances(f.data,split,a,b);B,C,N=d.shape;pool=f.data.pool_all(split,a,b)
            torch.cuda.synchronize();phase_wall['oracle_distance']+=time.perf_counter()-tick;tick=time.perf_counter()
            logits,mask=f.full_logits(split,a,b)
            assert bool(mask.gather(-1,pool).all()) and bool((pool.sort(-1).values[...,1:]!=pool.sort(-1).values[...,:-1]).all())
            ctrl=normalization_control(logits,mask,pool)
            past=past_delta_scores(f.data,x);random=random_scores(starts,C,N,f.data.device)
            pf=normalize(logits,mask,raw_single=f.n<=1)
            pp=normalize(logits.gather(-1,pool[:,:,None].expand(-1,-1,logits.shape[2],-1)),torch.ones_like(pool,dtype=torch.bool),raw_single=f.n<=1)
            # Full/P100 interventions use exactly identical Full logits. Native
            # P100 is a separate canonical row if FP32 encoding/order differs.
            variants=[('FULL',pf,d,mask,torch.arange(N,device=d.device).expand(B,C,N)),
                      ('P100_CONTROLLED',pp,d.gather(-1,pool),torch.ones_like(pool,dtype=torch.bool),pool),
                      ('P100_LOCAL_NORMALIZER',ctrl['local'],d.gather(-1,pool),torch.ones_like(pool,dtype=torch.bool),pool),
                      ('P100_FULL_NORMALIZER',ctrl['restricted'],d.gather(-1,pool),torch.ones_like(pool,dtype=torch.bool),pool)]
            if not f.full_trained:
                ns,ni=f.native_p100_logits(split,a,b)
                variants.append(('P100_NATIVE',normalize(ns,torch.ones_like(pool,dtype=torch.bool)),d.gather(-1,pool),torch.ones_like(pool,dtype=torch.bool),ni))
            torch.cuda.synchronize();phase_wall['retrieval_scores']+=time.perf_counter()-tick
            payload=dict(query_start=np.asarray(starts),metric_names=np.asarray(METRICS),feature_names=np.asarray(FEATURES),geometry_fields=np.asarray(GEOMETRY_FIELDS),pool_ids=asnumpy(pool),
                         head_mass=asnumpy(ctrl['mass']),implicit_head_weight=asnumpy(ctrl['weights']),
                         logZ_full=asnumpy(ctrl['logZ_full']),logZ_pool=asnumpy(ctrl['logZ_pool']))
            for name,score,dist,m,ids in variants:
                ev=evaluate(f.data,x,y,dist,score,m,ids)
                for k in ('metrics','selected_ids','oracle_ids','oracle100_ids','oracle100_mse','missed_ranks','false_ids','missed_ids','selected_mse','selected_scores'):
                    payload[name+'__'+k]=asnumpy(ev[k])
                if name in ('FULL','P100_CONTROLLED','P100_NATIVE'):
                    torch.cuda.synchronize();tick=time.perf_counter()
                    rnd=random if name=='FULL' else random.gather(-1,pool)
                    past_s=past if name=='FULL' else past.gather(-1,pool)
                    rid=ids.gather(-1,topk(rnd.masked_fill(~m,-float('inf')),10))
                    fs,fc=groups(f.data,x,ev,rid)
                    payload[name+'__group_feature_sum']=asnumpy(fs);payload[name+'__group_feature_count']=asnumpy(fc)
                    if split=='test':
                        ep=torch.tensor(epsilon['FULL' if name=='FULL' else 'P100'],device=dist.device)
                        threshold=ev['oracle100_mse'][...,9]+ep[None]
                        payload[name+'__near_optimal_count']=asnumpy(((dist<=threshold[...,None])&m).sum(-1))
                        payload[name+'__near_optimal_threshold']=asnumpy(threshold)
                    if name in ('FULL','P100_CONTROLLED'):
                        ge=geometry_sets(f,x,dist,m,ids,past_s,ev,rnd,raw_spaces=f.full_trained and f.n==0)
                        for k,v in ge.items():payload[name+'__geometry__'+k]=asnumpy(v)
                    torch.cuda.synchronize();phase_wall['features_geometry']+=time.perf_counter()-tick
                sampled=[i-a for i in range(a,b) if i in sample]
                if sampled:
                    tick=time.perf_counter()
                    rr=ranking.analyze(asnumpy(score[sampled]),asnumpy(dist[sampled]),asnumpy(m[sampled]),np.asarray(starts)[sampled])
                    for row in rr:row.update(split=split,variant=name,model=f.name)
                    records.extend(rr)
                    phase_wall['ranking']+=time.perf_counter()-tick
                if f.n==5 and name in ('FULL','P100_CONTROLLED'):
                    torch.cuda.synchronize();tick=time.perf_counter()
                    hs=logits if name=='FULL' else logits.gather(-1,pool[:,:,None].expand(-1,-1,5,-1))
                    hu=head_union(hs,score,m);payload[name+'__head_ids']=asnumpy(ids[:,:,None].expand(-1,-1,5,-1).gather(-1,hu['head_ids']))
                    payload[name+'__head_overlap']=asnumpy(hu['pair_overlap']);payload[name+'__union_size']=asnumpy(hu['union_size'])
                    headmetrics=[];headpred=[];headmse=[]
                    for h in range(5):
                        he=evaluate(f.data,x,y,dist,hs[:,:,h].masked_fill(~m,-float('inf')),m,ids)
                        headmetrics.append(he['metrics']);headpred.append(he['prediction']-y);headmse.append(he['selected_mse'])
                    payload[name+'__head_metrics']=asnumpy(torch.stack(headmetrics,2))
                    payload[name+'__head_selected_mse']=asnumpy(torch.stack(headmse,2))
                    residual=torch.stack(headpred,2);dot=torch.einsum('bcsh,bcth->bcst',residual,residual)/f.data.h
                    norm=residual.square().mean(-1).sqrt();payload[name+'__head_error_cosine']=asnumpy(dot/(norm[:,:,:,None]*norm[:,:,None,:]).clamp_min(1e-30))
                    # Keep original support normalization, changing only allowed set.
                    union_ev=evaluate(f.data,x,y,dist,score.masked_fill(~hu['union_mask'],-float('inf')),m,ids)
                    payload[name+'__union_deployable_metrics']=asnumpy(union_ev['metrics'])
                    # Oracle-in-union is explicitly diagnostic, not deployable.
                    union_oracle=-dist.masked_fill(~hu['union_mask'],float('inf'))
                    ue=evaluate(f.data,x,y,dist,union_oracle,m,ids)
                    payload[name+'__union_oracle_metrics']=asnumpy(ue['metrics'])
                    loo=[]
                    for h in range(5):
                        p=torch.softmax(hs.masked_fill(~m[:,:,None],-float('inf'))/.1,-1)[:,:, [j for j in range(5) if j!=h]].mean(2)
                        loo.append(evaluate(f.data,x,y,dist,p,m,ids)['metrics'])
                    payload[name+'__loo_metrics']=asnumpy(torch.stack(loo,2))
                    torch.cuda.synchronize();phase_wall['multihead']+=time.perf_counter()-tick
            # Per-query-channel support regret identity; no clipped negative values.
            rf=payload['FULL__metrics'][...,4];rp=payload['P100_CONTROLLED__metrics'][...,4]
            assert np.all(rf<=rp+2e-6)
            for name in ('P100_CONTROLLED','P100_LOCAL_NORMALIZER','P100_FULL_NORMALIZER')+(() if f.full_trained else ('P100_NATIVE',)):
                rm=payload[name+'__metrics'][...,0];poolreg=rp-rf;rankreg=rm-rp
                assert np.all(rankreg>=-2e-6)
                np.testing.assert_allclose(poolreg+rankreg,rm-rf,rtol=1e-10,atol=1e-10)
                payload[name+'__regret_decomposition']=np.stack([poolreg,rankreg,rm-rf],-1)
            save(target,payload);hashval=sha(target);dump(side,dict(status='PASS',identity=proof,sha256=hashval,queries=B,channels=C,at=now()))
            files[str(target.relative_to(out))]=hashval
            dump(out/'PHASE_WALL_CURRENT.json',dict(seconds=phase_wall,scope='synchronized wall time under shared GPU, not GPU kernel time',at=now()))
            # Ranking rows are checkpointed with each block, so resume loses none.
            rrpath=target.with_name(target.stem+'_ranking.json')
            for row in records:
                for k in ('spearman','kendall_tau_b'):
                    if not np.isfinite(row[k]):row[k]=None
            dump(rrpath,records);records=[]
            dump(out/'CURRENT.json',dict(phase='DIAGNOSTICS',model=f.name,split=split,queries=b,total=total,channels=C,at=now(),pid=os.getpid()))
            if a%256==0:
                obs=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
                observations.append(dict(at=now(),split=split,queries=b,observation=obs));print('MECHANISM02',f.data.cell,f.name,split,b,'/',total,obs,flush=True)
            seen_queries+=B
            del payload,logits,ctrl,variants;gc.collect()
        assert smoke or seen_queries==total
        dump(out/split/'SPLIT_GATE.json',dict(status='SMOKE_PASS' if smoke else 'PASS',queries=seen_queries,expected_queries=total,channels=f.data.c,units=seen_queries*f.data.c,seconds=time.perf_counter()-starttime,identity=proof))
        if split=='val':
            eps={}
            for label,name in [('FULL','FULL'),('P100','P100_CONTROLLED')]:
                values=[]
                for p in sorted((out/'val').glob('batch*.npz')):
                    with np.load(p) as z:values.append(z[name+'__oracle100_mse'][...,9].copy())
                eps[label]=np.maximum(1e-8,.01*np.median(np.concatenate(values),axis=0)).tolist()
            dump(out/'NEAROPTIMAL_VALIDATION.json',dict(**eps,source_split='val',test_used=False,identity=proof))
    dump(out/('ANALYSIS_SMOKE_GATE.json' if smoke else 'CORE_COMPLETE.json'),dict(status='SMOKE_PASS' if smoke else 'CORE_COMPLETE_NOT_ALL_PHASES',identity=proof,at=now(),files=files,
         query_counts={s:len(f.data.datasets[s]) for s in ('val','test')},channels=f.data.c,seconds=time.perf_counter()-begin,
         peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),observed_gpu=observations,
         phase_wall_seconds=phase_wall,timing_scope='synchronized shared-GPU wall; oracle_distance excludes oracle sorting inside metric evaluation; rankings exclude main metrics; disjoint named phases do not cover total',
         cpu_ram_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,source=f.identity(),
         pending=['validation-fixed geometry decision','feature/ambiguity/probe provenance reuse','paired bootstrap','integrated attribution/report']))

def main(args):
    os.chdir(ROOT) # Original data configuration uses repository-relative paths.
    observed=guard();torch.set_num_threads(1);data=Data(args.cell);rows=[r for r in inventory(args.cell) if r['architecture']==args.model and r['training_support']==args.training_support];assert len(rows)==1
    row=rows[0];out=OUT/'checkpoint_runs'/args.cell/(args.model+'_'+args.training_support);out.mkdir(parents=True,exist_ok=True)
    f=Frozen(data,row);proof=identity(row,data);torch.cuda.reset_peak_memory_stats()
    dump(out/'RESOURCE_START.json',dict(**observed,shared_gpu=True,timing_scope='not standalone throughput'))
    reproduce(f,out,proof)
    if args.mode=='analyze':analyze(f,out,proof)
    if args.mode=='smoke':
        import shutil
        smoke_out=OUT/'analysis_smoke'/args.cell/f.name;smoke_out.mkdir(parents=True,exist_ok=True)
        shutil.copy2(out/'REPRODUCTION_GATE.json',smoke_out/'REPRODUCTION_GATE.json')
        analyze(f,smoke_out,proof,smoke=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cell');p.add_argument('model',choices=['V0','V1','V5']);p.add_argument('training_support',choices=['P100','FULL']);p.add_argument('--mode',choices=['gate','analyze','smoke'],default='gate');a=p.parse_args()
    try:main(a)
    except BaseException as e:
        out=OUT/'checkpoint_runs'/a.cell/(a.model+'_'+a.training_support)
        dump(out/'FAILED.json',dict(status='INVALID_PENDING_REPAIR',at=now(),error=repr(e),traceback=traceback.format_exc(),pid=os.getpid(),
             peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
             peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None))
        dump(OUT/'00_Audit/IMMEDIATE_ISSUE.json',dict(status='BLOCKED',at=now(),cell=a.cell,model=a.model,support=a.training_support,error=repr(e)))
        raise
