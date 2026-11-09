"""All-query frozen score-fusion evaluation, shared encoding, atomic resumable blocks."""
import argparse,traceback
from bridge import *
from reproduce import verify as verify_reproduction

def canonical_logits(f,split,a,b,support):
    """Reference-only serial calls, never a production output or fallback."""
    from utils.full_candidate_bank import encode_raw_channel_first
    d=f.data;out=[];f.precision()
    if not hasattr(f,'fusion_ref_keys'):f.fusion_ref_keys={}
    for c in range(d.c):
        x,_,starts=d.channel_batch(split,a,b,c,future=False)
        if support=='P100' and not f.full_trained:
            ids=d.pool_all(split,a,b)[:,c]
            out.append(rt.scores(f.model,f.heads,x,d,c,ids,query_channel=0));continue
        if c not in f.fusion_ref_keys:
            h=d.bank_history(c)
            if f.full_trained:key=encode_raw_channel_first(f.model,h,0)
            else:
                parts=[]
                for j in range(0,len(h),3200):
                    xx=h[j:j+3200];n=len(xx)
                    if n<3200:xx=torch.cat([xx,xx.new_zeros(3200-n,d.h,1)])
                    parts.append(rt.encode_raw(f.model,xx,0)[:n])
                key=torch.cat(parts)
            f.fusion_ref_keys[c]=torch.nn.functional.normalize(key,dim=-1)
        z=encode_raw_channel_first(f.model,x,0) if f.full_trained else rt.encode_raw(f.model,x,0)
        q=f.heads(z);g=torch.einsum('bsd,nd->bsn',q,f.fusion_ref_keys[c])
        if support=='P100':g=g.gather(-1,d.pool_all(split,a,b)[:,c,None].expand(-1,5,-1))
        out.append(g)
    return torch.stack(out,1)

@torch.no_grad()
def smoke(d,f,prior,identity,before,dest):
    rows=[];start=time.perf_counter()
    for split in ('val','test'):
        N=len(d.datasets[split])
        for a in sorted(set([0,(N-1)//32*32])):
            b=min(a+32,N);x,y,_=d.query(split,a,b);raw=prior.batch(split,a,b);v,controlled=batch_logits(f,split,a,b)
            for support,(g,m,ids) in v.items():
                name=variant_name(f.support,support);p,chosen=e0_gate(g,m,ids,raw,name,d,x,y)
                ref=canonical_logits(f,split,a,b,support)
                # Report all raw-score deviations; exact official Top10 and E0
                # score reproduction on every previously stored head candidate is mandatory.
                refp=normalize(ref,m);refids=ids.gather(-1,topk(refp,10))
                assert torch.equal(chosen,refids),('direct canonical ordered ID mismatch',split,a,support)
                max_score_diff=float((p-refp).abs().max());max_cosine_diff=float((g-ref).abs().max())
                assert torch.allclose(p,refp,rtol=2e-6,atol=2e-8),('canonical E0 score',max_score_diff)
                fp=[]
                for method in ('E1','E2','E3','E4'):
                    gammas=GAMMAS if method=='E4' else [None]
                    for gam in gammas:
                        s=fuse(g,method,gam)
                        sliced=torch.cat([fuse(g[...,j:j+127],method,gam) for j in range(0,g.shape[-1],127)],-1)
                        assert torch.equal(s,sliced),('candidate chunk dependence',method,gam)
                        sliced=torch.cat([fuse(g[j:j+3],method,gam) for j in range(0,len(g),3)])
                        assert torch.equal(s,sliced),('query consumer chunk dependence',method,gam)
                        pick=topk(s.masked_fill(~m,-float('inf')),10)
                        s32=fuse(g,method,gam,dtype=torch.float32).masked_fill(~m,-float('inf'))
                        other=topk(s32,10)
                        boundary=topk(s.masked_fill(~m,-float('inf')),11);vals=s.gather(-1,boundary)
                        fp.append(dict(method=method,gamma=gam,ordered_fp32_fp64_different=int((pick!=other).any(-1).sum()),
                            min_boundary_gap=float((vals[...,9]-vals[...,10]).min())))
                        # Common-candidate support control uses the very same head cosines.
                        if support=='FULL':
                            pool=d.pool_all(split,a,b);shared=s.gather(-1,pool)
                            local=fuse(controlled,method,gam)
                            assert torch.equal(shared,local),('pure support dependence',method,gam)
                rows.append(dict(split=split,start=a,end=b,support=support,shape=list(g.shape),
                    all_channels=d.c,canonical_e0_ordered_ids_exact=True,max_canonical_e0_score_delta=max_score_diff,
                    max_canonical_raw_cosine_delta=max_cosine_diff,fp32_fp64=fp,chunk_invariance=True,support_control_exact=True))
                del ref,refp,p,s,s32
    gate=dict(status='PASS_FIRST_TAIL_GPU_SMOKE',identity=identity,rows=rows,at=now(),seconds=time.perf_counter()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),before=before,
        query_batch_semantics='Canonical encoder context B32, all channels intact; consumer chunks 3 and candidates127 exact',
        test_future_used_for_selection=False)
    dump(dest/'GPU_SMOKE_GATE.json',gate)
    print('GPU_SMOKE_PASS',d.cell,f.support,gate['seconds'],gate['peak_allocated_bytes'],flush=True)

def labels(split,gamma):
    return [('E0',None),('E1',None),('E2',None),('E3',None)]+[('E4',g) for g in (GAMMAS if split=='val' else [gamma])]

def label(method,gamma):return method if method!='E4' else 'E4_g'+str(gamma)

def verify_block(path,identity):
    side=json.loads(path.with_suffix('.json').read_text())
    assert side['identity']==identity and sha(path)==side['sha256']
    return side

@torch.no_grad()
def run(cell,training,smoke_only=False):
    d,f,prior,identity,before=setup(cell,training)
    dest=OUT/'checkpoints'/cell/('V5_'+training);dest.mkdir(parents=True,exist_ok=True)
    ip=dest/'IDENTITY.json'
    if ip.exists():assert json.loads(ip.read_text())==identity,'Code/data changed: preserve outputs and use an explicitly reviewed recovery'
    else:dump(ip,identity)
    sg=dest/'GPU_SMOKE_GATE.json'
    if sg.exists():assert json.loads(sg.read_text())['identity']==identity
    else:smoke(d,f,prior,identity,before,dest)
    if smoke_only:return
    verify_reproduction(d,f,prior,identity,dest)
    begun=time.perf_counter();hashes={};gamma=None;counts={};timer={};metric_sums={};units={};new_compute_seconds=0.
    for split in ('val','test'):
        if split=='test':
            rows=[]
            for gam in GAMMAS:
                name=label('E4',gam);key=('val',training,name)
                rows.append(dict(split='val',training_support=training,evaluation_support=training,gamma=gam,retmse=metric_sums[key]/units[key]))
            gamma=gamma_select(rows)
            gp=dict(status='FROZEN_BEFORE_TEST',chosen_gamma=gamma,rows=rows,selection='native validation only',checkpoint_sha256=identity['checkpoint']['checkpoint_sha256'],at=now())
            p=dest/'GAMMA_SELECTION.json'
            if p.exists():assert json.loads(p.read_text())['chosen_gamma']==gamma
            else:dump(p,gp)
        N=len(d.datasets[split]);count=0
        for a in range(0,N,32):
            b=min(a+32,N);p=dest/split/('batch%05d.npz'%(a//32))
            if p.exists():
                meta=verify_block(p,identity);out=read_npz(p)
            else:
                block_start=time.perf_counter();dump(dest/'CURRENT.json',dict(phase='FUSION_INFERENCE',split=split,query_start=a,total_queries=N,pid=os.getpid(),at=now(),gamma=gamma))
                raw=prior.batch(split,a,b);x,y,starts=d.query(split,a,b)
                torch.cuda.synchronize();t=time.perf_counter();variants,controlled=batch_logits(f,split,a,b);torch.cuda.synchronize()
                enc_seconds=time.perf_counter()-t
                out=dict(query_start=np.asarray(starts),pool_ids=d.pool_all(split,a,b).cpu().numpy())
                proof={};timings={'encode_and_cosine_seconds':enc_seconds};scores={};e0ids={}
                for support,(g,m,ids) in variants.items():
                    vn=variant_name(training,support);e0,e0id=e0_gate(g,m,ids,raw,vn,d,x,y);e0ids[support]=e0id
                    oracle,oldoret,rnd,oldrret,pr=prior.baseline(support,split,a,b);proof[support]=pr
                    om=candidate_distances(d,x,y,oracle);rm=candidate_distances(d,x,y,rnd)
                    np.testing.assert_allclose(om.double().mean(-1).cpu().numpy(),oldoret,atol=2e-6,rtol=2e-6)
                    np.testing.assert_allclose(rm.double().mean(-1).cpu().numpy(),oldrret,atol=2e-6,rtol=2e-6)
                    out[support+'__oracle_ids']=oracle.cpu().numpy();out[support+'__oracle_mse']=om.cpu().numpy()
                    out[support+'__random_ids']=rnd.cpu().numpy();out[support+'__random_mse']=rm.cpu().numpy()
                    head=torch.as_tensor(raw[vn+'__head_ids'],device=d.device)
                    scores[support]={}
                    for method,gam in labels(split,gamma):
                        name=label(method,gam);torch.cuda.synchronize();t=time.perf_counter()
                        s=e0 if method=='E0' else score(g,m,method,gam)
                        pos=topk(s,10);pick=ids.gather(-1,pos)
                        assert m.gather(-1,pos).all() and (pick.sort(-1).values[...,1:]!=pick.sort(-1).values[...,:-1]).all()
                        torch.cuda.synchronize();timings[support+'_'+name+'_fusion_topk_seconds']=time.perf_counter()-t
                        arr=metric_arrays(d,x,y,ids,pick,oracle,om,rnd,head,e0id)
                        for k,v in arr.items():out[support+'__'+name+'__'+k]=v
                        # Store only P100-restricted scalar scores for support comparison.
                        scores[support][name]=s.gather(-1,torch.as_tensor(out['pool_ids'],device=d.device)) if support=='FULL' else s
                        del s,arr
                pool=torch.as_tensor(out['pool_ids'],device=d.device)
                for method,gam in labels(split,gamma):
                    name=label(method,gam);fs=scores['FULL'][name];ps=scores['P100'][name]
                    ar=rank100(fs);br=rank100(ps)
                    out['STABILITY__'+name+'__common_p100_spearman']=(1-6*(ar-br).square().sum(-1)/(100*9999)).cpu().numpy()
                    out['STABILITY__'+name+'__common_p100_top10_overlap']=(topk(fs,10)[...,None]==topk(ps,10)[...,None,:]).any(-1).double().mean(-1).cpu().numpy()
                    if method!='E0':
                        cs=fuse(controlled,method,gam)
                        assert torch.equal(fs,cs),'E1-E4 changed under pure support restriction'
                        out['STABILITY__'+name+'__shared_cosine_score_max_delta']=np.zeros((b-a,d.c))
                    else:
                        local=normalize(controlled,torch.ones_like(pool,dtype=torch.bool))
                        ra=rank100(fs);rb=rank100(local)
                        out['STABILITY__'+name+'__shared_cosine_rank_spearman']=(1-6*(ra-rb).square().sum(-1)/(100*9999)).cpu().numpy()
                save(p,out);sec=time.perf_counter()-block_start;new_compute_seconds+=sec
                meta=dict(identity=identity,sha256=sha(p),query_start=a,query_end=b,at=now(),seconds=sec,timings=timings,baseline_sources=proof,
                    e0_ordered_ids_exact=True,e0_stored_scores_exact=True,e0_candidate_mse_exact=True,all_channels_single_batch=True)
                dump(p.with_suffix('.json'),meta)
                del variants,controlled,raw,scores,g,m,ids,e0
            assert np.array_equal(out['query_start'],d.datasets[split].starts[a:b])
            for support in ('FULL','P100'):
                for method,gam in labels(split,gamma):
                    name=label(method,gam);key=(split,support,name);v=out[support+'__'+name+'__retmse']
                    metric_sums[key]=metric_sums.get(key,0.)+float(v.sum(dtype=np.float64));units[key]=units.get(key,0)+v.size
            for k,v in meta['timings'].items():timer[k]=timer.get(k,0.)+v
            hashes[str(p.relative_to(dest))]=meta['sha256'];count+=b-a
            print(now(),cell,training,split,b,N,'gamma',gamma,flush=True)
        assert count==N;counts[split]=count
    native=metric_sums[('test',training,'E0')]/units[('test',training,'E0')]
    if training=='P100':official=float(f.row['canonical_retmse'])
    else:official=float(f.reference('test')['metrics']['retmse10'])
    assert np.isclose(native,official,atol=2e-6,rtol=2e-6),(native,official,'official E0 RetMSE mismatch')
    gate=dict(status='PASS_COMPLETE_ALL_QUERY_SCORE_FUSION',at=now(),identity=identity,queries=counts,channels=d.c,full_candidate_count=len(d.starts),
        p100_count=100,gamma=gamma,files=hashes,e0_ordered_id_mismatches=0,e0_stored_scores_exact=True,e0_native_retmse=native,
        e0_official_retmse=official,e0_metric_difference=native-official,all_channels_single_batch=True,
        elapsed_this_invocation_seconds=time.perf_counter()-begun,new_compute_seconds=new_compute_seconds,timings=timer,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),observed_gpu=snapshot(),before=before)
    dump(dest/'COMPLETE.json',gate);print('CHECKPOINT_COMPLETE',cell,training,gamma,native,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cell');p.add_argument('training',choices=['FULL','P100']);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:run(args.cell,args.training,args.smoke)
    except BaseException as e:
        dest=OUT/'checkpoints'/args.cell/('V5_'+args.training);dest.mkdir(parents=True,exist_ok=True)
        dump(dest/('FAILURE_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S')+'.json'),dict(at=now(),error=repr(e),traceback=traceback.format_exc(),pid=os.getpid(),preserved=True));raise
