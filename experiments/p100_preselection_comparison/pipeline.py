#!/usr/bin/env python3
"""Isolated P100 preselection experiment; shared CARTS source remains unchanged."""
import argparse, csv, hashlib, importlib, itertools, json, os, subprocess, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scripts.verify_v5_head_loss_stage2_correctness01 import CELLS
from scripts.train_j_shared_encoder_drift01 import build_model, state_hash
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_factorial_e2e01 import encode_raw, individual_utility_memsafe
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_retriever_pool01 import CandidatePoolCache, compute_scores
from utils.candidate_pool import compute_coarse_delta_last_cosine_scores, build_coarse_pool, CandidatePoolConfig, gather_candidate_values, pooled_future_mse
from utils.mean_mixture_selection import mean_mixture_topk_selection
from models.RelationStage1 import stable_topk_indices
from models.RelationStage2 import BaseForecastHead
from scripts.train_r_stage2_lambda01 import load_tensors

GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
METHODS=('baseline_cosine','paa16_cosine','saraf_time_aligned')
METRICS={'baseline_cosine':'delta_last_cosine','paa16_cosine':'paa16_delta_last_cosine','saraf_time_aligned':'saraf_time_aligned_alpha0.1'}
ORDER=('ETTh1_96','ETTh1_720','Weather_96','Weather_720')
def dump(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(obj,indent=2,default=str))
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def common(cell):
    c=CELLS[cell]; return SimpleNamespace(**c,patch_len=16,init_seed=0,top_k=10,batch_size=32)
def historical(cell,arm):
    ds,h=cell.split('_')
    return ROOT/f'results/TRACK-V-SHARED-TOP100-SIGNFIX01/{ds}/H{h}/seed0/pool_top100/{cell}/V{arm}'
def oldpool(cell):
    ds,h=cell.split('_'); return ROOT/f'results/TRACK-V-MULTIQUERY-GENERALIZATION01/{ds}/H{h}/seed0/pool_top100/shared_candidate_pool'
def selected(cell,arm):
    return json.loads((historical(cell,arm)/'checkpoint_selection/selected_checkpoint.json').read_text())['selected_checkpoint_path']
def fullselected(cell):
    ds,h=cell.split('_'); p=ROOT/f'results/TRACK-V-MEANMIX-CHECKPOINT-CORRECTION01/{ds}/H{h}/seed0/checkpoint_selection/selected_checkpoint.json'
    return json.loads(p.read_text())['selected_checkpoint_path']
def paa16(x):
    if x.shape[-1] not in (96,720): raise ValueError('PAA input must have length 96 or 720')
    return x.reshape(*x.shape[:-1],16,x.shape[-1]//16).mean(-1)
def time_features(stamps,device='cpu'):
    s=pd.DatetimeIndex(stamps)
    if s.hasnans: raise ValueError('Missing timestamp')
    return torch.tensor(np.stack([s.hour,s.dayofweek,s.month,s.minute],1),device=device,dtype=torch.float32)
def temporal_raw(q,k,length,steps_day):
    d=(q[:,None,:]-k[None,:,:]).abs()
    hd=torch.minimum(d[:,:,0],24-d[:,:,0]); md=torch.minimum(d[:,:,2],12-d[:,:,2]); nd=torch.minimum(d[:,:,3],60-d[:,:,3])
    short=length<steps_day
    out=(.6 if short else .3)*torch.exp(-hd/(2 if short else 4))
    out=out+.2*(d[:,:,1]==0)+.1*((q[:,None,1]>=5)==(k[None,:,1]>=5))
    out=out+(.2 if short else .4)*torch.exp(-md/(3 if short else 2))
    if length<steps_day//4: out=out+.1*torch.exp(-nd/15)
    return out
def validate_ids(ids,mask,k=100):
    assert ids.shape[-1]==k
    assert bool(mask.gather(1,ids).all())
    assert bool((ids.sort(-1).values.diff(dim=-1)>0).all())
def baseline_scores(x,memory_x,channel):
    # Canonical cached pools used the TF32 GEMM path for singleton tails.
    # Preserve arithmetic, epsilon and stable tie rules; pin only kernel geometry.
    if len(x) != 1:
        return compute_coarse_delta_last_cosine_scores(x,memory_x,channel)
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        return compute_coarse_delta_last_cosine_scores(x.repeat(32,1,1),memory_x,channel)[:1]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous

def select_scores(scores,mask):
    if not bool(torch.isfinite(scores).all()): raise ValueError('nonfinite preselection scores')
    ids=build_coarse_pool(scores,mask,100); validate_ids(ids,mask); return ids
def validate_timestamp_series(raw_dates):
    dates=pd.to_datetime(raw_dates,errors='raise')
    if dates.isna().any() or not dates.is_monotonic_increasing:
        raise ValueError('missing or decreasing timestamps')
    deltas=dates.diff().dropna().dt.total_seconds().to_numpy()
    positive=deltas[deltas>0]
    if not len(positive): raise ValueError('no positive timestamp intervals')
    values,counts=np.unique(positive,return_counts=True)
    nominal=float(values[counts.argmax()])
    if counts.max()/len(positive)<.99 or np.any(np.mod(positive,nominal)!=0):
        raise ValueError('ambiguous nominal sampling interval')
    nday=86400/nominal
    if nday!=int(nday): raise ValueError('sampling interval does not divide day')
    # Actual calendar values and row indices are never changed. Missing date
    # values still fail; duplicates/gaps are explicitly reported, not imputed.
    audit=dict(rows=len(dates),missing=int(dates.isna().sum()),duplicates=int(dates.duplicated().sum()),nominal_seconds=nominal,gaps=int((deltas>nominal).sum()),policy='actual CSV timestamps; original row order; no imputation/deduplication')
    return dates,int(nday),audit

def stamps_for(exp,args):
    raw=pd.read_csv(Path(args.root_path)/args.data_path,usecols=['date'])
    dates,nday,audit=validate_timestamp_series(raw.date)
    if audit['duplicates'] or audit['gaps']:
        print('TIMESTAMP_SOURCE_ANOMALIES '+json.dumps(audit),flush=True)
    dump(HERE/'reports'/f'timestamp_validation_{args.data}_{args.seq_len}.json',audit)
    starts=np.asarray(exp.memory_sampler.starts,dtype=np.int64)
    ix=starts+args.seq_len//2
    if ix.min()<0 or ix.max()>=len(dates): raise ValueError('memory timestamp out of bounds')
    return dates,starts,nday
def precompute(cell,method,smoke=False):
    out=HERE/('smoke/pools' if smoke else 'results/pools')/cell/method
    if (out/'COMPLETE.json').exists(): return out
    if method=='baseline_cosine' and not smoke and str(Path(sys.executable).absolute())!='/usr/bin/python3':
        # Reproduce canonical numerical execution, without changing training runtime.
        log=HERE/'logs'/f'{cell}_legacy_baseline_precompute.log'
        with log.open('a') as f:
            subprocess.check_call(['/usr/bin/python3','-u',str(HERE/'scripts/precompute_legacy_baseline.py'),cell],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
        assert (out/'COMPLETE.json').exists()
        return out
    exp,args,model=build_model(common(cell),torch.device('cuda')); del model
    exp.memory_x=exp.memory_x.to('cuda'); L=args.seq_len; C=args.enc_in
    dates,starts,steps=stamps_for(exp,args)
    kt=time_features(dates.iloc[starts+L//2],'cuda')
    features={}
    torch.cuda.synchronize(); t0=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    if method!='baseline_cosine':
        for c in range(C):
            x=exp.memory_x[:,:,c]
            features[c]=F.normalize(paa16(x-x[:,-1:]) if method=='paa16_cosine' else x-x.mean(-1,keepdim=True),dim=-1)
    torch.cuda.synchronize(); feature_sec=time.perf_counter()-t0
    info=[]
    for split in ('train','val','test'):
        ds,loader=exp._get_data(flag=split,shuffle=False)
        # All query global starts are provided by the same canonical cached pool.
        old=torch.load(oldpool(cell)/f'{split}.pt',map_location='cpu')
        expected=old['query_start_idx'].long(); mid=expected+L//2
        if int(mid.min())<0 or int(mid.max())>=len(dates): raise ValueError('query timestamp out of bounds')
        qtall=time_features(dates.iloc[mid.numpy()],'cuda')
        # Official full-split global maximum, streamed without changing its definition.
        norm=1.0
        if method=='saraf_time_aligned':
            norm=max(float(temporal_raw(qtall[i:i+32],kt,L,steps).max()) for i in range(0,len(qtall),32))
            norm=max(norm,1e-8)
        rows=[]; queries=[]; offset=0; torch.cuda.synchronize(); t1=time.perf_counter()
        for bi,(x,y,qs) in enumerate(loader):
            if smoke and bi>=1: break
            assert torch.equal(qs.cpu(),expected[offset:offset+len(qs)])
            x=x.float().cuda(); mask,_=exp._candidate_mask(qs); ids=[]
            tb=temporal_raw(qtall[offset:offset+len(qs)],kt,L,steps)/norm if method=='saraf_time_aligned' else None
            for c in range(C):
                if method=='baseline_cosine': score=baseline_scores(x,exp.memory_x,c)
                else:
                    q=x[:,:,c]
                    q=paa16(q-q[:,-1:]) if method=='paa16_cosine' else q-q.mean(-1,keepdim=True)
                    score=F.normalize(q,dim=-1)@features[c].T
                    if tb is not None: score=.9*score+.1*tb
                pick=select_scores(score,mask)
                if method=='baseline_cosine': assert torch.equal(pick.cpu(),old['pool_idx_global'][offset:offset+len(qs),c,:].long()), 'baseline IDs differ'
                ids.append(pick.cpu())
            rows.append(torch.stack(ids,1)); queries.append(qs.cpu()); offset+=len(qs)
        torch.cuda.synchronize()
        meta=dict(old['meta']); meta.update(candidate_pool_metric=METRICS[method],preselection_method=method,seed=0,time_alignment='input_midpoint',sampling_steps_per_day=steps,time_bonus_global_split_max=norm)
        out.mkdir(parents=True,exist_ok=True)
        torch.save(dict(query_start_idx=torch.cat(queries),pool_idx_global=torch.cat(rows).int(),meta=meta),out/f'{split}.pt')
        info.append(dict(split=split,queries=offset,search_seconds=time.perf_counter()-t1,normalization_scope='entire split query x train memory timestamps',time_bonus_global_max=norm,sha256=sha(out/f'{split}.pt')))
    dump(out/'COMPLETE.json',dict(cell=cell,method=method,smoke=smoke,feature_seconds=feature_sec,splits=info,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),concurrent_gpu=True))
    return out

def invoke(module,method,argv):
    """Reuse exact verified source; adapter ONLY validates the named pool metric."""
    m=importlib.import_module('scripts.'+module)
    class MethodPool(CandidatePoolCache):
        def __init__(self,path,expected):
            actual=torch.load(path,map_location='cpu')['meta']
            assert actual.get('candidate_pool_metric')==METRICS[method], 'wrong method cache'
            if method!='baseline_cosine': assert actual.get('preselection_method')==method
            required=dict(expected,candidate_pool_metric=METRICS[method])
            super().__init__(path,required)
    if hasattr(m,'CandidatePoolCache'): m.CandidatePoolCache=MethodPool
    if module!='train_v_sharedtop100_01' and hasattr(m,'build_model'):
        original_build=m.build_model
        def evaluation_build(cli,device):
            cli.batch_size=int(os.environ.get('P100_EVAL_BATCH','32'))
            return original_build(cli,device)
        m.build_model=evaluation_build
    # compute_scores does not use CandidatePoolConfig.metric; it consumes IDs already validated above.
    sys.argv=[module]+argv
    original_save=torch.save
    pool_path=Path(argv[argv.index('--candidate_pool_cache')+1]) if '--candidate_pool_cache' in argv else None
    pool_hashes={s:sha(pool_path/f'{s}.pt') for s in ('train','val','test')} if pool_path else None
    def provenance_save(obj,path,*aa,**kk):
        if isinstance(obj,dict) and 'config' in obj and 'model_state_dict' in obj:
            obj=dict(obj);obj['config']=dict(obj['config'],preselection_method=method,candidate_pool_metric=METRICS[method],candidate_pool_sha256=pool_hashes)
        return original_save(obj,path,*aa,**kk)
    torch.save=provenance_save
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t=time.perf_counter()
    try:m.main()
    finally:torch.save=original_save
    if '--out_dir' in argv:
        output=Path(argv[argv.index('--out_dir')+1])
        if '--cell' in argv and '--num_query_views' in argv:
            output=output/argv[argv.index('--cell')+1]/('V'+argv[argv.index('--num_query_views')+1])
        cp=output/'config.json'
        if cp.exists():
            config=json.loads(cp.read_text());config.update(preselection_method=method,candidate_pool_metric=METRICS[method],candidate_pool_sha256=pool_hashes);dump(cp,config)
    torch.cuda.synchronize()
    return dict(seconds=time.perf_counter()-t,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),concurrent_gpu=True)

def call(module,method,argv,log):
    command=[sys.executable,str(HERE/'pipeline.py'),'invoke','--module',module,'--method',method,'--']+list(map(str,argv))
    Path(log).parent.mkdir(parents=True,exist_ok=True)
    for evalbatch in (32,16,8,4):
        env=dict(os.environ,P100_EVAL_BATCH=str(evalbatch))
        with open(log,'a') as f:
            f.write('\nCOMMAND '+json.dumps(command)+' evaluation_batch='+str(evalbatch)+'\n'); f.flush()
            p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,env=env)
        if not p.returncode:return
        oom='out of memory' in Path(log).read_text().lower()
        if module=='train_v_sharedtop100_01' or not oom:break
        print('EVALUATION_OOM_RETRY',module,'next evaluation batch',evalbatch//2,flush=True)
    raise RuntimeError(f'{module} failed {p.returncode}; {log}; no unequal training batch retry permitted')

def args_for(cell):
    c=CELLS[cell]; return ['--reference_ckpt',c['reference_ckpt'],'--pred_len',c['pred_len'],'--seq_len',c['seq_len']]
def cache_build(cell,arm,method,pool,ckpt,out,log):
    a=args_for(cell)+['--num_query_views',arm,'--seed',0,'--candidate_pool_cache',pool,'--retriever_checkpoint',ckpt,'--out_dir',out]
    if arm==0:
        call('build_retrieval_cache_pool01',method,a+['--candidate_pool_mode','coarse_topk','--candidate_pool_size',100],log)
    else: call('build_v_meanmix_cache_pool01',method,a+['--tau_s',.1],log)

def verify_stage2(cell,cache,metrics):
    c=CELLS[cell]; exp,args=build_experiment(c['reference_ckpt'],dict(pred_len=c['pred_len'],seq_len=c['seq_len'],batch_size=32,seed=0))
    base=BaseForecastHead(seq_len=c['seq_len'],pred_len=c['pred_len'],channels=args.enc_in,mode='per_channel_linear').cuda()
    base.load_state_dict(torch.load(c['base_checkpoint'],map_location='cuda')['model_state_dict']); base.eval()
    _,B,R,Y=load_tensors(exp,base,cache,'test',torch.device('cuda'),args.enc_in)
    m=json.loads(Path(metrics).read_text()); fused=B+m['lambda']*(R-B)
    mse=float(((fused-Y)**2).mean()); mae=float((fused-Y).abs().mean())
    assert abs(mse-m['test_mse'])<1e-5 and abs(mae-m['test_mae'])<1e-5
    assert torch.allclose(B+0*(R-B),B) and torch.allclose(B+(R-B),R,atol=1e-6)
    return dict(mse=mse,mae=mae,lambda_value=m['lambda'],base_mse=float(((B-Y)**2).mean()),base_sha256=sha(c['base_checkpoint']),passed=True)

def reproduce(cell):
    out=HERE/'results/reproduction'/cell
    if (out/'PASS.json').exists(): return
    checks=[]
    for arm in (0,5):
        cache=out/f'V{arm}/cache'
        cache_build(cell,arm,'baseline_cosine',oldpool(cell),selected(cell,arm),cache,HERE/f'logs/repro_{cell}_V{arm}.log')
        fresh=json.loads((cache/'stage1_metrics.json').read_text()); original=json.loads((historical(cell,arm)/'cache/stage1_metrics.json').read_text())
        for k in ('retmse10','agg_mse10','recall10','ndcg10','D','C'):
            assert abs(float(fresh[k])-float(original[k]))<1e-5,(cell,arm,k,fresh[k],original[k])
        check=verify_stage2(cell,cache,historical(cell,arm)/'stage2_lambda_original/metrics.json')
        check.update(arm=arm,stage1=fresh,checkpoint=selected(cell,arm),checkpoint_sha256=sha(selected(cell,arm)))
        checks.append(check)
    dump(out/'PASS.json',dict(cell=cell,checks=checks,source='fresh Stage1 cache + existing saved lambda forward',seed=0))

def approximate_set_oracle(values,target,k=10):
    # Greedy without replacement, equal-weight partial-set MSE; no exact-optimum claim.
    B,N,H=values.shape; chosen=torch.zeros(B,N,dtype=torch.bool,device=values.device); total=torch.zeros(B,H,device=values.device)
    for j in range(k):
        score=(((total[:,None,:]+values)/(j+1)-target[:,None,:])**2).mean(-1).masked_fill(chosen,float('inf'))
        idx=stable_topk_indices(score,1,largest=False); chosen.scatter_(1,idx,True)
        total+=values.gather(1,idx[:,:,None].expand(-1,-1,H)).squeeze(1)
    return ((total/k-target)**2).mean(-1)

@torch.no_grad()
def quality(cell):
    out=HERE/'results/quality'/cell
    if (out/'metrics.json').exists(): return
    exp,args,model=build_model(common(cell),torch.device('cuda')); model.eval()
    fullpath=fullselected(cell); bl=torch.load(fullpath,map_location='cuda'); model.load_state_dict(bl['model_state_dict'])
    heads=SlotHeads(args.d_model,n_slots=5).cuda(); heads.load_state_dict(bl['slot_heads_state_dict']); heads.eval()
    _,loader=exp._get_data(flag='val',shuffle=False)
    # Fixed evenly spaced 32 queries, identical all methods, retained in output.
    xs=[];ys=[];ss=[]
    for x,y,s in loader: xs.append(x);ys.append(y);ss.append(s)
    X,Y,S=torch.cat(xs),torch.cat(ys),torch.cat(ss); take=torch.linspace(0,len(S)-1,min(32,len(S))).round().long().unique(); X,Y,S=X[take],Y[take],S[take]
    pools={}
    for m in METHODS:
        expected=dict(seq_len=args.seq_len,pred_len=args.pred_len,candidate_mask='raft',candidate_pool_size=100,candidate_pool_metric=METRICS[m])
        pools[m]=CandidatePoolCache(HERE/'results/pools'/cell/m/'val.pt',expected)
    vals={m:[] for m in METHODS}
    for c in range(args.enc_in):
        bank=torch.cat([encode_raw(model,exp.memory_x[i:i+1024],c) for i in range(0,len(exp.memory_x),1024)])
        bank=F.normalize(bank,dim=-1)
        for i in range(0,len(S),4):
            x,y,s=X[i:i+4].cuda(),Y[i:i+4].cuda(),S[i:i+4]; mask,_=exp._candidate_mask(s)
            scores=torch.einsum('bsd,nd->bsn',heads(encode_raw(model,x,c)),bank)
            mix,ph,pm=mean_mixture_topk_selection(scores,mask,.1,10)
            hid=stable_topk_indices(ph,10,largest=True)
            mem,off=memory_value(args,x,exp.memory_y,exp.memory_x_last,c); d=-individual_utility_memsafe(mem,off,y[:,:,c],1024)
            oracle=stable_topk_indices(d.masked_fill(~mask,float('inf')),10,largest=False)
            baseline=pools['baseline_cosine'].lookup(s,c,torch.device('cuda'))
            for m in METHODS:
                p=pools[m].lookup(s,c,torch.device('cuda')); v=gather_candidate_values(mem,p)+off[:,None,None]
                dp=d.gather(1,p); top=stable_topk_indices(dp,10,largest=False)
                retained=lambda ids: (ids.unsqueeze(-1)==p[:,None,:]).any(-1).float().mean(-1)
                rec=dict(oracle_coverage100=float(retained(oracle).mean()),mixture_retention100=float(retained(mix).mean()),best_individual_mse=float(dp.min(-1).values.mean()),individual_oracle_top10_mse=float(dp.gather(1,top).mean()),approximate_greedy_set_oracle_aggmse=float(approximate_set_oracle(v,y[:,:,c]).mean()),baseline_overlap100=float((baseline[:,:,None]==p[:,None,:]).any(-1).float().mean()))
                for h in range(5): rec[f'head{h+1}_retention100']=float(retained(hid[:,h]).mean())
                rec['n']=len(s); vals[m].append(rec)
    summary={}
    for m,rr in vals.items(): summary[m]={k:sum(r[k]*r['n'] for r in rr)/sum(r['n'] for r in rr) for k in rr[0] if k!='n'}
    dump(out/'metrics.json',dict(cell=cell,split='val',seed=0,query_global_starts=S.tolist(),full_checkpoint=fullpath,full_checkpoint_sha256=sha(fullpath),oracle='diagnostic only; approximate greedy distinct Top10',methods=summary))

def actual_method_batch(cell,method,pool):
    """Real gradient smoke on the same first chronological train batch for both architectures."""
    from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
    from scripts.train_k_multislot_predictive_retrieval01 import kl_loss_from_prob
    rows=[]
    for arm in (0,5):
        exp,args,model=build_model(common(cell),torch.device('cuda'))
        _,loader=exp._get_data(flag='train',shuffle=False);x,y,s=next(iter(loader));x=x.float().cuda();y=y.float().cuda()
        heads=SlotHeads(args.d_model,n_slots=arm).cuda() if arm else None
        cache=CandidatePoolCache(pool/'train.pt',dict(seq_len=args.seq_len,pred_len=args.pred_len,candidate_mask='raft',candidate_pool_size=100,candidate_pool_metric=METRICS[method]))
        mask,_=exp._candidate_mask(s);before=state_hash(model)
        scores,valid,ids=compute_scores(CandidatePoolConfig(mode='coarse_topk',size=100),model,heads,x,exp,0,mask,cache,s,torch.device('cuda'))
        validate_ids(ids,mask)
        mem,off=memory_value(args,x,exp.memory_y,exp.memory_x_last,0);dp=pooled_future_mse(gather_candidate_values(mem,ids),off,y[:,:,0])
        teacher=normalized_teacher_prob(dp,valid,.1); _,ph,pm=mean_mixture_topk_selection(scores,valid,.1,10)
        loss=kl_loss_from_prob(teacher,pm,valid);assert bool(torch.isfinite(loss));loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(bool(torch.isfinite(g).all()) for g in grads) and sum(float(g.abs().sum()) for g in grads)>0
        assert state_hash(model)==before, 'gradient-only smoke must preserve model parameters'
        rows.append(dict(arm=arm,loss=float(loss),init_state_hash=before,query_starts=s.tolist(),gradient_finite=True,selected_ids_valid=True))
    assert rows[0]['init_state_hash']==rows[1]['init_state_hash']
    dump(HERE/f'smoke/{cell}_{method}_gradient.json',dict(rows=rows,method=method,status='PASS'))

@torch.no_grad()
def head_report(cell,method,pool,ckpt,out):
    from scripts.train_patch_retrieval_expert01 import recall_at_k
    exp,args,model=build_model(common(cell),torch.device('cuda'));bl=torch.load(ckpt,map_location='cuda');model.load_state_dict(bl['model_state_dict']);model.eval()
    heads=SlotHeads(args.d_model,n_slots=5).cuda();heads.load_state_dict(bl['slot_heads_state_dict']);heads.eval()
    cache=CandidatePoolCache(pool/'test.pt',dict(seq_len=args.seq_len,pred_len=args.pred_len,candidate_mask='raft',candidate_pool_size=100,candidate_pool_metric=METRICS[method]))
    _,loader=exp._get_data(flag='test',shuffle=False);n=0;ret=torch.zeros(5);agg=ret.clone();mae=ret.clone();rec=ret.clone();cos=torch.zeros(5,5);ov=cos.clone();mixsum=0.;nearzero=0
    for x,y,s in loader:
        x=x.float().cuda();y=y.float().cuda();mask,_=exp._candidate_mask(s)
        for c in range(args.enc_in):
            scores,valid,ids=compute_scores(CandidatePoolConfig(mode='coarse_topk',size=100),model,heads,x,exp,c,mask,cache,s,torch.device('cuda'))
            mix,ph,pm=mean_mixture_topk_selection(scores,valid,.1,10);hi=stable_topk_indices(ph,10,largest=True)
            mem,off=memory_value(args,x,exp.memory_y,exp.memory_x_last,c);values=gather_candidate_values(mem,ids)+off[:,None,None]
            d=pooled_future_mse(gather_candidate_values(mem,ids),off,y[:,:,c]);oracle=stable_topk_indices(d,10,largest=False);errors=[]
            for h in range(5):
                ix=hi[:,h];pred=values.gather(1,ix[:,:,None].expand(-1,-1,args.pred_len)).mean(1);e=pred-y[:,:,c];errors.append(e)
                ret[h]+=d.gather(1,ix).mean(-1).sum().cpu();agg[h]+=(e**2).mean(-1).sum().cpu();mae[h]+=e.abs().mean(-1).sum().cpu();rec[h]+=recall_at_k(ix,oracle,10).sum().cpu()
            E=torch.stack(errors,1);dot=E@E.transpose(1,2);norm=E.norm(dim=-1);den=norm[:,:,None]*norm[:,None,:];nearzero+=int((den<=1e-12).sum());cs=torch.where(den<=1e-12,torch.zeros_like(dot),dot/den.clamp_min(1e-12));cos+=cs.sum(0).cpu()
            one=torch.zeros(len(x),5,100,device=x.device);one.scatter_(2,hi,1);ov+=(one@one.transpose(1,2)).sum(0).cpu()/10
            mp=values.gather(1,mix[:,:,None].expand(-1,-1,args.pred_len)).mean(1);mixsum+=float(((mp-y[:,:,c])**2).mean(-1).sum());n+=len(x)
    dump(out/'head_metrics_test.json',dict(cell=cell,method=method,split='test',seed=0,checkpoint_sha256=sha(ckpt),n_query_channel=n,head_retMSE=(ret/n).tolist(),head_aggMSE=(agg/n).tolist(),head_aggMAE=(mae/n).tolist(),head_recall10_restricted_pool=(rec/n).tolist(),head_top10_overlap=(ov/n).tolist(),head_error_cosine=(cos/n).tolist(),nearzero_norm_pair_count=nearzero,zero_rule='cosine0 when norm product <=1e-12',mixture_aggMSE=mixsum/n))

def provenance(cell,arm,method,ckpt,pool):
    bl=torch.load(ckpt,map_location='cpu')
    pool_hashes={s:sha(pool/f'{s}.pt') for s in ('train','val','test')}
    assert bl['config']['preselection_method']==method
    assert bl['config']['candidate_pool_metric']==METRICS[method]
    assert bl['config']['candidate_pool_sha256']==pool_hashes
    assert bl['config']['init_seed']==0 and bl['config']['loader_seed']==0
    return dict(cell=cell,seed=0,model=f'V{arm}',method=method,checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt),selected_epoch=bl['epoch'],pool_sha256=pool_hashes,git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),gpu_uuid=GPU,physical_gpu_index=1,training_batch_size=32,training_epochs=10,tau_s=.1,tau_t=.1,learning_rate=.001,shared_source_sha256=sha(ROOT/'scripts/train_v_sharedtop100_01.py'),note='Adapter changes only pool metadata verification and saved provenance; no model/loss/optimizer change.')

def summarize():
    rows=[]
    for cell in ORDER:
        for method in METHODS:
            for arm in (0,5):
                p=HERE/'results/runs'/cell/method/f'V{arm}'
                status='COMPLETED' if (p/'COMPLETE.json').exists() else ('RUNNING_OR_INTERRUPTED' if (p/'STARTED.json').exists() else 'NOT_RUN')
                r=dict(cell=cell,model=f'V{arm}',preselection=method,status=status)
                if status=='COMPLETED':
                    r.update(json.loads((p/'stage2/metrics.json').read_text())); s=json.loads((p/'cache/stage1_metrics.json').read_text()); r.update({k:s[k] for k in ('retmse10','agg_mse10','recall10')})
                rows.append(r)
    keys=sorted(set().union(*(r.keys() for r in rows))); (HERE/'reports').mkdir(exist_ok=True)
    with open(HERE/'reports/combined_results.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    dump(HERE/'reports/status.json',rows)
    completed=[r for r in rows if r['status']=='COMPLETED']
    lines=['# P100 Preselection 비교 진행 보고','주지표: Stage2 test MSE/MAE. seed0만 실행. 동시 GPU 경합 환경의 시간이며 독립 속도 비교 아님.','|Cell|Model|Preselection|Status|Stage2 MSE|Stage2 MAE|','|---|---|---|---|---|---|']
    for r in rows: lines.append('|'+ '|'.join(str(r.get(k,'—')) for k in ('cell','model','preselection','status','test_mse','test_mae'))+'|')
    lines+=['','미실행 수치를 추정하지 않음. Oracle은 preselection 입력이 아닌 validation 진단. 최적 모델 선택 및 hyperparameter tuning에 test 사용 금지.','학습/선택/캐시/Stage2의 명령과 로그는 logs/, 개별 provenance는 results/runs/ 아래 보존.']
    (HERE/'reports/FINAL_REPORT.md').write_text('\n'.join(lines))

def run():
    import fcntl
    lock=open(HERE/'pipeline.lock','w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU: raise RuntimeError('GPU UUID restriction missing')
    for cell in ORDER:
        dump(HERE/'reports/current.json',dict(cell=cell,phase='reproduction',pid=os.getpid(),seed=0))
        reproduce(cell)
        for method in METHODS: precompute(cell,method)
        for method in METHODS:actual_method_batch(cell,method,HERE/'results/pools'/cell/method)
        dump(HERE/'reports/current.json',dict(cell=cell,phase='quality',pid=os.getpid(),seed=0)); quality(cell)
        for method in METHODS:
            pool=HERE/'results/pools'/cell/method
            for arm in (0,5):
                out=HERE/'results/runs'/cell/method/f'V{arm}'
                if (out/'COMPLETE.json').exists(): continue
                dump(out/'STARTED.json',dict(pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())));summarize()
                dump(HERE/'reports/current.json',dict(cell=cell,method=method,model=f'V{arm}',phase='training',pid=os.getpid(),seed=0))
                log=HERE/f'logs/{cell}_{method}_V{arm}.log'; trainroot=out/'training'; ckroot=HERE/'checkpoints'/cell/method/f'V{arm}'
                call('train_v_sharedtop100_01',method,args_for(cell)+['--num_query_views',arm,'--cell',cell,'--candidate_pool_cache',pool,'--train_epochs',10,'--disable_early_stopping','--batch_size',32,'--out_dir',trainroot,'--checkpoints',ckroot],log)
                ckdir=ckroot/cell/f'V{arm}'
                if arm==5:
                    call('eval_v5_meanmix_checkpoint_selection_pool01',method,args_for(cell)+['--ckpt_dir',ckdir,'--epochs',','.join(map(str,range(1,11))),'--num_query_views',5,'--seed',0,'--tau_s',.1,'--candidate_pool_cache',pool,'--out_dir',out/'selection'],log)
                    ckpt=Path(json.loads((out/'selection/selected_checkpoint.json').read_text())['selected_checkpoint_path'])
                else: ckpt=ckdir/'checkpoint_best_retmse.pth'
                manifest=provenance(cell,arm,method,ckpt,pool); dump(out/'PROVENANCE.json',manifest)
                conf=json.loads((trainroot/cell/f'V{arm}'/'config.json').read_text())
                initfile=HERE/'configs'/f'{cell}_controlled_init.json'
                if initfile.exists():assert json.loads(initfile.read_text())['init_hash']==conf['init_hash']
                else:dump(initfile,dict(init_hash=conf['init_hash'],cell=cell,seed=0))
                cache_build(cell,arm,method,pool,ckpt,out/'cache',log)
                call('train_r_stage2_lambda01',method,args_for(cell)+['--seed',0,'--base_checkpoint',CELLS[cell]['base_checkpoint'],'--cache_dir',out/'cache','--out_dir',out/'stage2'],log)
                check=verify_stage2(cell,out/'cache',out/'stage2/metrics.json'); dump(out/'COMPLETE.json',dict(status='COMPLETED',independent_stage2=check,provenance=manifest));summarize()
                if arm==5:head_report(cell,method,pool,ckpt,out)
        print('CELL_COMPLETED',cell,flush=True)
    dump(HERE/'reports/current.json',dict(status='COMPLETED',seed=0));summarize()

def smoke():
    # Actual one batch canonical training smoke is executed only after CPU tests pass.
    cell='ETTh1_96'; pool=oldpool(cell)
    for arm in (0,5):
        call('train_v_sharedtop100_01','baseline_cosine',args_for(cell)+['--num_query_views',arm,'--cell',cell,'--candidate_pool_cache',pool,'--smoke_test','--out_dir',HERE/'smoke/trainer','--checkpoints',HERE/'smoke/checkpoints'],HERE/f'logs/smoke_V{arm}.log')
    for method in METHODS: precompute(cell,method,smoke=True)
    dump(HERE/'smoke/REAL_BATCH_PASS.json',dict(models=[0,5],methods=METHODS,seed=0,gpu_uuid=GPU))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['invoke','smoke','run','reproduce','quality','precompute']);p.add_argument('--module');p.add_argument('--method',default='baseline_cosine');p.add_argument('--cell',default='ETTh1_96');a,rest=p.parse_known_args()
    if rest and rest[0]=='--':rest=rest[1:]
    os.chdir(ROOT)
    try:
        if a.action=='invoke':
            resource=invoke(a.module,a.method,rest)
            # Sidecar follows each subprocess log, stdout always identifies exact module/method.
            print('RESOURCE_METRICS '+json.dumps(dict(module=a.module,method=a.method,**resource)),flush=True)
        elif a.action=='smoke':smoke()
        elif a.action=='run':run()
        elif a.action=='reproduce':reproduce(a.cell)
        elif a.action=='quality':quality(a.cell)
        elif a.action=='precompute':precompute(a.cell,a.method)
    except Exception as e:
        if a.action=='run':dump(HERE/'reports/FAILED.json',dict(error=repr(e),traceback=traceback.format_exc(),time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),status='FAILED'))
        raise
