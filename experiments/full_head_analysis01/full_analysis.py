#!/usr/bin/env python3
"""Frozen FULL V5 diagnostics; standalone files, no legacy source mutation."""
import argparse, csv, hashlib, itertools, json, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.RelationStage1 import stable_topk_indices
from models.RelationStage2 import BaseForecastHead
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_factorial_e2e01 import individual_utility_memsafe
from scripts.train_patch_retrieval_expert01 import recall_at_k, ndcg_at_k
from utils.full_candidate_bank import encode_raw_channel_first, compute_scores_full_grad_channel_first
from utils.mean_mixture_selection import mean_mixture_topk_selection
from scripts.verify_v5_head_loss_stage2_correctness01 import CELLS
from scripts.train_r_stage2_lambda01 import GlobalLambdaGate, load_tensors, eval_gate

GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
OUT=ROOT/'results/TRACK-V5-FULL-HEAD-ANALYSIS01'
SUBSETS=[s for n in range(1,6) for s in itertools.combinations(range(5),n)]
NAME=lambda s: ''.join(str(h+1) for h in s)
PRIMARY=[(h,) for h in range(5)]+[tuple(range(5))]+[tuple(h for h in range(5) if h!=j) for j in range(5)]

def dump(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def csv_write(path,rows):
    if not rows:return
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def selection(p,subset,mask,k=10):
    # Preserve canonical tensor layout/reduction for All5; advanced indexing
    # changes float32 accumulation layout and can flip near-tied candidates.
    selected=p if tuple(subset)==tuple(range(p.shape[1])) else p[:,list(subset)]
    prob=selected.mean(1).masked_fill(~mask,float('-inf'))
    return stable_topk_indices(prob,k,largest=True)
def independent_raft_mask(query_starts,memory_starts,seq_len,pred_len):
    q=np.asarray(query_starts)[:,None];m=np.asarray(memory_starts)[None,:]
    belongs=(q==m).any(-1)
    return (~belongs[:,None]) | (np.abs(m-q)>=seq_len+pred_len)
def decomp(y,truth):
    e=y-truth[:,None];agg=(y.mean(1)-truth).square().mean(-1)
    d=e.square().mean(-1).mean(-1)/y.size(1)
    dots=torch.einsum('bih,bjh->bij',e.double(),e.double())/e.size(-1)
    c=(dots.sum((-1,-2))-dots.diagonal(dim1=-2,dim2=-1).sum(-1))/y.size(1)**2
    return agg,d,c.to(agg.dtype)
def error_pairs(errors):
    errors=errors.double()
    dot=torch.einsum('bih,bjh->bij',errors.double(),errors.double()).to(errors.dtype)
    norm=errors.norm(dim=-1);den=norm[:,:,None]*norm[:,None,:]
    cos=torch.where(den>1e-12,dot/den.clamp_min(1e-12),torch.zeros_like(dot))
    return dot,norm,cos,den<=1e-12

def setup(cell,batch=16):
    cfg=dict(CELLS[cell]);ds,h=cell.split('_');h=int(h)
    canonical=ROOT/f'results/TRACK-V-MEANMIX-CHECKPOINT-CORRECTION01/{ds}/H{h}/seed0'
    selected=json.loads((canonical/'checkpoint_selection/selected_checkpoint.json').read_text())
    assert 'pool_top100' not in selected['selected_checkpoint_path']
    overrides=dict(pred_len=h,seq_len=h,batch_size=batch,seed=0,top_k=10,relation_encoder_type='mlp',
        relation_self_fill='linear',relation_input_space='delta_last',relation_teacher_space='delta_last',
        relation_value_space='delta_last',candidate_mask='raft',patch_len=16,stride=16)
    exp,args=build_experiment(cfg['reference_ckpt'],overrides);exp._ensure_memory()
    model=exp.model.module if hasattr(exp.model,'module') else exp.model
    model.to('cuda').eval();bl=torch.load(selected['selected_checkpoint_path'],map_location='cuda')
    model.load_state_dict(bl['model_state_dict']);heads=SlotHeads(int(args.d_model),n_slots=5).cuda().eval()
    heads.load_state_dict(bl['slot_heads_state_dict'])
    for p in list(model.parameters())+list(heads.parameters()):p.requires_grad_(False)
    base=BaseForecastHead(seq_len=h,pred_len=h,channels=int(args.enc_in),mode='per_channel_linear').cuda().eval()
    base.load_state_dict(torch.load(cfg['base_checkpoint'],map_location='cuda')['model_state_dict'])
    for p in base.parameters():p.requires_grad_(False)
    return cfg,canonical,selected,exp,args,model,heads,base

@torch.no_grad()
def bank(model,memory,c,chunk):
    if chunk==0:
        return F.normalize(encode_raw_channel_first(model,memory,c),dim=-1)
    return F.normalize(torch.cat([encode_raw_channel_first(model,memory[i:i+chunk],c) for i in range(0,len(memory),chunk)]),dim=-1)

@torch.no_grad()
def smoke(cell,batch,chunk):
    torch.backends.cuda.matmul.allow_tf32=True
    cfg,canonical,sel,exp,args,model,heads,base=setup(cell,batch)
    _,loader=exp._get_data(flag='val',shuffle=False);x,y,start=next(iter(loader));x=x.float().cuda();y=y.float().cuda()
    mask,_=exp._candidate_mask(start);k=bank(model,exp.memory_x,0,chunk)
    independent=independent_raft_mask(start.numpy(),exp.memory_sampler.starts,args.seq_len,args.pred_len)
    assert np.array_equal(mask.cpu().numpy(),independent)
    trainset,_=exp._get_data(flag='train',shuffle=False)
    assert np.array_equal(exp.memory_sampler.starts,trainset.starts),'memory must be train split only'
    scores=torch.einsum('bsd,nd->bsn',heads(encode_raw_channel_first(model,x,0)),k)
    old=compute_scores_full_grad_channel_first(model,heads,x,exp.memory_x,0)
    assert torch.allclose(scores,old,atol=2e-6,rtol=1e-5),(scores-old).abs().max()
    picks,p,_=mean_mixture_topk_selection(scores,mask,sel['tau_s'],10)
    oldpick,_,_=mean_mixture_topk_selection(old,mask,sel['tau_s'],10)
    assert torch.equal(picks,oldpick),'cached/chunked bank selection differs'
    assert torch.equal(picks,selection(p,tuple(range(5)),mask))
    assert mask.gather(1,picks).all();assert (picks.sort().values.diff(dim=-1)>0).all()
    mem,offset=memory_value(args,x,exp.memory_y,exp.memory_x_last,0)
    vals=mem[picks]+offset[:,None,None];agg,d,c=decomp(vals,y[:,:,0])
    assert torch.allclose(agg,d+c,atol=1e-5)
    result=dict(status='PASS',cell=cell,n_queries=len(x),candidate_count=len(exp.memory_x),
       max_cached_score_difference=float((scores-old).abs().max()),top10_exact=True,full_support=True,
       seed=0,gpu_uuid=GPU,batch=batch,chunk=chunk,canonical_checkpoint=sel['selected_checkpoint_path'],independent_raft_mask=True,train_only_memory=True)
    dump(OUT/cell/'real_one_batch_smoke.json',result);print('[SMOKE]',result,flush=True)

@torch.no_grad()
def pass_split(cell,split,parts,exp,args,model,heads,tau,batch,chunk,out,details=False,save_cache=False):
    torch.backends.cuda.matmul.allow_tf32=True
    dataset,loader=exp._get_data(flag=split,shuffle=False);n=len(dataset);C=int(args.enc_in);H=int(args.pred_len)
    accum={NAME(s):dict(retmse10=0.,agg_mse10=0.,agg_mae10=0.,recall10=0.,ndcg10=0.,D=0.,C=0.) for s in parts}
    caches={};starts=np.zeros(n,dtype=np.int64)
    if save_cache:
        for s in parts:
            name=NAME(s)
            if s in PRIMARY:
                p=out/'cache'/name;p.mkdir(parents=True,exist_ok=True)
                caches[name]=np.lib.format.open_memmap(p/f'{split}.npy',mode='w+',dtype='float32',shape=(n,H,C))
    query_metrics=np.zeros((n,C,len(parts),3),dtype=np.float32) if details else None
    pair_sums={}; union_sums=np.zeros(6);unique_sums=np.zeros(5);mix_union=0.;norm_sums=np.zeros(5)
    for c in range(C):
        keys=bank(model,exp.memory_x,c,chunk);offset_rows=0
        for bi,(x,y,start) in enumerate(loader):
            x=x.float().cuda();truth=y.float().cuda()[:,:,c];b=len(x);mask,counts=exp._candidate_mask(start)
            assert mask.shape[-1]==len(exp.memory_x) and (mask.sum(-1)>=10).all()
            assert np.array_equal(mask.cpu().numpy(),independent_raft_mask(start.numpy(),exp.memory_sampler.starts,args.seq_len,args.pred_len))
            q=heads(encode_raw_channel_first(model,x,c));scores=torch.einsum('bsd,nd->bsn',q,keys)
            mix,probs,_=mean_mixture_topk_selection(scores,mask,tau,10)
            mem,off=memory_value(args,x,exp.memory_y,exp.memory_x_last,c)
            d_raw=-individual_utility_memsafe(mem,off,truth,chunk or 4096)
            oracle=stable_topk_indices(d_raw.masked_fill(~mask,float('inf')),10,largest=False)
            headidx=[];headpred=[]
            for si,s in enumerate(parts):
                name=NAME(s);idx=selection(probs,s,mask)
                assert mask.gather(1,idx).all()
                values=mem[idx]+off[:,None,None];pred=values.mean(1)
                agg,d,cv=decomp(values,truth)
                assert torch.allclose(agg,d+cv,atol=1e-4)
                mae=(pred-truth).abs().mean(-1);ret=d_raw.gather(1,idx).mean(-1)
                recall=recall_at_k(idx,oracle,10);ndcg=ndcg_at_k(idx,d_raw,mask,10)
                vals=dict(retmse10=ret,agg_mse10=agg,agg_mae10=mae,recall10=recall,ndcg10=ndcg,D=d,C=cv)
                for key,v in vals.items():accum[name][key]+=float(v.double().sum())
                if details:query_metrics[offset_rows:offset_rows+b,c,si]=torch.stack([agg,mae,recall],-1).cpu().numpy()
                if name in caches:caches[name][offset_rows:offset_rows+b,:,c]=pred.cpu().numpy()
                if len(s)==1:headidx.append(idx);headpred.append(pred)
            if details:
                E=torch.stack(headpred,1)-truth[:,None];dots,norm,cos,zero=error_pairs(E)
                norm_sums+=norm.double().sum(0).cpu().numpy()
                for i,j in itertools.combinations(range(5),2):
                    ov=(headidx[i][:,:,None]==headidx[j][:,None,:]).any(-1).float().mean(-1)
                    one=(headidx[i][:,0]==headidx[j][:,0]).float()
                    vals=torch.stack([ov,one,dots[:,i,j],cos[:,i,j],(cos[:,i,j]<0).float(),zero[:,i,j].float()],-1)
                    pair_sums.setdefault((i,j),np.zeros(6));pair_sums[(i,j)]+=vals.double().sum(0).cpu().numpy()
                members=torch.stack([torch.zeros(b,len(keys),device='cuda',dtype=torch.bool).scatter_(1,idx,True) for idx in headidx],1)
                union=members.any(1);union_sums[0]+=float(union.sum(1).double().sum())
                for nh in range(1,6):
                    union_sums[nh]+=float(members[:,:nh].any(1).gather(1,oracle).float().mean(1).double().sum())
                for h in range(5):unique_sums[h]+=float((members[:,h] & (members.sum(1)==1)).sum(1).double().sum())
                mix_union+=float(union.gather(1,mix).float().mean(1).double().sum())
            starts[offset_rows:offset_rows+b]=start.numpy();offset_rows+=b
        assert offset_rows==n
        print(f'[PASS] {cell}/{split} channel={c+1}/{C} queries={n}',flush=True)
        del keys
    rows=[dict(cell=cell,split=split,seed=0,subset=name,head_count=len(name),candidate_count=len(exp.memory_x),**{k:v/(n*C) for k,v in vals.items()}) for name,vals in accum.items()]
    dump(out/f'{split}_metrics.json',rows)
    if details:
        np.save(out/f'{split}_subset_query_metrics.npy',query_metrics)
        pairrows=[dict(cell=cell,split=split,head1=i+1,head2=j+1,**dict(zip(['top10_overlap','top1_overlap','raw_dot','error_cosine','negative_cosine_fraction','zero_norm_fraction'],v/(n*C)))) for (i,j),v in pair_sums.items()]
        dump(out/f'{split}_head_pairs.json',pairrows)
        dump(out/f'{split}_union.json',dict(cell=cell,split=split,union_size=union_sums[0]/(n*C),oracle_coverage_by_heads=(union_sums[1:]/(n*C)).tolist(),unique_candidates=(unique_sums/(n*C)).tolist(),mixture_top10_in_head_union=mix_union/(n*C),head_error_norm=(norm_sums/(n*C)).tolist()))
    for name,arr in caches.items():
        arr.flush();torch.save(dict(query_start_idx=torch.from_numpy(starts.copy()),relation_outputs=torch.from_numpy(arr),channels=list(range(C)),pred_len=H,split=split),out/'cache'/name/f'{split}.pt')
    return rows

@torch.no_grad()
def reproduce_stage2(exp,args,base,canonical):
    torch.backends.cuda.matmul.allow_tf32=False
    starts,B,R,Y=load_tensors(exp,base,canonical/'cache','test','cuda',int(args.enc_in))
    gate=GlobalLambdaGate().cuda();bl=torch.load(canonical/'stage2/checkpoint.pth',map_location='cuda');gate.load_state_dict(bl['gate_state_dict'])
    mse,mae,lam=eval_gate(gate,B,R,Y);old=json.loads((canonical/'stage2/metrics.json').read_text())
    assert abs(mse-old['test_mse'])<1e-5 and abs(mae-old['test_mae'])<1e-5,(mse,mae,old)
    assert torch.allclose(B+0*(R-B),B) and torch.allclose(B+(R-B),R,atol=1e-6)
    return dict(status='PASS',mse=mse,mae=mae,lambda_value=lam,recorded=old,n_test=len(Y),stored_gate_sha256=sha(canonical/'stage2/checkpoint.pth'))

@torch.no_grad()
def newcache_coupling(exp,args,base,out,canonical):
    torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    for split in ['train','val','test']:
        new=torch.load(out/'cache/12345'/f'{split}.pt',map_location='cpu');old=torch.load(canonical/'cache'/f'{split}.pt',map_location='cpu')
        assert torch.equal(new['query_start_idx'],old['query_start_idx']),f'{split}: query alignment'
        diff=float((new['relation_outputs']-old['relation_outputs']).abs().max())
        assert torch.allclose(new['relation_outputs'],old['relation_outputs'],atol=1e-5,rtol=1e-5),(split,diff)
        rows.append(dict(split=split,query_ids_exact=True,max_prediction_difference=diff))
        del new,old
    _,B,R,Y=load_tensors(exp,base,out/'cache/12345','test','cuda',int(args.enc_in))
    gate=GlobalLambdaGate().cuda();gate.load_state_dict(torch.load(canonical/'stage2/checkpoint.pth',map_location='cuda')['gate_state_dict'])
    mse,mae,lam=eval_gate(gate,B,R,Y);expected=json.loads((canonical/'stage2/metrics.json').read_text())
    assert abs(mse-expected['test_mse'])<1e-5 and abs(mae-expected['test_mae'])<1e-5
    dump(out/'newcache_stored_lambda_verification.json',dict(status='PASS',cache_comparisons=rows,mse=mse,mae=mae,lambda_value=lam))

@torch.no_grad()
def historical_comparators(cell,cfg,exp,args,base,canonical,out):
    torch.backends.cuda.matmul.allow_tf32=False
    ds,h=cell.split('_');rows=[]
    _,B,_,Y=load_tensors(exp,base,canonical/'cache','test','cuda',int(args.enc_in))
    rows.append(dict(cell=cell,model='Base',status='VERIFIED',source_kind='existing_checkpoint_forward',test_mse=float((B-Y).square().mean()),test_mae=float((B-Y).abs().mean()),base_sha256=sha(cfg['base_checkpoint'])))
    del B,Y
    for arm in ['V0','V1','V2','V5']:
        if arm=='V5':location=canonical;cache=location/'cache'
        elif arm=='V2':location=canonical/'V2';cache=location/'cache'
        else:location=ROOT/f'results/TRACK-V-MEANMIX-INFERENCE01/{ds}/H{h}/seed0/{arm}';cache=location
        metric=location/'stage2/metrics.json'
        if not metric.exists() or not (cache/'test.pt').exists():
            rows.append(dict(cell=cell,model=arm,status='MISSING',source=str(metric)));continue
        starts,B,R,Y=load_tensors(exp,base,cache,'test','cuda',int(args.enc_in));old=json.loads(metric.read_text())
        gate=GlobalLambdaGate().cuda();gate.load_state_dict(torch.load(location/'stage2/checkpoint.pth',map_location='cuda')['gate_state_dict'])
        mse,mae,lam=eval_gate(gate,B,R,Y)
        assert abs(mse-old['test_mse'])<1e-5 and abs(mae-old['test_mae'])<1e-5,(arm,'same base/split historical mismatch')
        rows.append(dict(cell=cell,model=arm,status='VERIFIED',source_kind='reused_historical_metrics_verified_forward_not_new_training',source=str(metric),source_sha256=sha(metric),base_sha256=sha(cfg['base_checkpoint']),seed=0,split='test',n_test=len(Y),test_mse=mse,test_mae=mae,lambda_value=lam,inference='mean_mixture_full',selection_note='V2/V5 corrected validation Mean-Mixture; V0/V1 single-head mixture equals original selection; V0-to-V1 adds projection'))
        del B,R,Y;torch.cuda.empty_cache()
    csv_write(out/'INDEPENDENT_MODELS.csv',rows)

def fit_all(cell,cfg,exp,args,base,out,canonical_lambda):
    torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    for subset in PRIMARY:
        name=NAME(subset);cache=out/'cache'/name;dest=out/'stage2'/name
        command=[sys.executable,'scripts/train_r_stage2_lambda01.py','--reference_ckpt',cfg['reference_ckpt'],'--pred_len',str(args.pred_len),'--seq_len',str(args.seq_len),'--seed','0','--base_checkpoint',cfg['base_checkpoint'],'--cache_dir',str(cache),'--out_dir',str(dest)]
        with open(out/f'stage2_{name}.log','w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        m=json.loads((dest/'metrics.json').read_text());starts,B,R,Y=load_tensors(exp,base,cache,'test','cuda',int(args.enc_in))
        fused=B+canonical_lambda*(R-B)
        rows.append(dict(cell=cell,subset=name,seed=0,split='test',primary_mse=m['test_mse'],primary_mae=m['test_mae'],primary_lambda=m['lambda'],secondary_fixed_lambda=canonical_lambda,secondary_mse=float((fused-Y).square().mean()),secondary_mae=float((fused-Y).abs().mean()),checkpoint=str(dest/'checkpoint.pth')))
        del B,R,Y,fused;torch.cuda.empty_cache()
        csv_write(out/'HEAD_STAGE2_MSE_MAE.csv',rows)
        print('[STAGE2]',rows[-1],flush=True)
    return rows

def summarize_cell(cell,out):
    val=json.loads((out/'val_metrics.json').read_text());test=json.loads((out/'test_metrics.json').read_text());lookup={r['subset']:r for r in test};all5=lookup['12345']
    winner=min(val,key=lambda r:r['agg_mse10'])['subset']
    q=np.load(out/'test_subset_query_metrics.npy',mmap_mode='r')
    summary=dict(cell=cell,val_selected_subset=winner,test_selected_subset=lookup[winner],oracle_test_aggmse=float(q[:,:,:,0].min(-1).mean()),oracle_test_aggmae_independent_objective=float(q[:,:,:,1].min(-1).mean()),oracle_not_deployable=True)
    dump(out/'subset_selection.json',summary)
    csv_write(out/'HEAD_QUALITY.csv',[r for r in val+test if len(r['subset'])==1 or r['subset']=='12345'])
    csv_write(out/'HEAD_SUBSET_ANALYSIS.csv',val+test)
    loo=[]
    for split,rs in [('val',val),('test',test)]:
        ref=next(r for r in rs if r['subset']=='12345')
        for r in rs:
            if len(r['subset'])==4:loo.append(dict(r,removed_head=next(h for h in '12345' if h not in r['subset']),delta_aggmse=r['agg_mse10']-ref['agg_mse10'],delta_aggmae=r['agg_mae10']-ref['agg_mae10']))
    csv_write(out/'HEAD_LEAVE_ONE_OUT.csv',loo)
    count=[]
    for k in range(1,6):
        vs=[r for r in val if len(r['subset'])==k];best=min(vs,key=lambda r:r['agg_mse10'])['subset'];ts=[r for r in test if len(r['subset'])==k]
        count.append(dict(cell=cell,head_count=k,mean_test_aggmse=np.mean([r['agg_mse10'] for r in ts]),mean_test_recall=np.mean([r['recall10'] for r in ts]),val_selected_subset=best,test_selected_aggmse=lookup[best]['agg_mse10']))
    csv_write(out/'HEAD_COUNT.csv',count)
    pair=sum([json.loads((out/f'{s}_head_pairs.json').read_text()) for s in ['val','test']],[])
    csv_write(out/'HEAD_OVERLAP.csv',pair);csv_write(out/'HEAD_ERROR_COMPLEMENTARITY.csv',pair)
    csv_write(out/'HEAD_UNION_COVERAGE.csv',[json.loads((out/f'{s}_union.json').read_text()) for s in ['val','test']])

def run_cell(cell,batch,chunk,gate_only=False):
    out=OUT/cell;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'COMPLETE.json').exists(),'already completed; refuse duplicate'
    cfg,canonical,sel,exp,args,model,heads,base=setup(cell,batch)
    torch.cuda.reset_peak_memory_stats();t0=time.time()
    prov=dict(cell=cell,seed=0,candidate_support='FULL',candidate_count=len(exp.memory_x),gpu_uuid=GPU,physical_gpu=1,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),reference=cfg['reference_ckpt'],checkpoint=sel['selected_checkpoint_path'],checkpoint_sha256=sha(sel['selected_checkpoint_path']),base_checkpoint=cfg['base_checkpoint'],base_sha256=sha(cfg['base_checkpoint']),selection=sel,batch=batch,chunk=chunk,timing='concurrent shared GPU; not isolated speed comparison')
    prov.update(python_executable=sys.executable,torch_version=torch.__version__,cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),retrieval_matmul_allow_tf32=True,stage2_parent_forward_matmul_allow_tf32=False,stage2_training_subprocess_default_matmul_allow_tf32=True,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,precision_evidence='legacy_parity_metrics_probe.json: original Python1.12/CUDA11.3 exactly reproduces all test R and stored-lambda metrics; parent forward TF32 False; original Stage2 subprocess uses legacy default True')
    dump(out/'CHECKPOINT_PROVENANCE.json',prov)
    st2=reproduce_stage2(exp,args,base,canonical)
    repro=pass_split(cell,'test',[tuple(range(5))],exp,args,model,heads,sel['tau_s'],batch,chunk,out/'reproduction')
    old=json.loads((canonical/'cache/stage1_metrics.json').read_text())
    for key in ['retmse10','agg_mse10','recall10','ndcg10','D','C']:
        assert abs(repro[0][key]-old[key])<2e-6,(key,repro[0][key],old[key])
    dump(out/'baseline_reproduction.json',dict(status='PASS',stage1=repro[0],stage1_recorded=old,stage2=st2))
    (out/'BASELINE_REPRODUCTION.md').write_text('# Canonical reproduction\nPASS: full test Stage1 and saved lambda forward Stage2; source JSON attached.\n')
    if gate_only:
        print('[CANONICAL GATE PASS]',cell,flush=True)
        return
    for split in ['train','val','test']:
        pass_split(cell,split,PRIMARY if split=='train' else SUBSETS,exp,args,model,heads,sel['tau_s'],batch,chunk,out,details=split!='train',save_cache=True)
    newcache_coupling(exp,args,base,out,canonical)
    stage2=fit_all(cell,cfg,exp,args,base,out,st2['lambda_value']);summarize_cell(cell,out)
    historical_comparators(cell,cfg,exp,args,base,canonical,out)
    dump(out/'COMPLETE.json',dict(status='COMPLETED',cell=cell,stage2_configurations=len(stage2),wall_seconds=time.time()-t0,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
    print('[CELL COMPLETE]',cell,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cell',choices=list(CELLS));ap.add_argument('--smoke',action='store_true');ap.add_argument('--gate-only',action='store_true');ap.add_argument('--batch',type=int,default=32);ap.add_argument('--chunk',type=int,default=0);a=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU,'must use physical GPU1 UUID only'
    actual=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    assert any(line.split(',')[0].strip()=='1' and line.split(',')[1].strip()==GPU for line in actual.splitlines()),'physical GPU UUID mismatch'
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    if a.smoke:smoke(a.cell,a.batch,a.chunk);return
    cells=[a.cell] if a.cell else ['ETTh1_96','ETTh1_720','Weather_96','Weather_720']
    for cell in cells:
        smoke(cell,a.batch,a.chunk);run_cell(cell,a.batch,a.chunk,a.gate_only)
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'FAILURE.txt').write_text(traceback.format_exc());raise
