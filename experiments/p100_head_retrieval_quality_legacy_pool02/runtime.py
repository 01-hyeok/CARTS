"""Isolated Stage1 experiment runtime; canonical model/loss/data functions reused."""
import argparse
import copy
import csv
import datetime as dt
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE)]
from preparation import (GPU_UUID, sha256_file, state_hash, seeded_slot_heads,
                         initialize_common_encoder, make_training_loader,
                         batch_order_hash, validate_pool, require_teacher_resolution)
from data_provider.data_factory import data_provider
from utils.relation_memory import RelationMemorySampler
from models.RelationStage1 import Model, stable_topk_indices
from scripts.train_factorial_e2e01 import encode_raw
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_k_multislot_predictive_retrieval01 import kl_loss_from_prob
from utils.candidate_pool import build_coarse_pool, compute_coarse_delta_last_cosine_scores, pooled_future_mse
from utils.mean_mixture_selection import mean_mixture_topk_selection
from scripts.precompute_candidate_pool01 import precompute_split

OUT = ROOT / 'results/TRACK-P100-HEAD-RETRIEVAL-QUALITY01-LEGACYPOOL02'
CKPTS = ROOT / 'checkpoints/track_p100_head_retrieval_quality01_legacy_pool02'
FIELDS = ['retmse', 'agg_mse', 'agg_mae', 'recall', 'regret', 'last_win', 'last_tie']

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False, default=str) + '\n')
    os.replace(tmp, path)

def atomic_torch(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)

def config():
    cfg = json.loads((HERE/'configs/protocol.json').read_text())
    assert require_teacher_resolution(cfg) == 'zscore_future_mse'
    assert cfg['stage2_enabled'] is False
    return cfg

def gpu_guard():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU_UUID, 'Only physical GPU1 UUID may be exposed'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    text = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'], text=True)
    assert any(line.strip() == f'1, {GPU_UUID}' for line in text.splitlines())
    return dict(physical_index=1, uuid=GPU_UUID, logical_index=0, name=torch.cuda.get_device_name(0))

def source_hashes():
    files = list(HERE.glob('*.py')) + [ROOT/x for x in (
        'models/RelationStage1.py','data_provider/data_loader.py','data_provider/data_factory.py',
        'utils/relation_memory.py','utils/candidate_pool.py','utils/mean_mixture_selection.py',
        'scripts/train_factorial_e2e01.py','scripts/train_horizon_retrieval_expert01.py',
        'scripts/train_k_multislot_predictive_retrieval01.py','scripts/precompute_candidate_pool01.py')]
    return {str(f.relative_to(ROOT)):sha256_file(f) for f in sorted(files)}

def load_args(cell, seed=0):
    cfg = config()
    ref = cfg['references'][cell]
    assert sha256_file(ref['path']) == ref['sha256']
    payload = torch.load(ref['path'], map_location='cpu', weights_only=False)
    args = dict(payload['args'])  # Configuration only: never load reference model weights.
    h = int(cell.rsplit('_',1)[1])
    args.update(pred_len=h, seq_len=h, batch_size=32, seed=seed, top_k=10,
                relation_encoder_type='mlp', relation_self_fill='linear',
                relation_input_space='delta_last', relation_teacher_space='delta_last',
                relation_value_space='delta_last', candidate_mask='raft', patch_len=16, stride=16,
                use_gpu=False, use_multi_gpu=False, num_workers=0)
    assert args['task_name']=='stage1_relation' and args['features']=='M'
    assert args['teacher_mse_space']=='normalized' and args['augmentation_ratio']==0
    assert args['d_model']==128 and args['d_ff']==256 and args['dropout']==0.1
    return SimpleNamespace(**args)

class CellData:
    """Canonical datasets and zero-copy CPU window views; compact train series on GPU."""
    def __init__(self, cell, device='cpu'):
        self.cell, self.args, self.device = cell, load_args(cell), torch.device(device)
        data_info=config()['data'][cell.rsplit('_',1)[0]]
        assert sha256_file(data_info['path'])==data_info['sha256'], 'Pinned dataset changed'
        self.h = self.args.pred_len
        self.datasets = {s:data_provider(self.args,s,shuffle=False)[0] for s in ('train','val','test')}
        self.sampler = RelationMemorySampler(self.datasets['train'],self.h,self.h,'raft')
        self.starts = self.sampler.starts
        assert np.array_equal(self.starts,np.arange(len(self.starts))), 'This track requires canonical contiguous train starts'
        self.c = self.args.enc_in
        self.series = torch.from_numpy(np.array(self.datasets['train'].data_x,dtype=np.float32,copy=True)).to(device)
        self.future_series = torch.from_numpy(np.array(self.datasets['train'].data_y,dtype=np.float32,copy=True)).to(device)
        assert torch.equal(self.series,self.future_series)
        self.offsets = torch.arange(self.h,device=device)
        self.pool_dir = OUT/'pools'/cell
        self.pools = {}
        expected = config()['references'][cell]['expected_query_counts']
        for split,ds in self.datasets.items():
            assert len(ds)==expected[split]
            assert ds.data_x.shape[1]==self.c
            assert np.array_equal(ds.starts, np.arange(len(ds))+ds.base_dataset.border1)
            assert np.isfinite(ds.data_x).all() and np.isfinite(ds.data_y).all()
            assert np.array_equal(ds.base_dataset.scaler.mean_,self.datasets['train'].base_dataset.scaler.mean_)
            assert np.array_equal(ds.base_dataset.scaler.scale_,self.datasets['train'].base_dataset.scaler.scale_)
        self.data_info = dict(cell=cell,counts={s:len(d) for s,d in self.datasets.items()},
            channels=self.c,channel_names=self.datasets['train'].channel_names,
            starts={s:[int(d.starts[0]),int(d.starts[-1])] for s,d in self.datasets.items()},
            scaler_train_rows=len(self.datasets['train'].data_x),
            scaler_mean=self.datasets['train'].base_dataset.scaler.mean_.tolist(),
            scaler_scale=self.datasets['train'].base_dataset.scaler.scale_.tolist(),
            normalized_space=True, candidate_count=len(self.starts), seq_len=self.h, pred_len=self.h)

    def loader(self, split, batch=256, seed=None):
        if seed is not None:
            return make_training_loader(self.datasets[split],batch,seed)
        return DataLoader(self.datasets[split],batch_size=batch,shuffle=False,num_workers=0,drop_last=False)

    def mask(self, starts):
        # Canonical RAFT sampler remains the authority. Eval train-bank endpoints
        # are additionally asserted in validate_pool, not silently filtered.
        return self.sampler.valid_mask_batch(np.asarray(starts,dtype=np.int64))[0].to(self.device)

    def histories(self, ids, channel, all_channels=False):
        pos = ids[...,None] + self.offsets
        if all_channels:
            return self.series[pos]
        return self.series[:,channel][pos][...,None]

    def values(self, ids, channel):
        return self.future_series[:,channel][ids[...,None]+self.h+self.offsets] - self.series[:,channel][ids+self.h-1][...,None]

    def bank_history(self, channel):
        # [N,L,1] strided view over a compact [T] channel, then per-channel copy.
        return self.series[:,channel].unfold(0,self.h,1)[:len(self.starts)].contiguous()[...,None]

    def pool(self, split, starts, channel):
        if split not in self.pools:
            self.pools[split]=np.load(self.pool_dir/f'{split}.npy',mmap_mode='r')
        rows = np.asarray(starts,dtype=np.int64)-int(self.datasets[split].starts[0])
        arr = self.pools[split][rows,channel,:]
        return torch.from_numpy(np.array(arr,dtype=np.int64,copy=True)).to(self.device)

    def batch(self, split, rows):
        ds=self.datasets[split]
        items=[ds[int(i)] for i in rows]
        return (torch.tensor(np.stack([x[0] for x in items]),device=self.device).float(),
                torch.tensor(np.stack([x[1] for x in items]),device=self.device).float(),
                torch.tensor([x[2] for x in items]))

    def channel_batch(self, split, start, end, c, future=True):
        ds=self.datasets[split]
        rows=np.arange(start,end)[:,None]+np.arange(self.h)[None,:]
        x=torch.from_numpy(np.asarray(ds.data_x[:,c][rows],dtype=np.float32)).to(self.device)[...,None]
        y=None if not future else torch.from_numpy(np.asarray(ds.data_y[:,c][rows+self.h],dtype=np.float32)).to(self.device)[...,None]
        return x,y,torch.from_numpy(ds.starts[start:end].copy())

def model_init(cell, seed, nheads, device):
    args=load_args(cell,seed)
    model, state, init_hash=initialize_common_encoder(lambda:Model(args).float(),seed)
    heads=seeded_slot_heads(args.d_model,nheads,seed) if nheads else None
    head_hash=state_hash(heads) if heads is not None else None
    return model.to(device), heads.to(device) if heads is not None else None, init_hash, head_hash

def scores(model, heads, x, data, c, ids, reference_all_channels=False, query_channel=None):
    zq=encode_raw(model,x,c if query_channel is None else query_channel)
    b,p=ids.shape
    xx=data.histories(ids,c,all_channels=reference_all_channels)
    nc=xx.shape[-1]
    zk=encode_raw(model,xx.reshape(b*p,data.h,nc),c if reference_all_channels else 0).reshape(b,p,-1)
    zk=F.normalize(zk,dim=-1)
    if heads is None:
        return torch.einsum('bd,bpd->bp',F.normalize(zq,dim=-1),zk).unsqueeze(1)
    return torch.einsum('bsd,bpd->bsp',heads(zq),zk)

def kl(scores_, values, qlast, target):
    mask=torch.ones(scores_.shape[0],scores_.shape[-1],device=scores_.device,dtype=torch.bool)
    d=pooled_future_mse(values,qlast,target)
    teacher=normalized_teacher_prob(d,mask,0.1)
    prob=torch.softmax(scores_/0.1,-1).mean(1)
    return kl_loss_from_prob(teacher,prob,mask),d,teacher

def legacy_pool_dir(cell):
    ds,h=cell.rsplit('_',1)
    if ds in ('ETTh1','Weather') and int(h) in (96,720):
        return ROOT/f'results/TRACK-V-MULTIQUERY-GENERALIZATION01/{ds}/H{h}/seed0/pool_top100/shared_candidate_pool'
    return None

def ordered_equal(actual,expected):
    a,b=np.asarray(actual),np.asarray(expected)
    if a.shape!=b.shape or not np.array_equal(a,b):
        raise AssertionError('Full ordered candidate IDs differ; set equality is insufficient')
    return True

def load_legacy_payload(data,split):
    source=legacy_pool_dir(data.cell)
    assert source is not None
    path=source/f'{split}.pt'
    blob=torch.load(path,map_location='cpu',weights_only=False)
    baseline=ROOT/f'experiments/p100_preselection_comparison/results/pools/{data.cell}/baseline_cosine/{split}.pt'
    other=torch.load(baseline,map_location='cpu',weights_only=False)
    ids=blob['pool_idx_global'].numpy()
    ordered_equal(ids,other['pool_idx_global'].numpy())
    assert np.array_equal(blob['query_start_idx'].numpy(),data.datasets[split].starts)
    assert np.array_equal(other['query_start_idx'].numpy(),data.datasets[split].starts)
    meta=blob['meta']
    for k,v in [('seq_len',data.h),('pred_len',data.h),('candidate_pool_size',100),
                ('candidate_mask','raft'),('candidate_pool_metric','delta_last_cosine')]:
        assert meta[k]==v,(path,k,meta[k],v)
    assert ids.shape==(len(data.datasets[split]),data.c,100)
    assert meta['memory_starts_fingerprint']==hashlib.sha256(data.starts.tobytes()).hexdigest()
    return blob,dict(source=str(path),source_sha256=sha256_file(path),baseline=str(baseline),
                     baseline_sha256=sha256_file(baseline),legacy_vs_baseline_ordered_exact=True)

class CanonicalPoolAdapter:
    """Only the data/memory interface of the original precompute function.

    Its batch32, query_chunk256, full [N,L,C] input layout, normalization,
    matmul and stable sorting are reused verbatim. No future values enter scoring.
    """
    def __init__(self,data):
        self.data=data
        self.memory_x=torch.from_numpy(data.sampler.memory_x).float().to(data.device)
    def _get_data(self,flag,shuffle=False):
        assert shuffle is False
        return self.data.datasets[flag],self.data.loader(flag,batch=32)
    def _candidate_mask(self,starts):
        return self.data.sampler.valid_mask_batch(starts.cpu().numpy())[0].to(self.data.device),None

@torch.no_grad()
def canonical_sample_pool(data,split,rows,adapter=None):
    rows=np.asarray(rows,dtype=np.int64)
    if legacy_pool_dir(data.cell) is not None:
        blob,_=load_legacy_payload(data,split)
        return blob['pool_idx_global'][torch.from_numpy(rows)].long().to(data.device)
    adapter=adapter or CanonicalPoolAdapter(data)
    result=torch.empty(len(rows),data.c,100,dtype=torch.long,device=data.device)
    ds=data.datasets[split]
    for batch_start in sorted(set((rows//32*32).tolist())):
        x,_,starts=data.batch(split,range(batch_start,min(batch_start+32,len(ds))))
        mask=adapter._candidate_mask(starts)[0]
        for c in range(data.c):
            ids=build_coarse_pool(compute_coarse_delta_last_cosine_scores(x,adapter.memory_x,c),mask,100)
            for i,row in enumerate(rows):
                if batch_start<=row<batch_start+len(x):result[i,c]=ids[int(row)-batch_start]
    return result

@torch.no_grad()
def generate_pools(cell):
    gpu_guard()
    data=CellData(cell,'cuda:0')
    dest=data.pool_dir
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():
        validate_pools(data,check_scores=True)
        return
    begin=time.monotonic()
    old=legacy_pool_dir(cell)
    adapter=None if old is not None else CanonicalPoolAdapter(data)
    parity={}
    for split,ds in data.datasets.items():
        if old is not None:
            blob,origin=load_legacy_payload(data,split)
            starts,ids=blob['query_start_idx'],blob['pool_idx_global']
            kind='immutable_legacy_ordered_ID_copy'
        else:
            # Invoke the original implementation rather than approximate its
            # GEMM batch shape or input layout in an optimized replacement.
            starts,ids=precompute_split(adapter,data.args,split,list(range(data.c)),100,data.device,256)
            origin=dict(source='scripts/precompute_candidate_pool01.py::precompute_split',
                        source_sha256=sha256_file(ROOT/'scripts/precompute_candidate_pool01.py'),
                        loader_batch_size=32,query_chunk_size=256,full_memory_layout=True)
            kind='new_canonical_precompute_batch32'
        assert np.array_equal(starts.numpy(),ds.starts)
        a=np.lib.format.open_memmap(dest/f'{split}.npy',mode='w+',dtype=np.int32,shape=tuple(ids.shape))
        a[:]=ids.numpy();a.flush()
        ordered_equal(a,ids.numpy())
        parity[split]=dict(**origin,full_ordered_ID_parity=True,query_channel_rows=len(ds)*data.c,
                           candidate_ID_elements=int(ids.numel()))
        del a,ids
        if old is not None:del blob
        print(f'POOL {cell} {split} {kind} FULL ORDERED PARITY PASS seconds={time.monotonic()-begin:.1f}',flush=True)
    del adapter
    write_json(dest/'metadata.json',dict(**data.data_info,created_at=now(),data_sha256=config()['data'][cell.rsplit('_',1)[0]]['sha256'],
               source_hashes=source_hashes(),method='canonical raw delta-last history cosine',future_blind=True,
               pool_origin=kind,ordered_parity=parity,loader_batch_size=32,query_chunk_size=256,
               mask='RAFT offline nonoverlap train; train-only evaluation bank',wall_seconds=time.monotonic()-begin))
    report=validate_pools(data,check_scores=False)
    write_json(dest/'COMPLETE.json',report)

@torch.no_grad()
def validate_pools(data,check_scores=True):
    hashes={}
    metadata=json.loads((data.pool_dir/'metadata.json').read_text())
    parity={}
    adapter=None
    if check_scores and legacy_pool_dir(data.cell) is None:adapter=CanonicalPoolAdapter(data)
    for split,ds in data.datasets.items():
        path=data.pool_dir/f'{split}.npy'
        arr=np.load(path,mmap_mode='r')
        assert arr.shape==(len(ds),data.c,100)
        for row in range(0,len(ds),128):
            validate_pool(arr[row:row+128],ds.starts[row:row+128],data.starts,data.h,data.h,split)
        if legacy_pool_dir(data.cell) is not None:
            blob,origin=load_legacy_payload(data,split)
            ordered_equal(arr,blob['pool_idx_global'].numpy())
            parity[split]=dict(**origin,full_ordered_ID_parity=True)
            del blob
        elif check_scores:
            rows=sorted(set([0,len(ds)//2,len(ds)-1]))
            expected=canonical_sample_pool(data,split,rows,adapter)
            ordered_equal(arr[rows],expected.cpu().numpy())
            parity[split]=dict(full_ordered_ID_parity_at_generation=metadata['ordered_parity'][split]['full_ordered_ID_parity'],
                               canonical_batch32_sample_rows=rows)
        else:
            assert metadata['ordered_parity'][split]['full_ordered_ID_parity']
            parity[split]=metadata['ordered_parity'][split]
        hashes[split]=sha256_file(path)
    return dict(status='PASS',checked_at=now(),all_candidate_ids_eligible_unique=True,
                all_channels=True,ordered_parity=parity,hashes=hashes,**data.data_info)

def metric_rows(d, values, qlast, target, picks, oracle, last_mse):
    selected=values.gather(1,picks[:,:,None].expand(-1,-1,target.shape[-1]))+qlast[:,None,None]
    err=selected.mean(1)-target
    ret=d.gather(1,picks).mean(1)
    oret=d.gather(1,oracle).mean(1)
    mse=err.square().mean(1)
    return torch.stack([ret,mse,err.abs().mean(1),
        (picks[:,:,None]==oracle[:,None,:]).any(-1).float().mean(1),ret-oret,
        (mse<last_mse).float(),(mse==last_mse).float()],1)

def configurations(nheads, detailed):
    allheads=tuple(range(max(1,nheads)))
    out=[('All',allheads,10)]
    if detailed:
        out += [(f'All_K{k}',allheads,k) for k in (1,5,20)]
        if nheads>1:
            for size in range(1,nheads+1):
                for sub in itertools.combinations(range(nheads),size):
                    if sub!=allheads:out.append(('H'+'+'.join(str(h+1) for h in sub),sub,10))
    return out

@torch.no_grad()
def evaluate(data,model,heads,split,detailed=False,dest=None,baselines=False):
    model.eval()
    if heads is not None:heads.eval()
    nheads=0 if heads is None else heads.W.shape[0]
    cfgs=configurations(nheads,detailed)
    ds=data.datasets[split]
    sums=np.zeros((len(cfgs),len(FIELDS)),dtype=np.float64)
    last_sum=np.zeros(2,dtype=np.float64)
    count=0
    pair_sums={};union_sum=0.;gap_zero=0
    baseline_sums=np.zeros((22,len(FIELDS)),dtype=np.float64) if baselines else None
    for c in range(data.c):
        chunks=[]; idchunks=[]; baseline_chunks=[]; lastchunks=[]; inputchunks=[]; ties=[]
        gens=[torch.Generator(device='cpu').manual_seed(990000+1000*c+j) for j in range(20)]
        for start in range(0,len(ds),32):
            end=min(start+32,len(ds));x,y,starts=data.channel_batch(split,start,end,c)
            ids=data.pool(split,starts,c)
            sc=scores(model,heads,x,data,c,ids,query_channel=0)
            mask=torch.ones_like(ids,dtype=torch.bool)
            _,ph,pm=mean_mixture_topk_selection(sc,mask,.1,10)
            values=data.values(ids,c);qlast=x[:,-1,0];target=y[:,:,0]
            d=pooled_future_mse(values,qlast,target)
            oracle_order=stable_topk_indices(d,100,largest=False)
            sd=d.gather(1,oracle_order);gap=sd[:,10]-sd[:,9]
            gap_zero+=int((gap==0).sum());ties.append(gap.cpu().numpy())
            le=qlast[:,None]-target
            lm=le.square().mean(1);la=le.abs().mean(1)
            last_sum+=torch.stack([lm,la],1).double().sum(0).cpu().numpy()
            this=[];picked=[]
            for name,subset,k in cfgs:
                prob=ph[:,list(subset)].mean(1)
                pick=stable_topk_indices(prob,k,largest=True)
                met=metric_rows(d,values,qlast,target,pick,oracle_order[:,:k],lm)
                assert bool((met[:,4]>=-1e-5).all())
                this.append(met.cpu().numpy())
                padded=torch.full((len(x),20),-1,dtype=torch.int32,device=data.device)
                padded[:,:k]=ids.gather(1,pick).int()
                picked.append(padded.cpu().numpy())
            mat=np.stack(this,1)
            sums+=mat.astype(np.float64).sum(0)
            count+=len(x)
            if nheads>1 and detailed:
                hs=stable_topk_indices(ph,10,largest=True)
                for a,b in itertools.combinations(range(nheads),2):
                    overlap=(hs[:,a,:,None]==hs[:,b,None,:]).any(-1).float().mean(-1)
                    pair_sums[(a,b)]=pair_sums.get((a,b),0.)+float(overlap.double().sum())
                sorted_ids=hs.reshape(len(x),-1).sort(-1).values
                union_sum+=float((1+(sorted_ids[:,1:]!=sorted_ids[:,:-1]).sum(-1)).double().sum())
            if baselines:
                bm=[]
                for j in range(22):
                    if j==0:pick=oracle_order[:,:10]
                    elif j==1:pick=torch.arange(10,device=data.device).expand(len(x),10)
                    else:
                        pick=torch.stack([torch.randperm(100,generator=gens[j-2])[:10] for _ in range(len(x))]).to(data.device)
                    bm.append(metric_rows(d,values,qlast,target,pick,oracle_order[:,:10],lm).cpu().numpy())
                ba=np.stack(bm,1);baseline_sums+=ba.astype(np.float64).sum(0);baseline_chunks.append(ba)
            if dest is not None:
                chunks.append(mat);idchunks.append(np.stack(picked,1));lastchunks.append(torch.stack([lm,la],1).cpu().numpy())
                inputchunks.append(torch.stack([x[:,:,0].std(1,unbiased=False),x[:,1:,0].sub(x[:,:-1,0]).abs().mean(1)],1).cpu().numpy())
        if dest is not None:
            dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
            payload=dict(query_start=ds.starts,channel=np.int32(c),metrics=np.concatenate(chunks),
                candidate_ids=np.concatenate(idchunks),last=np.concatenate(lastchunks),
                input_features=np.concatenate(inputchunks),oracle_k10_gap=np.concatenate(ties),
                config_names=np.array([v[0] for v in cfgs]),metric_names=np.array(FIELDS))
            if baselines:payload['baselines']=np.concatenate(baseline_chunks)
            tmp=dest/f'channel{c:03d}.tmp.npz'
            np.savez_compressed(tmp,**payload);os.replace(tmp,dest/f'channel{c:03d}.npz')
        if detailed and (c%8==0 or c==data.c-1):
            print(f'EVAL {data.cell} {split} channel={c+1}/{data.c}',flush=True)
    assert count==len(ds)*data.c
    last_mean=last_sum/count
    rows=[]
    for i,(name,sub,k) in enumerate(cfgs):
        row=dict(name=name,heads=[x+1 for x in sub],k=k,**dict(zip(FIELDS,(sums[i]/count).tolist())))
        row['last_skill']=None if last_mean[0]<=1e-12 else 1-row['agg_mse']/last_mean[0]
        rows.append(row)
    result=dict(cell=data.cell,split=split,query_count=len(ds),channels=data.c,unit_count=count,
                rows=rows,last_mse=float(last_mean[0]),last_mae=float(last_mean[1]),
                oracle_k10_boundary_tie_fraction=gap_zero/count)
    if nheads>1 and detailed:
        result['diversity']=dict(pair_overlap={f'H{a+1}_H{b+1}':v/count for (a,b),v in pair_sums.items()},
            mean_overlap=float(np.mean(list(pair_sums.values())))/count,union_size=union_sum/count,normalized_union=union_sum/count/(10*nheads))
    if baselines:
        bm=baseline_sums/count
        result['baselines']={'P100 Individual-MSE Oracle@10':dict(zip(FIELDS,bm[0].tolist())),
            'Raw cosine Top-10':dict(zip(FIELDS,bm[1].tolist())),
            'Random Top-10':dict(zip(FIELDS,bm[2:].mean(0).tolist())),
            'Random MC draw standard deviation':dict(zip(FIELDS,bm[2:].std(0,ddof=1).tolist()))}
        for key,row in result['baselines'].items():
            row['last_skill']=None if last_mean[0]<=1e-12 else (row['agg_mse']/last_mean[0] if key=='Random MC draw standard deviation' else 1-row['agg_mse']/last_mean[0])
        result['random_draw_metrics']=bm[2:].tolist()
    return result

def ensure_init(cell,seed,model,heads,initial_hash,head_hash):
    p=OUT/'initialization'/cell/f'seed{seed}'
    p.mkdir(parents=True,exist_ok=True)
    meta=p/'common.json'
    if meta.exists():assert json.loads(meta.read_text())['model_hash']==initial_hash
    else:
        atomic_torch(p/'common_model.pth',{k:v.detach().cpu() for k,v in model.state_dict().items()})
        write_json(meta,dict(model_hash=initial_hash,seed=seed,cell=cell,reference_role='configuration only',created_at=now()))
    if heads is not None:
        hp=p/f'heads{heads.W.shape[0]}.json'
        obj=dict(hash=head_hash,per_head_hashes=[state_hash({'W':w}) for w in heads.W.detach().cpu()],seed=seed)
        if hp.exists():assert json.loads(hp.read_text())==obj
        else:write_json(hp,obj)

def train_one(run_id):
    gpu=gpu_guard();cfg=config()
    runs=json.loads((HERE/'configs/run_manifest.json').read_text())
    row=next(x for x in runs if x['run_id']==run_id)
    cell=f"{row['dataset']}_{row['horizon']}";seed=row['seed'];nheads=row['num_query_heads']
    dest=ROOT/row['output_path'];ckpt=ROOT/row['checkpoint_path']
    dest.mkdir(parents=True,exist_ok=True);ckpt.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():
        done=json.loads((dest/'COMPLETE.json').read_text());assert done['status']=='COMPLETE'
        assert sha256_file(done['selected_checkpoint'])==done['checkpoint_sha256']
        print('ALREADY COMPLETE',run_id,flush=True);return
    gate=json.loads((HERE/'reports/RUNTIME_GATES.json').read_text())
    assert gate['status']=='PASS' and gate['source_hashes']==source_hashes()
    poolmeta=json.loads((OUT/'pools'/cell/'COMPLETE.json').read_text())
    assert poolmeta['status']=='PASS'
    for split,digest in poolmeta['hashes'].items():
        assert sha256_file(OUT/'pools'/cell/f'{split}.npy')==digest,'Pinned common pool changed'
    data=CellData(cell,'cuda:0')
    model,heads,ih,hh=model_init(cell,seed,nheads,'cuda:0')
    ensure_init(cell,seed,model,heads,ih,hh)
    loader=data.loader('train',32,seed)
    params=list(model.parameters())+(list(heads.parameters()) if heads is not None else [])
    opt=torch.optim.Adam(params,lr=.001)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    history=[];start_epoch=1;elapsed_prior=0.
    files=sorted(ckpt.glob('epoch[0-9][0-9].pth'))
    if files:
        saved=torch.load(files[-1],map_location='cpu',weights_only=False)
        assert saved['source_hashes']==source_hashes() and saved['pool_hashes']==poolmeta['hashes']
        model.load_state_dict(saved['model'])
        if heads is not None:heads.load_state_dict(saved['heads'])
        opt.load_state_dict(saved['optimizer'])
        loader.generator.set_state(saved['loader_rng'])
        torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        history=saved['history'];start_epoch=saved['epoch']+1;elapsed_prior=saved['elapsed_seconds']
    provenance=dict(run=row,teacher=cfg['teacher'],source_hashes=source_hashes(),gpu=gpu,
        initial_model_hash=ih,head_hash=hh,pool_hashes=poolmeta['hashes'],reference_warm_start=False,
        selection='minimum validation Mean-Mixture RetMSE@10; first epoch on exact tie',stage2=False,
        logical_batch=32,channel_chunk=1,candidate_encode_chunk=3200,training_dtype='float32',created_at=now())
    write_json(dest/'provenance.json',provenance)
    begin=time.monotonic()
    for epoch in range(start_epoch,11):
        model.train()
        if heads is not None:heads.train()
        order=[];train_sum=0.;train_n=0;epstart=time.monotonic()
        for bi,(x,y,starts) in enumerate(loader):
            order.append(starts.clone());x=x.float().to(data.device);y=y.float().to(data.device)
            opt.zero_grad(set_to_none=True);bl=0.
            for c in range(data.c):
                ids=data.pool('train',starts,c)
                sc=scores(model,heads,x,data,c,ids)
                loss,_,_=kl(sc,data.values(ids,c),x[:,-1,c],y[:,:,c])
                assert torch.isfinite(loss),f'nonfinite loss {run_id} {epoch} {bi} {c}'
                (loss/data.c).backward();bl+=float(loss.detach())/data.c
            assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
            opt.step();train_sum+=bl*len(x);train_n+=len(x)
            if bi%100==0:
                print(f'TRAIN {run_id} epoch={epoch} batch={bi}/{len(loader)} kl={bl:.6f} elapsed={time.monotonic()-epstart:.1f}',flush=True)
                write_json(dest/'current.json',dict(status='TRAINING',run_id=run_id,epoch=epoch,batch=bi,pid=os.getpid(),updated_at=now()))
        assert train_n==len(data.datasets['train']) and len(order)==row['optimizer_steps_per_epoch']
        bh=batch_order_hash(order)
        shared=OUT/'initialization'/cell/f'seed{seed}'/f'epoch{epoch:02d}_batch_hash.json'
        if shared.exists():assert json.loads(shared.read_text())['sha256']==bh
        else:write_json(shared,dict(sha256=bh,n_queries=train_n,n_batches=len(order)))
        val=evaluate(data,model,heads,'val')
        h=dict(epoch=epoch,train_kl=train_sum/train_n,validation=val,batch_hash=bh,train_queries=train_n,
               epoch_seconds=time.monotonic()-epstart)
        history.append(h)
        saved=dict(epoch=epoch,model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            heads={k:v.detach().cpu() for k,v in heads.state_dict().items()} if heads is not None else None,
            optimizer=opt.state_dict(),history=history,loader_rng=loader.generator.get_state(),cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            source_hashes=source_hashes(),pool_hashes=poolmeta['hashes'],elapsed_seconds=elapsed_prior+time.monotonic()-begin)
        atomic_torch(ckpt/f'epoch{epoch:02d}.pth',saved)
        write_json(dest/'epochs.json',history)
        print(f'EPOCH COMPLETE {run_id} {epoch}/10 val_retmse={val["rows"][0]["retmse"]:.9f} val_agg={val["rows"][0]["agg_mse"]:.9f}',flush=True)
    selected=min(history,key=lambda x:(x['validation']['rows'][0]['retmse'],x['epoch']))
    selected_path=ckpt/f'epoch{selected["epoch"]:02d}.pth'
    saved=torch.load(selected_path,map_location='cpu',weights_only=False);model.load_state_dict(saved['model'])
    if heads is not None:heads.load_state_dict(saved['heads'])
    result={}
    for split in ('val','test'):
        result[split]=evaluate(data,model,heads,split,detailed=True,dest=dest/'queries'/split,baselines=(nheads==0 and seed==0))
        write_json(dest/f'{split}_metrics.json',result[split])
    assert abs(result['val']['rows'][0]['retmse']-selected['validation']['rows'][0]['retmse'])<1e-10
    selection=dict(epoch=selected['epoch'],validation_retmse=selected['validation']['rows'][0]['retmse'],
                   selected_checkpoint=str(selected_path),sha256=sha256_file(selected_path),test_used_for_selection=False)
    write_json(dest/'checkpoint_selection.json',selection)
    torch.cuda.synchronize()
    complete=dict(status='COMPLETE',run_id=run_id,completed_at=now(),selected_checkpoint=str(selected_path),
        checkpoint_sha256=selection['sha256'],selected_epoch=selected['epoch'],official_test=result['test']['rows'][0],
        query_count=result['test']['query_count'],channels=data.c,unit_count=result['test']['unit_count'],
        elapsed_seconds=elapsed_prior+time.monotonic()-begin,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),gpu=gpu,stage2=False,reused_training=False)
    write_json(dest/'metrics.json',complete)
    write_json(dest/'COMPLETE.json',complete)
    print('RUN COMPLETE',json.dumps(complete),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['pool','train']);ap.add_argument('target')
    args=ap.parse_args()
    if args.action=='pool':generate_pools(args.target)
    else:train_one(args.target)
