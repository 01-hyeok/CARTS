"""Dedicated read-only adapter. No changes to earlier experiment modules.

Every production encoder call contains every channel. Serial channel calls exist
only in the explicitly named canonical reference gate, never as a fallback.
"""
import csv,hashlib,inspect,json,sys,textwrap
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from protocol import ROOT
sys.path[:0]=[str(ROOT/'experiments/retrieval_failure_attribution01'),
             str(ROOT/'experiments/retrieval_all_channel_tail08'),str(ROOT)]
import diagnostic_adapter as old
import packed_batch
from packed_batch import PackedChannelFull
from all_channel_ops import distance_all
from utils.full_candidate_bank import encode_raw_channel_first

Data=old.Data
rt=old.rt
BRIDGE={'ETTh1_96','ETTh1_720','Weather_96','Weather_720'}

def inventory(cell):
    p=ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02/00_Audit/CHECKPOINT_AUDIT.csv'
    return [r for r in csv.DictReader(p.open()) if r['cell']==cell]

class LogitBatch(PackedChannelFull):
    def __init__(self,frozen):
        # Same inherited operations, now also allow an independently P100-trained
        # parameter set to be evaluated on Full. Do not relabel its provenance.
        self.frozen=frozen;self.data=frozen.data;self.keys=None;self.shapes=[]
        self.lt=None;self.blas=None;self.split64=None;self.p100_score_lt=None
        assert frozen.model.encoder.encoder_type=='mlp' and not frozen.model.training
    @torch.no_grad()
    def build_keys(self):
        f=self.frozen;d=self.data;f.precision();n=len(d.starts)
        rows=n if f.full_trained else 3200
        step=rows//d.c;parts=[]
        for a in range(0,n,step):
            b=min(a+step,n);pos=torch.arange(a,b,device=d.device)[:,None]+d.offsets
            x=d.series[pos].permute(0,2,1).contiguous()
            parts.append(self.encode(x,rows))
        self.keys=torch.cat(parts).permute(1,0,2).contiguous()
        self.shapes.append(dict(axis='candidate',channel_batch=d.c,encoder_rows=rows,
                                candidate_chunk=step,all_channels=True))

# Reuse the exact validated tail kernels; expose logits before normalization.
# A source assertion prevents silently using a changed upstream implementation.
_src=textwrap.dedent(inspect.getsource(PackedChannelFull.probability))
_old="return scores[:,:,0].masked_fill(~mask,-float('inf')) if f.heads is None else global_mixture(scores,mask)"
assert _src.count(_old)==1
_src=_src.replace('def probability(', 'def logits(').replace(_old,'return scores,mask')
_score_line='else:scores=score_all(q,self.keys)'
assert _src.count(_score_line)==1
_src=_src.replace(_score_line,'''elif f.full_trained and f.n==1 and not torch.backends.cuda.matmul.allow_tf32:
        # Full S1 uses canonical Bx128 @ 128xN GEMM. PyTorch channel bmm
        # changes accumulation despite bitwise identical query/key embeddings.
        # One Lt operation with batchCount=ALL channels preserves that GEMM.
        if self.lt is None:self.lt=CublasLtBatch()
        c,s,dim=d.c,q.shape[2],q.shape[3]
        qq=q.permute(1,0,2,3).contiguous().reshape(c,b*s,dim)
        scores=self.lt.score(qq,self.keys,68,1,0).reshape(c,b,s,-1).permute(1,0,2,3).contiguous()
    else:scores=score_all(q,self.keys)''')
_ns=dict(packed_batch.__dict__);exec(compile(_src,__file__+'::validated_logits','exec'),_ns)
LogitBatch.logits=_ns['logits']

# P100 V0 has FP32 short batches that the legacy Full/TF32 path did not cover.
# Preserve the first Linear's canonical B-row GEMM and both B-row norm
# reductions while submitting ALL channels in one Lt batch. No cache/future use.
_p100_short=textwrap.dedent(inspect.getsource(PackedChannelFull.short_query))
_first='(compute==68 and layer.in_features==720)'
_norm='exact_torch2_tail=(compute==68 and h==720 and b==7 and c==21 and self.frozen.heads is None)'
assert _p100_short.count(_first)==1 and _p100_short.count(_norm)==1
_p100_short=_p100_short.replace(_first,'(compute==68)').replace(_norm,'exact_torch2_tail=(compute==68 and self.frozen.heads is None)')
_short_ns=dict(packed_batch.__dict__);exec(compile(_p100_short,__file__+'::p100_short_query','exec'),_short_ns)
_p100_short_query=_short_ns['short_query']

def p100_v0_short_query(batch,x):
    assert not batch.frozen.full_trained and batch.frozen.heads is None
    assert not torch.backends.cuda.matmul.allow_tf32 and x.shape[0]<x.shape[1]
    return _p100_short_query(batch,x)

class Frozen:
    def __init__(self,data,row):
        self.data=data;self.row=row;self.full_trained=row['training_support']=='FULL'
        self.support=row['training_support'];self.name=row['architecture']+'_'+self.support
        assert rt.sha256_file(row['checkpoint'])==row['checkpoint_sha256']
        self.n={'V0':0,'V1':1,'V5':5}[row['architecture']]
        self.model,self.heads,_,_=rt.model_init(data.cell,0,self.n,'cuda:0')
        p=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
        self.model.load_state_dict(p['model_state_dict'] if self.full_trained else p['model'],strict=True)
        if self.heads is not None:self.heads.load_state_dict(p['slot_heads_state_dict'] if self.full_trained else p['heads'],strict=True)
        self.model.eval()
        if self.heads is not None:self.heads.eval()
        for p in list(self.model.parameters())+([] if self.heads is None else list(self.heads.parameters())):p.requires_grad_(False)
        self.batch=LogitBatch(self);self.refs={};self.canonical_keys={}
        self.tf32=self.full_trained and (self.n>0 or data.cell not in BRIDGE)
        # Immutable-cache numerical audit, not a claim about historical runtime.
        # E96/E720/W96/W720 V1 first/tail caches match Torch2 FP32 exactly.
        # Historical runtime stays UNKNOWN; each full population remains gated.
        if self.full_trained and self.n==1 and data.cell in {'ETTh1_96','ETTh1_720','Weather_96','Weather_720'}:self.tf32=False
        self.cache_manifest=None
    def precision(self):
        torch.backends.cuda.matmul.allow_tf32=self.tf32
        torch.backends.cudnn.allow_tf32=self.full_trained
    def identity(self):
        assert rt.sha256_file(self.row['checkpoint'])==self.row['checkpoint_sha256']
        return dict(**self.row,torch_version=torch.__version__,matmul_tf32=self.tf32,
                    original_runtime='UNKNOWN',all_channel_batch=True,shapes=self.batch.shapes[:3])
    def reference(self,split):
        if split not in self.refs:
            if not self.full_trained:
                ids=[];met=[]
                for c in range(self.data.c):
                    with np.load(Path(self.row['evidence'])/'queries'/split/f'channel{c:03d}.npz') as z:
                        assert np.array_equal(z['query_start'],self.data.datasets[split].starts)
                        ids.append(z['candidate_ids'][:,0,:10].copy());met.append(z['metrics'][:,0].copy())
                self.refs[split]=dict(ids=np.stack(ids,1),metrics=np.stack(met,1))
            else:
                ds,h=self.data.cell.rsplit('_',1);arm=self.row['architecture']
                base=ROOT/(f'results/TRACK-V-MEANMIX-CHECKPOINT-CORRECTION01/{ds}/H{h}/seed0/cache' if arm=='V5'
                           else f'results/TRACK-V-MULTIQUERY-GENERALIZATION01/{ds}/H{h}/seed0/{arm}/cache')
                z=torch.load(base/f'{split}.pt',map_location='cpu',weights_only=False)
                assert np.array_equal(z['query_start_idx'].numpy(),self.data.datasets[split].starts)
                self.refs[split]=dict(prediction=z['relation_outputs'].numpy().transpose(0,2,1),
                                      metrics=json.loads((base/'stage1_metrics.json').read_text()))
        return self.refs[split]
    @torch.no_grad()
    def full_logits(self,split,a,b):
        self.precision();d=self.data
        if self.full_trained and self.n==0 and d.cell in BRIDGE:
            dest=old.OUT/'full_v0_canonical_scores'/d.cell
            if self.cache_manifest is None:
                self.cache_manifest=json.loads((dest/'COMPLETE.json').read_text())
                assert self.cache_manifest['status']=='PASS'
                assert self.cache_manifest['provenance']['checkpoint_sha256']==self.row['checkpoint_sha256']
            p=dest/split/f'batch{a//32:05d}.npz'
            assert rt.sha256_file(p)==self.cache_manifest['file_hashes'][str(p.relative_to(dest))]
            with np.load(p) as z:
                assert np.array_equal(z['query_start'],d.datasets[split].starts[a:b])
                s=torch.as_tensor(z['scores'].copy(),device=d.device)[:,:,None]
            mask=d.mask(d.datasets[split].starts[a:b])[:,None].expand(b-a,d.c,-1)
            return s,mask
        return self.batch.logits(split,a,b)
    @torch.no_grad()
    def native_p100_logits(self,split,a,b):
        assert not self.full_trained
        self.precision();d=self.data;x,_,starts=d.query(split,a,b);B,C,H=x.shape
        # Canonical query row count B32 with every C in each padded call.
        if B<C:q=p100_v0_short_query(self.batch,x) if self.heads is None else self.batch.short_query(x)
        else:
            q=torch.cat([self.batch.encode(x[i:i+max(1,B//C)],B,project=True) for i in range(0,B,max(1,B//C))])
        ids=d.pool_all(split,a,b);rows=B*100;step=max(1,rows//C)
        # Candidate/query rows may be chunked. C is never divided, including B=1.
        hist=d.histories_all(ids).permute(0,2,1,3).reshape(B*100,C,H)
        key=torch.cat([self.batch.encode(hist[i:i+step],rows) for i in range(0,B*100,step)])
        key=key.reshape(B,100,C,-1).permute(0,2,1,3).contiguous()
        # The original pooled score uses a B-element strided batch. Expanding
        # that batch to B*C changes the CUDA reduction kernel even with bitwise
        # identical q/key tensors. Pack all C with query chunking, preserving B.
        if B>=C:
            parts=[];step=B//C
            for i in range(0,B,step):
                n=min(step,B-i);r=n*C;qq=q[i:i+n].reshape(r,q.shape[2],-1);kk=key[i:i+n].reshape(r,100,-1)
                if r<B:
                    qq=torch.cat([qq,qq.new_zeros(B-r,qq.shape[1],qq.shape[2])])
                    kk=torch.cat([kk,kk.new_zeros(B-r,100,kk.shape[2])])
                score=(torch.einsum('bd,bpd->bp',qq[:,0],kk)[:,None] if self.heads is None
                       else torch.einsum('bsd,bpd->bsp',qq,kk))
                parts.append(score[:r].reshape(n,C,q.shape[2],100))
            result=torch.cat(parts)
        elif self.heads is None:
            # Preserve canonical per-query 1x128 @ 128x100 shape and B-count
            # heuristic; one actual batchCount=B*C includes every channel.
            # Torch2 CUDA12 canonical bmm dispatches GEMV here. The inherited
            # CUDA11 Lt handle uses different accumulation even for exact q/k.
            # Match the runtime library and canonical B heuristic, keeping the
            # actual operation batch B*C. Encoder kernels remain unchanged.
            assert int(torch.version.cuda.split('.')[0])==12
            if self.batch.p100_score_lt is None:
                from cublaslt_batch import CublasLtBatch
                self.batch.p100_score_lt=CublasLtBatch('libcublasLt.so.12')
            result=self.batch.p100_score_lt.score(q.reshape(B*C,1,-1),key.reshape(B*C,100,-1),68,B,0).reshape(B,C,1,100)
        else:
            result=torch.einsum('bcsd,bcpd->bcsp',q,key)
        return result,ids
    @torch.no_grad()
    def canonical_full_ids(self,split,a,b):
        """REFERENCE ONLY. Never supplies production output on a mismatch."""
        assert self.full_trained;self.precision();d=self.data;parts=[]
        for c in range(d.c):
            x,_,starts=d.channel_batch(split,a,b,c,future=False)
            if c not in self.canonical_keys:
                self.canonical_keys[c]=F.normalize(encode_raw_channel_first(self.model,d.bank_history(c),0),dim=-1)
            z=encode_raw_channel_first(self.model,x,0)
            q=F.normalize(z,dim=-1)[:,None] if self.heads is None else self.heads(z)
            s=(torch.matmul(q[:,0],self.canonical_keys[c].T)[:,None] if self.heads is None
               else torch.einsum('bsd,nd->bsn',q,self.canonical_keys[c]))
            mask=d.mask(starts)
            # Original independent S1 raw ranking; no multihead RR cache reused.
            p=s[:,0].masked_fill(~mask,-float('inf')) if self.n<=1 else torch.softmax(s.masked_fill(~mask[:,None],-float('inf'))/.1,-1).mean(1)
            parts.append(rt.stable_topk_indices(p,10,largest=True))
        return torch.stack(parts,1)

def topk(score,k):
    shape=score.shape;return rt.stable_topk_indices(score.reshape(-1,shape[-1]),k,largest=True).reshape(*shape[:-1],k)

def normalize(logits,mask,raw_single=False):
    z=logits.masked_fill(~mask[:,:,None],-float('inf'))
    if raw_single and z.shape[2]==1:return z[:,:,0]
    if z.shape[-1]>100 and z.shape[2]>1:
        # CUDA softmax vector-load alignment depends on row position when N is
        # not divisible by four. C-first keeps the original B/S row order for
        # each channel, while still issuing ONE full-channel softmax operation.
        # Query padding (never candidates/channels) preserves row alignment at
        # short tails too. Padded rows are discarded before any TopK or metric.
        B,C,S,N=z.shape;v=z.permute(1,0,2,3).contiguous();pad=(-B)%4
        if pad:v=torch.cat([v,v.new_zeros(C,pad,S,N)],1)
        return torch.softmax(v/.1,-1).mean(2)[:,:B].permute(1,0,2).contiguous()
    return torch.softmax(z/.1,-1).mean(2)

@torch.no_grad()
def distances(data,split,a,b,chunk=128):
    x,y,starts=data.query(split,a,b);B,C,_=x.shape;N=len(data.starts)
    d=torch.empty(B,C,N,device=data.device)
    for i in range(0,N,chunk):
        ids=torch.arange(i,min(i+chunk,N),device=data.device).expand(B,C,-1)
        d[...,i:i+chunk]=distance_all(data.values_all(ids),x[:,:,-1],y)
    return x,y,starts,d
