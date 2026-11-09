"""Read-only frozen model/data adapters. Uses existing CARTS numerical paths."""
import os,sys,json,csv,time,hashlib
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/data/pjh_workspace/CARTS')
HERE=ROOT/'experiments/retrieval_failure_attribution01'
OUT=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01'
OLD=ROOT/'experiments/p100_head_retrieval_quality_legacy_pool02'
GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
if int(torch.__version__.split('.')[0])<2:
    original_load=torch.load
    def compatible_load(*args,**kwargs):
        kwargs.pop('weights_only',None)
        return original_load(*args,**kwargs)
    torch.load=compatible_load
sys.path[:0]=[str(OLD),str(OLD/'validation04'),str(ROOT)]
import runtime as rt
from cached_validation import CandidateCache
from utils.full_candidate_bank import encode_raw_channel_first,compute_scores_full_grad_channel_first
from scripts.train_factorial_e2e01 import arm_score


def inventory(cell,support):
    ds,h=cell.rsplit('_',1)
    rows=list(csv.DictReader((OUT/'00_Audit/CHECKPOINT_INVENTORY.csv').open()))
    return [x for x in rows if x['dataset']==ds and x['horizon']==h and x['training_support']==support]


def snapshot_hashes():
    return {str(p.relative_to(ROOT)):rt.sha256_file(p) for p in sorted(list(HERE.glob('*.py'))+[HERE/'PROTOCOL.json'])}


class Frozen:
    def __init__(self,data,row,evaluation_support):
        assert row['status']=='VALID' and rt.sha256_file(row['checkpoint'])==row['checkpoint_sha256']
        self.data=data;self.row=row;self.support=evaluation_support;self.full_trained=row['training_support']=='FULL'
        self.name=row['architecture']+('_FullTrained' if self.full_trained else '_P100Trained')
        if evaluation_support=='FULL' and not self.full_trained:self.name+='_SupportTransfer'
        n=0 if row['architecture']=='V0' else 5
        self.model,self.heads,_,_=rt.model_init(data.cell,0,n,'cuda:0')
        payload=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
        self.model.load_state_dict(payload['model_state_dict'] if self.full_trained else payload['model'],strict=True)
        if self.heads is not None:self.heads.load_state_dict(payload['slot_heads_state_dict'] if self.full_trained else payload['heads'],strict=True)
        self.model.eval()
        if self.heads is not None:self.heads.eval()
        for p in list(self.model.parameters())+([] if self.heads is None else list(self.heads.parameters())):p.requires_grad_(False)
        self.caches=[CandidateCache(rt.scores) for _ in range(data.c)]
        self.keys={};self.p100_reference={};self.full_reference={};self.pick_mismatches=0;self.score_probe_count=0;self.reproduction=[]
        self.enabled=True

    def precision(self):
        torch.backends.cuda.matmul.allow_tf32=self.full_trained
        torch.backends.cudnn.allow_tf32=self.full_trained

    def reference(self,split):
        if self.support=='P100':
            if split not in self.p100_reference:
                ids=[];metrics=[]
                for c in range(self.data.c):
                    with np.load(Path(self.row['evidence'])/'queries'/split/f'channel{c:03d}.npz') as a:
                        assert np.array_equal(a['query_start'],self.data.datasets[split].starts)
                        ids.append(a['candidate_ids'][:,0,:10].copy());metrics.append(a['metrics'][:,0,:].copy())
                self.p100_reference[split]=(np.stack(ids,1),np.stack(metrics,1))
            return self.p100_reference[split]
        if self.full_trained:
            if split not in self.full_reference:
                ds,h=self.data.cell.rsplit('_',1)
                if self.row['architecture']=='V5':base=ROOT/f'results/TRACK-V-MEANMIX-CHECKPOINT-CORRECTION01/{ds}/H{h}/seed0/cache'
                else:base=ROOT/f'results/TRACK-V-MULTIQUERY-GENERALIZATION01/{ds}/H{h}/seed0/V0/cache'
                a=torch.load(base/f'{split}.pt',map_location='cpu',weights_only=False)
                assert np.array_equal(a['query_start_idx'].numpy(),self.data.datasets[split].starts)
                self.full_reference[split]=(a['relation_outputs'].numpy(),json.loads((base/'stage1_metrics.json').read_text()))
            return self.full_reference[split]
        return None

    @torch.no_grad()
    def probability(self,split,start,end):
        """Always encode query in canonical B32 groups; never microbatch the encoder."""
        assert start%32==0 and end==min(start+32,len(self.data.datasets[split]))
        self.precision();out=[];data=self.data
        for c in range(data.c):
            x,_,starts=data.channel_batch(split,start,end,c,future=False)
            if self.support=='P100':
                ids=data.pool(split,starts,c);mask=torch.ones_like(ids,dtype=torch.bool)
                sc=self.caches[c](self.model,self.heads,x,data,c,ids,query_channel=0)
                _,ph,_=rt.mean_mixture_topk_selection(sc,mask,.1,10)
                prob=ph[:,list(range(ph.shape[1]))].mean(1)
                expected=self.reference(split)[0][start:end,c]
                chosen=ids.gather(1,rt.stable_topk_indices(prob,10,largest=True)).cpu().numpy()
                if not np.array_equal(chosen,expected):
                    # Full canonical fallback, never an epsilon-based tie repair.
                    sc=rt.scores(self.model,self.heads,x,data,c,ids,query_channel=0)
                    _,ph,_=rt.mean_mixture_topk_selection(sc,mask,.1,10)
                    prob=ph[:,list(range(ph.shape[1]))].mean(1)
                    chosen=ids.gather(1,rt.stable_topk_indices(prob,10,largest=True)).cpu().numpy()
                    self.caches[c].fallback=True;self.pick_mismatches+=1
                assert np.array_equal(chosen,expected),(self.name,split,c,start,'canonical candidate ID mismatch')
            else:
                # [N,L,1] only one channel at a time, never [Q,N,H,C].
                if c not in self.keys:
                    history=data.bank_history(c)
                    if self.full_trained:
                        k=encode_raw_channel_first(self.model,history,0)
                    else:
                        chunks=[]
                        for a in range(0,len(history),3200):
                            z=history[a:a+3200];n=len(z)
                            if n<3200:z=torch.cat([z,torch.zeros(3200-n,data.h,1,device=data.device)])
                            chunks.append(rt.encode_raw(self.model,z,0)[:n])
                        k=torch.cat(chunks)
                    self.keys[c]=rt.F.normalize(k,dim=-1);del history
                zq=encode_raw_channel_first(self.model,x,0) if self.full_trained else rt.encode_raw(self.model,x,0)
                q=rt.F.normalize(zq,dim=-1)[:,None] if self.heads is None else self.heads(zq)
                sc=(torch.matmul(q[:,0],self.keys[c].T).unsqueeze(1) if self.heads is None else torch.einsum('bsd,nd->bsn',q,self.keys[c]))
                mask=data.mask(starts)
                # Full valid-candidate axis is retained in the score tensor;
                # the softmax is global, never normalized independently by chunk.
                _,ph,prob=rt.mean_mixture_topk_selection(sc,mask,.1,10)
                if self.full_trained and self.heads is None:
                    # Historical V0 uses single-slot raw-score argmax/TopK.
                    # Softmax is monotone mathematically but can merge FP32 ties.
                    prob=sc[:,0].masked_fill(~mask,-float('inf'))
                if not self.full_trained:
                    # This is a same-frozen-parameter support transfer row.
                    prob=ph[:,list(range(ph.shape[1]))].mean(1).masked_fill(~mask,-float('inf'))
                if split=='val' and start==0 and self.full_trained:
                    direct=compute_scores_full_grad_channel_first(self.model,self.heads,x,data.bank_history(c),0) if self.heads is not None else arm_score(encode_raw_channel_first(self.model,x,0),encode_raw_channel_first(self.model,data.bank_history(c),0),None).unsqueeze(1)
                    if self.heads is None:canonical=rt.stable_topk_indices(direct[:,0].masked_fill(~mask,-float('inf')),10,largest=True)
                    else:canonical,_,_=rt.mean_mixture_topk_selection(direct,mask,.1,10)
                    assert torch.equal(canonical,rt.stable_topk_indices(prob,10,largest=True)),(self.name,c,'full direct path ID gate')
                    self.score_probe_count+=len(x)
            out.append(prob)
        return torch.stack(out,1)

    def immutable_check(self):
        assert rt.sha256_file(self.row['checkpoint'])==self.row['checkpoint_sha256']
        return dict(model=self.name,checkpoint=self.row['checkpoint'],sha256=self.row['checkpoint_sha256'],
            training_support=self.row['training_support'],evaluation_support=self.support,
            original_checkpoint_unchanged=True,p100_canonical_fallback_batches=self.pick_mismatches,
            full_direct_path_probe_queries=self.score_probe_count,metric_reproduction=self.reproduction)


class Data(rt.CellData):
    def __init__(self,cell,device='cuda:0'):
        super().__init__(cell,device)
        self.cpu_pool={s:np.load(self.pool_dir/f'{s}.npy',mmap_mode='r') for s in ('train','val','test')}
        done=json.loads((self.pool_dir/'COMPLETE.json').read_text())
        assert done['status']=='PASS' and done['all_candidate_ids_eligible_unique'] and done['all_channels']
        for s,a in self.cpu_pool.items():
            assert a.shape==(len(self.datasets[s]),self.c,100)
            assert rt.sha256_file(self.pool_dir/f'{s}.npy')==done['hashes'][s]
        self.pool_hashes=done['hashes']

    def query(self,split,start,end):
        ds=self.datasets[split];rows=np.arange(start,end)[:,None]+np.arange(self.h)[None]
        x=torch.as_tensor(np.asarray(ds.data_x[rows],np.float32),device=self.device).permute(0,2,1).contiguous()
        y=torch.as_tensor(np.asarray(ds.data_y[rows+self.h],np.float32),device=self.device).permute(0,2,1).contiguous()
        return x,y,ds.starts[start:end]

    def histories_all(self,ids):
        """ids [B,C,P], return [B,C,P,H], all channels together."""
        c=torch.arange(self.c,device=self.device)[None,:,None,None]
        return self.series[ids[...,None]+self.offsets,c]

    def values_all(self,ids):
        c=torch.arange(self.c,device=self.device)[None,:,None,None]
        return self.future_series[ids[...,None]+self.h+self.offsets,c]-self.series[ids+self.h-1,torch.arange(self.c,device=self.device)[None,:,None]][...,None]

    def pool_all(self,split,start,end):
        return torch.as_tensor(np.array(self.cpu_pool[split][start:end],np.int64,copy=True),device=self.device)
