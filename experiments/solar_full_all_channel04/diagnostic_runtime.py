"""Frozen Solar followup with the same all-channel inference as training validation."""
import sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
import solar_full_train as train
from evaluation import FrozenFull
sys.path.insert(0,str(train.ROOT/'experiments/retrieval_all_channel03'))
import runtime_all_channels as allc
run=allc.run;adapter=allc.adapter
OUT=train.ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01-SOLAR-ALL-CHANNEL04'
SUPP=train.ROOT/'results/TRACK-SOLAR-FULL-ALL-CHANNEL04-DIAGNOSTIC-SUPPLEMENT'

def inventory(cell,support):
    assert support=='FULL';rows=[]
    for m in ('V0','V5'):
        dest=train.OUT/cell/m/'seed0';meta=train.read(dest/'COMPLETE.json')
        assert meta['status']=='COMPLETE' and meta['source_hashes']==train.source_hashes()
        rows.append(dict(status='VALID',architecture=m,training_support='FULL',checkpoint=meta['checkpoint'],checkpoint_sha256=meta['checkpoint_sha256'],evidence=str(dest),selected_epoch=meta['selection']['epoch'],selection_criterion='validation Mean-Mixture RetMSE after ten epochs',seed=0))
    return rows

class SolarFrozen(adapter.Frozen):
    def precision(self):
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def reference(self,split):
        if split not in self.full_reference:
            cache=Path(self.row['evidence'])/'cache'
            assert np.array_equal(np.load(cache/f'{split}_query_start.npy'),self.data.datasets[split].starts)
            self.full_reference[split]=(np.load(cache/f'{split}_prediction.npy',mmap_mode='r'),train.read(cache/f'{split}_metrics.json'))
        return self.full_reference[split]
    @torch.no_grad()
    def probability(self,split,a,b):
        self.precision()
        if not hasattr(self,'accelerator'):
            g=train.read(train.OUT/'SMOKE_GATE.json');self.accelerator=FrozenFull(self.data,self.model,self.heads,g['query_chunk'],g['candidate_chunk'])
        p=self.accelerator.probability(split,a,b)
        ids=adapter.rt.stable_topk_indices(p.reshape(-1,p.shape[-1]),10,largest=True).reshape(b-a,self.data.c,10)
        cache=Path(self.row['evidence'])/'cache';expected=np.load(cache/f'{split}_ids.npy',mmap_mode='r')[a:b]
        assert np.array_equal(ids.cpu().numpy(),expected),(self.name,split,a,'Solar canonical ordered IDs mismatch; no serial fallback')
        return p
    def immutable_check(self):
        assert adapter.rt.sha256_file(self.row['checkpoint'])==self.row['checkpoint_sha256']
        return dict(model=self.name,checkpoint=self.row['checkpoint'],sha256=self.row['checkpoint_sha256'],original_checkpoint_unchanged=True,training_support='FULL',evaluation_support='FULL',
            channel_batch=137,original_generation_runtime='Torch2.4.1 TF32=False; documented all-channel04',metric_reproduction=self.reproduction)

def snapshot():
    d=allc.snapshot();d.update({str(train.HERE/k):v for k,v in train.source_hashes().items()});return d

def configure():
    run.OUT=OUT;run.snapshot_hashes=snapshot;adapter.snapshot_hashes=snapshot
    return run

@torch.no_grad()
def reproduce(data,models,support,out):
    assert support=='FULL';g=train.read(train.OUT/'SMOKE_GATE.json');rows=[];begin=time.perf_counter()
    for split in ('val','test'):
        sums={m.name:np.zeros(3) for m in models};count=0
        for a in range(0,len(data.datasets[split]),32):
            b=min(a+32,len(data.datasets[split]));x,y,_=data.query(split,a,b)
            for m in models:
                p=m.probability(split,a,b);pick=adapter.rt.stable_topk_indices(p.reshape(-1,p.shape[-1]),10,largest=True).reshape(b-a,data.c,10)
                # Process original eval Q chunks to preserve arithmetic shapes.
                for q in range(0,b-a,g['query_chunk']):
                    e=min(q+g['query_chunk'],b-a);vals=train.core.selected_values(data,pick[q:e],x[q:e,:,-1]);pred=vals.mean(2)
                    assert np.array_equal(pred.permute(0,2,1).cpu().numpy(),m.reference(split)[0][a+q:a+e]),'stored prediction changed'
                    ret=(vals-y[q:e,:,None]).square().mean(-1).mean(-1);err=pred-y[q:e]
                    met=torch.stack([ret,err.square().mean(-1),err.abs().mean(-1)],-1)
                    old=np.load(Path(m.row['evidence'])/'cache'/f'{split}_query_metrics.npy',mmap_mode='r')[a+q:a+e,:,:3]
                    assert np.array_equal(met.cpu().numpy(),old),'stored query metrics changed'
                    sums[m.name]+=met.reshape(-1,3).double().sum(0).cpu().numpy()
            count+=(b-a)*data.c
        assert count==len(data.datasets[split])*data.c
        for m in models:
            expected=m.reference(split)[1];got=sums[m.name]/count
            assert np.array_equal(got,np.array([expected[k] for k in ('retmse10','agg_mse10','agg_mae10')]))
            rows.append(dict(model=m.name,split=split,status='PASS',queries=len(data.datasets[split]),channels=data.c,units=count,ordered_ids_exact=True,predictions_exact=True,query_metrics_exact=True,metrics=got.tolist()))
    train.dump(out/'CANONICAL_REPRODUCTION.json',dict(status='PASS',rows=rows,seconds=time.perf_counter()-begin,all_channel_batch=True))
run.reproduce=reproduce
