"""Exact-shape candidate encoder cache for frozen validation only.

No approximate TopK margin threshold is used. Full B32 candidate encoding always
has the legacy 3200-row GEMM shape; short final batches use the original path.
Query encoding, head scores, mixture, TopK and metrics remain canonical.
"""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import runtime as rt
import torch


class CandidateCache:
    def __init__(self,original):
        self.original=original;self.channel=None;self.bank=None;self.fallback=False
        self.stats=dict(channels_cached=0,channel_probe_mismatches=0,full_batches_cached=0,
            short_batches_canonical=0,channel_fallback_batches=0,cache_build_seconds=0.,
            candidate_encode_rows=3200,pool_budget=100,exact_probe_elements=0)

    @torch.no_grad()
    def __call__(self,model,heads,x,data,c,ids,reference_all_channels=False,query_channel=None):
        assert not model.training and not reference_all_channels and ids.shape[-1]==100
        if len(x)!=32:
            self.stats['short_batches_canonical']+=1
            return self.original(model,heads,x,data,c,ids,reference_all_channels,query_channel)
        if self.channel!=c:
            self.bank=None;self.channel=c;self.fallback=False
            begin=time.perf_counter();history=data.bank_history(c);chunks=[]
            for start in range(0,len(data.starts),3200):
                block=history[start:start+3200];n=len(block)
                if n<3200:
                    block=torch.cat([block,torch.zeros((3200-n,data.h,1),device=data.device,dtype=block.dtype)])
                chunks.append(rt.encode_raw(model,block,0)[:n])
            self.bank=torch.cat(chunks);del history,chunks
            # Conservative channel fallback if even the initial real batch
            # does not match exactly, rather than assuming kernel equivalence.
            reference=rt.encode_raw(model,data.histories(ids,c).reshape(3200,data.h,1),0)
            cached=self.bank[ids].reshape(3200,-1)
            self.stats['exact_probe_elements']+=reference.numel()
            if not torch.equal(reference,cached):
                self.fallback=True;self.bank=None;self.stats['channel_probe_mismatches']+=1
            else:self.stats['channels_cached']+=1
            self.stats['cache_build_seconds']+=time.perf_counter()-begin
        if self.fallback:
            self.stats['channel_fallback_batches']+=1
            return self.original(model,heads,x,data,c,ids,reference_all_channels,query_channel)
        zq=rt.encode_raw(model,x,c if query_channel is None else query_channel)
        zk=rt.F.normalize(self.bank[ids],dim=-1)
        self.stats['full_batches_cached']+=1
        if heads is None:
            return torch.einsum('bd,bpd->bp',rt.F.normalize(zq,dim=-1),zk).unsqueeze(1)
        return torch.einsum('bsd,bpd->bsp',heads(zq),zk)


@torch.no_grad()
def evaluate(data,model,heads,split,detailed=False,dest=None,baselines=False,trace=None):
    assert split=='val' and not detailed and dest is None and not baselines
    model.eval()
    if heads is not None:heads.eval()
    original=rt.scores;cache=CandidateCache(original)
    def wrapped(*args,**kwargs):
        score=cache(*args,**kwargs)
        if trace is not None:trace(*args,score=score,**kwargs)
        return score
    rt.scores=wrapped
    try:result=rt.evaluate(data,model,heads,split)
    finally:rt.scores=original
    result['validation_acceleration']=dict(method='canonical-shape candidate bank cache',**cache.stats)
    return result
