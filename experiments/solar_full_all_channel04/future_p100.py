"""Future P100 calls only. Explicitly unactivated pending full saved-ID GPU gates."""
import numpy as np
import torch
import torch.nn.functional as F
import batch_core as core

def pooled_scores(model,heads,x,history):
    # Query B,C,H and histories B,C,P,H: each encoder call contains every C.
    b,c,p,h=history.shape;assert x.shape==(b,c,h)
    q=core.encode(model,x);key=F.normalize(core.encode(model,history),dim=-1)
    q=F.normalize(q,dim=-1)[:,:,None] if heads is None else heads(q.reshape(b*c,-1)).reshape(b,c,-1,q.shape[-1])
    return torch.einsum('bcsd,bcpd->bcsp',q,key)

def pooled_probability(score):return torch.softmax(score/.1,-1).mean(2)

class FutureP100:
    def __init__(self,frozen,query_chunk=1,candidate_chunk=100):
        assert frozen.support=='P100' and not frozen.full_trained
        assert 0<query_chunk<=32 and 0<candidate_chunk<=100
        self.frozen=frozen;self.query_chunk=query_chunk;self.candidate_chunk=candidate_chunk
        self.activation='NOT_ACTIVATED_FULL_VALIDATION_TEST_ID_GATE_REQUIRED'
    @torch.no_grad()
    def probability(self,split,a,b):
        f=self.frozen;data=f.data;f.precision();parts=[]
        for start in range(a,b,self.query_chunk):
            end=min(start+self.query_chunk,b);x,_,starts=data.query(split,start,end);ids=data.pool_all(split,start,end)
            assert ids.shape==(end-start,data.c,100)
            assert bool((ids.sort(-1).values[...,1:]!=ids.sort(-1).values[...,:-1]).all())
            assert bool(data.mask(starts)[:,None].expand(-1,data.c,-1).gather(-1,ids).all())
            # Keep candidate ID/order; chunking precedes a single global P100 softmax.
            qs=core.encode(f.model,x);B,C,D=qs.shape
            qs=F.normalize(qs,dim=-1)[:,:,None] if f.heads is None else f.heads(qs.reshape(B*C,D)).reshape(B,C,-1,D)
            scores=[]
            for p in range(0,100,self.candidate_chunk):
                key=F.normalize(core.encode(f.model,data.histories_all(ids[...,p:p+self.candidate_chunk])),dim=-1)
                scores.append(torch.einsum('bcsd,bcpd->bcsp',qs,key))
            prob=pooled_probability(torch.cat(scores,-1));pick=core.rt.stable_topk_indices(prob.reshape(B*C,100),10,largest=True).reshape(B,C,10)
            actual=ids.gather(-1,pick).cpu().numpy();expected=f.reference(split)[0][start:end]
            assert np.array_equal(actual,expected),(data.cell,split,start,'P100 all-channel ordered ID mismatch; no serial fallback')
            parts.append(prob)
        return torch.cat(parts)
