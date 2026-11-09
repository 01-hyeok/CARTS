"""Concurrent all-channel execution without numerically-changing CUDA capture.

Every channel is submitted before any channel join. Python loops enqueue work;
they do not synchronize channels. This is deliberately not a flattened GEMM.
"""
import torch
import torch.nn.functional as F
from utils.full_candidate_bank import encode_raw_channel_first
from concurrent_canonical import ConcurrentCanonicalFull
from all_channel_ops import score_all,global_mixture


class ConcurrentStreamsFull(ConcurrentCanonicalFull):
    @torch.no_grad()
    def probability(self,split,start,end):
        f=self.frozen;d=self.data;f.precision()
        if self.keys is None:self.build_keys()
        x,_,starts=d.query(split,start,end);b=end-start
        if not hasattr(self,'streams'):self.streams=[torch.cuda.Stream() for _ in range(d.c)]
        current=torch.cuda.current_stream();outputs=[]
        for c,stream in enumerate(self.streams):
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                z=encode_raw_channel_first(f.model,x[:,c,:,None].contiguous(),0)
                outputs.append(F.normalize(z,dim=-1)[:,None] if f.heads is None else f.heads(z))
        for stream in self.streams:current.wait_stream(stream)
        queries=torch.stack(outputs,1)
        scores=score_all(queries,self.keys)
        mask=d.mask(starts)[:,None,:].expand(b,d.c,len(d.starts))
        self.shapes.append(dict(kind='concurrent_all_channels_without_capture',channels_submitted_before_join=d.c,
                               query_input=list(x.shape),query_embeddings=list(queries.shape),
                               score_global_softmax=list(scores.shape),channel_subset=False))
        return scores[:,:,0].masked_fill(~mask,-float('inf')) if f.heads is None else global_mixture(scores,mask)
