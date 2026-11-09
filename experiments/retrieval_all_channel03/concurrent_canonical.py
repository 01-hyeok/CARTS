"""All-channel CUDA graph dispatch preserving canonical GEMM shapes.

This is concurrent channel execution, not a flattened [B*C,L] GEMM. All C
channels are submitted in one graph replay. No channel subset/fallback exists.
"""
import torch
import torch.nn.functional as F
from utils.full_candidate_bank import encode_raw_channel_first
from all_channel_ops import score_all,global_mixture


class ConcurrentCanonicalFull:
    def __init__(self,frozen):
        assert frozen.support=='FULL' and frozen.full_trained
        self.frozen=frozen;self.data=frozen.data;self.keys=None
        self.graphs={};self.shapes=[]

    @torch.no_grad()
    def build_keys(self):
        f=self.frozen;d=self.data;f.precision()
        # All channels resident and launched on independent streams before join.
        current=torch.cuda.current_stream();streams=[torch.cuda.Stream() for _ in range(d.c)]
        histories=[d.bank_history(c) for c in range(d.c)]
        outputs=[]
        for c,stream in enumerate(streams):
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                outputs.append(F.normalize(encode_raw_channel_first(f.model,histories[c],0),dim=-1))
        for stream in streams:current.wait_stream(stream)
        self.keys=torch.stack(outputs,0)
        self.shapes.append(dict(kind='concurrent_candidate_encoder',channels_submitted_together=d.c,
                                per_channel_input=[len(d.starts),d.h,1],output=list(self.keys.shape),
                                channel_synchronization='one join after all C launches'))

    @torch.no_grad()
    def query_graph(self,b):
        f=self.frozen;d=self.data;f.precision()
        inputs=torch.zeros(d.c,b,d.h,1,device=d.device)
        streams=[torch.cuda.Stream() for _ in range(d.c)]
        capture=torch.cuda.Stream();current=torch.cuda.current_stream()
        # Warm up every per-channel shape outside capture on side streams.
        for _ in range(3):
            for c,stream in enumerate(streams):
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    z=encode_raw_channel_first(f.model,inputs[c],0)
                    q=F.normalize(z,dim=-1)[:,None] if f.heads is None else f.heads(z)
            for stream in streams:current.wait_stream(stream)
        current.synchronize()
        graph=torch.cuda.CUDAGraph();outputs=[]
        with torch.cuda.graph(graph,stream=capture):
            for c,stream in enumerate(streams):
                stream.wait_stream(capture)
                with torch.cuda.stream(stream):
                    z=encode_raw_channel_first(f.model,inputs[c],0)
                    outputs.append(F.normalize(z,dim=-1)[:,None] if f.heads is None else f.heads(z))
            for stream in streams:capture.wait_stream(stream)
            queries=torch.stack(outputs,1)
        current.wait_stream(capture)
        self.graphs[b]=(inputs,graph,queries,streams,capture)
        self.shapes.append(dict(kind='all_channel_query_graph',graph_replays_per_batch=1,
                                channels_submitted_together=d.c,per_channel_input=[b,d.h,1],
                                stacked_output=list(queries.shape),flattened_gemm=False))

    @torch.no_grad()
    def probability(self,split,start,end):
        f=self.frozen;d=self.data;f.precision()
        if self.keys is None:self.build_keys()
        b=end-start
        if b not in self.graphs:self.query_graph(b)
        inputs,graph,queries,streams,capture=self.graphs[b]
        x,_,starts=d.query(split,start,end)
        inputs.copy_(x.permute(1,0,2)[...,None]);graph.replay()
        scores=score_all(queries,self.keys)
        mask=d.mask(starts)[:,None,:].expand(b,d.c,len(d.starts))
        self.shapes.append(dict(kind='batched_score_global_softmax',shape=list(scores.shape)))
        return scores[:,:,0].masked_fill(~mask,-float('inf')) if f.heads is None else global_mixture(scores,mask)
