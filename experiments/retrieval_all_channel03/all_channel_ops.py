"""Frozen full-support operations. The complete channel axis is always retained.

Only query and candidate axes may be chunked. No per-channel fallback exists.
These operations are experimental until a saved canonical-ID gate passes.
"""
import torch
import torch.nn.functional as F


def channel_linear(x, layer):
    """[C,B,D] -> [C,B,E] in one batched GEMM, with shared weights."""
    c, b, _ = x.shape
    weight = layer.weight.t()[None].expand(c, -1, -1)
    if layer.bias is None:
        return torch.bmm(x, weight)
    bias = layer.bias[None, None].expand(c, b, -1)
    return torch.baddbmm(bias, x, weight)


def channel_sequence(x, seq):
    for layer in seq:
        if isinstance(layer, torch.nn.Linear):
            x = channel_linear(x, layer)
        else:
            assert not isinstance(layer, torch.nn.Dropout) or not layer.training
            x = layer(x)
    return x


def encode_all(model, history, mode='flat'):
    """history [B,C,L], all C encoded by each GPU operation."""
    from models.RelationStage1 import transform_relation_features
    b, c, length = history.shape
    views = transform_relation_features(history.permute(0, 2, 1), model.relation_input_space)
    rows = torch.stack(views, dim=2).permute(0, 3, 2, 1).contiguous()
    enc = model.encoder
    if mode == 'flat':
        return enc(rows.reshape(b*c, rows.shape[2], rows.shape[3])).reshape(b,c,-1)
    assert mode == 'grouped' and enc.encoder_type == 'mlp' and enc.self_fill == 'linear'
    x = rows.permute(1,0,2,3).reshape(c,b,-1).contiguous()
    x = channel_sequence(x, enc.encoder)
    x = enc.norm(x)
    x = channel_sequence(x, enc.proj)
    if enc.retrieval_similarity != 'l2':
        x = F.normalize(x,dim=-1)
    return x.permute(1,0,2).contiguous()


def score_all(q, keys):
    """q [B,C,S,D], keys [C,N,D] -> score [B,C,S,N]."""
    b,c,s,d=q.shape
    scores=torch.bmm(q.permute(1,0,2,3).reshape(c,b*s,d),keys.transpose(1,2))
    return scores.reshape(c,b,s,keys.shape[1]).permute(1,0,2,3).contiguous()


def global_mixture(scores, mask, temperature=.1):
    """Single global candidate softmax for each query/channel/head."""
    assert mask.shape == scores.shape[:2]+scores.shape[-1:]
    return torch.softmax(scores.masked_fill(~mask[:,:,None,:],-float('inf'))/temperature,-1).mean(2)


def distance_all(values, qlast, target):
    """Reuse canonical distance arithmetic on [B*C,P,H], no channel loop."""
    from utils.candidate_pool import pooled_future_mse
    b,c,p,h=values.shape
    return pooled_future_mse(values.reshape(b*c,p,h),qlast.reshape(b*c),target.reshape(b*c,h)).reshape(b,c,p)


class AllChannelFull:
    def __init__(self,frozen,mode='flat',candidate_chunk=1024):
        assert frozen.support=='FULL'
        self.frozen=frozen;self.data=frozen.data;self.mode=mode
        self.chunk=candidate_chunk;self.keys=None;self.shapes=[]

    @torch.no_grad()
    def build_keys(self):
        f=self.frozen;d=self.data;f.precision();parts=[]
        for start in range(0,len(d.starts),self.chunk):
            stop=min(start+self.chunk,len(d.starts))
            pos=torch.arange(start,stop,device=d.device)[:,None]+d.offsets
            history=d.series[pos].permute(0,2,1).contiguous()
            self.shapes.append(dict(kind='candidate_encoder',input=list(history.shape)))
            parts.append(F.normalize(encode_all(f.model,history,self.mode),dim=-1))
        self.keys=torch.cat(parts,0).permute(1,0,2).contiguous()
        assert self.keys.shape[:2]==(d.c,len(d.starts))

    @torch.no_grad()
    def probability(self,split,start,end):
        f=self.frozen;d=self.data;f.precision()
        if self.keys is None:self.build_keys()
        x,_,starts=d.query(split,start,end)
        self.shapes.append(dict(kind='query_encoder',input=list(x.shape)))
        z=encode_all(f.model,x,self.mode);b,c,dim=z.shape
        q=F.normalize(z,dim=-1)[:,:,None] if f.heads is None else f.heads(z.reshape(b*c,dim)).reshape(b,c,-1,dim)
        scores=score_all(q,self.keys)
        mask=d.mask(starts)[:,None,:].expand(b,c,len(d.starts))
        self.shapes.append(dict(kind='score_global_softmax',input=list(scores.shape),mask=list(mask.shape)))
        return scores[:,:,0].masked_fill(~mask,-float('inf')) if f.heads is None else global_mixture(scores,mask)
