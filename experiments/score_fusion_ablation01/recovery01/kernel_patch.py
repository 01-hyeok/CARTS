"""Exact canonical B1 full-trained V5 scores, all seven channels in one batch.

Scope is only ETTh1 H192/H336/H720 Full-trained V5, one-query tail.
The earlier all-channel Lt score path preserved Top10 but changed raw cosine.
Use canonical five-row head normalization and strided bmm as proven for E96.
"""
import inspect,textwrap,types
import torch
import packed_batch
from cublaslt_batch import CublasLtBatch
from exact_short import canonical_norm128

CELLS={'ETTh1_192','ETTh1_336','ETTh1_720'}
def applicable(cell,training,heads,batch):return cell in CELLS and training=='FULL' and heads==5 and batch==1

def install(f):
    original=f.full_logits
    src=textwrap.dedent(inspect.getsource(packed_batch.PackedChannelFull.short_query))
    src=src[:src.index('    if self.frozen.heads is None:')]+'    return z\n'
    ns=dict(packed_batch.__dict__);exec(compile(src,__file__+'::canonical_encoder_prefix','exec'),ns);prefix=ns['short_query']
    @torch.no_grad()
    def exact(self,split,a,b):
        d=self.data
        if not applicable(d.cell,self.support,self.n,b-a):return original(split,a,b)
        assert self.tf32 and d.c==7
        self.precision()
        if self.batch.keys is None:self.batch.build_keys()
        if not hasattr(self.batch,'fusion_recovery_lt'):self.batch.fusion_recovery_lt=CublasLtBatch()
        x,_,starts=d.query(split,a,b);z=prefix(self.batch,x)
        w=self.heads.W.reshape(-1,128)[None].expand(d.c,-1,-1).contiguous()
        v=self.batch.fusion_recovery_lt.score(z.permute(1,0,2).contiguous(),w,77,1,0).reshape(d.c,5,128)
        q=canonical_norm128(v,5)
        g=torch.bmm(q,self.batch.keys.transpose(1,2))[None]
        m=d.mask(starts)[:,None].expand(1,d.c,-1)
        return g.contiguous(),m
    f.full_logits=types.MethodType(exact,f)
