"""Proven E96 Full V5 one-row kernel, with all seven channels batched."""
import sys,inspect,textwrap,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import d123_full_population as pop
import packed_batch,torch
import torch.nn.functional as F
from cublaslt_batch import CublasLtBatch
from exact_short import canonical_norm128
from scope import applies,choose
OUT=pop.OUT/'population_recovery06'


def install():
    path=OUT/'NORM_PROBE_GATE.json'
    proof=json.loads(path.read_text());qmode,normmode=choose(proof)
    assert pop.sha(HERE/'probe_norm.py')==proof['source_sha256']
    prior=json.loads((pop.OUT/'population_recovery05/PROBE_GATE.json').read_text())
    assert all(r['keys_exact'] and r['encoder_exact'] for r in prior['rows'])
    assert pop.sha(HERE.parent/'population_recovery05/probe.py')==prior['source_sha256']
    source=textwrap.dedent(inspect.getsource(packed_batch.PackedChannelFull.short_query))
    source=source[:source.index('    if self.frozen.heads is None:')]+'    return z\n'
    ns=dict(packed_batch.__dict__);exec(compile(source,__file__+'::encoder_prefix','exec'),ns)
    prefix=ns['short_query'];original=pop.Frozen.full_logits

    @torch.no_grad()
    def exact(self,split,a,b):
        if not applies(self.data.cell,self.row['training_support'],self.n,b-a):
            return original(self,split,a,b)
        self.precision();d=self.data
        assert self.tf32 and d.c==7 and len(d.starts)==8449
        assert self.row==proof['checkpoint'] and pop.sha(self.row['checkpoint'])==self.row['checkpoint_sha256']
        if self.batch.keys is None:self.batch.build_keys()
        if not hasattr(self.batch,'recovery06_lt'):
            self.batch.recovery06_lt=CublasLtBatch()
        lt=self.batch.recovery06_lt
        x,_,starts=d.query(split,a,b);z=prefix(self.batch,x)
        assert (qmode,normmode)==('lt11_tf32','canonical5')
        weight=self.heads.W.reshape(-1,128)[None].expand(d.c,-1,-1).contiguous()
        zz=z.permute(1,0,2).contiguous()
        v=lt.score(zz,weight,77,1,0).reshape(d.c,5,128)
        q=canonical_norm128(v,5)[None]
        qq=q.permute(1,0,2,3).reshape(d.c,5,128).contiguous()
        v=torch.bmm(qq,self.batch.keys.transpose(1,2))
        mask=d.mask(starts)[:,None].expand(1,d.c,-1)
        return v[:,None].permute(1,0,2,3).contiguous(),mask
    pop.Frozen.full_logits=exact
    return dict(probe_gate_sha256=pop.sha(path),query_mode=qmode,norm_mode=normmode,score_mode='torch_bmm',
                scope='ETTh1_96/V5_FULL/B1 only',source_hashes={n:pop.sha(HERE/n) for n in ['scope.py','exact_patch.py']})
