"""Process-local, exact-kernel patch for two proven FP32 tails only."""
import sys,json,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'cublas_recovery02'))
spec=importlib.util.spec_from_file_location('_tail_reference02',HERE.parent/'cublas_recovery02/probe.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
sys.path.insert(0,str(HERE.parent/'cublas_recovery03'))
from source_contract import transform
from scope import applies
import packed_batch,adapter,torch
ROOT=prior.ROOT;bridge=prior.bridge
OUT=ROOT/'results/TRACK-HEAD-SHIFT-DIAGNOSTIC01/extensions_d5_d6/cublas_recovery04'

def install():
    proof_path=OUT.parent/'cublas_recovery03/QUERY_PROBE.json'
    proof=json.loads(proof_path.read_text())
    assert proof['status']=='PASS_EXACT_QUERY_AND_SCORE_FOUND' and proof['chosen']=='cuda12_lt_exact_norm'
    for n,h in proof['source_hashes'].items():assert bridge.sha(HERE.parent/'cublas_recovery03'/n)==h
    ns=dict(packed_batch.__dict__);exec(compile(transform(adapter._p100_short,proof['chosen']),str(HERE/'weather_patch.py')+'::exact_query','exec'),ns)
    query=ns['short_query'];original=bridge.Frozen.full_logits
    @torch.no_grad()
    def full_logits(self,split,a,b):
        if not applies(self.data.cell,self.row['training_support'],self.n,b-a):return original(self,split,a,b)
        self.precision();assert not self.tf32
        assert self.data.c==21 and len(self.data.starts)==36216
        if self.n==1:assert self.row['checkpoint_sha256']==proof['checkpoint_sha256']
        if self.batch.keys is None:self.batch.build_keys()
        x,_,starts=self.data.query(split,a,b)
        q=query(self.batch,x) if self.n==1 and b-a==7 else prior.batched_query(self,split,a,b)
        qq=q[:,:,0].permute(1,0,2).contiguous()
        if b-a==7:
            if not hasattr(self.batch,'recovery_split'):self.batch.recovery_split=prior.Split64()
            score=self.batch.recovery_split(qq,self.batch.keys)
        else:
            if not hasattr(self.batch,'recovery_lt12'):self.batch.recovery_lt12=prior.CublasLtBatch('libcublasLt.so.12')
            score=self.batch.recovery_lt12.score(qq,self.batch.keys,68,1,0)
        mask=self.data.mask(starts)[:,None].expand(b-a,self.data.c,-1)
        return score.permute(1,0,2)[:,:,None].contiguous(),mask
    bridge.Frozen.full_logits=full_logits
    return original
