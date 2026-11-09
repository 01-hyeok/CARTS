import sys
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/retrieval_failure_attribution01'))
import diagnostic_adapter as adapter
import torch
from full_precision_policy import precision_policy

old_snapshot=adapter.snapshot_hashes
old_immutable=adapter.Frozen.immutable_check
original_probability=adapter.Frozen.probability
def precision(self):
    p=precision_policy(self.row['training_support'],self.row['architecture'])
    torch.backends.cuda.matmul.allow_tf32=p['matmul_tf32']
    torch.backends.cudnn.allow_tf32=p['cudnn_tf32']
def immutable(self):
    r=old_immutable(self)
    r['precision_policy']=precision_policy(self.row['training_support'],self.row['architecture'])
    r['precision_repair']='Full V0 matmul TF32=False; Full V5=True; each checked against stored canonical predictions'
    if self.support=='FULL' and self.full_trained and self.heads is None:
        import json
        p=adapter.OUT/'full_v0_canonical_scores'/self.data.cell/'COMPLETE.json'
        g=json.loads(p.read_text())
        r['canonical_score_cache']=str(p)
        r['canonical_score_gate_sha256']=adapter.rt.sha256_file(p)
        r['score_runtime']=g['provenance']['torch']
        r['candidate_score_support']='all bank candidates, original order; no filtering beyond original mask'
    return r
def snapshot():
    out=old_snapshot()
    out.update({str(p.relative_to(ROOT)):adapter.rt.sha256_file(p) for p in HERE.glob('*.py')})
    return out
def probability(self,split,start,end):
    if self.support=='FULL' and self.full_trained and self.heads is None:
        import json,numpy as np
        dest=adapter.OUT/'full_v0_canonical_scores'/self.data.cell
        done=json.loads((dest/'COMPLETE.json').read_text())
        assert done['status']=='PASS' and done['provenance']['checkpoint_sha256']==self.row['checkpoint_sha256']
        path=dest/split/f'batch{start//32:05d}.npz'
        assert adapter.rt.sha256_file(path)==done['file_hashes'][str(path.relative_to(dest))]
        with np.load(path) as z:
            assert np.array_equal(z['query_start'],self.data.datasets[split].starts[start:end])
            scores=torch.as_tensor(z['scores'].copy(),device=self.data.device)
        assert tuple(scores.shape)==(end-start,self.data.c,len(self.data.starts))
        return scores
    return original_probability(self,split,start,end)
adapter.Frozen.precision=precision
adapter.Frozen.immutable_check=immutable
adapter.Frozen.probability=probability
adapter.snapshot_hashes=snapshot
