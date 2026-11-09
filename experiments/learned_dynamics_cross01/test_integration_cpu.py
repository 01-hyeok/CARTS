"""CPU integration on actual Train feature caches; never loads query futures."""
import sys,json,copy,io
import numpy as np
import torch
from experiment_common import *
from learned_scores import *
from train_models import Features,Arm
from final_evaluation import ranks,corr_ranks
sys.path.insert(0,str(ROOT/'experiments/retrieval_failure_mechanism02'))
from adapter import Data
from scipy.stats import rankdata,spearmanr

def main():
    torch.set_num_threads(1);torch.manual_seed(0);checks=[]
    v=torch.tensor([[3.,3.,1.,7.,99.],[2.,2.,2.,2.,2.]])
    m=torch.tensor([[True,True,True,True,False],[True,True,True,True,True]])
    r=ranks(v,m)
    for i in range(2):np.testing.assert_allclose(r[i,m[i]].numpy(),rankdata(v[i,m[i]].numpy())-1)
    _,defined=corr_ranks(r,r,m);assert bool(defined[0]) and not bool(defined[1])
    for cell in CELLS:
        d=Data(cell,'cpu');features=Features(d,OUT/'features'/cell);rows=np.array([0,len(d.datasets['train'])//2,len(d.datasets['train'])-1])
        ids=torch.tensor(np.array(d.cpu_pool['train'][rows],copy=True))
        for kind in ('L2','DYN','CROSS'):
            x=features.pair('train',rows,ids,kind)
            assert all(torch.isfinite(y).all() for y in x) if isinstance(x,tuple) else torch.isfinite(x).all()
        mask=torch.ones_like(ids,dtype=torch.bool);p=torch.softmax(torch.randn(ids.shape),-1)
        # Synthetic teacher only. Actual data target futures are not accessed here.
        teacher=torch.softmax(torch.randn(ids.shape,dtype=torch.float64),-1)
        for name in ARMS:
            torch.manual_seed(0);arm=Arm(name);pair=features.pair('train',rows,ids,arm.kind);arm.calibrate(pair)
            raw=arm.raw_chunks(features,'train',rows,ids,31);loss=kl_from_log_probability(teacher,arm.logprob(p,raw,mask),mask)
            loss.backward();assert torch.isfinite(loss)
            if arm.fused:assert arm.fusion.beta.grad is not None and torch.isfinite(arm.fusion.beta.grad)
            else:assert any(v.grad is not None and v.grad.abs().sum()>0 for v in arm.scorer.parameters())
        checks.append(dict(cell=cell,all_channels=d.c,train_rows=rows.tolist(),arms=6,finite=True))
        del features,d
    gate=dict(status='PASS',actual_train_feature_checks=checks,metrics_rank_ties='matches scipy rankdata; constants undefined',source_hashes=sources(),at=now(),scope='CPU integration only; GPU baseline and train save/load gates still required')
    dump(OUT/'CPU_IMPLEMENTATION_GATE.json',gate);print(json.dumps(gate,indent=2))

if __name__=='__main__':main()
