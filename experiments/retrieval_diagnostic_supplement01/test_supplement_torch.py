import sys
from supplement_common import ORIGINAL
sys.path.insert(0,str(ORIGINAL))
import torch
from supplement_inference import rank_detail

def test_detail_known_ranks_scores_and_ties():
    s=torch.arange(20,dtype=torch.float32).reshape(1,1,20)
    ids=torch.arange(100,120).reshape(1,1,20);mask=torch.ones_like(s,dtype=torch.bool)
    oracle=ids[...,:10];picked,ranks,stats=rank_detail(s,ids,mask,oracle)
    assert torch.equal(picked,ids[...,10:].flip(-1))
    assert ranks.tolist()==[[list(range(20,10,-1))]]
    assert stats.tolist()==[[[45.,145.,10.,10.]]]
def test_empty_and_masked():
    s=torch.zeros(2,1,20);ids=torch.arange(20).expand(2,1,20);mask=torch.ones_like(s,dtype=torch.bool);mask[...,19]=False
    pick,ranks,stat=rank_detail(s,ids,mask,ids[...,:10])
    assert torch.equal(pick,ids[...,:10]) and not ranks.any() and not stat.any()
def test_oracle_label_does_not_change_ranking():
    s=torch.rand(2,3,100);ids=torch.randperm(100).expand(2,3,100);mask=torch.ones_like(s,dtype=torch.bool)
    a=rank_detail(s,ids,mask,ids[...,:10])[0];b=rank_detail(s,ids,mask,ids[...,-10:])[0]
    assert torch.equal(a,b)
