import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pytest
import torch
import pandas as pd
from pipeline import paa16, temporal_raw, time_features, select_scores, validate_ids, approximate_set_oracle
from utils.candidate_pool import compute_coarse_delta_last_cosine_scores, build_coarse_pool

@pytest.mark.parametrize('length',[96,720])
def test_paa_exact(length):
    x=torch.arange(length).float()[None,:]
    out=paa16(x)
    assert out.shape==(1,16)
    assert torch.equal(out,x.reshape(1,16,length//16).mean(-1))
def test_paa_rejects_other_length():
    with pytest.raises(ValueError):paa16(torch.zeros(1,100))
def test_count_unique_mask_and_forbidden_ids():
    s=torch.arange(140).float()[None].repeat(2,1); m=torch.ones_like(s,dtype=torch.bool);m[:,110:]=False
    ids=select_scores(s,m);validate_ids(ids,m);assert ids.max()<110
def test_insufficient_pool_fails():
    with pytest.raises(ValueError):select_scores(torch.zeros(1,110),torch.arange(110)[None]<99)
@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_nonfinite_fails(bad):
    s=torch.zeros(1,110);s[0,0]=bad
    with pytest.raises(ValueError):select_scores(s,torch.ones_like(s,dtype=torch.bool))
def test_baseline_identical_and_batch_invariant():
    torch.manual_seed(1);q=torch.randn(5,96,2);k=torch.randn(120,96,2);mask=torch.ones(5,120,dtype=torch.bool)
    all_=compute_coarse_delta_last_cosine_scores(q,k,0)
    expected=build_coarse_pool(all_,mask,100)
    actual=torch.cat([select_scores(compute_coarse_delta_last_cosine_scores(q[i:i+1],k,0),mask[i:i+1]) for i in range(5)])
    assert torch.equal(expected,actual)
def test_timestamp_midpoint_and_split_global_alignment():
    dates=pd.date_range('2020-01-01',periods=1000,freq='10min'); starts=torch.tensor([0,200,500]);L=96
    f=time_features(dates[starts.numpy()+L//2]);ref=time_features([dates[int(s)+48] for s in starts]);assert torch.equal(f,ref)
def test_timestamp_missing_fails():
    with pytest.raises(ValueError):time_features([pd.NaT])
def test_circular_time_wrap():
    q=torch.tensor([[23.,0,12,50]]);k=torch.tensor([[1.,0,1,10]])
    val=temporal_raw(q,k,96,144)
    expected=.6*torch.exp(torch.tensor(-1.))+.2+.1+.2*torch.exp(torch.tensor(-1/3))
    assert torch.allclose(val,expected)
def test_time_batch_invariant_global_normalization():
    f=time_features(pd.date_range('2020-01-01',periods=16,freq='10min'))
    a=temporal_raw(f,f,96,144);norm=a.max();b=torch.cat([temporal_raw(f[i:i+1],f,96,144)/norm for i in range(len(f))]);assert torch.equal(a/norm,b)
def test_pearson_centering_affine_invariance():
    import torch.nn.functional as F
    x=torch.arange(96).float()[None];y=3*x+7
    assert torch.allclose(F.normalize(x-x.mean(-1,keepdim=True),dim=-1),F.normalize(y-y.mean(-1,keepdim=True),dim=-1))
def test_oracle_is_diagnostic_only_signature():
    import inspect
    assert 'future' not in inspect.signature(select_scores).parameters
def test_greedy_set_oracle_exact_constant_case():
    v=torch.ones(2,100,96)*3;target=torch.ones(2,96)*3
    assert torch.equal(approximate_set_oracle(v,target),torch.zeros(2))
def test_raft_train_exclusion_and_validation_memory_only():
    import numpy as np
    from utils.relation_memory import RelationMemorySampler
    sampler=object.__new__(RelationMemorySampler);sampler.starts=np.arange(1000);sampler.seq_len=96;sampler.pred_len=96;sampler.mask_mode='raft'
    mask,_=sampler.valid_mask_batch([500,1500])
    assert not bool(mask[0,309:692].any())
    assert bool(mask[0,:309].all()) and bool(mask[0,692:].all())
    assert bool(mask[1].all()) and mask.shape[1]==1000
def test_teacher_sign_lower_future_mse_gets_higher_probability():
    from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
    d=torch.tensor([[0.,1.,2.]])
    p=normalized_teacher_prob(d,torch.ones_like(d,dtype=torch.bool),.1)
    assert p[0,0]>p[0,1]>p[0,2]

def test_actual_timestamp_duplicate_gap_preserves_row_mapping():
    from pipeline import validate_timestamp_series
    dates=pd.Series(pd.date_range('2020-01-01',periods=1001,freq='10min'))
    dates.iloc[20]=dates.iloc[19]
    parsed,steps,audit=validate_timestamp_series(dates.astype(str))
    assert parsed.equals(dates)
    assert steps==144 and audit['duplicates']==1 and audit['gaps']==1
def test_timestamp_missing_or_decreasing_still_fails():
    from pipeline import validate_timestamp_series
    with pytest.raises(ValueError):validate_timestamp_series(pd.Series(['2020-01-02',None]))
    with pytest.raises(ValueError):validate_timestamp_series(pd.Series(['2020-01-02','2020-01-01']))
