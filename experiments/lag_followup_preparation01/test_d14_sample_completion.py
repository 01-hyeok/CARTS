import numpy as np
from d14_sample_completion import rerank_metrics,paired_ci,correlation

def test_identity_and_undefined():
    o=list(range(10));d={i:float(i) for i in range(12)}
    r=rerank_metrics(o,o,o,d)
    assert r['ndcg10']==1 and r['missed_oracle_recovery_rate'] is None and r['false_selected_promotion_rate'] is None

def test_recovery_and_false_promotion():
    d={i:float(i) for i in range(14)};o=list(range(10));b=list(range(8))+[10,11];s=list(range(9))+[12]
    r=rerank_metrics(s,b,o,d)
    assert r['missed_oracle_recovery_rate']==.5 and r['false_selected_promotion_rate']==.5
    assert r['false_selected_promoted_fraction_of_k']==.1 and r['ndcg10']<1

def test_missing_groups_stay_missing():
    r=paired_ci(np.full((3,2),np.nan),np.arange(3),2)
    assert r['delta'] is None and not r['sufficient']

def test_timestamp_joint_resampling_constant_difference():
    r=paired_ci(np.ones((8,3))*2,np.arange(8)*10,10)
    assert r['delta']==r['ci_low']==r['ci_high']==2 and r['blocks']==8

def test_undefined_constant_correlation():
    assert correlation([1,1,1],[0,1,2]) is None
    assert np.isclose(correlation([1,2,3],[3,2,1]),-1)
