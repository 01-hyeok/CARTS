import numpy as np
from .d56_metrics import *

def test_entropy_size_and_equal_distributions():
    z=np.zeros(10);s=lag_summary(np.arange(10)-5,z,z,z,z)
    assert s['unique_best_lags']==10 and s['sign_coverage']==3
    np.testing.assert_allclose(s['useful_candidate_lag_entropy'],np.log(10))
    assert hist_distance({'0':2,'1':2},{'0':1,'1':1})['total_variation']==0

def test_empty_groups_are_not_zero_effect():
    assert lag_summary([],[],[],[],[])=={'count':0}
    assert hist_distance({}, {'0':1})['total_variation'] is None

def test_validation_has_no_test_target_argument():
    rng=np.random.RandomState(0);s=rng.rand(4,3,20);m=rng.rand(4,3,20)
    x=select_validation(s,s*.1,s*.2,m,'ShiftOnly')
    assert x['selected']['beta']==0 and not x['test_used']
    assert len(x['grid'])==len(ALPHAS)

def test_recoverability_identity_and_monotonicity():
    d=np.arange(1,31,dtype=float);ids=np.arange(30)[::-1];rows=[]
    for R in [10,20,30]:
        selected=ids[:R];m=d[selected]
        v=d6_unit(selected,m,np.arange(R)[::-1],np.zeros(R),np.zeros(R),np.arange(10),d[:10],
             dict(alpha=0.,beta=0.,score_scale=1.,recent_scale=1.),10)
        rows.append(dict(R=R,**v))
        np.testing.assert_allclose(v['total_regret'],v['candidate_limitation_regret']+v['reranking_regret'])
    validate_r_curve(rows)
    assert rows[-1]['oracle10_coverage']==1

def test_future_perturbation_does_not_change_selected_ids():
    ids=np.arange(30);score=np.linspace(1,0,30);gain=np.sin(ids);recent=np.cos(ids)**2
    sel=dict(alpha=.3,beta=.1,score_scale=.2,recent_scale=1.)
    first=d6_unit(ids,ids+1.,score,gain,recent,ids[:10],ids[:10]+1.,sel)
    second=d6_unit(ids,ids+100.,score,gain,recent,ids[:10],ids[:10]+100.,sel)
    assert first['selected_ids']==second['selected_ids']

def test_loss_tail_uses_timestamp_means_and_zero_denominator():
    assert loss_tail_contribution(np.zeros((32,7)),.01) is None
    x=np.zeros((32,7));x[-1]=1
    assert loss_tail_contribution(x,.01)==1

def test_oracle_must_be_same_support_and_not_clamped():
    import pytest
    with pytest.raises(AssertionError,match='negative regret'):
        d6_unit(np.arange(10),np.zeros(10),np.arange(10)[::-1],np.zeros(10),np.zeros(10),np.arange(10),np.ones(10),
                dict(alpha=0.,beta=0.,score_scale=1.,recent_scale=1.))
