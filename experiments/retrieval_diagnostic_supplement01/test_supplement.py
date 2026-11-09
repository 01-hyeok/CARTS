import numpy as np
from supplement_common import thresholds,strata,interval_status,pooled_rank_stats

def test_validation_only_thresholds_and_nan():
    val=np.array([[0.,10.],[1.,20.],[2.,30.],[3.,40.]])
    t=thresholds(val);assert np.array_equal(t,[[1.,2.],[20.,30.]])
    labels=strata(np.array([[100.,5.],[np.nan,25.]]),t)
    assert np.array_equal(labels,[[2,0],[-1,1]])
def test_tied_features_do_not_create_fake_groups():
    a=np.ones((4,2));assert np.all(strata(a,thresholds(a))==0)
def test_undefined_period():
    a=np.full((10,2),np.nan);assert np.all(strata(a,thresholds(a))==-1)
def test_ci_degeneracy():
    assert interval_status(2161,2880)[0]=='INSUFFICIENT_TIME_BLOCKS'
    assert interval_status(2545,1344)[0]=='INSUFFICIENT_TIME_BLOCKS'
    assert interval_status(2785,384)[1]>7
def test_pooled_quantiles_not_mean_query_quantiles():
    ranks=np.array([11,12,100,100,100,100,100]);h=np.bincount(ranks,minlength=101)
    s=pooled_rank_stats(h)
    assert s['pooled_median']==np.median(ranks)
    assert s['pooled_p90']==np.quantile(ranks,.9)
    assert s['pooled_mean']==ranks.mean()
def test_empty_missed():assert pooled_rank_stats(np.zeros(101,int))['pooled_median'] is None
def test_histogram_linear_interpolation():
    x=np.array([11,12,20,99]);s=pooled_rank_stats(np.bincount(x,minlength=101))
    assert s['pooled_median']==16 and np.isclose(s['pooled_p90'],75.3)
