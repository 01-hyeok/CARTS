import numpy as np
from core import features,bins,swap_masks,block_ci,delta_decomposition

def test_constant_features_finite():
 f=features(np.ones((3,96,2)));assert f.shape==(3,2,4) and np.isfinite(f).all() and np.array_equal(f,np.zeros_like(f))
def test_trend_direction_and_input_only():
 x=np.arange(96,dtype=float)[None,:,None];a=features(x);b=features(-x);assert a[0,0,0]>0 and b[0,0,0]<0
 assert np.allclose(a,features(x+500),atol=1e-6)
def test_periodicity_signal():
 x=np.sin(2*np.pi*np.arange(96)/12)[None,:,None];assert features(x)[0,0,2]>.9

def test_validation_cuts_do_not_use_test():
 v=np.arange(120).reshape(10,3,4);a,labels=bins(v,v);b,_=bins(v,v*1e9);assert np.array_equal(a,b) and labels.max()==2

def test_swap_zero_and_cardinality():
 a=np.array([[1,2,3],[3,2,1]]);b=np.array([[1,2,4],[3,2,1]]);i,o=swap_masks(a,b);assert np.array_equal(i.sum(-1),[1,0]);assert np.array_equal(i.sum(-1),o.sum(-1))
def test_delta_identity():
 r=np.random.default_rng(7);e=r.normal(size=(4,20));d=r.normal(size=(4,20));a,b=delta_decomposition(e,d,.4);assert np.allclose(a+b,np.mean((e+.4*d)**2-e**2,axis=-1))
def test_block_paired_channels_and_seed():
 x=np.tile([1.,-1.],(120,1));a=block_ci(x,block=10,reps=100);b=block_ci(x,block=10,reps=100);assert a==b and a['mean']==0 and a['ci_low']==0

def test_low_block_count_and_channel_macro():
 d=np.array([[1.,10.],[3.,20.]]);m=np.array([[1,1],[1,0]],bool);a=block_ci(d,m,block=2,reps=100);assert a['mean']==6 and a['family_low'] is None and a['status']=='LOW_BLOCK_COUNT'

def test_identical_set_rounding_is_not_a_win():
 from core import comparison_events
 changed,gain,loss=comparison_events(np.array([[1e-7,-.2]]),np.array([[[3,1,2],[1,2,4]]]),np.array([[[1,2,3],[1,2,3]]]))
 assert changed.tolist()==[[False,True]]
 assert gain.tolist()==[[False,False]]
 assert loss.tolist()==[[False,True]]

def test_stratum_bootstrap_preserves_channel_composition():
 from core import block_ci
 d=np.array([[1.,999.],[999.,3.]])
 mask=np.array([[True,False],[False,True]])
 a=block_ci(d,mask,block=1,reps=1000)
 assert a["mean"]==2. and a["ci_low"]==2. and a["ci_high"]==2.
 assert 0<a["bootstrap_valid_draws"]<1000
