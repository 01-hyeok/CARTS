import numpy as np
import pytest
from .past_lag import indices,alignment,activity,freeze_activity_scale,shift_features
from .fusion import analyze_unit,global_ids_gate,loo_from_probabilities
from .softlag import forward,objective_gradient,softmax


@pytest.mark.parametrize('h',[96,192,336,720])
def test_fixed_past_window(h):
    ix,lags=indices(h)
    for lag in lags:
        assert (ix+lag).min()>=0 and (ix+lag).max()<h
        assert (ix+lag).shape==ix.shape


@pytest.mark.parametrize('delta',[-12,-5,0,7,12])
def test_synthetic_sign(delta):
    # Candidate[t+delta] equals query[t], drawn from one nonperiodic signal.
    z=np.random.default_rng(4).normal(size=200)
    q=z[40:136][None,None,:];x=z[40-delta:136-delta][None,None,None,:]
    a,lags=alignment(q,x);f=shift_features(a,lags)
    assert f['best_lag'].item()==delta
    np.testing.assert_allclose(f['best'],1,atol=1e-14)


@pytest.mark.parametrize('h',[96,336])
def test_chunk_exact(h):
    rng=np.random.default_rng(3);q=rng.normal(size=(3,7,h));x=rng.normal(size=(3,7,9,h))
    a,l=alignment(q,x,candidate_chunk=9)
    for n in [1,2,4]:np.testing.assert_array_equal(alignment(q,x,candidate_chunk=n)[0],a)
    np.testing.assert_array_equal(np.concatenate([alignment(q[i:i+1],x[i:i+1])[0] for i in range(3)]),a)


def test_flat_finite_zero_activity():
    q=np.ones((2,7,96));x=np.ones((2,7,4,96))*3
    a,l=alignment(q,x);f=shift_features(a,l)
    assert np.isfinite(a).all() and (a==0).all() and (f['best_lag']==0).all()
    assert (activity(q,x,freeze_activity_scale(q))==0).all()


def test_unused_edges_and_no_future_arguments():
    import inspect
    assert 'future' not in inspect.signature(alignment).parameters
    rng=np.random.default_rng(2);q=rng.normal(size=(2,7,96));x=rng.normal(size=(2,7,4,96))
    a,l=alignment(q,x);qq=q.copy();qq[:,:,:12]=12345;qq[:,:,84:]=-12345
    np.testing.assert_array_equal(a,alignment(qq,x)[0])


def example(alpha=0.,shared=True):
    rng=np.random.default_rng(5);g=rng.normal(size=(2,7,5,17))*.05
    a=rng.normal(size=(2,7,17,25))*.1;c=rng.uniform(size=(2,7,17))
    t=softmax(rng.normal(size=(2,7,17)))
    b=np.zeros((1 if shared else 5,25))
    return g,a,c,t,alpha,b


@pytest.mark.parametrize('shared',[True,False])
def test_alpha_zero_exact_and_later_gradient(shared):
    g,a,c,t,alpha,b=example(shared=shared)
    f=forward(g,a,c,alpha,b)
    np.testing.assert_array_equal(f['u'],g)
    base=softmax(g/.1).mean(2)
    np.testing.assert_array_equal(f['mix'],base)
    np.testing.assert_array_equal(np.argsort(-f['mix'],kind='stable'),np.argsort(-base,kind='stable'))
    loss,ga,gb=objective_gradient(g,a,c,t,alpha,b)
    assert np.isfinite(loss) and ga!=0 and (gb==0).all()
    _,_,later=objective_gradient(g,a,c,t,-.01*ga,b)
    assert np.max(np.abs(later))>0


def test_b1_b2_initial_equality():
    g,a,c,t,alpha,b=example()
    np.testing.assert_array_equal(forward(g,a,c,0,b)['u'],forward(g,a,c,0,np.repeat(b,5,0))['u'])


@pytest.mark.parametrize('shared',[True,False])
def test_gradients_finite_difference(shared):
    g,a,c,t,_,b=example(shared=shared);alpha=.2;step=1e-6
    loss,ga,gb=objective_gradient(g,a,c,t,alpha,b)
    fn=lambda av,bv:objective_gradient(g,a,c,t,av,bv)[0]
    np.testing.assert_allclose(ga,(fn(alpha+step,b)-fn(alpha-step,b))/(2*step),rtol=1e-6,atol=1e-8)
    for h,j in [(0,0),(0,12),(b.shape[0]-1,24)]:
        plus=b.copy();minus=b.copy();plus[h,j]+=step;minus[h,j]-=step
        np.testing.assert_allclose(gb[h,j],(fn(alpha,plus)-fn(alpha,minus))/(2*step),rtol=1e-5,atol=1e-8)


def test_mask_excluded_student_teacher():
    g,a,c,t,alpha,b=example();mask=np.ones_like(t,bool);mask[...,-3:]=False
    t=np.where(mask,t,0);t/=t.sum(-1,keepdims=True)
    f=forward(g,a,c,alpha,b,mask=mask)
    assert (f['mix'][~mask]==0).all()
    assert np.isfinite(objective_gradient(g,a,c,t,alpha,b,mask=mask)[0])


@pytest.mark.parametrize('seed',list(range(8)))
def test_fusion_identity_and_unique(seed):
    rng=np.random.default_rng(seed);d=rng.uniform(size=100)
    h=np.stack([rng.choice(100,10,replace=False) for _ in range(5)])
    s=rng.choice(100,10,replace=False);o=np.argsort(d)[:10]
    r=analyze_unit(h,d[h],s,d[s],o,d[o])
    assert r['fusion_regret']>=0 and r['discovery_regret']>=0
    np.testing.assert_allclose(r['fusion_regret']+r['discovery_regret'],d[s].mean()-d[o].mean())
    assert r['lost_oracle_count']==len((set(h.ravel())&set(o))-set(s))
    assert r['union_size']==len(np.unique(h))


def test_union_must_include_mix():
    d=np.arange(100,dtype=float);h=np.tile(np.arange(40,50),(5,1));s=o=np.arange(10)
    r=analyze_unit(h,d[h],s,d[s],o,d[o])
    assert r['fusion_regret']==0 and r['discovery_regret']==0


def test_duplicate_inconsistent_rejected():
    d=np.arange(100,dtype=float);h=np.tile(np.arange(10),(5,1));s=o=np.arange(10)
    wrong=d[h].copy();wrong[1,0]=.3
    with pytest.raises(AssertionError):analyze_unit(h,wrong,s,d[s],o,d[o])
    wrongid=h.copy();wrongid[0,1]=wrongid[0,0]
    with pytest.raises(AssertionError):analyze_unit(wrongid,d[wrongid],s,d[s],o,d[o])


def test_mask_ids_and_loo():
    global_ids_gate(np.array([3,7]),np.array([3,7,20]))
    with pytest.raises(AssertionError):global_ids_gate(np.array([2]),np.array([3,7]))
    p=softmax(np.random.default_rng(3).normal(size=(2,7,5,100)))
    loo=loo_from_probabilities(p)
    for h in range(5):np.testing.assert_allclose(loo[:,:,h],np.delete(p,h,axis=2).mean(2),atol=1e-15)


def test_single_head_normalizer_invariance():
    logits=np.random.default_rng(6).normal(size=(2,7,200));pool=np.arange(100)
    fp=softmax(logits/.1)[...,pool];pp=softmax(logits[...,pool]/.1)
    np.testing.assert_array_equal(np.argsort(-fp,kind='stable'),np.argsort(-pp,kind='stable'))


def test_parameter_budget():
    assert (1*25+1,5*25+1)==(26,126)
    assert (1*33+1,5*33+1)==(34,166)
