import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from score_fusion import numpy_fusion,gamma_select,GAMMAS

@pytest.mark.parametrize('method',['E1','E2','E3','E4'])
def test_support_and_chunk_invariance(method):
    z=np.random.default_rng(0).normal(size=(7,3,5,101)).astype('float32')
    full=numpy_fusion(z,method,.1)
    ix=np.array([1,4,70,22,100])
    assert np.array_equal(full[...,ix],numpy_fusion(z[...,ix],method,.1))
    for q in [1,2,4]:
        assert np.array_equal(full,np.concatenate([numpy_fusion(z[i:i+q],method,.1) for i in range(0,7,q)]))
    assert np.array_equal(full,np.concatenate([numpy_fusion(z[...,i:i+17],method,.1) for i in range(0,101,17)],-1))

@pytest.mark.parametrize('gamma',GAMMAS)
def test_lse_reference(gamma):
    z=np.random.default_rng(2).uniform(-1,1,(2,5,11))
    assert np.allclose(numpy_fusion(z,'E4',gamma),gamma*np.log(np.exp(z/gamma).mean(-2)),atol=1e-14,rtol=1e-14)

def test_mean_max_top2():
    z=np.arange(30,dtype=float).reshape(5,6)/30
    assert np.array_equal(numpy_fusion(z,'E1'),z.mean(0))
    assert np.array_equal(numpy_fusion(z,'E2'),z[-1])
    assert np.array_equal(numpy_fusion(z,'E3'),(z[-1]+z[-2])/2)

def test_limits_and_finite():
    z=np.random.default_rng(1).uniform(-1,1,(4,5,100))
    assert np.max(abs(numpy_fusion(z,'E4',1e-7)-numpy_fusion(z,'E2'))) < 2e-7
    assert np.max(abs(numpy_fusion(z,'E4',1e6)-numpy_fusion(z,'E1'))) < 1e-6
    assert np.isfinite(numpy_fusion(z,'E4',1e-300)).all()

def test_selection_native_validation_only():
    r=[dict(split='val',training_support='FULL',evaluation_support='FULL',gamma=g,retmse=1.) for g in GAMMAS]
    assert gamma_select(r)==min(GAMMAS)
    r[0]['split']='test'
    with pytest.raises(AssertionError):gamma_select(r)
    r[0]['split']='val';r[0]['evaluation_support']='P100'
    with pytest.raises(AssertionError):gamma_select(r)
