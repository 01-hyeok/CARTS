"""CPU arithmetic/orientation checks; these do not assert GPU kernel parity."""
import ast
from pathlib import Path
import numpy as np
import pytest


@pytest.mark.parametrize('channels,batch', [(7,1),(21,7),(21,28),(21,32),(137,1),(137,7)])
def test_all_channel_packed_diagonal_retains_same_dots(channels,batch):
    rng=np.random.default_rng(0);d,n=8,17
    q=rng.integers(-4,5,(channels,batch,d)).astype(np.float32)
    keys=rng.integers(-4,5,(channels,n,d)).astype(np.float32)
    expected=np.matmul(q,keys.transpose(0,2,1))
    all_pairs=q.reshape(channels*batch,d) @ keys.reshape(channels*n,d).T
    idx=np.arange(channels)
    got=all_pairs.reshape(channels,batch,channels,n)[idx,:,idx,:]
    assert got.shape==(channels,batch,n)
    assert np.array_equal(got,expected)
    transposed=np.matmul(keys,q.transpose(0,2,1)).transpose(0,2,1)
    assert np.array_equal(transposed,expected)


def test_candidate_backends_have_no_channel_loop_or_sync():
    tree=ast.parse(Path(__file__).with_name('score_backends.py').read_text())
    assert not any(isinstance(n,(ast.For,ast.While,ast.ListComp,ast.GeneratorExp)) for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Attribute) and n.attr=='synchronize' for n in ast.walk(tree))


def test_boundary_waiter_cannot_release_production_or_solar():
    tree=ast.parse(Path(__file__).with_name('run_after_boundary.py').read_text())
    strings=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
    assert 'diagnose_weather720.py' in strings
    assert not any(s.endswith(('full_channel_run.py','solar_full_controller.py')) for s in strings)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen']
    assert len(calls)==1
