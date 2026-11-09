import hashlib,json
import numpy as np
import pytest
from .d2_saved_blocks import run


def make_source(tmp_path,channels=7,variant='FULL'):
    base=tmp_path/'source';part=base/'test';part.mkdir(parents=True)
    identity={'checkpoint_sha256':'synthetic-fixture-only','channels':channels}
    (base/'REPRODUCTION_GATE.json').write_text(json.dumps({'identity':identity}))
    gate={'status':'PASS','identity':identity,'queries':2,'channels':channels,'units':2*channels}
    (part/'SPLIT_GATE.json').write_text(json.dumps(gate))
    rng=np.random.default_rng(44);d=rng.uniform(size=(2,channels,100))
    heads=np.broadcast_to(np.arange(50).reshape(5,10),(2,channels,5,10))
    mix=np.broadcast_to(np.arange(40,50),(2,channels,10))
    oracle=np.argsort(d,axis=-1)[...,:10]
    headmse=np.take_along_axis(np.broadcast_to(d[:,:,None,:],(2,channels,5,100)),heads,-1)
    payload={'query_start':np.array([500,501]),variant+'__head_ids':heads,variant+'__head_selected_mse':headmse,
        variant+'__selected_ids':mix,variant+'__selected_mse':np.take_along_axis(d,mix,-1),
        variant+'__oracle_ids':oracle,variant+'__oracle100_mse':np.sort(d,-1)}
    p=part/'batch00000.npz';np.savez_compressed(p,**payload)
    side={'status':'PASS','identity':identity,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    p.with_suffix('.json').write_text(json.dumps(side))
    return base


@pytest.mark.parametrize('channels',[7,21,137])
def test_full_channels_saved_path(tmp_path,channels):
    base=make_source(tmp_path,channels);out=tmp_path/'out';run(base,out)
    g=json.loads((out/'CPU_SET_GATE.json').read_text())
    assert g['units']==2*channels and g['channels']==channels
    assert g['status']=='PARTIAL_CPU_SET_DIAGNOSTICS_PASS'
    assert len(g['pending'])==3
    with pytest.raises(AssertionError):run(base,out)


def test_native_not_substituted(tmp_path):
    base=make_source(tmp_path,variant='P100_CONTROLLED');out=tmp_path/'out'
    with pytest.raises(RuntimeError,match='MISSING'):run(base,out,variant='P100_NATIVE')
    assert not out.exists()


def test_bad_hash_rejected(tmp_path):
    base=make_source(tmp_path);p=base/'test/batch00000.npz'
    p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(AssertionError):run(base,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_incomplete_population_rejected(tmp_path):
    base=make_source(tmp_path);p=base/'test/SPLIT_GATE.json';g=json.loads(p.read_text());g['queries']=3;p.write_text(json.dumps(g))
    with pytest.raises(AssertionError):run(base,tmp_path/'out')
    assert not (tmp_path/'out').exists()
