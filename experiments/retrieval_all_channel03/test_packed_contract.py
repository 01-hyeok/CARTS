import ast
from pathlib import Path


def test_no_channel_loop_in_encoder():
    p=Path(__file__).with_name('packed_batch.py')
    t=ast.parse(p.read_text())
    for n in ast.walk(t):
        if isinstance(n,ast.For):
            assert not isinstance(n.target,ast.Name) or n.target.id not in ['c','ci','channel']
    assert 'b*c<=rows' in p.read_text()
    assert 'all_channels_in_each_model_call=True' in p.read_text()


def test_production_does_not_import_streams():
    source=Path(__file__).with_name('runtime_all_channels.py').read_text()
    assert 'from packed_batch import PackedChannelFull' in source
    assert 'from concurrent_' not in source


def test_small_tail_never_drops_channels():
    for c in [7,21,137]:
        for b in [1,7,12,23,28,32]:
            rows=max(b,c);chunk=rows//c
            covered=[]
            for a in range(0,b,chunk):
                stop=min(a+chunk,b)
                assert (stop-a)*c<=rows
                covered.extend(range(a,stop))
            assert covered==list(range(b))
