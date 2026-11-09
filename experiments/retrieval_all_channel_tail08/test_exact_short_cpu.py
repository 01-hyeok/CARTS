import ast
from pathlib import Path
import numpy as np
import pytest
import torch
from exact_short import canonical_norm128,canonical_sum_squares128


@pytest.mark.parametrize('rows',[1,2,3,7,8,15,16,20,28,31,32])
def test_norm_matches_mathematical_definition(rows):
    g=torch.Generator().manual_seed(90+rows)
    x=torch.randn(21,rows,128,generator=g)
    x[0,0]=0
    got=canonical_norm128(x,rows)
    reference=x.double()/x.double().norm(2,dim=-1,keepdim=True).clamp_min(1e-12)
    assert got.shape==x.shape and torch.isfinite(got).all()
    assert torch.allclose(got.double(),reference,atol=1e-7,rtol=1e-6)
    assert torch.equal(got[0,0],torch.zeros(128))


def test_norm_specific_canonical_tree():
    x=torch.randn(21,7,128,generator=torch.Generator().manual_seed(8))
    raw=x.numpy();expected=np.empty((21,7,1),np.float32)
    # Independent literal per-row reference to installed Reduce.cuh order.
    for c in range(21):
        for b in range(7):
            v=raw[c,b]*raw[c,b]
            for width in (64,32):v=v[:width]+v[width:2*width]
            for shift in (1,2,4,8,16):v=v[:-shift]+v[shift:]
            expected[c,b,0]=v[0]
    # Compare the actual ordered FP32 reduction exactly. CPU NumPy vs Torch
    # final division is not the CUDA canonical contract (different vectorized
    # reciprocal implementations); full GPU normalization is tested exactly.
    assert np.array_equal(canonical_sum_squares128(x,7).numpy(),expected)


def test_no_production_channel_loops():
    for name in ['packed_batch.py','exact_short.py']:
        tree=ast.parse(Path(__file__).with_name(name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,(ast.For,ast.comprehension)):
                iterator=node.iter
                if isinstance(iterator,ast.Call) and isinstance(iterator.func,ast.Name) and iterator.func.id=='range':
                    assert not any((isinstance(arg,ast.Name) and arg.id in ['c','C']) or (isinstance(arg,ast.Attribute) and arg.attr=='c') for arg in iterator.args)


def test_kernel_full_domain_and_explicit_rounding():
    source=Path(__file__).with_name('split64.cu').read_text()
    assert 'channels*queries*candidates' in source
    assert '__fmaf_rn' in source and '__fadd_rn' in source
    assert 'for(int c=' not in source
