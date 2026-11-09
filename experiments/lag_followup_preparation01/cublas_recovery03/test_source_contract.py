from source_contract import transform
import pytest

SOURCE="def short_query(self,x):\n    exact_torch2_tail=(compute==68 and self.frozen.heads is None)\n    return self.frozen.heads(z.reshape(b*c,dim)).reshape(b,c,-1,dim)\n"
@pytest.mark.parametrize('mode',['flat_exact_norm','cuda12_lt_exact_norm','cuda11_lt_exact_norm','cuda11_blas_exact_norm'])
def test_scoped_shape_and_compilation(mode):
    s=transform(SOURCE,mode);compile(s,'test','exec')
    assert 'self.frozen.n==1' in s and 'canonical_norm128(projected,b)' in s
    assert 'for c in' not in s and 'compute==68' in s

def test_no_silent_unknown_source_or_policy():
    with pytest.raises(AssertionError):transform('changed source','flat_exact_norm')
    with pytest.raises(AssertionError):transform(SOURCE,'serial_fallback')
