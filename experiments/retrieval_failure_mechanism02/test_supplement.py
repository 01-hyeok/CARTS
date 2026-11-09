import unittest
import numpy as np
from supplement import input_features,freeze_strata,apply_strata
class Supplement(unittest.TestCase):
    def test_constant_input_finite(self):
        x=np.ones((3,7,96));f=input_features(x);self.assertEqual(f.shape,(3,7,5));self.assertTrue(np.isfinite(f).all());self.assertTrue((f==0).all())
    def test_validation_threshold_fixed(self):
        v=np.arange(60).reshape(4,3,5);cut=freeze_strata(v);copy=cut.copy();a=apply_strata(np.ones((2,3,5))*1e6,cut)
        self.assertTrue((a==2).all());np.testing.assert_array_equal(cut,copy)
    def test_trend_sign_invariant(self):
        x=np.arange(96)[None,None,:];np.testing.assert_allclose(input_features(x),input_features(-x))
if __name__=='__main__':unittest.main()
