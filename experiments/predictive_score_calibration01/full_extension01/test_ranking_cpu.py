import unittest,json,sys
from pathlib import Path
import numpy as np
from ranking_metrics import tie_rank,pairs,metrics,aggregate

class RankingTests(unittest.TestCase):
    def test_average_ties(self):
        np.testing.assert_array_equal(tie_rank(np.array([[4.,2.,2.,8.]])),[[2.,.5,.5,3.]])
    def test_perfect_ranking_and_local_oracle(self):
        d=np.arange(20,dtype=float)[None];m=aggregate(metrics(-d,d));self.assertAlmostEqual(m['spearman'],1.);self.assertEqual(m['pair_accuracy'],1.);self.assertEqual(m['local_oracle_recall10'],1.);self.assertEqual(m['ndcg10'],1.)
    def test_constant_score_is_undefined_and_half_pair_credit(self):
        d=np.arange(20,dtype=float)[None];m=aggregate(metrics(np.zeros_like(d),d));self.assertIsNone(m['spearman']);self.assertEqual(m['pair_accuracy'],.5)
    def test_future_ties_excluded(self):
        d=np.ones((1,20));m=aggregate(metrics(np.arange(20)[None],d));self.assertIsNone(m['pair_accuracy']);self.assertEqual(m['pair_denominator'],0);self.assertEqual(m['future_ties'],190)
    def test_large_full_pair_sampling_is_bounded_and_repeatable(self):
        a,b=pairs(36504);aa,bb=pairs(36504);self.assertEqual(len(a),512);np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb);self.assertTrue((a<b).all());self.assertEqual(len(set(zip(a,b))),512)

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RankingTests))
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x') as f:json.dump(dict(passed=r.wasSuccessful(),tests=r.testsRun,failures=len(r.failures),errors=len(r.errors)),f,indent=2)
    raise SystemExit(0 if r.wasSuccessful() else 1)
