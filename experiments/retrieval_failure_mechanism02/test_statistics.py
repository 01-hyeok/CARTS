import unittest,importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('mechanism_statistics',Path(__file__).with_name('mechanism_statistics.py'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class Statistics(unittest.TestCase):
    def test_channels_move_together(self):
        x=np.arange(20.)[:,None]*np.array([[1.,-1.]])
        r=mod.paired_blocks(x,4,repeats=40)
        self.assertEqual(float(r['mean']),0);self.assertEqual(float(r['ci_low']),0);self.assertEqual(float(r['ci_high']),0)
    def test_insufficient_blocks_flagged(self):
        r=mod.paired_blocks(np.ones((100,7)),200,repeats=20)
        self.assertFalse(r['sufficient']);self.assertEqual(r['effective_blocks'],1)
    def test_reproducible(self):
        x=np.random.default_rng(0).normal(size=(90,7));a=mod.paired_blocks(x,10,repeats=20);b=mod.paired_blocks(x,10,repeats=20)
        self.assertEqual(a['ci_low'],b['ci_low'])
    def test_tail_weighted(self):
        x=np.arange(11.)[:,None]*np.ones((1,3));r=mod.paired_blocks(x,4,repeats=20)
        self.assertEqual(float(r['mean']),5.)

if __name__=='__main__':unittest.main()
