import unittest
import numpy as np
import torch
from geometry import describe,validation_choice,adjusted_rand,FIELDS

class Geometry(unittest.TestCase):
    def test_null_permitted(self):
        x=torch.zeros(2,3,10,8);s,l=describe(x)
        self.assertTrue(torch.isfinite(s[...,:9]).all());self.assertEqual(validation_choice(s.numpy(),s.numpy()),1)
    def test_all_channel_shape(self):
        torch.manual_seed(1);s,l=describe(torch.randn(2,7,10,8))
        self.assertEqual(tuple(s.shape),(2,7,len(FIELDS)));self.assertEqual(tuple(l.shape),(2,7,2,10))
    def test_ari_label_permutation(self):
        a=torch.tensor([[[0,0,1,1,2,2]]]);b=2-a
        self.assertEqual(float(adjusted_rand(a,b,3)),1.)
    def test_stability_and_control_required(self):
        a=np.zeros((3,7,len(FIELDS)));r=a.copy();a[...,3:6]=[.8,4,.79]
        self.assertEqual(validation_choice(a,r),1)
        a[...,5]=.9;r[...,3]=.75
        self.assertEqual(validation_choice(a,r),1)
        r[...,3]=.1;self.assertEqual(validation_choice(a,r),2)

if __name__=='__main__':unittest.main()
