"""CPU tensor regressions; these are NOT a GPU/canonical reproduction gate."""
import unittest
import numpy as np
import torch
import gpu_metrics as gm

class FakeData:
    h=8
    def __init__(self,values):self.v=values
    def values_all(self,ids):
        b,c,p=ids.shape
        return self.v[torch.arange(b)[:,None,None],torch.arange(c)[None,:,None],ids]

class TensorContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0);self.B=3;self.C=7;self.N=112
        self.x=torch.randn(3,7,8);self.y=torch.randn(3,7,8);self.v=torch.randn(3,7,112,8)
        self.data=FakeData(self.v);self.d=(self.v+self.x[:,:,-1,None,None]-self.y[:,:,None]).square().mean(-1)
        self.ids=torch.arange(112).expand(3,7,-1);self.mask=torch.ones_like(self.ids,dtype=torch.bool)
    def test_oracle_identity_and_axes(self):
        a=gm.evaluate(self.data,self.x,self.y,self.d,-self.d,self.mask,self.ids)
        m=dict(zip(gm.METRICS,a['metrics'].unbind(-1)))
        self.assertEqual(a['metrics'].shape,(3,7,len(gm.METRICS)))
        self.assertTrue(torch.equal(m['ranking_regret'],torch.zeros(3,7,dtype=torch.float64)))
        self.assertTrue(torch.equal(m['recall10'],torch.ones(3,7,dtype=torch.float64)))
        self.assertTrue(torch.equal(m['ndcg100_inverse_mse'],torch.ones(3,7,dtype=torch.float64)))
    def test_regret_and_exchange(self):
        a=gm.evaluate(self.data,self.x,self.y,self.d,torch.randn_like(self.d),self.mask,self.ids)
        m=dict(zip(gm.METRICS,a['metrics'].unbind(-1)))
        torch.testing.assert_close(m['ranking_regret'],m['exchange_regret'],rtol=1e-10,atol=1e-10)
        self.assertTrue(bool((m['ranking_regret']>=0).all()))
    def test_normalization_identity(self):
        l=torch.randn(3,7,5,112);a=gm.normalization_control(l,self.mask,self.ids[...,:100])
        torch.testing.assert_close(a['weights'].sum(-1),torch.ones(3,7,dtype=torch.float64))
        expected=torch.softmax((l/.1).double(),-1)[...,:100].mean(2)
        torch.testing.assert_close(a['restricted'],expected,rtol=1e-12,atol=2e-15)
    def test_single_head_invariant(self):
        l=torch.randn(3,7,1,112);a=gm.normalization_control(l,self.mask,self.ids[...,:100])
        self.assertTrue(torch.equal(gm.topk(a['local'],10),gm.topk(a['restricted'],10)))
    def test_union_uses_original_scores(self):
        l=torch.randn(3,7,5,112);p=torch.softmax(l,-1).mean(2)
        a=gm.head_union(l,p,self.mask)
        self.assertTrue(torch.equal(a['deployable'],gm.topk(p.masked_fill(~a['union_mask'],-float('inf')),10)))
        self.assertTrue(bool((a['union_size']<=50).all()))

if __name__=='__main__':unittest.main()
