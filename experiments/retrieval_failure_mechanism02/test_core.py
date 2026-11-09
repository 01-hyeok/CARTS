import unittest
import numpy as np
from io_utils import id_bytes
import core as c
from protocol import PROTOCOL, CELLS

class Contracts(unittest.TestCase):
    def test_id_hash_dtype_and_order(self):
        x=np.arange(30).reshape(3,10)
        self.assertEqual(id_bytes(x.astype('int32')),id_bytes(x.astype('int64')))
        self.assertNotEqual(id_bytes(x),id_bytes(x[:,::-1]))
    def setUp(self):
        self.r=np.random.default_rng(0)
        self.d=self.r.uniform(0,4,(4,120));self.s=self.r.normal(size=(4,120));self.m=np.ones((4,120),bool)
    def test_oracle_regret_exchange(self):
        x=c.metrics(self.s,self.d,self.m)
        np.testing.assert_allclose(x['regret'],x['exchange'])
        self.assertTrue((x['retmse']>=x['oracle_retmse']).all())
    def test_oracle_is_one(self):
        x=c.metrics(-self.d,self.d,self.m)
        np.testing.assert_array_equal(x['recall'],1);np.testing.assert_allclose(x['ndcg'],1)
    def test_constant_distances_no_nan(self):
        x=c.metrics(self.s,np.zeros_like(self.d),self.m)
        np.testing.assert_array_equal(x['ndcg'],1)
    def test_mask(self):
        self.m[:,:20]=False
        self.assertTrue((c.topk(self.s,self.m)>=20).all())
    def test_less_than_k_rejected(self):
        self.m[:,:115]=False
        with self.assertRaises(AssertionError):c.topk(self.s,self.m)
    def test_stable_ties(self):
        np.testing.assert_array_equal(c.topk(np.zeros_like(self.s),self.m)[0],np.arange(10))
    def test_subset_order(self):
        ids=np.broadcast_to(np.arange(120),(4,120));p=np.broadcast_to(np.arange(99,-1,-1),(4,100))
        c.check_subset(ids,self.m,p)
    def test_subset_duplicate_rejected(self):
        p=np.zeros((4,100),int)
        with self.assertRaises(AssertionError):c.check_subset(np.broadcast_to(np.arange(120),(4,120)),self.m,p)
    def test_subset_mask_rejected(self):
        self.m[:,7]=False
        with self.assertRaises(AssertionError):c.check_subset(np.broadcast_to(np.arange(120),(4,120)),self.m,np.broadcast_to(np.arange(100),(4,100)))
    def test_softmax_denominator_identity(self):
        logits=self.r.normal(size=(4,5,120));mask=np.broadcast_to(self.m[:,None],logits.shape)
        x=c.normalization_control(logits,mask,np.broadcast_to(np.arange(100),(4,100)))
        np.testing.assert_allclose(x['head_weight'].sum(-1),1)
        self.assertTrue((x['head_mass']<=1+1e-12).all())
    def test_single_head_rank_invariance(self):
        logits=self.r.normal(size=(4,1,120));c.normalization_control(logits,np.ones_like(logits,bool),np.broadcast_to(np.arange(100),(4,100)))
    def test_decomposition(self):
        x=c.decomposition(np.ones(3),np.ones(3)*2,np.ones(3)*4,'P100')
        self.assertAlmostEqual(x['pool_share'],1/3)
    def test_full_not_forced_above_p100(self):
        c.decomposition(np.ones(3),np.ones(3)*4,np.ones(3)*2,'FULL')
    def test_no_clamp(self):
        with self.assertRaises(AssertionError):c.decomposition(np.ones(3),np.ones(3)*4,np.ones(3)*2,'P100')
    def test_zero_total_undefined(self):
        self.assertIsNone(c.decomposition(np.ones(3),np.ones(3),np.ones(3),'P100')['pool_share'])
    def test_union_original_score(self):
        h=np.array([np.arange(10),np.arange(5,15)]);s=np.arange(20)
        np.testing.assert_array_equal(c.union_topk(h,s),np.arange(14,4,-1))
    def test_worst_query_is_timestamp(self):
        x=c.worst_timestamp_share(np.array([[4.,0.],[1.,1.]]),.5);self.assertAlmostEqual(x,2/3)
    def test_probe_train_boundary(self):
        c.train_label_gate([0,8],2,2,0,12)
        with self.assertRaises(AssertionError):c.train_label_gate([9],2,2,0,12)
    def test_epsilon_val_frozen(self):
        np.testing.assert_allclose(c.freeze_near_optimal_epsilon([[0,1],[0,3]]),[1e-8,.02])
    def test_scope_and_v1_required(self):
        self.assertEqual(len(CELLS),8);self.assertIn('V1',PROTOCOL['models'])
        self.assertIn('same frozen',PROTOCOL['support_intervention'])

if __name__=='__main__':unittest.main()
