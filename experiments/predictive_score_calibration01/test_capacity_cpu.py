"""P2 controls, loss/gradient, masking, train-only normalization tests."""
import os,sys,json,unittest,inspect
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,numpy as np
from calibrators import ResidualCalibrator,teacher_kl
torch.set_num_threads(1)

class Controls(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0);self.x=torch.randn(3,2,100,38,dtype=torch.float64);self.p=torch.randn(3,2,100).softmax(-1);self.m=torch.ones_like(self.p,dtype=torch.bool)
    def arm(self,active):
        torch.manual_seed(0);a=ResidualCalibrator(38,active);a.normalizer.fit(self.x,'train');return a
    def test_same_capacity_and_initialization(self):
        a=self.arm([1]*5+[0]*33);b=self.arm([1]*38)
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        for x,y in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(x,y))
        self.assertTrue(torch.equal(a.score(self.p,self.x,self.m,True),b.score(self.p,self.x,self.m,True)))
    def test_neutral_features_and_loss(self):
        a=self.arm([1]*5+[0]*33);a.beta.data.fill_(.2);other=self.x.clone();other[...,5:]+=1000
        self.assertTrue(torch.equal(a.score(self.p,self.x,self.m),a.score(self.p,other,self.m)))
        teacher=torch.randn_like(self.p).softmax(-1).double();loss=teacher_kl(teacher,a.logprob(self.p,self.x,self.m),self.m);loss.backward()
        self.assertEqual(a.layers[0].weight.grad[:,5:].abs().sum().item(),0)
        self.assertGreater(a.layers[0].weight.grad[:,:5].abs().sum().item(),0)
    def test_invalid_future_fit_rejected(self):
        a=ResidualCalibrator(38)
        with self.assertRaises(ValueError):a.normalizer.fit(self.x,'val')
        self.assertFalse(a.normalizer.fitted.item())
        self.assertNotIn('future',inspect.signature(a.score).parameters)
        self.assertNotIn('teacher',inspect.signature(a.score).parameters)
    def test_global_chunk_score_loss_gradient(self):
        a=self.arm([1]*38);a.beta.data.fill_(.1);b=self.arm([1]*38);b.load_state_dict(a.state_dict())
        teacher=torch.randn_like(self.p).softmax(-1).double()
        sa=a.score(self.p,self.x,self.m);sb=torch.cat([b.score(self.p[...,i:i+17],self.x[...,i:i+17,:],self.m[...,i:i+17]) for i in range(0,100,17)],-1)
        self.assertTrue(torch.allclose(sa,sb,rtol=1e-12,atol=1e-12))
        la=teacher_kl(teacher,(sa/.1).log_softmax(-1),self.m);lb=teacher_kl(teacher,(sb/.1).log_softmax(-1),self.m)
        self.assertTrue(torch.allclose(la,lb,rtol=1e-12,atol=1e-12));la.backward();lb.backward()
        for pa,pb in zip(a.parameters(),b.parameters()):self.assertTrue(torch.allclose(pa.grad,pb.grad,rtol=1e-10,atol=1e-10))
    def test_train_donor_permutation_preserves_marginal(self):
        n=101;rows=np.random.RandomState(0).permutation(n);c=np.random.RandomState(0).permutation(100)
        x=np.arange(n*100).reshape(n,100);shuffled=x[rows][:,c]
        np.testing.assert_array_equal(np.sort(x.ravel()),np.sort(shuffled.ravel()))
        self.assertFalse(np.array_equal(x,shuffled))

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    p=Path('/data/pjh_workspace/CARTS/results/TRACK-PREDICTIVE-SCORE-CALIBRATION01/P2_CPU_TESTS.json')
    with p.open('x') as f:json.dump(dict(passed=r.wasSuccessful(),tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),actual_training=False),f,indent=2)
    raise SystemExit(0 if r.wasSuccessful() else 1)
