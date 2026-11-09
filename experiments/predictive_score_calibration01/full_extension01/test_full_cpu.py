"""Full and cross-support mathematics verified without CUDA/data changes."""
import sys,unittest,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
import torch
from calibrators import ResidualCalibrator,FixedHeadWeights,teacher_kl
from streaming import assemble,global_logprob,stable_topk,candidate_coverage
from train_pair_permutation import TrainPairPermutation
torch.set_num_threads(1)

class FullTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7);self.x=torch.randn(2,3,137,38,dtype=torch.float64);self.m=torch.ones(2,3,137,dtype=torch.bool);self.m[...,11:17]=False
        logits=torch.randn(2,3,137,dtype=torch.float64);self.p=logits.masked_fill(~self.m,-torch.inf).softmax(-1)
    def model(self):
        torch.manual_seed(0);a=ResidualCalibrator(38);a.normalizer.fit(self.x[self.m],'train');a.beta.data.fill_(.2);return a
    def test_masked_global_full_loss_gradients(self):
        a=self.model();b=self.model();teacher=torch.randn_like(self.p).masked_fill(~self.m,-torch.inf).softmax(-1)
        sa=a.score(self.p,self.x,self.m)
        sb=assemble(137,19,lambda i,j:b.score(self.p[...,i:j],self.x[...,i:j,:],self.m[...,i:j]))
        pa=global_logprob(sa,self.m);pb=global_logprob(sb,self.m)
        self.assertTrue(torch.allclose(pa[self.m],pb[self.m],atol=1e-12,rtol=1e-12))
        la=teacher_kl(teacher,pa,self.m);lb=teacher_kl(teacher,pb,self.m);la.backward();lb.backward()
        self.assertTrue(torch.allclose(la,lb,atol=1e-12,rtol=1e-12))
        for x,y in zip(a.parameters(),b.parameters()):self.assertTrue(torch.allclose(x.grad,y.grad,atol=1e-10,rtol=1e-10))
        self.assertEqual(pb.exp()[~self.m].sum().item(),0)
    def test_chunk_support_cannot_drop_or_duplicate(self):
        self.assertEqual(candidate_coverage(self.m,[(0,19),(19,137)]),int(self.m.sum()))
        with self.assertRaises(ValueError):candidate_coverage(self.m,[(0,19),(20,137)])
        with self.assertRaises(ValueError):candidate_coverage(self.m,[(0,20),(19,137)])
    def test_global_normalization_is_not_per_chunk_softmax(self):
        score=torch.tensor([[0.,1.,8.,9.]],dtype=torch.float64);mask=torch.ones_like(score,dtype=torch.bool)
        correct=global_logprob(score,mask,1).exp();incorrect=torch.cat([score[:,:2].softmax(-1),score[:,2:].softmax(-1)],-1)
        self.assertFalse(torch.allclose(correct,incorrect));self.assertEqual(incorrect.sum().item(),2.)
    def test_cross_support_does_not_refit_normalizer(self):
        a=self.model();before={k:v.clone() for k,v in a.state_dict().items()};narrow=slice(0,100)
        mask=self.m[...,narrow];p=self.p[...,narrow];p=p/p.sum(-1,keepdim=True)
        a.score(p,self.x[...,narrow,:],mask)
        for k,v in before.items():self.assertTrue(torch.equal(v,a.state_dict()[k]))
        with self.assertRaises(ValueError):a.normalizer.fit(self.x,'val')
    def test_epoch_zero_full_ordered_ids_and_probability(self):
        a=self.model();a.beta.data.zero_()
        self.assertTrue(torch.equal(stable_topk(a.score(self.p,self.x,self.m,True)),stable_topk(self.p.masked_fill(~self.m,-torch.inf))))
        self.assertTrue(torch.allclose(a.logprob(self.p,self.x,self.m).exp(),self.p,atol=1e-12,rtol=1e-12))
    def test_fixed_head_masked_distribution(self):
        g=self.x[...,:5];a=FixedHeadWeights();p=(g.transpose(-1,-2)/.1).masked_fill(~self.m.unsqueeze(-2),-torch.inf).softmax(-1).mean(-2)
        self.assertTrue(torch.allclose(a.logprob(p,g,self.m).exp(),p,rtol=1e-12,atol=1e-12))
    def test_full_train_pair_shuffle_is_exact_bijection(self):
        n,h=17,2;p=TrainPairPermutation(n,h);q=torch.arange(n);c=torch.arange(n).expand(n,1,n)
        dq,dc,valid=p.map(q,c,'train');source=(q[:,None,None]*n+c)[valid];target=(dq*n+dc)[valid]
        self.assertTrue(torch.equal(torch.sort(source).values,torch.sort(target).values))
        self.assertFalse(torch.equal(source,target));self.assertEqual(int(valid.sum()),p.total)
        eq,ec,_=p.map(torch.tensor([0,1]),c[:2],'test',torch.tensor([100,101]))
        self.assertTrue((eq<n).all() and (ec<n).all())

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FullTests))
    result=dict(passed=r.wasSuccessful(),tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),torch_version=torch.__version__,gpu_used=False,actual_full_data=False)
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x') as f:json.dump(result,f,indent=2)
    raise SystemExit(0 if r.wasSuccessful() else 1)
