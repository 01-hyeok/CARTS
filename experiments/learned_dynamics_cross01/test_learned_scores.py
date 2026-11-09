"""Run after learned_scores.py in the same namespace for remote stdin testing."""
import io
import json
import sys
import unittest
if 'fit_ar' not in globals():
    from learned_scores import *
sys.path.insert(0, '/data/pjh_workspace/CARTS')
from utils.mean_mixture_selection import mean_mixture_topk_selection
from models.RelationStage1 import stable_topk_indices
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)

    def fixtures(self):
        q = fit_ar(torch.randn(3,96).cumsum(-1))
        c = fit_ar(torch.randn(23,96).cumsum(-1))
        n = FixedStandardizer(6).fit(c['phi'], 'train')
        return {k:v[:,None] for k,v in q.items()}, {k:v[None] for k,v in c.items()}, n

    def baseline(self, count=23):
        heads=torch.randn(3,5,count)
        mask=torch.rand(3,count)>.2
        mask[:,:10]=True
        picks,_,p=mean_mixture_topk_selection(heads,mask,.1,10)
        return mask,p,picks

    def test_cross_formula_and_constants(self):
        qpast=torch.randn(3,96).cumsum(-1)
        cpast=torch.randn(23,96).cumsum(-1)
        q,c=fit_ar(qpast),fit_ar(cpast)
        d=qpast.double().diff(dim=-1).unfold(-1,5,1)[:,-16:]
        X=torch.cat((torch.ones_like(d[...,:1]),d[...,:4].flip(-1)),-1)
        direct=(torch.einsum('qtp,cp->qct',X,c['coef'])-d[:,None,:,4]).square().mean(-1)
        actual=cross_error({k:v[:,None] for k,v in q.items()},{k:v[None] for k,v in c.items()})
        torch.testing.assert_close(actual,direct,rtol=1e-10,atol=1e-10)
        for kind in (torch.ones(2,96),torch.arange(96).expand(2,-1)):
            self.assertTrue(torch.isfinite(fit_ar(kind)['phi']).all())

    def test_pair_shapes_and_learnability(self):
        q,c,n=self.fixtures()
        for features,dim in ((dynamics_pairs(q,c,n),15),(cross_pairs(q,c,n),18)):
            self.assertEqual(features.shape,(3,23,dim))
            m=PairMLP(dim)
            m.standardizer.fit(features,'train')
            s=m(features)
            (s*torch.randn_like(s)).sum().backward()
            self.assertGreater(float(m.layers[0].weight.grad.norm()),0)
        m=DynamicsL2()
        m(n(q['phi']),n(c['phi'])).sum().backward()
        self.assertTrue((F.softplus(m.raw_weight)>0).all())
        self.assertGreater(float(m.raw_weight.grad.norm()),0)

    def test_official_beta_zero_probability_and_order(self):
        for count in (23,100,503):
            mask,p,picks=self.baseline(count)
            new=torch.randn_like(p,dtype=torch.float64)
            f=Fusion();f.calibration.fit(new[mask][:,None],'train')
            actual=f.log_probability(p,new,mask).exp()
            torch.testing.assert_close(actual,p.double(),rtol=0,atol=2e-7)
            score=f.ranking_score(p,new,mask)
            self.assertTrue(torch.equal(score,p.masked_fill(~mask,-torch.inf)))
            self.assertTrue(torch.equal(stable_topk_indices(score,10,largest=True),picks))
        # Tied FP32 scores remain byte-identical on the baseline ranking path.
        p=torch.full((1,100),.01);mask=torch.ones_like(p,dtype=torch.bool)
        self.assertTrue(torch.equal(f.ranking_score(p,p,mask),p))

    def test_beta_zero_training_gradient_and_second_step(self):
        mask,p,_=self.baseline()
        m=nn.Linear(6,1).double();features=torch.randn(3,23,6,dtype=torch.float64)
        f=Fusion();f.calibration.fit(m(features).detach().squeeze(-1)[mask][:,None],'train')
        target=normalized_teacher_prob(torch.rand_like(p),mask,.1).double()
        opt=torch.optim.AdamW(list(m.parameters())+list(f.parameters()),lr=.001)
        for step in range(2):
            opt.zero_grad()
            loss=kl_from_log_probability(target,f.log_probability(p,m(features).squeeze(-1),mask),mask)
            loss.backward()
            self.assertGreater(float(f.beta.grad.abs()),0)
            if step==0:self.assertEqual(float(m.weight.grad.norm()),0)
            else:self.assertGreater(float(m.weight.grad.norm()),0)
            opt.step()

    def test_chunk_value_gradient_and_global_teacher(self):
        q,c,n=self.fixtures();features=cross_pairs(q,c,n)
        m=PairMLP(18);m.standardizer.fit(features,'train')
        mask,p,_=self.baseline()
        teacher=normalized_teacher_prob(torch.rand_like(p),mask,.1).double()
        f=Fusion();f.calibration.fit(m(features).detach()[mask][:,None],'train')
        with torch.no_grad():f.beta.fill_(.2)
        results=[]
        for chunk in (1,7,23):
            m.zero_grad();f.zero_grad()
            s=assembled_scores(23,chunk,lambda a,b:m(features[:,a:b]))
            loss=kl_from_log_probability(teacher,f.log_probability(p,s,mask),mask)
            loss.backward()
            results.append((s.detach(),loss.detach(),m.layers[0].weight.grad.clone(),f.beta.grad.clone()))
        for r in results[1:]:
            for a,b in zip(results[0],r):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
        self.assertTrue(teacher[~mask].eq(0).all())

    def test_train_only_calibration_and_serialization(self):
        m=PairMLP(15);x=torch.randn(2,23,15,dtype=torch.float64)
        with self.assertRaises(ValueError):m.standardizer.fit(x,'val')
        with self.assertRaises(ValueError):m.standardizer.fit(x,'test')
        m.standardizer.fit(x,'train')
        with self.assertRaises(ValueError):m.standardizer.fit(x,'train')
        stream=io.BytesIO();torch.save(m.state_dict(),stream);stream.seek(0)
        restored=PairMLP(15);restored.load_state_dict(torch.load(stream,map_location='cpu'))
        self.assertTrue(torch.equal(m(x),restored(x)))

    def test_new_score_support_invariance(self):
        q,c,n=self.fixtures();features=dynamics_pairs(q,c,n)
        m=PairMLP(15);m.standardizer.fit(features,'train')
        subset=torch.tensor([1,2,9,11])
        torch.testing.assert_close(m(features)[:,subset],m(features[:,subset]),rtol=1e-12,atol=1e-12)

    def test_temporal_masks_and_empty_prefix(self):
        q=torch.tensor([0,100,300]);c=torch.arange(1000)
        legacy=eligibility(q,c,96,96,'legacy_raft','train')
        causal=eligibility(q,c,96,96,'known_at_forecast_origin','train')
        self.assertTrue(legacy[0,192]);self.assertFalse(causal[0].any())
        self.assertTrue(((c+192)[None].expand_as(causal)[causal] <= (q+96)[:,None].expand_as(causal)[causal]).all())
        with self.assertRaises(ValueError):masked_log_probability(torch.zeros(1,3),torch.zeros(1,3,dtype=torch.bool))

    def test_bad_probability_and_raw_encoder_path(self):
        with self.assertRaises(ValueError):log_v5_score(torch.tensor([[0.,1.]]),torch.ones(1,2,dtype=torch.bool))
        encoder=nn.Linear(6,6);scorer=DynamicsL2();q,c,n=self.fixtures()
        scorer(n(q['phi']),n(c['phi'])).sum().backward()
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    print(json.dumps({'scope':'CPU synthetic tests with direct official V5/teacher calls; no dataset baseline reproduction',
                      'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful()}))
    sys.exit(not result.wasSuccessful())
