"""CPU synthetic protocol tests, not real model performance evidence."""
import json, unittest
from pathlib import Path
import torch
torch.set_num_threads(1)
torch.manual_seed(0)

class ProtocolTests(unittest.TestCase):
    def test_distribution_kl_and_gradient(self):
        raw=torch.randn(3,5,23,dtype=torch.float64)
        mask=torch.ones(3,23,dtype=torch.bool); mask[:,-3:]=False
        p=(raw/.1).masked_fill(~mask[:,None],-torch.inf).softmax(-1).mean(1)
        base=.1*p.masked_fill(~mask,1).log()
        beta=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
        residual=torch.randn_like(base,requires_grad=True)
        logp=((base+beta*residual)/.1).masked_fill(~mask,-torch.inf).log_softmax(-1)
        self.assertTrue(torch.allclose(logp.exp(),p,rtol=1e-12,atol=1e-12))
        teacher=torch.randn_like(base).masked_fill(~mask,-torch.inf).softmax(-1)
        loss=(teacher*(teacher.clamp_min(1e-300).log()-logp.masked_fill(~mask,0))).sum(-1).mean()
        direct=(teacher*(teacher.clamp_min(1e-300).log()-p.clamp_min(1e-300).log())).sum(-1).mean()
        self.assertTrue(torch.allclose(loss,direct,atol=1e-12,rtol=1e-12))
        loss.backward();self.assertGreater(abs(beta.grad.item()),0)
        self.assertEqual(residual.grad.abs().sum().item(),0)
        self.assertFalse(torch.allclose((base/.2).masked_fill(~mask,-torch.inf).softmax(-1),p))

    def test_global_chunk_normalization_and_gradients(self):
        x=torch.randn(4,37,dtype=torch.float64,requires_grad=True)
        y=x.detach().clone().requires_grad_()
        teacher=torch.randn_like(x).softmax(-1)
        full=x.log_softmax(-1)
        chunks=list(y.split(7,-1))
        normalizer=torch.logsumexp(torch.stack([c.logsumexp(-1) for c in chunks],-1),-1)
        chunked=torch.cat([c-normalizer[:,None] for c in chunks],-1)
        self.assertTrue(torch.allclose(full,chunked,atol=1e-12,rtol=1e-12))
        a=-(teacher*full).sum();b=-(teacher*chunked).sum();a.backward();b.backward()
        self.assertTrue(torch.allclose(x.grad,y.grad,atol=1e-12,rtol=1e-12))

    def test_zero_last_layer_gate_learns(self):
        gate=torch.nn.Linear(4,1).double();delta=torch.nn.Linear(6,1).double()
        torch.nn.init.zeros_(delta.weight);torch.nn.init.zeros_(delta.bias)
        q=torch.randn(8,4,dtype=torch.float64);z=torch.randn(8,6,dtype=torch.float64)
        correction=gate(q).sigmoid()*delta(z)
        self.assertEqual(correction.abs().sum().item(),0)
        correction.square().sum().backward() # zero residual objective has no signal by itself
        delta.zero_grad();gate.zero_grad()
        correction=gate(q).sigmoid()*delta(z)
        (correction-torch.ones_like(correction)).square().sum().backward()
        self.assertGreater(delta.weight.grad.abs().sum().item(),0)
        self.assertEqual(gate.weight.grad.abs().sum().item(),0)

    def test_uniform_gate_only_reproduces_current_heads(self):
        head=torch.randn(2,5,17,dtype=torch.float64)
        residual=torch.randn_like(head)
        original=head.softmax(-1).mean(1)
        corrected=(head+residual).softmax(-1)
        gate=torch.zeros(2,5,dtype=torch.float64).softmax(-1)
        uniform=(gate[:,:,None]*corrected).sum(1)
        self.assertTrue(torch.allclose(uniform,corrected.mean(1)))
        self.assertFalse(torch.allclose(uniform,original))

    def test_global_beta_is_reparameterization(self):
        layer=torch.nn.Linear(5,1).double();x=torch.randn(7,5,dtype=torch.float64);beta=2.3
        folded=torch.nn.functional.linear(x,layer.weight*beta,layer.bias*beta)
        self.assertTrue(torch.allclose(beta*layer(x),folded))

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests))
    out=Path('/data/pjh_workspace/CARTS/results/TRACK-PREDICTIVE-SCORE-CALIBRATION01')
    with (out/'CPU_SYNTHETIC_TESTS.json').open('x') as f:
        json.dump(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),real_data=False,gpu_used=False),f,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
