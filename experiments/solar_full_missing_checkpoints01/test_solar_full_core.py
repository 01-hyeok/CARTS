import copy
import pytest
import torch
import solar_full_core as f

@pytest.mark.parametrize('h',[96,192,336,720])
@pytest.mark.parametrize('nheads',[0,5])
def test_group_forward_loss_gradient_update_matches_channel_reference(h,nheads):
    # No real training/data/GPU. Explicit dropout-off numerical equivalence.
    model,heads,_,_=f.rt.model_init(f'Solar_{h}',0,nheads,'cpu')
    model.eval()
    if heads is not None:heads.eval()
    other=copy.deepcopy(model);oh=copy.deepcopy(heads)
    torch.manual_seed(7);b,c,n=2,3,31
    x=torch.randn(b,c,h);bank=torch.randn(c,n,h);values=torch.randn(c,n,h);y=torch.randn(b,c,h)
    mask=torch.ones(b,n,dtype=torch.bool);mask[0,:2]=False
    distance=f.teacher_distances(values,x[:,:,-1],y,7)
    direct=((values[None]+x[:,:,-1,None,None]-y[:,:,None])**2).mean(-1)
    assert torch.equal(distance,direct)
    score=f.scores_group(model,heads,x,bank)
    loss=f.objective(score,distance,mask,heads);loss.backward()
    total=0
    for j in range(c):
        reference=f.scores_group(other,oh,x[:,j:j+1],bank[j:j+1])
        torch.testing.assert_close(score[:,j:j+1],reference,rtol=2e-5,atol=2e-6)
        part=f.objective(reference,distance[:,j:j+1],mask,oh)/c;part.backward();total+=float(part.detach())
    assert float(loss)==pytest.approx(total,rel=2e-5,abs=2e-6)
    a=list(model.parameters())+([] if heads is None else list(heads.parameters()))
    z=list(other.parameters())+([] if oh is None else list(oh.parameters()))
    for p,q in zip(a,z):
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=3e-6)
    # SGD is used only for this update-equivalence check. Actual trainer uses Adam.
    opt=torch.optim.SGD(a,lr=.001);op2=torch.optim.SGD(z,lr=.001);opt.step();op2.step()
    for p,q in zip(a,z):torch.testing.assert_close(p,q,rtol=2e-5,atol=2e-6)

def test_global_teacher_and_student_normalization():
    torch.manual_seed(3);s=torch.randn(2,3,5,127);d=torch.rand(2,3,127);mask=torch.ones(2,127,dtype=torch.bool);mask[:,-7:]=False
    valid=mask[:,None].expand(2,3,127).reshape(6,127)
    teacher=f.rt.normalized_teacher_prob(d.reshape(6,127),valid,.1)
    assert torch.allclose(teacher.sum(-1),torch.ones(6))
    p=torch.softmax((s/.1).masked_fill(~mask[:,None,None],-float('inf')),-1).mean(2)
    assert torch.allclose(p.sum(-1),torch.ones(2,3))
    assert torch.equal(p[...,-7:],torch.zeros_like(p[...,-7:]))
    # Per-chunk normalizations would sum to two and must never substitute this.
    assert p.shape[-1]==127

def test_dropout_group_path_determinism_with_rng_restore():
    model,heads,_,_=f.rt.model_init('Solar_96',0,5,'cpu');model.train();heads.train()
    x=torch.randn(2,3,96);bank=torch.randn(3,31,96);state=f.rng_snapshot()
    a=f.scores_group(model,heads,x,bank);f.rng_restore(state);b=f.scores_group(model,heads,x,bank)
    assert torch.equal(a,b)

def test_bank_views_cover_every_candidate_and_channel():
    class D:pass
    data=D();data.h=5;data.series=torch.arange(30*3,dtype=torch.float32).reshape(30,3)
    data.future_series=data.series;data.starts=list(range(21))
    history,values=f.bank_views(data,[0,1,2])
    assert history.shape==(3,21,5) and values.shape==(3,21,5)
    for c in range(3):
        for i in range(21):
            assert torch.equal(history[c,i],data.series[i:i+5,c])
            assert torch.equal(values[c,i],data.series[i+5:i+10,c]-data.series[i+4,c])
