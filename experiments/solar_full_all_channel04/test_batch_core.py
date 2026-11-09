import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import batch_core as f

def data_fixture(h,c=3,n=23):
    torch.manual_seed(122);series=torch.randn(n+2*h-1,c)
    return SimpleNamespace(h=h,c=c,starts=np.arange(n),series=series,future_series=series,
        offsets=torch.arange(h),device=torch.device('cpu'),mask=lambda starts:torch.ones(len(starts),n,dtype=torch.bool))

@pytest.mark.parametrize('h',[96,192,336,720])
@pytest.mark.parametrize('heads',[0,5])
def test_all_channel_chunk_loss_gradients_and_step(h,heads):
    d=data_fixture(h);model,head,_,_=f.rt.model_init('Solar_'+str(h),0,heads,'cpu')
    model.eval()
    if head is not None:head.eval()
    ref=copy.deepcopy(model);rh=copy.deepcopy(head)
    torch.manual_seed(64);x=torch.randn(3,d.c,h);y=torch.randn_like(x)
    history,values=f.original.bank_views(d,list(range(d.c)))
    direct=f.objective(f.original.scores_group(ref,rh,x,history),f.original.teacher_distances(values,x[:,:,-1],y),d.mask(range(3)),rh)
    direct.backward();loss=f.backward_batch(d,model,head,x,y,np.arange(3),3,23)
    assert loss==pytest.approx(float(direct),abs=2e-6,rel=2e-5)
    p=list(model.parameters())+([] if head is None else list(head.parameters()))
    q=list(ref.parameters())+([] if rh is None else list(rh.parameters()))
    for a,b in zip(p,q):
        assert (a.grad is None)==(b.grad is None)
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=3e-6,rtol=2e-4)
    # Exact one-step contract, no optimizer step per query or candidate chunk.
    opt=torch.optim.Adam(p,lr=.001);opt.step()
    reference_opt=torch.optim.Adam(q,lr=.001);reference_opt.step()
    for a,b in zip(p,q):assert torch.equal(a,b),'Production full grouped update must be exact on CPU'
    assert all(int(v['step'])==1 for v in opt.state.values())

@pytest.mark.parametrize('query_chunk,candidate_chunk',[(1,5),(2,7),(3,23)])
def test_dropout_mask_mapping_and_rng_recompute(query_chunk,candidate_chunk):
    d=data_fixture(96);model,heads,_,_=f.rt.model_init('Solar_96',0,5,'cpu');model.train();heads.train()
    other=copy.deepcopy(model);oh=copy.deepcopy(heads);x=torch.randn(3,3,96);y=torch.randn_like(x)
    history,values=f.original.bank_views(d,[0,1,2]);before=torch.get_rng_state()
    direct=f.objective(f.original.scores_group(other,oh,x,history),f.original.teacher_distances(values,x[:,:,-1],y),d.mask(range(3)),oh)
    direct.backward();expected_rng=torch.get_rng_state();torch.set_rng_state(before)
    loss=f.backward_batch(d,model,heads,x,y,np.arange(3),query_chunk,candidate_chunk)
    assert torch.equal(torch.get_rng_state(),expected_rng),'Chunk/recompute must not consume additional dropout draws'
    assert loss==pytest.approx(float(direct),abs=2e-6,rel=2e-5)
    for p,q in zip(list(model.parameters())+list(heads.parameters()),list(other.parameters())+list(oh.parameters())):
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=2e-4)

def test_all_137_channels_and_every_candidate_in_chunks():
    d=data_fixture(96,137,19)
    h=torch.cat([f.history_chunk(d,a,min(a+5,19)) for a in range(0,19,5)],1)
    v=torch.cat([f.values_chunk(d,a,min(a+5,19)) for a in range(0,19,5)],1)
    refh,refv=f.original.bank_views(d,list(range(137)))
    assert h.shape==v.shape==(137,19,96)
    assert torch.equal(h,refh) and torch.equal(v,refv)
    x=torch.randn(2,137,96);y=torch.randn_like(x)
    assert torch.equal(f.distances(d,x,y,5),f.original.teacher_distances(refv,x[:,:,-1],y,5))

def test_global_softmax_not_candidate_chunk_softmax():
    score=torch.randn(2,137,5,101);mask=torch.ones(2,101,dtype=torch.bool);mask[:,-1]=False
    p=f.probability(score,mask,object())
    assert p.shape==(2,137,101) and torch.equal(p[...,-1],torch.zeros(2,137))
    torch.testing.assert_close(p.sum(-1),torch.ones(2,137))

@pytest.mark.parametrize('q,p',[(16,23),(32,7),(1,1)])
def test_failed_chunk_training_cannot_be_activated(q,p):
    with pytest.raises(AssertionError,match='BLOCKED'):f.validate_training_shape(data_fixture(96),q,p)
