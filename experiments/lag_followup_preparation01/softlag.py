"""B1/B2 cache-only objective and independent NumPy gradient reference.

Encoder/head scores are immutable inputs; only scalar alpha and lag bias train.
Teacher is an externally verified probability array from the original pipeline.
No local teacher reimplementation is used to define the actual experiment.
"""
import numpy as np


def softmax(x):
    e=np.exp(x-np.max(x,axis=-1,keepdims=True));return e/e.sum(-1,keepdims=True)


def forward(g, a, activity, alpha, bias, tau_a=.1, tau_s=.1, mask=None):
    g,a,c=np.asarray(g,np.float64),np.asarray(a,np.float64),np.asarray(activity,np.float64)
    assert g.ndim==4 and g.shape[2]==5 and a.shape[:3]==g.shape[:2]+g.shape[3:]
    assert c.shape==a.shape[:-1] and np.asarray(bias).shape in [(1,a.shape[-1]),(5,a.shape[-1])]
    w=softmax(a[:,:,None]/tau_a+np.asarray(bias)[None,None,:,None,:])
    ah=(w*a[:,:,None]).sum(-1)
    correction=float(alpha)*c[:,:,None]*(ah-a[...,a.shape[-1]//2][:,:,None])
    u=g+correction
    m=np.ones(g.shape[:2]+g.shape[3:],bool) if mask is None else np.asarray(mask,bool)
    assert m.shape==c.shape and m.any(-1).all()
    p=softmax(np.where(m[:,:,None],u/tau_s,-np.inf));mix=p.mean(2)
    return dict(u=u,p=p,mix=mix,w=w,ah=ah,correction=correction,mask=m)


def objective_gradient(g,a,c,teacher,alpha,bias,tau_a=.1,tau_s=.1,mask=None):
    f=forward(g,a,c,alpha,bias,tau_a,tau_s,mask)
    t=np.asarray(teacher,np.float64);assert t.shape==f['mix'].shape and (t>=0).all()
    np.testing.assert_allclose(t.sum(-1),1,atol=1e-12)
    assert (t[~f['mask']]==0).all()
    mix=f['mix'];ratio=np.divide(t,mix,out=np.zeros_like(t),where=mix>0)
    logratio=np.zeros_like(t);np.log(np.divide(t,mix,out=np.ones_like(t),where=t>0),out=logratio,where=t>0)
    loss=float((t*logratio).sum(-1).mean())
    p=f['p'];grad_u=p*((p*ratio[:,:,None]).sum(-1,keepdims=True)-ratio[:,:,None])/(5*tau_s*t.shape[0]*t.shape[1])
    delta=f['ah']-a[...,a.shape[-1]//2][:,:,None]
    grad_alpha=float((grad_u*c[:,:,None]*delta).sum())
    grad_bias=(grad_u[...,None]*float(alpha)*c[:,:,None,:,None]*f['w']*(a[:,:,None]-f['ah'][...,None])).sum((0,1,3))
    if bias.shape[0]==1:grad_bias=grad_bias.sum(0,keepdims=True)
    return loss,grad_alpha,grad_bias


def torch_parameters(headwise,lags,device='cpu',dtype=None):
    import torch
    dtype=torch.float64 if dtype is None else dtype
    return (torch.nn.Parameter(torch.zeros((),device=device,dtype=dtype)),
            torch.nn.Parameter(torch.zeros(5 if headwise else 1,lags,device=device,dtype=dtype)))


def torch_forward(g,a,c,alpha,bias,mask,tau_a=.1,tau_s=.1):
    import torch
    assert not g.requires_grad and not a.requires_grad and not c.requires_grad
    w=torch.softmax(a[:,:,None]/tau_a+bias[None,None,:,None,:],-1)
    ah=(w*a[:,:,None]).sum(-1)
    u=g+alpha*c[:,:,None]*(ah-a[...,a.shape[-1]//2][:,:,None])
    return u,torch.softmax((u/tau_s).masked_fill(~mask[:,:,None],-float('inf')),-1).mean(2)
