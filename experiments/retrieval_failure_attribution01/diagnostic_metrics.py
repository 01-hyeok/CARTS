"""Bounded tensor statistics used by retrieval diagnostics (no model training)."""
import math
import numpy as np
import torch
from diagnostic_core import stable_order, support_metrics, ndcg_inverse_mse

FEATURES=['raw_cosine','delta_last_cosine','pearson','paa16_cosine','diff1_cosine',
          'history_mse','znorm_mse','mean_difference','std_difference','trend_difference',
          'diff_variance_difference','local_change_difference','acf1_difference',
          'low_frequency_power_difference','dominant_period_difference']
FEATURES += [side+'_'+k for side in ['candidate','query'] for k in ['mean','std','trend_slope','diff_variance','local_change','acf1','low_frequency_power_ratio','dominant_period']]
FEATURES += ['candidate_volatility','query_volatility','candidate_last_relative_mean','query_last_relative_mean']
PROBE_FEATURES=list(range(14))  # undefined dominant periods are excluded, not imputed from test.
METRICS=['retmse','oracle_retmse','ranking_regret','exchange_regret','recall','boundary_error','legacy_ndcg10',
         'ndcg_inverse_mse','ndcg50','ndcg100','recall50','recall100','spearman',
         'kendall','pairwise_ranking_accuracy','kendall_pair_count','student_tied_pairs','teacher_tied_pairs','pearson_score_negative_mse','spearman_student_top50','spearman_student_top100',
         'oracle_boundary_gap','student_boundary_gap','student_top1_topk_gap',
         'missed_count','missed_rank_mean','missed_rank_median','missed_rank_p90','missed_rank_percentile_mean',
         'missed_11_20','missed_21_50','missed_51_100','missed_101_500','missed_501_1000','missed_1001_plus',
         'boundary_11_20_exchange','agg_mse','agg_mae','last_mse','last_mae',
         'past10_distance','past10_future_dispersion','retmse_fp32_compat']


def temporal(x):
    """Past-only row statistics. Periods require at least two observed cycles."""
    mean=x.mean(-1);z=x-mean[...,None];std=z.square().mean(-1).sqrt()
    t=torch.arange(x.shape[-1],device=x.device,dtype=x.dtype);t=t-t.mean()
    slope=(z*t).sum(-1)/t.square().sum()
    diff=x[...,1:]-x[...,:-1];dv=diff.var(-1,unbiased=False)
    local=diff[...,-min(8,len(t)-1):].abs().mean(-1)
    acf=(z[...,1:]*z[...,:-1]).mean(-1)/(std.square()+1e-12)
    power=torch.fft.rfft(z,dim=-1).abs().square();power[...,0]=0
    total=power.sum(-1);low=power[...,1:max(2,x.shape[-1]//16+1)].sum(-1)/(total+1e-12)
    reliable=power.clone();reliable[...,:2]=0
    freq=reliable.argmax(-1)
    period=torch.where((total>1e-12)&(freq>=2),x.shape[-1]/freq.clamp_min(1),torch.full_like(total,float('nan')))
    return torch.stack([mean,std,slope,dv,local,acf,low,period],-1)


def features(q,c,qt=None,ct=None):
    """q [U,L], c [U,P,L]; no future arguments. Result [U,P,F]."""
    import torch.nn.functional as F
    from diagnostic_core import paa16
    def cos(a,b):return (F.normalize(a,dim=-1)[:,None]*F.normalize(b,dim=-1)).sum(-1)
    qz=q-q.mean(-1,keepdim=True);cz=c-c.mean(-1,keepdim=True)
    qd=q-q[:,-1:];cd=c-c[...,-1:]
    # population variance is fixed before observing new diagnostic outcomes.
    qn=qz/(qz.square().mean(-1,keepdim=True).sqrt()+1e-6)
    cn=cz/(cz.square().mean(-1,keepdim=True).sqrt()+1e-6)
    qt=temporal(q) if qt is None else qt;ct=temporal(c) if ct is None else ct
    a=[cos(q,c),cos(qd,cd),cos(qz,cz),cos(paa16(qd),paa16(cd)),
       cos(q[:,1:]-q[:,:-1],c[...,1:]-c[...,:-1]),
       (q[:,None]-c).square().mean(-1),(qn[:,None]-cn).square().mean(-1)]
    a += [(qt[:,None,i]-ct[...,i]).abs() for i in range(8)]
    a += [ct[...,i] for i in range(8)]+[qt[:,None,i].expand(c.shape[:2]) for i in range(8)]
    a += [(c[...,1:]-c[...,:-1]).square().mean(-1).sqrt(),(q[:,1:]-q[:,:-1]).square().mean(-1).sqrt()[:,None].expand(c.shape[:2]),c.mean(-1)-c[...,-1],(q.mean(-1)-q[:,-1])[:,None].expand(c.shape[:2])]
    return torch.stack(a,-1)


def average_ranks(x,mask):
    """Exact average ranks for exact ties, ignoring masked entries."""
    n=x.shape[-1];order=stable_order(x,mask,False);v=x.gather(1,order)
    valid=mask.gather(1,order);change=torch.ones_like(valid)
    change[:,1:]=(v[:,1:]!=v[:,:-1])|(~valid[:,1:])
    group=change.long().cumsum(-1)-1
    pos=torch.arange(1,n+1,device=x.device,dtype=torch.float64).expand_as(x)
    sums=torch.zeros_like(x,dtype=torch.float64).scatter_add_(1,group,pos*valid)
    count=torch.zeros_like(x,dtype=torch.float64).scatter_add_(1,group,valid.double())
    av=sums.gather(1,group)/count.gather(1,group).clamp_min(1)
    result=torch.zeros_like(x,dtype=torch.float64).scatter_(1,order,av)
    return result.masked_fill(~mask,0)


def pearson_masked(x,y,mask):
    n=mask.sum(-1).clamp_min(1);x=x.double();y=y.double()
    xx=(x-(x*mask).sum(-1)[:,None]/n[:,None])*mask
    yy=(y-(y*mask).sum(-1)[:,None]/n[:,None])*mask
    den=(xx.square().sum(-1)*yy.square().sum(-1)).sqrt()
    return torch.where(den>1e-15,(xx*yy).sum(-1)/den,torch.full_like(den,float('nan')))


def spearman(s,d,mask):return pearson_masked(average_ranks(s,mask),average_ranks(-d,mask),mask)


PAIR_CACHE={}
def pair_indices(n,device,full):
    key=(n,str(device),full)
    if key in PAIR_CACHE:return PAIR_CACHE[key]
    if full:
        # Fixed position pairs selected without seeing any score or future.
        rng=np.random.default_rng(0);pairs=set()
        target=min(4096,n*(n-1)//2)
        while len(pairs)<target:
            a,b=map(int,rng.integers(0,n,2))
            if a!=b:pairs.add((min(a,b),max(a,b)))
        a,b=np.array(sorted(pairs)).T
        a=torch.as_tensor(a,device=device);b=torch.as_tensor(b,device=device)
    else:a,b=torch.triu_indices(n,n,offset=1,device=device)
    PAIR_CACHE[key]=(a,b);return a,b

def kendall(s,d,mask,full=False):
    a,b=pair_indices(s.shape[-1],s.device,full)
    valid=mask[:,a]&mask[:,b]
    sx=torch.sign(s[:,a]-s[:,b]);sy=torch.sign(d[:,b]-d[:,a])
    den=((valid&(sx!=0)).sum(-1).double()*(valid&(sy!=0)).sum(-1).double()).sqrt()
    numerator=(sx*sy*valid).double().sum(-1)
    return torch.where(den>0,numerator/den,torch.full_like(den,float('nan')))

def pairwise_accuracy(s,d,mask,full=False):
    a,b=pair_indices(s.shape[-1],s.device,full);valid=mask[:,a]&mask[:,b]&(d[:,a]!=d[:,b])
    product=torch.sign(s[:,a]-s[:,b])*torch.sign(d[:,b]-d[:,a]);den=valid.sum(-1)
    correct=(((product>0).float()+.5*(product==0).float())*valid).sum(-1)
    return torch.where(den>0,correct/den,torch.full_like(correct,float('nan')))


def query_metrics(score,d,mask,full=False):
    m=support_metrics(score,d,mask);order=stable_order(score,mask);oracle_order=stable_order(d,mask,False)
    rel=(d.masked_fill(~mask,-float('inf')).max(-1,keepdim=True).values-d).masked_fill(~mask,0)
    disc=1/torch.log2(torch.arange(2,12,device=d.device,dtype=d.dtype))
    m['legacy_ndcg10']=(rel.gather(1,order[:,:10])*disc).sum(-1)/(rel.gather(1,oracle_order[:,:10])*disc).sum(-1).clamp_min(1e-12)
    for k in (50,100):
        assert bool((mask.sum(-1)>=k).all())
        m['ndcg'+str(k)]=ndcg_inverse_mse(d,order,mask,k)
        m['recall'+str(k)]=(order[:,:k,None]==oracle_order[:,None,:k]).any(-1).float().mean(-1)
        sub=order[:,:k];m['spearman_student_top'+str(k)]=spearman(score.gather(1,sub),d.gather(1,sub),torch.ones_like(sub,dtype=torch.bool))
    m['spearman']=spearman(score,d,mask);m['kendall']=kendall(score,d,mask,full)
    m['pairwise_ranking_accuracy']=pairwise_accuracy(score,d,mask,full)
    pa,pb=pair_indices(score.shape[-1],score.device,full);pv=mask[:,pa]&mask[:,pb]
    m['kendall_pair_count']=pv.sum(-1);m['student_tied_pairs']=(pv&(score[:,pa]==score[:,pb])).sum(-1);m['teacher_tied_pairs']=(pv&(d[:,pa]==d[:,pb])).sum(-1)
    m['pearson_score_negative_mse']=pearson_masked(score,-d,mask)
    ranks=m['missed_ranks'];num=m['missed_mask'].sum(-1);m['missed_count']=num
    m['missed_rank_mean']=ranks.sum(-1)/num.clamp_min(1)
    sr=ranks.double().masked_fill(~m['missed_mask'],float('inf')).sort(-1).values
    m['missed_rank_median']=(sr.gather(1,((num-1).clamp_min(0)//2)[:,None])+sr.gather(1,(num//2)[:,None]))[:,0]/2
    m['missed_rank_p90']=sr.gather(1,(torch.ceil(num.float()*.9).long()-1).clamp_min(0)[:,None]).squeeze(1)
    for key in ('missed_rank_mean','missed_rank_median','missed_rank_p90'):
        m[key]=torch.where(num>0,m[key],torch.full_like(m[key],float('nan')))
    m['missed_rank_percentile_mean']=m['missed_rank_mean']/mask.sum(-1)
    for lo,hi,name in [(11,20,'11_20'),(21,50,'21_50'),(51,100,'51_100'),(101,500,'101_500'),(501,1000,'501_1000'),(1001,numeric_max(),'1001_plus')]:
        m['missed_'+name]=((ranks>=lo)&(ranks<=hi)).sum(-1)
    false_d=d.gather(1,m['selected']).double().masked_fill(~m['false_selected_mask'],-float('inf')).sort(-1,descending=True).values
    missed_d=d.gather(1,m['oracle']).double().masked_fill(~m['missed_mask'],float('inf'))
    ix=torch.sort(missed_d,dim=-1,stable=True).indices;md=missed_d.gather(1,ix);mr=ranks.gather(1,ix)
    eligible=torch.arange(10,device=d.device)[None]<num[:,None]
    delta=torch.where(eligible,(false_d-md)/10,torch.zeros_like(md))
    m['boundary_11_20_exchange']=(delta*((mr>=11)&(mr<=20))).sum(-1)
    assert torch.allclose(delta.sum(-1),m['ranking_regret'],atol=2e-5,rtol=2e-5)
    return m


def numeric_max():return 2**62


class MomentRidge:
    """Streaming train-only sufficient statistics, fixed three-alpha validation selection."""
    def __init__(self,dimension=14):
        self.d=dimension;self.n=0;self.sx=np.zeros(dimension);self.sxx=np.zeros((dimension,dimension));self.sxy=np.zeros(dimension);self.sy=0.
        self.alphas=(.1,1.,10.);self.selected=None
    def update(self,x,y,split):
        assert split=='train';x=np.asarray(x,np.float64);y=np.asarray(y,np.float64)
        assert x.shape==(len(y),self.d) and np.isfinite(x).all() and np.isfinite(y).all()
        self.n+=len(y);self.sx+=x.sum(0);self.sxx+=x.T@x;self.sxy+=x.T@y;self.sy+=y.sum()
    def finish(self):
        assert self.n>self.d
        self.mean=self.sx/self.n;self.ymean=self.sy/self.n
        cov=self.sxx-np.outer(self.sx,self.sx)/self.n
        self.scale=np.sqrt(np.maximum(np.diag(cov)/self.n,1e-24))
        gram=cov/self.scale[:,None]/self.scale[None,:]
        rhs=(self.sxy-self.sx*self.ymean)/self.scale
        self.weights=np.stack([np.linalg.solve(gram+a*np.eye(self.d),rhs) for a in self.alphas])
        return self
    def predict(self,x):
        if torch.is_tensor(x):
            mean=torch.as_tensor(self.mean,device=x.device,dtype=x.dtype);scale=torch.as_tensor(self.scale,device=x.device,dtype=x.dtype)
            w=torch.as_tensor(self.weights,device=x.device,dtype=x.dtype)
            return torch.matmul((x-mean)/scale,w.T)+self.ymean
        return ((np.asarray(x)-self.mean)/self.scale)@self.weights.T+self.ymean
    def choose(self,losses,split):
        assert split=='val';self.selected=int(np.argmin(losses));return self.selected
    def state(self):return dict(mean=self.mean.tolist(),scale=self.scale.tolist(),weights=self.weights.tolist(),ymean=self.ymean,n=self.n,alphas=list(self.alphas),selected=self.selected)
