"""Pure CPU D5/D6 metrics. Future labels define diagnostic groups only."""
import numpy as np

ALPHAS=(0.,.01,.03,.1,.3,1.)
BETAS=(0.,.01,.03,.1)
NEAR_FRACTION=.05
LAG_MARGIN_EPS=.01  # cosine units; fixed before reading new test diagnostics
FLAT_STD_EPS=1e-8

def lag_summary(lag,gain,margin,entropy,std):
    a=np.asarray(lag);n=len(a)
    if not n:return dict(count=0)
    bins,freq=np.unique(a,return_counts=True);p=freq/n
    return dict(count=n,best_lag_mean=float(a.mean()),best_lag_median=float(np.median(a)),
       best_lag_histogram={str(int(k)):int(v) for k,v in zip(bins,freq)},
       absolute_lag_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(abs(a),return_counts=True))},
       zero_fraction=float(np.mean(a==0)),positive_fraction=float(np.mean(a>0)),negative_fraction=float(np.mean(a<0)),
       unique_best_lags=len(bins),sign_coverage=int(np.any(a<0))+int(np.any(a==0))+int(np.any(a>0)),
       useful_candidate_lag_entropy=float(-np.sum(p*np.log(p))),
       shift_gain_mean=float(np.mean(gain)),shift_gain_median=float(np.median(gain)),
       shift_gain_p90=float(np.quantile(gain,.9)),shift_gain_p95=float(np.quantile(gain,.95)),
       top12_margin_mean=float(np.mean(margin)),lag_attention_entropy_mean=float(np.mean(entropy)),
       effective_lags_mean=float(np.mean(np.exp(entropy))),
       near_tied_candidate_fraction=float(np.mean(np.asarray(margin)<=LAG_MARGIN_EPS)),
       flat_candidate_fraction=float(np.mean(np.asarray(std)<=FLAT_STD_EPS)))

def hist_distance(a,b):
    keys=sorted(set(a)|set(b));x=np.array([a.get(k,0) for k in keys],float);y=np.array([b.get(k,0) for k in keys],float)
    if not x.sum() or not y.sum():return dict(total_variation=None,jensen_shannon=None)
    x/=x.sum();y/=y.sum();m=(x+y)/2
    js=.5*sum(np.sum(t[t>0]*np.log(t[t>0]/m[t>0])) for t in [x,y])
    return dict(total_variation=float(abs(x-y).sum()/2),jensen_shannon=float(js))

def scores_to_ids(score,ids,k=10):
    order=np.argsort(-np.asarray(score),axis=-1,kind='stable')[...,:k]
    return np.take_along_axis(ids,order,-1),order

def select_validation(score,gain,recent,mse,kind):
    # No test parameters exist in this API. All channels retain identical axes.
    scale=np.maximum(1e-8,np.std(score,axis=(0,2)))
    recscale=np.maximum(1e-8,np.median(recent,axis=(0,2)))
    trials=[]
    for a in ALPHAS:
        for b in ((0.,) if kind=='ShiftOnly' else BETAS):
            s=score+scale[None,:,None]*(a*gain-b*recent/recscale[None,:,None])
            ix=np.argsort(-s,axis=-1,kind='stable')[...,:10]
            ret=np.take_along_axis(mse,ix,-1).astype(np.float64).mean()
            trials.append(dict(alpha=a,beta=b,val_retmse=float(ret)))
    chosen=min(trials,key=lambda r:(r['val_retmse'],r['alpha'],r['beta']))
    return dict(selected=chosen,score_scale=scale.tolist(),recent_scale=recscale.tolist(),grid=trials,test_used=False)

def d6_unit(ids,mse,score,gain,recent,oracle_ids,oracle_mse,selection,near_support_count=None):
    ids=np.asarray(ids);mse=np.asarray(mse,np.float64)
    assert len(ids)>=10 and len(np.unique(ids))==len(ids)
    a,b=selection['alpha'],selection['beta']
    adjusted=score+selection['score_scale']*(a*gain-b*recent/selection['recent_scale'])
    chosen,ix=scores_to_ids(adjusted,ids)
    if a==b==0:assert np.array_equal(chosen,ids[:10])
    r0=float(mse[:10].mean());rr=float(np.sort(mse,kind='stable')[:10].mean())
    rs=float(np.asarray(oracle_mse,np.float64).mean());rt=float(mse[ix].mean())
    limitation=rr-rs;reranking=rt-rr
    assert limitation>=-1e-10 and reranking>=-1e-10,'negative regret; do not clamp'
    np.testing.assert_allclose(rt-rs,limitation+reranking,atol=1e-12,rtol=1e-12)
    near=mse<=float(np.max(oracle_mse))+max(1e-8,NEAR_FRACTION*float(np.max(oracle_mse)))
    return dict(original_retmse=r0,reranked_retmse=rt,oracle_at_r=rr,support_oracle_retmse=rs,
        candidate_limitation_regret=limitation,reranking_regret=reranking,total_regret=rt-rs,
        recoverable_headroom=r0-rr,oracle10_coverage=float(np.isin(oracle_ids,ids).mean()),
        near_oracle_coverage=float(near.sum()/near_support_count) if near_support_count else None,
        near_oracle_support_count=near_support_count,
        recall=float(np.isin(chosen,oracle_ids).mean()),win=int(rt<r0),tie=int(rt==r0),loss=int(rt>r0),
        selected_ids=chosen.tolist())

def validate_r_curve(rows):
    ordered=sorted(rows,key=lambda r:r['R'])
    assert all(a['oracle10_coverage']<=b['oracle10_coverage'] for a,b in zip(ordered,ordered[1:]))
    assert all(a['oracle_at_r']+1e-10>=b['oracle_at_r'] for a,b in zip(ordered,ordered[1:]))
    if ordered and ordered[0]['R']==10:
        first=ordered[0]
        np.testing.assert_allclose(first['original_retmse'],first['oracle_at_r'],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(first['original_retmse'],first['reranked_retmse'],atol=1e-12,rtol=1e-12)

def loss_tail_contribution(delta,fraction):
    # timestamp means over ALL channels, then positive deterioration only.
    d=np.asarray(delta).mean(1);positive=np.maximum(d,0);n=max(1,int(np.ceil(len(d)*fraction)))
    denom=positive.sum()
    return None if denom==0 else float(np.sort(positive)[-n:].sum()/denom)
