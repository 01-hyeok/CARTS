"""D1/D2 exact candidate-set diagnostics from immutable ID/distance arrays."""
import numpy as np


def analyze_unit(head_ids, head_mse, mix_ids, mix_mse, oracle_ids, oracle_mse,
                 epsilon_fractions=(.01,.05,.10)):
    h,hd,s,sd,o,od = map(np.asarray,(head_ids,head_mse,mix_ids,mix_mse,oracle_ids,oracle_mse))
    assert h.shape==hd.shape==(5,10) and s.shape==sd.shape==o.shape==od.shape==(10,)
    assert all(len(np.unique(x))==10 for x in [*h,s,o])
    assert all(np.isfinite(x).all() and (x>=0).all() for x in [hd,sd,od])
    assert all(np.issubdtype(x.dtype,np.integer) and (x>=0).all() for x in [h,s,o])
    dist={}
    for ids,ds in [(h.ravel(),hd.ravel()),(s,sd),(o,od)]:
        for i,d in zip(ids,ds):
            if int(i) in dist:
                # Same precomputed per-candidate errors must agree exactly.
                assert dist[int(i)]==float(d),'same candidate has inconsistent future MSE'
            dist[int(i)]=float(d)
    sets=[set(map(int,x)) for x in h];u=set.union(*sets);mix=set(map(int,s));oracle=set(map(int,o))
    union_mix=u|mix;choice=sorted(union_mix,key=lambda i:(dist[i],i))[:10]
    r_mix=float(sd.mean());r_c=float(np.mean([dist[i] for i in choice]));r_o=float(od.mean())
    fusion=r_mix-r_c;discovery=r_c-r_o
    assert fusion>=-1e-12 and discovery>=-1e-12,'invalid Oracle/candidate distance contract'
    np.testing.assert_allclose(fusion+discovery,r_mix-r_o,atol=1e-12,rtol=1e-12)
    lost=(u&oracle)-mix
    records=[]
    for j,hs in enumerate(sets):
        exclusive=hs-set.union(*(sets[k] for k in range(5) if k!=j))
        row=dict(head=j+1,exclusive_count=len(exclusive),exclusive_oracle_hits=len(exclusive&oracle),
                 head_ret=float(hd[j].mean()),head_recall=len(hs&oracle)/10)
        for eps in epsilon_fractions:
            useful={i for i in exclusive if dist[i]<=float(od.max())+max(1e-8,eps*float(od.max()))}
            key=str(eps)
            row['near_count_'+key]=len(useful)
            row['near_rate_'+key]=len(useful)/10
            row['near_mean_mse_'+key]=np.mean([dist[i] for i in useful]) if useful else None
        records.append(row)
    near_lost={}
    for eps in epsilon_fractions:
        near_lost[str(eps)]=len({i for i in u-mix if dist[i]<=float(od.max())+max(1e-8,eps*float(od.max()))})
    found=len(u&oracle)
    return dict(union_size=len(u),union_oracle_hits=found,final_oracle_hits=len(mix&oracle),
                lost_oracle_ids=sorted(lost),lost_oracle_count=len(lost),
                lost_oracle_fraction=len(lost)/found if found else None,
                lost_mean_mse=np.mean([dist[i] for i in lost]) if lost else None,
                lost_near_counts=near_lost,oracle_C_ids=choice,r_mix=r_mix,r_oracle_C=r_c,
                r_oracle=r_o,fusion_regret=fusion,discovery_regret=discovery,
                heads=records,negative_clamp=False,
                missing_fields=['lost_mixture_rank','lost_all_head_scores','lost_all_head_probabilities'])


def global_ids_gate(ids, valid_ids):
    a=np.asarray(ids);v=np.asarray(valid_ids)
    assert np.issubdtype(a.dtype,np.integer) and len(np.unique(v))==len(v)
    assert np.isin(a,v).all(),'invalid mask or ID mapping'


def loo_from_probabilities(p):
    p=np.asarray(p,np.float64)
    assert p.shape[-2]==5 and (p>=0).all()
    np.testing.assert_allclose(p.sum(-1),1,atol=1e-12)
    result=np.stack([(p.sum(-2)-p[...,h,:])/4 for h in range(5)],-2)
    np.testing.assert_allclose(result.sum(-1),1,atol=1e-12)
    return result
