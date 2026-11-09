"""Read-only numerical diagnosis; all channel production vs reference-only formulas."""
from pathlib import Path
import sys,time,json,hashlib
ROOT=Path('/data/pjh_workspace/CARTS')
sys.path.insert(0,str(ROOT/'experiments/retrieval_all_channel_tail08'))
import runtime_all_channels as runtime
r=runtime.run;t=r.torch;np=r.np

@t.no_grad()
def main():
    r.rt.gpu_guard();d=r.Data('Solar_96');cal=r.Calendar(d);a=len(d.datasets['test'])-1
    x,y,starts=d.query('test',a,a+1);B,C,H=x.shape;N=len(d.starts);mask=d.mask(starts)[:,None].expand(B,C,N)
    actual=t.empty(B,C,N,4,device=d.device);ref=t.empty_like(actual)
    from diagnostic_core import paa16,stable_order
    import torch.nn.functional as F
    def cos(q,k):return (F.normalize(q,dim=-1)[:,None]*F.normalize(k,dim=-1)).sum(-1)
    qm=x.reshape(B*C,H).mean(-1).reshape(B,C);qmr=t.stack([x[:,c].contiguous().mean(-1) for c in range(C)],1)
    sample=[]
    for k in range(0,N,128):
        ids=t.arange(k,min(k+128,N),device=d.device).expand(B,C,-1);hist=d.histories_all(ids)
        actual[...,k:k+128,:]=r.feature_block(d,x,starts,ids,cal)[...,:4]
        for c in range(C):
            q=x[:,c].contiguous();v=hist[:,c].contiguous();qz=q-q.mean(-1,keepdim=True);vz=v-v.mean(-1,keepdim=True);qd=q-q[:,-1:];vd=v-v[...,-1:]
            ref[:,c,k:k+128]=t.stack([cos(q,v),cos(qd,vd),cos(qz,vz),cos(paa16(qd),paa16(vd))],-1)
        if k==0:
            sample.append(dict(hist_shape=list(hist.shape),query_mean_maxdiff=float((qm-qmr).abs().max())))
    result=dict(cell=d.cell,split='test',query_row=a,query_start=int(starts[0]),query_shape=list(x.shape),full_candidates=N,rows=[],sample=sample,torch=t.__version__)
    for j,name in enumerate(r.BASE_NAMES[1:]):
        scores=actual[...,j];rs=ref[...,j]
        ap=stable_order(scores.reshape(B*C,N),mask.reshape(B*C,N))[...,:10]
        rp=stable_order(rs.reshape(B*C,N),mask.reshape(B*C,N))[...,:10]
        bad=t.where((ap!=rp).any(-1))[0].tolist();details=[]
        for c in bad:
            ids=t.unique(t.cat([ap[c],rp[c]]));qq=x[0,c]
            details.append(dict(channel=c,ordered_actual=ap[c].tolist(),ordered_reference=rp[c].tolist(),set_equal=bool(t.equal(ap[c].sort()[0],rp[c].sort()[0])),
                query_min=float(qq.min()),query_max=float(qq.max()),query_std=float(qq.std()),mean_batched=float(qm[0,c]),mean_reference=float(qmr[0,c]),
                candidates=ids.tolist(),actual=scores[0,c,ids].tolist(),reference=rs[0,c,ids].tolist(),max_score_diff=float((scores[0,c]-rs[0,c]).abs().max())))
        result['rows'].append(dict(method=name,ordered_mismatch_channels=len(bad),max_score_difference=float((scores-rs).abs().max()),different_score_count=int((scores!=rs).sum()),details=details))
    out=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/SOLAR_DIAG11';out.mkdir(parents=True,exist_ok=True)
    r.dump(out/'DIAGNOSIS.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
