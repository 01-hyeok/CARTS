"""Reference-only numerical audit of a known failing batch, never a fallback."""
import os,json
import torch
import torch.nn.functional as F
import numpy as np
from adapter import Data,Frozen,inventory,rt,normalize,topk
from worker import guard,ROOT,OUT
from io_utils import dump,now,asnumpy

def main():
    os.chdir(ROOT);guard();torch.set_num_threads(1);d=Data('ETTh1_96');row=next(r for r in inventory(d.cell) if r['architecture']=='V0' and r['training_support']=='P100');f=Frozen(d,row);f.precision()
    split='val';a=352;b=384;x,_,starts=d.query(split,a,b);B,C,H=x.shape;ids=d.pool_all(split,a,b)
    nq=torch.cat([f.batch.encode(x[i:i+max(1,B//C)],B,project=True) for i in range(0,B,max(1,B//C))])[:,:,0]
    hist=d.histories_all(ids).permute(0,2,1,3).reshape(B*100,C,H);rows=B*100;step=rows//C
    nk=torch.cat([f.batch.encode(hist[i:i+step],rows) for i in range(0,B*100,step)]).reshape(B,100,C,-1).permute(0,2,1,3).contiguous()
    cq=[];ck=[];cs=[]
    for c in range(C):
        xx,_,_=d.channel_batch(split,a,b,c,future=False)
        z=rt.encode_raw(f.model,xx,0);cq.append(F.normalize(z,dim=-1))
        hh=d.histories(ids[:,c],c);kk=F.normalize(rt.encode_raw(f.model,hh.reshape(B*100,H,1),0).reshape(B,100,-1),dim=-1)
        ck.append(kk);cs.append(rt.scores(f.model,f.heads,xx,d,c,ids[:,c],query_channel=0))
    cq=torch.stack(cq,1);ck=torch.stack(ck,1);cs=torch.stack(cs,1)
    ns,_=f.native_p100_logits(split,a,b)
    # Disentangle encoding from batched dot-product and its operand layout.
    candidates={'production':ns,'canonical_encoded_batched':torch.einsum('bcd,bcpd->bcp',cq,ck)[:,:,None],
                'native_encoded_reduce':(nq[:,:,None]*nk).sum(-1)[:,:,None],
                'canonical_encoded_reduce':(cq[:,:,None]*ck).sum(-1)[:,:,None]}
    packed=[]
    for offset in range(0,B,max(1,B//C)):
        end=min(offset+max(1,B//C),B);n=(end-offset)*C
        q=nq[offset:end].reshape(n,-1);k=nk[offset:end].reshape(n,100,-1)
        q=torch.cat([q,torch.zeros(B-n,q.shape[-1],device=q.device)])
        k=torch.cat([k,torch.zeros(B-n,100,k.shape[-1],device=k.device)])
        p=torch.einsum('bd,bpd->bp',q,k)[:n].reshape(end-offset,C,1,100);packed.append(p)
    candidates['canonical_B32_packed_allchannels']=torch.cat(packed)
    old=f.reference(split)['ids'][a:b];result=dict(at=now(),q_max=float((nq-cq).abs().max()),key_max=float((nk-ck).abs().max()),cases={})
    for name,s in {'canonical':cs,**candidates}.items():
        p=normalize(s,torch.ones(B,C,100,device=d.device,dtype=torch.bool));actual=asnumpy(ids.gather(-1,topk(p,10)))
        mismatch=np.argwhere(np.any(actual!=old,axis=-1));details=[]
        for q,c in mismatch[:10]:
            details.append(dict(query=int(starts[q]),channel=int(c),got=actual[q,c].tolist(),expected=old[q,c].tolist(),
                                logits_max=float((s-cs)[q,c].abs().max())))
        result['cases'][name]=dict(score_max=float((s-cs).abs().max()),mismatch_qc=len(mismatch),details=details)
    dump(OUT/'00_Audit/P100_BATCH352_NUMERICAL_AUDIT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
