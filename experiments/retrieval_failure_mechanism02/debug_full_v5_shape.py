"""Reference-only audit of Full V5 E96 validation batch 1984."""
import os,json
import torch
import torch.nn.functional as F
import numpy as np
from adapter import Data,Frozen,inventory,topk,normalize,encode_raw_channel_first
from worker import ROOT,OUT,guard
from io_utils import dump,now,asnumpy
from cublaslt_batch import CublasLtBatch

@torch.no_grad()
def main():
    os.chdir(ROOT);guard();torch.set_num_threads(1);d=Data('ETTh1_96')
    row=next(r for r in inventory(d.cell) if r['architecture']=='V5' and r['training_support']=='FULL')
    f=Frozen(d,row);f.precision();f.batch.build_keys();a=1984;b=2016;x,y,starts=d.query('val',a,b);B,C,H=x.shape
    q=torch.cat([f.batch.encode(x[i:i+B//C],B,project=True) for i in range(0,B,B//C)])
    cq=[];ck=[];cs=[];cp=[];chead=[]
    mask=d.mask(starts)[:,None].expand(B,C,-1)
    for c in range(C):
        xx,_,_=d.channel_batch('val',a,b,c,future=False)
        cq.append(f.heads(encode_raw_channel_first(f.model,xx,0)))
        ck.append(F.normalize(encode_raw_channel_first(f.model,d.bank_history(c),0),dim=-1))
        cs.append(torch.einsum('bsd,nd->bsn',cq[-1],ck[-1]))
        chead.append(torch.softmax(cs[-1].masked_fill(~mask[:,c,None],-float('inf'))/.1,-1));cp.append(chead[-1].mean(1))
    cq=torch.stack(cq,1);ck=torch.stack(ck);cs=torch.stack(cs,1);cp=torch.stack(cp,1)
    native,mask=f.full_logits('val',a,b);lt=CublasLtBatch();qq=q.permute(1,0,2,3).contiguous().reshape(C,B*5,128)
    cases={'production':native,'canonical_score_batched_normalization':cs}
    for choice in range(3):
        try:cases['Lt_choice'+str(choice)]=lt.score(qq,f.batch.keys,77,1,choice).reshape(C,B,5,-1).permute(1,0,2,3).contiguous()
        except Exception as e:print('UNAVAILABLE',choice,repr(e),flush=True)
    expected=topk(cp,10);report=dict(at=now(),q_max=float((q-cq).abs().max()),key_max=float((f.batch.keys-ck).abs().max()),cases={})
    for name,s in cases.items():
        p=normalize(s,mask);ids=topk(p,10);mismatch=(ids!=expected).any(-1);detail=[]
        for qr,c in asnumpy(mismatch.nonzero())[:5]:
            detail.append(dict(query=int(starts[qr]),channel=int(c),expected=asnumpy(expected[qr,c]).tolist(),actual=asnumpy(ids[qr,c]).tolist(),
              score_difference=float((s-cs)[qr,c].abs().max()),prob_difference=float((p-cp)[qr,c].abs().max())))
        report['cases'][name]=dict(score_max=float((s-cs).abs().max()),prob_max=float((p-cp).abs().max()),ordered_mismatch_qc=int(mismatch.sum()),
          mismatch_sets=int(((ids[...,None]==expected[:,:,None]).any(-1).sum(-1)!=10).sum()),details=detail)
    zh=native.masked_fill(~mask[:,:,None],-float('inf'))/.1
    head=torch.softmax(zh,-1);chead=torch.stack(chead,1)
    packed=[]
    for offset in range(0,B,B//C):
        stop=min(offset+B//C,B);n=(stop-offset)*C;v=zh[offset:stop].reshape(n,5,-1)
        if n<B:v=torch.cat([v,v.new_zeros(B-n,5,v.shape[-1])])
        packed.append(torch.softmax(v,-1).mean(1)[:n].reshape(stop-offset,C,-1))
    channel_first=native.permute(1,0,2,3).contiguous().masked_fill(~mask.permute(1,0,2)[:,:,None],-float('inf'))
    channel_first_prob=torch.softmax(channel_first/.1,-1).mean(2).permute(1,0,2).contiguous()
    for name,p in {'channel_first':channel_first_prob,'packed_query_rows':torch.cat(packed),'canonical_head_batched_mean':chead.mean(2),'explicit_mean':(head[:,:,0]+head[:,:,1]+head[:,:,2]+head[:,:,3]+head[:,:,4])/5}.items():
        report['cases'][name]=dict(prob_max=float((p-cp).abs().max()),ordered_mismatch_qc=int((topk(p,10)!=expected).any(-1).sum()))
    report['head_probability_max']=float((head-chead).abs().max())
    report['channel_first_head_max']=float((torch.softmax(channel_first/.1,-1).permute(1,0,2,3)-chead).abs().max())
    dump(OUT/'00_Audit/FULL_V5_E96_BATCH1984_SHAPE_AUDIT.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
