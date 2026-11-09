"""Known-failure layer/shape audit, reference only and never a fallback."""
import os,json
import torch
import torch.nn.functional as F
import numpy as np
from adapter import Data,Frozen,inventory,topk,encode_raw_channel_first
from worker import ROOT,OUT,guard
from io_utils import dump,now,asnumpy
from cublaslt_batch import CublasLtBatch

@torch.no_grad()
def main():
    os.chdir(ROOT);guard();torch.set_num_threads(1);d=Data('ETTh1_96')
    row=next(r for r in inventory(d.cell) if r['architecture']=='V1' and r['training_support']=='FULL')
    f=Frozen(d,row);f.precision();f.batch.build_keys();a=32;b=64;x,y,starts=d.query('val',a,b);B,C,H=x.shape
    q=torch.cat([f.batch.encode(x[i:i+B//C],B,project=True) for i in range(0,B,B//C)])
    cq=[];ck=[];cs=[]
    for c in range(C):
        xx,_,_=d.channel_batch('val',a,b,c,future=False)
        cq.append(f.heads(encode_raw_channel_first(f.model,xx,0)))
        ck.append(F.normalize(encode_raw_channel_first(f.model,d.bank_history(c),0),dim=-1))
        cs.append(torch.einsum('bsd,nd->bsn',cq[-1],ck[-1]))
    cq=torch.stack(cq,1);ck=torch.stack(ck);cs=torch.stack(cs,1);native,mask=f.full_logits('val',a,b)
    cases={'production':native};lt=CublasLtBatch()
    qq=q.permute(1,0,2,3).contiguous().reshape(C,B,128)
    for choice in range(4):
        try:cases['Lt_choice'+str(choice)]=lt.score(qq,f.batch.keys,68,1,choice).permute(1,0,2)[:,:,None]
        except Exception as e:print('UNAVAILABLE',choice,repr(e),flush=True)
    expected=topk(cs[:,:,0].masked_fill(~mask,-float('inf')),10)
    report=dict(at=now(),q_max=float((q-cq).abs().max()),key_max=float((f.batch.keys-ck).abs().max()),cases={})
    for name,s in cases.items():
        ids=topk(s[:,:,0].masked_fill(~mask,-float('inf')),10)
        report['cases'][name]=dict(score_max=float((s-cs).abs().max()),ordered_mismatch_qc=int((ids!=expected).any(-1).sum()),
          mismatch_sets=int(((ids[...,None]==expected[:,:,None]).any(-1).sum(-1)!=10).sum()))
    dump(OUT/'00_Audit/FULL_V1_E96_BATCH32_SHAPE_AUDIT.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
