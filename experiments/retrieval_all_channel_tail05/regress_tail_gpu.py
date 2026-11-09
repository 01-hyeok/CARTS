import sys,os,json,time,argparse
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');sys.path[:0]=[str(ROOT/'experiments/retrieval_full_precision_repair02'),str(ROOT)]
import full_precision_runtime
import diagnostic_adapter as a
import torch
import torch.nn.functional as F
from packed_batch import PackedChannelFull
OUT=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/TAIL05'
@torch.no_grad()
def main(cell,model):
    a.rt.gpu_guard();d=a.Data(cell);f=a.Frozen(d,next(r for r in a.inventory(cell,'FULL') if r['architecture']==model),'FULL');fast=PackedChannelFull(f);f.precision();rows=[]
    keys=torch.stack([F.normalize(a.encode_raw_channel_first(f.model,d.bank_history(c),0),dim=-1) for c in range(d.c)])
    for B in range(1,33):
        f.precision();x,y,starts=d.query('val',0,B);refs=[];qrefs=[]
        for c in range(d.c):
            z=a.encode_raw_channel_first(f.model,x[:,c,:,None].contiguous(),0)
            qq=F.normalize(z,dim=-1)[:,None] if f.heads is None else f.heads(z);qrefs.append(qq)
            sc=qq[:,0]@keys[c].T if f.heads is None else torch.einsum('bsd,nd->bsn',qq,keys[c])
            mask=d.mask(starts)
            refs.append(sc.masked_fill(~mask,-float('inf')) if f.heads is None else a.rt.mean_mixture_topk_selection(sc,mask,.1,10)[2])
        ref=torch.stack(refs,1);got=fast.probability('val',0,B);top=lambda p:a.rt.stable_topk_indices(p.reshape(B*d.c,-1),10,largest=True).reshape(B,d.c,10)
        old=top(ref);new=top(got);vold=(d.values_all(old)+x[:,:,-1,None,None]).mean(2);vnew=(d.values_all(new)+x[:,:,-1,None,None]).mean(2)
        delta=(got-ref).abs();delta=delta[torch.isfinite(delta)]
        row=dict(batch=B,channels=d.c,candidates=len(d.starts),ordered=int((old!=new).any(-1).sum()),set_mismatch=int((old.sort(-1).values!=new.sort(-1).values).any(-1).sum()),score_maxdiff=float(delta.max()),prediction_maxdiff=float((vold-vnew).abs().max()))
        rows.append(row);print(cell,model,json.dumps(row),flush=True)
        a.rt.write_json(OUT/(cell+'_'+model+'_ALL_TAIL_GPU.json'),dict(status='PASS' if B==32 and all(r['ordered']==0 for r in rows) else 'PARTIAL_OR_FAILED',rows=rows,checkpoint_sha256=f.row['checkpoint_sha256'],all_channels_together=True,canonical_serial_reference_only=True))
    assert all(r['ordered']==0 and r['prediction_maxdiff']==0 for r in rows),'Tail regression failed; no activation'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',default='Weather_192');p.add_argument('--model',default='V0');z=p.parse_args();main(z.cell,z.model)
