import sys,os,json,subprocess
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');HERE=Path(__file__).parent
sys.path[:0]=[str(ROOT/'experiments/retrieval_full_precision_repair02'),str(ROOT)]
import full_precision_runtime
import diagnostic_adapter as a
import torch
import torch.nn.functional as F
from packed_batch import PackedChannelFull
from all_channel_ops import score_all,channel_linear
from cublaslt_batch import CublasLtBatch
OUT=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/TAIL05'

@torch.no_grad()
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.GPU;a.rt.gpu_guard()
    d=a.Data('Weather_336');row=next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V0');f=a.Frozen(d,row,'FULL');f.precision()
    end=len(d.datasets['val']);start=end//32*32;x,y,starts=d.query('val',start,end);B,C,H=x.shape
    ref=f.probability('val',start,end);fast=PackedChannelFull(f);got=fast.probability('val',start,end)
    key=torch.stack([f.keys[c] for c in range(C)]);q=torch.stack([F.normalize(a.encode_raw_channel_first(f.model,x[:,c,:,None].contiguous(),0),dim=-1) for c in range(C)],1)[:,:,None]
    mask=d.mask(starts)[:,None].expand(B,C,-1)
    def assess(sc):
        sc=sc.masked_fill(~mask,-float('inf'));old=a.rt.stable_topk_indices(ref.reshape(B*C,-1),10,largest=True);new=a.rt.stable_topk_indices(sc.reshape(B*C,-1),10,largest=True)
        delta=(sc-ref).abs();delta=delta[torch.isfinite(delta)]
        return dict(max_score_difference=float(delta.max()),ordered_mismatch=int((new!=old).any(-1).sum()),set_mismatch=int((new.sort(-1).values!=old.sort(-1).values).any(-1).sum()))
    result=dict(cell=d.cell,model='V0',batch=B,channels=C,tf32=torch.backends.cuda.matmul.allow_tf32,actual=assess(got),keys_max_difference=float((key-fast.keys).abs().max()),query_max_difference=float((q-fast.short_query(x)).abs().max()),canonical_query_batched_score=assess(score_all(q,key)[:,:,0]))
    layers={n:m for n,m in f.model.encoder.named_modules() if isinstance(m,(torch.nn.Linear,torch.nn.LayerNorm,torch.nn.GELU))};saved={n:[] for n in layers};hooks=[]
    for n,m in layers.items():
        def hook(m,inp,out,n=n):saved[n].append((inp[0].clone(),out.clone()))
        hooks.append(m.register_forward_hook(hook))
    for c in range(C):a.encode_raw_channel_first(f.model,x[:,c,:,None].contiguous(),0)
    for h in hooks:h.remove()
    lt=CublasLtBatch();records=[]
    for n,m in layers.items():
        inp=torch.stack([z[0] for z in saved[n]]);expected=torch.stack([z[1] for z in saved[n]]);variants=[]
        for tf32 in (True,False):
            torch.backends.cuda.matmul.allow_tf32=tf32;out=m(inp.reshape(C*B,-1)).reshape(C,B,-1)
            variants.append(dict(mode='flat',tf32=tf32,maxdiff=float((out-expected).abs().max()),exact=torch.equal(out,expected)))
        f.precision()
        if isinstance(m,torch.nn.Linear):
            for compute in (77,68):
                for choice in range(8):
                    try:
                        out=lt.linear(inp,m,compute,1,choice);torch.cuda.synchronize()
                        variants.append(dict(mode='lt_batch',compute=compute,choice=choice,maxdiff=float((out-expected).abs().max()),exact=torch.equal(out,expected)))
                    except RuntimeError as e:variants.append(dict(mode='lt_batch',compute=compute,choice=choice,error=str(e)))
        records.append(dict(layer=n,shape=list(inp.shape),variants=variants))
    result['layers']=records;a.rt.write_json(OUT/'WEATHER336_V0_TAIL_DIAGNOSIS.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()

