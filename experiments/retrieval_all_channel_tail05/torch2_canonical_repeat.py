from remaining_tail_preflight import *
import torch.nn.functional as F
@torch.no_grad()
def probe():
 a.rt.gpu_guard();d=a.Data('Weather_720');f=a.Frozen(d,next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V0'),'FULL');f.precision()
 n=len(d.datasets['val']);b=n%32;start=n-b;x,_,starts=d.query('val',start,n)
 keys=torch.stack([F.normalize(a.encode_raw_channel_first(f.model,d.bank_history(c),0),dim=-1) for c in range(d.c)])
 q=torch.stack([F.normalize(a.encode_raw_channel_first(f.model,x[:,c,:,None].contiguous(),0),dim=-1) for c in range(d.c)])
 mask=d.mask(starts)[None];top=lambda t:a.rt.stable_topk_indices(t.reshape(-1,len(d.starts)),10,largest=True)
 base=None;rows=[]
 for repeat in range(10):
  score=torch.stack([q[c]@keys[c].T for c in range(d.c)]).masked_fill(~mask,-float('inf'));torch.cuda.synchronize()
  if base is None:base=score.clone();ids=top(base)
  new=top(score);delta=(score-base).abs();delta=delta[torch.isfinite(delta)]
  row=dict(repeat=repeat,maxdiff=float(delta.max()),bitwise_equal=torch.equal(score,base),ordered_mismatches=int((new!=ids).any(-1).sum()),set_mismatches=int((new.sort(-1).values!=ids.sort(-1).values).any(-1).sum()))
  rows.append(row);print(json.dumps(row),flush=True)
 a.rt.write_json(OUT/'TORCH2_CANONICAL_REPEAT.json',dict(rows=rows,cell=d.cell,split='val_tail7',same_input_weights_precision=True,tf32=False,torch=torch.__version__))
if __name__=='__main__':probe()
