from remaining_tail_preflight import *
import torch.nn.functional as F
from cublaslt_batch import CublasLtBatch
from all_channel_ops import score_all
@torch.no_grad()
def probe():
 a.rt.gpu_guard();d=a.Data('Weather_720');f=a.Frozen(d,next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V0'),'FULL');f.precision()
 end=len(d.datasets['val']);start=end//32*32;x,y,starts=d.query('val',start,end);B,C,H=x.shape
 ref=canonical.original_probability(f,'val',start,end);keys=torch.stack([f.keys[c] for c in range(C)])
 q=torch.stack([F.normalize(a.encode_raw_channel_first(f.model,x[:,c,:,None].contiguous(),0),dim=-1) for c in range(C)])
 fast=PackedChannelFull(f);qfast=fast.short_query(x).squeeze(2).permute(1,0,2).contiguous();mask=d.mask(starts)[None]
 print('QUERY_DIFF',float((qfast-q).abs().max()),flush=True)
 expected=ref.permute(1,0,2);rows=[]
 for lib in ['libcublasLt.so.11','libcublasLt.so.12']:
  try:lt=CublasLtBatch(lib)
  except OSError as e:print(str(e));continue
  for choice in range(16):
   try:
    got=lt.score(q,keys,68,1,choice).masked_fill(~mask,-float('inf'));delta=(got-expected).abs();delta=delta[torch.isfinite(delta)]
    row=dict(lib=lib,choice=choice,maxdiff=float(delta.max()),exact=torch.equal(got,expected))
   except RuntimeError as e:row=dict(lib=lib,choice=choice,error=str(e))
   rows.append(row);print(json.dumps(row),flush=True)
 a.rt.write_json(OUT/'TORCH2_SCORE_KERNEL_AUDIT.json',dict(query_maxdiff=float((qfast-q).abs().max()),rows=rows))
 extra=[];lt=CublasLtBatch('libcublasLt.so.12')
 for choice in [0,1,3]:
  for splitk in [0,1,2,4,8]:
   for reduction in [0,1,2,4]:
    try:
     got=lt.score(q,keys,68,1,choice,{2:splitk,3:reduction}).masked_fill(~mask,-float('inf'));delta=(got-expected).abs();delta=delta[torch.isfinite(delta)]
     row=dict(choice=choice,splitk=splitk,reduction=reduction,maxdiff=float(delta.max()),exact=torch.equal(got,expected))
    except RuntimeError as e:row=dict(choice=choice,splitk=splitk,reduction=reduction,error=str(e))
    extra.append(row)
 print('CONFIG',json.dumps([v for v in extra if 'maxdiff' in v]),flush=True)
 a.rt.write_json(OUT/'TORCH2_SCORE_CONFIG_AUDIT.json',dict(rows=extra))
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:q[0]@keys[0].T;torch.cuda.synchronize()
 p.export_chrome_trace(str(OUT/'TORCH2_CANONICAL_SCORE_TRACE.json'))
 trace=json.load(open(OUT/'TORCH2_CANONICAL_SCORE_TRACE.json'));print('KERNELS',json.dumps([v.get('name') for v in trace['traceEvents'] if v.get('cat','').lower()=='kernel']))
if __name__=='__main__':probe()
