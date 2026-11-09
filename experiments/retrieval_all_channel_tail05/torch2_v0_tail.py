from remaining_tail_preflight import *
@torch.no_grad()
def probe():
 assert int(torch.__version__.split('.')[0])>=2
 a.rt.gpu_guard();d=a.Data('Weather_720');f=a.Frozen(d,next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V0' and r['status']=='VALID'),'FULL');fast=PackedChannelFull(f);rows=[]
 for split in ['val','test']:
  n=len(d.datasets[split]);b=n%32;start=n-b;x,y,starts=d.query(split,start,n)
  f.precision();ref=canonical.original_probability(f,split,start,n);got=fast.probability(split,start,n)
  top=lambda t:a.rt.stable_topk_indices(t.reshape(b*d.c,-1),10,largest=True).reshape(b,d.c,10)
  old=top(ref);new=top(got);pv=(d.values_all(new)+x[:,:,-1,None,None]).mean(2)
  saved=torch.as_tensor(f.reference(split)[0][start:n].transpose(0,2,1),device=d.device)
  row=dict(cell=d.cell,model=f.name,split=split,queries=b,channels=d.c,candidates=len(d.starts),torch=torch.__version__,tf32=torch.backends.cuda.matmul.allow_tf32,ordered_mismatches=int((old!=new).any(-1).sum()),prediction_maxdiff=float((pv-saved).abs().max()),stored_prediction_pass=torch.allclose(pv,saved,atol=1e-5,rtol=1e-5),full_score_cache_export='NOT_STARTED_DEFERRED_TO_NORMAL_CELL_ORDER',all_channels_gpu_batch=True)
  rows.append(row);print(json.dumps(row),flush=True);assert row['ordered_mismatches']==0 and row['stored_prediction_pass'],row
 a.rt.write_json(OUT/'WEATHER720_V0_TORCH2_TAIL_GATE.json',dict(status='PASS',rows=rows))
if __name__=='__main__':probe()
