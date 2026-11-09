import sys,json,time
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');sys.path[:0]=[str(ROOT/'experiments/retrieval_full_precision_repair02'),str(ROOT)]
import full_precision_runtime as canonical
import diagnostic_adapter as a
import torch
from packed_batch import PackedChannelFull
from all_channel_ops import distance_all
from utils.candidate_pool import pooled_future_mse
OUT=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/TAIL05'
@torch.no_grad()
def main():
 a.rt.gpu_guard();records=[]
 for cell in ['Weather_192','Weather_336','Weather_720','Solar_96','Solar_192','Solar_336','Solar_720']:
  d=a.Data(cell);rows=[r for r in a.inventory(cell,'FULL') if r['status']=='VALID' and not(cell=='Weather_720' and r['architecture']=='V0')];models=[a.Frozen(d,r,'FULL') for r in rows]
  adapters={m.name:PackedChannelFull(m) for m in models if not(m.heads is None and canonical.uses_score_cache(cell))}
  for split in ['val','test']:
   n=len(d.datasets[split]);b=n%32 or 32;start=n-b;x,y,starts=d.query(split,start,n)
   for m in models:
    m.precision();reference=m.probability(split,start,n)
    got=adapters[m.name].probability(split,start,n) if m.name in adapters else m.probability(split,start,n)
    pick=lambda t:a.rt.stable_topk_indices(t.reshape(b*d.c,-1),10,largest=True).reshape(b,d.c,10)
    ids=pick(got);expected=pick(reference);pv=(d.values_all(ids)+x[:,:,-1,None,None]).mean(2)
    saved=torch.as_tensor(m.reference(split)[0][start:n].transpose(0,2,1),device=d.device)
    pred_ok=torch.allclose(pv,saved,atol=1e-5,rtol=1e-5)
    row=dict(cell=cell,model=m.name,split=split,queries=b,channels=d.c,full_candidates=len(d.starts),ordered_mismatches=int((ids!=expected).any(-1).sum()),max_prediction_diff=float((pv-saved).abs().max()),stored_prediction_tolerance_pass=pred_ok,all_channels_gpu_batch=True,score_cache=m.name not in adapters)
    records.append(row);print(json.dumps(row),flush=True);assert row['ordered_mismatches']==0 and pred_ok,row
   if not models:
    # Solar Full-trained models are still MISSING. Check the actual data-only
    # path with all 137 channels; no fake model-parity PASS is emitted.
    for k in sorted(set([0,max(0,len(d.starts)-128)])):
     ids=torch.arange(k,min(k+128,len(d.starts)),device=d.device).expand(b,d.c,-1)
     values=d.values_all(ids);got=distance_all(values,x[:,:,-1],y)
     ref=torch.stack([pooled_future_mse(values[:,c].contiguous(),x[:,c,-1],y[:,c].contiguous()) for c in range(d.c)],1)
     assert torch.equal(got,ref),(cell,split,k,'data-only exact distance')
     row=dict(cell=cell,split=split,queries=b,channels=d.c,full_candidates=len(d.starts),candidate_chunk_start=k,candidate_chunk=ids.shape[-1],value_tensor_shape=list(values.shape),distance_bitwise_equal=True,all_channels_gpu_batch=True,learned_model_parity='MISSING_CHECKPOINT_NOT_TESTED',scope='real first/last candidate chunk smoke; complete Full bank retained for production')
     records.append(row);print(json.dumps(row),flush=True)
  del adapters,models,d
  torch.cuda.empty_cache()
  a.rt.write_json(OUT/'REMAINING_TAIL_PREFLIGHT.json',dict(status='IN_PROGRESS',rows=records))
 a.rt.write_json(OUT/'REMAINING_TAIL_PREFLIGHT.json',dict(status='PASS_WITH_SOLAR_MODEL_PARITY_NOT_APPLICABLE_MISSING_CHECKPOINT',rows=records))
if __name__=='__main__':main()
