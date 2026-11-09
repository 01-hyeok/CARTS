"""Validation-only numerical reconstruction audit; no metric optimization."""
import sys,json,gc
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS')
sys.path.insert(0,str(ROOT/'experiments/retrieval_failure_attribution01'))
import diagnostic_adapter as ad
import torch,numpy as np
from scripts.build_t2_true_original_kl_cache01 import compute_scores_true_original_kl
from scripts.train_t_pure_multislot01 import round_robin_topk_selection
@torch.no_grad()
def main():
 ad.rt.gpu_guard();torch.set_num_threads(2);out=[]
 for cell in sys.argv[1:]:
  data=ad.rt.CellData(cell,'cuda:0');row=next(r for r in ad.inventory(cell,'FULL') if r['architecture']=='V0' and r['status']=='VALID');m=ad.Frozen(data,row,'FULL')
  expected=m.reference('val')[0][:32]
  for tf32 in [True,False]:
   torch.backends.cuda.matmul.allow_tf32=tf32;torch.backends.cudnn.allow_tf32=True;maxdiff=0.;mismatch=0
   for c in range(data.c):
    x,y,starts=data.channel_batch('val',0,32,c);mask=data.mask(starts)
    score=compute_scores_true_original_kl(m.model,x,data.bank_history(c),0);ids=round_robin_topk_selection(score,mask,10)
    values=data.values(torch.arange(len(data.starts),device=data.device),c);pred=(values[ids]+x[:,-1,0,None,None]).mean(1).cpu().numpy();ref=expected[:,:,c]
    maxdiff=max(maxdiff,float(np.abs(pred-ref).max()));mismatch+=int((~np.isclose(pred,ref,atol=1e-5,rtol=1e-5)).any(-1).sum())
   r=dict(cell=cell,torch=torch.__version__,tf32=tf32,queries=32,channels=data.c,candidates=len(data.starts),max_prediction_diff=maxdiff,mismatched_query_channels=mismatch,checkpoint_sha256=row['checkpoint_sha256'],source='canonical V0 scorer, validation rows0:32, no future score input',provisional_requires_full_val_test_gate=True)
   out.append(r);print(json.dumps(r),flush=True)
  del m,data;gc.collect();torch.cuda.empty_cache()
 dest=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/FULL_REPAIR02';dest.mkdir(parents=True,exist_ok=True)
 ad.rt.write_json(dest/('V0_validation_runtime_audit_'+torch.__version__.split('+')[0]+'.json'),out)
if __name__=='__main__':main()
