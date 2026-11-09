"""Actual cell/model/split tail and full-batch checks before full split gates."""
import os,sys,json,fcntl,subprocess,time
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments/retrieval_full_precision_repair02'),str(ROOT)]
import full_precision_runtime as canonical
import diagnostic_adapter as a
import torch
from packed_batch import PackedChannelFull
OUT=a.OUT/'00_Audit/TAIL08'


@torch.no_grad()
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.GPU
    assert a.GPU not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    a.rt.gpu_guard();d=a.Data('Weather_720');torch2=int(torch.__version__.split('.')[0])>=2
    arch='V0' if torch2 else 'V5';dest=OUT/('SMOKE_'+arch+'.json');assert not dest.exists()
    row=next(r for r in a.inventory(d.cell,'FULL') if r['architecture']==arch and r['status']=='VALID')
    frozen=a.Frozen(d,row,'FULL');fast=PackedChannelFull(frozen);rows=[];begin=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for split,start,end in [('val',0,32),('val',4544,4551),('test',9792,9820)]:
        got=fast.probability(split,start,end);ref=canonical.original_probability(frozen,split,start,end)
        pick=lambda v:a.rt.stable_topk_indices(v.reshape(-1,len(d.starts)),10,largest=True).reshape(end-start,d.c,10)
        actual,expected=pick(got),pick(ref);finite=torch.isfinite(got)&torch.isfinite(ref)
        delta=float((got[finite]-ref[finite]).abs().max());mismatch=int((actual!=expected).any(-1).sum())
        assert mismatch==0,(arch,split,start,mismatch,delta)
        x,_,_=d.query(split,start,end)
        pred=(d.values_all(actual)+x[:,:,-1,None,None]).reshape((end-start)*d.c,10,d.h).mean(1).reshape(end-start,d.c,d.h)
        saved=torch.as_tensor(frozen.reference(split)[0][start:end].transpose(0,2,1),device=d.device)
        assert torch.allclose(pred,saved,atol=1e-5,rtol=1e-5)
        r=dict(model=arch,split=split,start=start,end=end,queries=end-start,channels=d.c,candidates=len(d.starts),
               score_bitwise_equal=torch.equal(got,ref),score_maxdiff=delta,ordered_mismatches=mismatch,
               max_saved_prediction_diff=float((pred-saved).abs().max()),torch=torch.__version__,tf32=torch.backends.cuda.matmul.allow_tf32)
        rows.append(r);print(json.dumps(r),flush=True)
    canonical.old_immutable(frozen)
    a.rt.write_json(dest,dict(status='PASS_SMOKE_NOT_FULL_RELEASE',rows=rows,seconds=time.perf_counter()-begin,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        checkpoint_sha256=row['checkpoint_sha256'],source_hashes={p.name:a.rt.sha256_file(p) for p in HERE.iterdir() if p.suffix in ['.py','.so','.cu']},
        all_channels_one_batch=True,production_serial_fallback=False,tolerance_changed=False,original_generation_runtime='UNKNOWN'))


if __name__=='__main__':main()
