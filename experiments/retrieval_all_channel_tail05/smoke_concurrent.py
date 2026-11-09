import sys,os,json,time,argparse
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS')
sys.path[:0]=[str(ROOT/'experiments/retrieval_full_precision_repair02'),str(ROOT)]
import full_precision_runtime
import diagnostic_adapter as a
import torch
from concurrent_canonical import ConcurrentCanonicalFull

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--cell',default='Weather_96');p.add_argument('--model',default='V5');args=p.parse_args()
    dest=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/ALL_CHANNEL03'
    pause=json.loads((dest/'PAUSE.json').read_text())
    for pid in [pause['worker_pid'],pause['controller_pid']]:assert Path('/proc/%d/stat'%pid).read_text().split(') ')[1].split()[0]=='T'
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.GPU
    a.rt.gpu_guard();d=a.Data(args.cell);row=next(r for r in a.inventory(d.cell,'FULL') if r['architecture']==args.model);f=a.Frozen(d,row,'FULL');f.precision()
    def measure(fn):
        torch.cuda.synchronize();t=time.perf_counter();v=fn();torch.cuda.synchronize();return v,time.perf_counter()-t
    ref,_=measure(lambda:f.probability('val',0,32));_,refsec=measure(lambda:f.probability('val',0,32))
    torch.cuda.reset_peak_memory_stats();fast=ConcurrentCanonicalFull(f)
    got,cold=measure(lambda:fast.probability('val',0,32));_,sec=measure(lambda:fast.probability('val',0,32))
    top=lambda v:a.rt.stable_topk_indices(v.reshape(-1,v.shape[-1]),10,largest=True)
    equal=torch.equal(top(got),top(ref));diff=(got-ref).abs();finite=torch.isfinite(diff)
    r=dict(status='PASS_BATCH_ONLY' if equal else 'FAIL',cell=args.cell,model=args.model,
           implementation='one multi-stream CUDA graph replay over all channels; canonical per-channel GEMM shapes retained; batched scores/global softmax',
           queries=32,channels=d.c,candidates=len(d.starts),ordered_top10_mismatch_units=int((top(got)!=top(ref)).any(-1).sum()),
           max_probability_difference=float(diff[finite].max()),canonical_warm_seconds=refsec,concurrent_cold_seconds=cold,concurrent_warm_seconds=sec,
           peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
           shapes=fast.shapes,full_validation='NOT_RUN',original_generation_runtime='UNKNOWN',sequential_fallback=False)
    a.rt.write_json(dest/(args.cell+'_'+args.model+'_concurrent_smoke.json'),r);print(json.dumps(r),flush=True)
    if not equal:raise SystemExit(2)

if __name__=='__main__':main()
