"""One GPU gate at a time. Never activates an unverified runtime."""
import os,sys,json,time,argparse,traceback
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS')
sys.path[:0]=[str(ROOT/'experiments/retrieval_full_precision_repair02'),str(ROOT)]
import full_precision_runtime
import diagnostic_adapter as a
from all_channel_ops import AllChannelFull,distance_all
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--cell',default='Weather_96');p.add_argument('--model',default='V5');p.add_argument('--mode',choices=['flat','grouped'],default='flat');p.add_argument('--chunk',type=int,default=1024);x=p.parse_args()
    out=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/ALL_CHANNEL03';out.mkdir(exist_ok=True)
    pause=json.loads((out/'PAUSE.json').read_text())
    for pid in [pause['controller_pid'],pause['worker_pid']]:
        assert Path('/proc/%d/stat'%pid).read_text().split(') ')[1].split()[0]=='T','Original job must remain paused during GPU gate'
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.GPU
    a.rt.gpu_guard();d=a.Data(x.cell)
    row=next(r for r in a.inventory(x.cell,'FULL') if r['architecture']==x.model and r['status']=='VALID')
    f=a.Frozen(d,row,'FULL');f.precision()
    def measure(fn):
        torch.cuda.synchronize();t=time.perf_counter();value=fn();torch.cuda.synchronize();return value,time.perf_counter()-t
    ref,refcold=measure(lambda:f.probability('val',0,32))
    _,refwarm=measure(lambda:f.probability('val',0,32))
    torch.cuda.reset_peak_memory_stats()
    fast=AllChannelFull(f,x.mode,x.chunk)
    got,cold=measure(lambda:fast.probability('val',0,32))
    _,warm=measure(lambda:fast.probability('val',0,32))
    picks=a.rt.stable_topk_indices(got.reshape(-1,got.shape[-1]),10,largest=True).reshape(32,d.c,10)
    expected=a.rt.stable_topk_indices(ref.reshape(-1,ref.shape[-1]),10,largest=True).reshape_as(picks)
    mismatch=(picks!=expected).any(-1);element=(picks!=expected).sum().item()
    xq,yq,starts=d.query('val',0,32)
    val=a.Data.values_all(d,picks);oldval=a.Data.values_all(d,expected)
    pred=(val+xq[:,:,-1,None,None]).mean(2);oldpred=(oldval+xq[:,:,-1,None,None]).mean(2)
    original=f.reference('val')[0][:32].transpose(0,2,1)
    ids=torch.arange(min(128,len(d.starts)),device=d.device).expand(4,d.c,-1)
    values=d.values_all(ids);newdist=distance_all(values,xq[:4,:,-1],yq[:4])
    olddist=torch.stack([a.rt.pooled_future_mse(values[:,c].contiguous(),xq[:4,c,-1].contiguous(),yq[:4,c].contiguous()) for c in range(d.c)],1)
    score_diff=(got-ref).abs();finite=torch.isfinite(score_diff)
    sorted_ref=torch.sort(ref,dim=-1,descending=True).values
    rows=mismatch.nonzero().cpu().tolist()
    report=dict(status='PASS_BATCH_ONLY' if not element and torch.equal(newdist,olddist) else 'FAIL',cell=x.cell,model=x.model,mode=x.mode,chunk=x.chunk,torch=torch.__version__,precision=full_precision_runtime.precision_policy('FULL',x.model,x.cell),queries=32,channels=d.c,candidates=len(d.starts),candidate_reduction=False,ordered_top10_mismatch_query_channels=int(mismatch.sum()),ordered_top10_mismatch_elements=element,mismatch_examples=[dict(query=q,channel=c,canonical=expected[q,c].cpu().tolist(),batched=picks[q,c].cpu().tolist(),score_gap10_11=float(sorted_ref[q,c,9]-sorted_ref[q,c,10])) for q,c in rows[:5]],max_score_diff=float(score_diff[finite].max()),max_prediction_difference_vs_sequential=float((pred-oldpred).abs().max()),max_prediction_difference_vs_stored=float(abs(pred.cpu().numpy()-original).max()),distance_bit_exact=torch.equal(newdist,olddist),distance_max_difference=float((newdist-olddist).abs().max()),reference_cold_seconds=refcold,reference_warm_seconds=refwarm,batched_cold_seconds=cold,batched_warm_seconds=warm,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),shapes=fast.shapes[:2]+fast.shapes[-4:],encoder=str(f.model.encoder),head=str(f.heads),checkpoint_sha256=row['checkpoint_sha256'],full_validation_gate='NOT_RUN',original_generation_runtime='UNKNOWN',sequential_fallback=False)
    a.rt.write_json(out/('%s_%s_%s_chunk%d.json'%(x.cell,x.model,x.mode,x.chunk)),report)
    print(json.dumps(report,default=str),flush=True)
    if report['status']=='FAIL':sys.exit(2)

if __name__=='__main__':main()
