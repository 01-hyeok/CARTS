import sys,os,json,time,argparse,hashlib
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS')
sys.path[:0]=[str(ROOT/'experiments/retrieval_full_precision_repair02'),str(ROOT)]
import full_precision_runtime
import diagnostic_adapter as a
import torch
from packed_batch import PackedChannelFull as ConcurrentCanonicalFull

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cell',default='Weather_96');ap.add_argument('--model',default='V5');args=ap.parse_args()
    dest=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/ALL_CHANNEL03'
    pause=json.loads((dest/'PAUSE.json').read_text())
    for pid in [pause['worker_pid'],pause['controller_pid']]:assert Path('/proc/%d/stat'%pid).read_text().split(') ')[1].split()[0]=='T'
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.GPU
    a.rt.gpu_guard();d=a.Data(args.cell);row=next(r for r in a.inventory(d.cell,'FULL') if r['architecture']==args.model)
    f=a.Frozen(d,row,'FULL');fast=ConcurrentCanonicalFull(f);rows=[];torch.cuda.reset_peak_memory_stats()
    for split in ['val','test']:
        mismatch=0;set_mismatch=0;units=0;maxdiff=0.;maxpred=0.;cold=0.;refsec=0.;fastsec=0.;examples=[];h1=hashlib.sha256();h2=hashlib.sha256()
        for start in range(0,len(d.datasets[split]),32):
            end=min(start+32,len(d.datasets[split]));b=end-start
            torch.cuda.synchronize();t=time.perf_counter();ref=f.probability(split,start,end);torch.cuda.synchronize();refsec+=time.perf_counter()-t
            torch.cuda.synchronize();t=time.perf_counter();got=fast.probability(split,start,end);torch.cuda.synchronize();fastsec+=time.perf_counter()-t
            top=lambda v:a.rt.stable_topk_indices(v.reshape(-1,v.shape[-1]),10,largest=True).reshape(b,d.c,10)
            old=top(ref);new=top(got);bad=(old!=new).any(-1);mismatch+=int(bad.sum());units+=b*d.c
            set_mismatch+=int((old.sort(-1).values!=new.sort(-1).values).any(-1).sum())
            delta=(got-ref).abs();finite=torch.isfinite(delta);maxdiff=max(maxdiff,float(delta[finite].max()))
            x,y,_=d.query(split,start,end)
            oldpred=(d.values_all(old)+x[:,:,-1,None,None]).mean(2);newpred=(d.values_all(new)+x[:,:,-1,None,None]).mean(2)
            maxpred=max(maxpred,float((newpred-oldpred).abs().max()))
            h1.update(old.cpu().numpy().tobytes());h2.update(new.cpu().numpy().tobytes())
            if len(examples)<8:
                for q,c in bad.nonzero().cpu().tolist()[:8-len(examples)]:examples.append(dict(query=start+q,channel=c,canonical=old[q,c].cpu().tolist(),concurrent=new[q,c].cpu().tolist()))
            if start%512==0:print('PROGRESS',args.cell,split,end,'/',len(d.datasets[split]),'ordered_mismatch',mismatch,'set_mismatch',set_mismatch,flush=True)
            a.rt.write_json(dest/'PACKED_LT_GATE_CURRENT.json',dict(pid=os.getpid(),cell=args.cell,model=args.model,split=split,completed_queries=end,total_queries=len(d.datasets[split]),ordered_mismatch_units=mismatch,set_mismatch_units=set_mismatch))
        rows.append(dict(split=split,queries=len(d.datasets[split]),channels=d.c,units=units,ordered_mismatch_units=mismatch,set_mismatch_units=set_mismatch,max_probability_diff=maxdiff,max_prediction_diff=maxpred,canonical_seconds=refsec,concurrent_seconds=fastsec,canonical_id_sha256=h1.hexdigest(),concurrent_id_sha256=h2.hexdigest(),examples=examples,status='PASS' if mismatch==0 else 'FAIL'))
        a.rt.write_json(dest/(args.cell+'_'+args.model+'_PACKED_LT_FULL_GATE.json'),dict(status='FAIL' if mismatch else 'PARTIAL_VALIDATION_PASS',rows=rows,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),sequential_fallback=False))
        if mismatch:
            print('FAILED_CANONICAL_ORDERED_ID_GATE',json.dumps(rows[-1]),flush=True);raise SystemExit(2)
    report=dict(status='PASS',rows=rows,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),sequential_fallback=False,checkpoint_sha256=row['checkpoint_sha256'],shapes=fast.shapes[:2],original_generation_runtime='UNKNOWN')
    a.rt.write_json(dest/(args.cell+'_'+args.model+'_PACKED_LT_FULL_GATE.json'),report);print('FULL_GATE_PASS',json.dumps(report),flush=True)

if __name__=='__main__':main()
