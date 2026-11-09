"""Actual first/tail Oracle-sharing comparison; no training or result overwrites."""
import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/data/pjh_workspace/CARTS')
sys.path.insert(0,str(ROOT/'experiments/retrieval_failure_mechanism02'))
import torch
import worker as canonical
import gpu_metrics
import resources
from cache_store import atomic_json,file_sha,exact_equal
from oracle_reuse import build_cached_evaluate,validate_oracle_inputs


def runtime_manifest():
    libraries={}
    for line in Path('/proc/self/maps').read_text().splitlines():
        path=line.split()[-1]
        if path.startswith('/') and any(x in path for x in ['libcublas','libcudart','libcuda.so']):
            p=Path(path)
            if p.is_file() and path not in libraries:libraries[path]=file_sha(p)
    deps={}
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and str(path).startswith(str(ROOT)) and str(path).endswith('.py'):
            p=Path(path)
            if p.is_file():deps[str(p)]=file_sha(p)
    return dict(torch_version=torch.__version__,cuda_runtime=torch.version.cuda,
        device_uuid=canonical.GPU_UUID if hasattr(canonical,'GPU_UUID') else os.environ['CUDA_VISIBLE_DEVICES'],
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,dtype='float32 inputs; original evaluate FP64 metrics',
        blas_libraries_sha256=libraries,external_dependencies_sha256=deps)


@torch.no_grad()
def run(cell,model,support,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    os.chdir(ROOT);before=canonical.guard();torch.set_num_threads(1)
    start=time.perf_counter();data=canonical.Data(cell)
    row=next(r for r in canonical.inventory(cell) if r['architecture']==model and r['training_support']==support)
    f=canonical.Frozen(data,row);proof=canonical.identity(row,data)
    optimized=build_cached_evaluate(gpu_metrics);records=[]
    atomic_json(out/'START.json',dict(status='RUNNING',pid=os.getpid(),gpu_before=before,identity=proof,
                                    started=canonical.now(),shared_gpu=True))
    for split in ['val','test']:
        total=len(data.datasets[split])
        for a in sorted({0,((total-1)//32)*32}):
            resources.during();b=min(a+32,total)
            x,y,q,d=canonical.distances(data,split,a,b);B,C,N=d.shape
            assert C==data.c and N==len(data.starts)
            native=canonical.native_batch(f,split,a,b,d,x,y)
            expected=f.reference(split)
            if support=='P100':
                assert np.array_equal(native['selected_ids'].cpu().numpy(),expected['ids'][a:b])
                delta=float(np.max(np.abs(canonical.canonical_p100_metrics(native).cpu().numpy()-expected['metrics'][a:b,:,:4])))
                assert delta<=2e-6
            else:
                assert torch.equal(native['selected_ids'],f.canonical_full_ids(split,a,b))
                delta=float(np.max(np.abs(native['prediction'].cpu().numpy()-expected['prediction'][a:b])))
                assert np.allclose(native['prediction'].cpu().numpy(),expected['prediction'][a:b],atol=1e-5,rtol=1e-5)
            del native
            logits,mask=f.full_logits(split,a,b);pool=data.pool_all(split,a,b)
            assert bool(mask.gather(-1,pool).all())
            poolmask=torch.ones_like(pool,dtype=torch.bool);dp=d.gather(-1,pool)
            ids=torch.arange(N,device=d.device).expand(B,C,N)
            ctrl=canonical.normalization_control(logits,mask,pool)
            pf=canonical.normalize(logits,mask,raw_single=f.n<=1)
            lp=logits.gather(-1,pool[:,:,None].expand(-1,-1,logits.shape[2],-1))
            pp=canonical.normalize(lp,poolmask,raw_single=f.n<=1)
            cases=[('FULL',pf,d,mask,ids),('POOL',pp,dp,poolmask,pool),
                   ('POOL',ctrl['local'],dp,poolmask,pool),('POOL',ctrl['restricted'],dp,poolmask,pool)]
            if support=='P100':
                ns,ni=f.native_p100_logits(split,a,b);assert torch.equal(ni,pool)
                cases.append(('POOL',canonical.normalize(ns,poolmask),dp,poolmask,pool))
            if f.n==5:
                for name,score,dist,m,gids,hs in [('FULL',pf,d,mask,ids,logits),('POOL',pp,dp,poolmask,pool,lp)]:
                    union=canonical.head_union(hs,score,m)['union_mask']
                    for h in range(5):cases.append((name,hs[:,:,h].masked_fill(~m,-float('inf')),dist,m,gids))
                    cases.append((name,score.masked_fill(~union,-float('inf')),dist,m,gids))
                    cases.append((name,-dist.masked_fill(~union,float('inf')),dist,m,gids))
                    for h in range(5):
                        score_loo=torch.softmax(hs.masked_fill(~m[:,:,None],-float('inf'))/.1,-1)[:,:,[j for j in range(5) if j!=h]].mean(2)
                        cases.append((name,score_loo,dist,m,gids))
            def oracles():
                return {'FULL':canonical.topk(-d.masked_fill(~mask,float('inf')),100),
                        'POOL':canonical.topk(-dp.masked_fill(~poolmask,float('inf')),100)}
            saved=oracles()
            validate_oracle_inputs(d,mask,ids,saved['FULL']);validate_oracle_inputs(dp,poolmask,pool,saved['POOL'])
            for name,score,dist,m,gids in cases:
                reference=gpu_metrics.evaluate(data,x,y,dist,score,m,gids)
                current=optimized(data,x,y,dist,score,m,gids,cached_oracle=saved[name])
                assert reference.keys()==current.keys()
                for key in reference:
                    assert exact_equal(reference[key].cpu().numpy(),current[key].cpu().numpy()),(split,a,name,key)
            del reference,current,saved
            times={'original':[],'shared_oracle':[]};peaks={'original':[],'shared_oracle':[]}
            # AB then BA mitigates (but cannot remove) Solar contention variation.
            for order in [('original','shared_oracle'),('shared_oracle','original')]:
                for mode in order:
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                    saved=oracles() if mode=='shared_oracle' else None
                    for name,score,dist,m,gids in cases:
                        result=(gpu_metrics.evaluate(data,x,y,dist,score,m,gids) if mode=='original' else
                                optimized(data,x,y,dist,score,m,gids,cached_oracle=saved[name]))
                    torch.cuda.synchronize();times[mode].append(time.perf_counter()-tick)
                    peaks[mode].append(dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()))
                    del result,saved
            rec=dict(split=split,start=a,end=b,query_shape=list(x.shape),channels=C,full_candidates=N,p100=100,k=10,
                     native_ordered_ids_exact=True,canonical_metric_max_diff=delta,oracle_bit_exact=True,
                     all_evaluation_fields_bit_exact=True,cases=len(cases),old_oracle_calls=len(cases),new_oracle_calls=2,
                     seconds=times,median_seconds={k:statistics.median(v) for k,v in times.items()},peak_bytes=peaks,
                     gpu_observation=resources.during(),encoder_shapes=f.identity().get('shapes'),at=canonical.now())
            records.append(rec);atomic_json(out/'PROGRESS.json',dict(status='RUNNING',records=records))
            print('CACHE14_GPU',cell,model,support,split,a,b,rec['median_seconds'],flush=True)
            del cases,logits,ctrl,lp,pf,pp,d,dp,x,y;gc.collect()
            if time.perf_counter()-start>240:
                raise RuntimeError('minimal benchmark budget exceeded; preserve results and resume original queue')
    old=sum(r['median_seconds']['original'] for r in records)
    new=sum(r['median_seconds']['shared_oracle'] for r in records)
    atomic_json(out/'GPU_ORACLE_GATE.json',dict(status='EXACT_PASS',at=canonical.now(),pid=os.getpid(),cell=cell,model=model,
        training_support=support,identity=proof,records=records,runtime=runtime_manifest(),
        new_code_sha256={p.name:file_sha(p) for p in HERE.glob('*.py')},wall_seconds=time.perf_counter()-start,
        measured_original_metric_seconds=old,measured_shared_metric_seconds=new,
        metric_phase_improvement_fraction=1-new/old,positive_measured_saving=new<old,
        end_to_end_speedup='UNMEASURED',production_activated=False,shared_gpu=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--model',required=True)
    p.add_argument('--support',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    try:run(a.cell,a.model,a.support,a.out)
    except BaseException as e:
        import traceback
        dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True)
        atomic_json(dest/'FAILED.json',dict(status='FAILED_NOT_APPLIED',at=canonical.now(),error=repr(e),traceback=traceback.format_exc()))
        raise
