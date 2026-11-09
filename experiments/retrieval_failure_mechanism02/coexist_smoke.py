"""One short actual GPU coexistence/reproduction gate, never a training run."""
import argparse,os,time,traceback
import numpy as np
import torch
from worker import guard,Data,inventory,Frozen,native_batch,distances,OUT,ROOT,identity,canonical_p100_metrics
from io_utils import dump,now,asnumpy
import resources

def main(cell,model,support):
    os.chdir(ROOT)
    start=guard();torch.set_num_threads(1);data=Data(cell)
    row=next(r for r in inventory(cell) if r['architecture']==model and r['training_support']==support)
    f=Frozen(data,row);torch.cuda.reset_peak_memory_stats();rows=[];begin=time.perf_counter()
    for split in ('val','test'):
        n=len(data.datasets[split])
        for a in sorted({0,(n//32)*32 if n%32 else n-32}):
            b=min(a+32,n);x,y,q,d=distances(data,split,a,b);ev=native_batch(f,split,a,b,d,x,y)
            actual=asnumpy(ev['selected_ids']);ref=f.reference(split)
            if support=='P100':
                assert np.array_equal(actual,ref['ids'][a:b]),(split,a,'ordered native IDs mismatch')
                delta=float(abs(asnumpy(canonical_p100_metrics(ev))-ref['metrics'][a:b,:,:4]).max())
                assert delta<=2e-6,(split,a,delta)
            else:
                assert np.array_equal(actual,asnumpy(f.canonical_full_ids(split,a,b))),(split,a,'ordered canonical IDs mismatch')
                delta=float(abs(asnumpy(ev['prediction'])-ref['prediction'][a:b]).max())
                assert np.allclose(asnumpy(ev['prediction']),ref['prediction'][a:b],atol=1e-5,rtol=1e-5),(cell,model,split,a,'stored prediction mismatch',delta)
            rows.append(dict(split=split,start=a,end=b,query_shape=list(x.shape),channels_in_single_batch=data.c,
                             full_candidates=len(data.starts),p100_candidates=100,final_k=10,ordered_ids_exact=True,max_difference=delta))
    torch.cuda.synchronize();end=resources.during()
    gate=dict(status='PASS',at=now(),cell=cell,model=model,support=support,rows=rows,identity=identity(row,data),
              gpu_before=start,gpu_after=end,shared_gpu=True,wall_seconds=time.perf_counter()-begin,
              peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
              actual_pid=os.getpid(),full_population_gate='NOT_RUN_THIS_IS_FIRST_AND_TAIL_SMOKE_ONLY')
    dump(OUT/'00_Audit'/f'COEXIST_SMOKE_{cell}_{model}_{support}.json',gate)
    print(gate,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cell');p.add_argument('model');p.add_argument('support');a=p.parse_args()
    try:main(a.cell,a.model,a.support)
    except BaseException as e:
        stamp=now().replace(':','').replace('.','')
        dump(OUT/'00_Audit'/f'COEXIST_SMOKE_{a.cell}_{a.model}_{a.support}_FAILED_{stamp}.json',dict(status='FAIL',at=now(),error=repr(e),traceback=traceback.format_exc(),pid=os.getpid()))
        raise
