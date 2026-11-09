"""Full val/test canonical ordered IDs; no production results written."""
import sys,time,json,traceback,fcntl,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(1,str(HERE.parent))
from exact_patch import pop,install,OUT
import torch


@torch.no_grad()
def main():
    before=pop.start_gpu();patch=install();d=pop.Data('ETTh1_96')
    row=next(r for r in pop.inventory(d.cell) if r['architecture']=='V5' and r['training_support']=='FULL')
    f=pop.Frozen(d,row);f.precision();begin=time.perf_counter();counts={}
    for split in ['val','test']:
        count=0;N=len(d.datasets[split])
        for a in range(0,N,32):
            b=min(a+32,N);g,mask=f.full_logits(split,a,b);pool=d.pool_all(split,a,b)
            full=pop.topk(pop.normalize(g,mask),10)
            pg=g.gather(-1,pool[:,:,None].expand(-1,-1,5,-1));pm=torch.ones_like(pool,dtype=torch.bool)
            controlled=pool.gather(-1,pop.topk(pop.normalize(pg,pm),10))
            for support,ids in [('FULL',full),('P100',controlled)]:
                ref=pop.canonical_cross_ids(f,split,a,b,support)
                assert torch.equal(ids,ref),(split,a,b,support,int((ids!=ref).sum()))
                assert mask.gather(-1,ids).all()
                ordered=ids.sort(-1).values;assert (ordered[...,1:]!=ordered[...,:-1]).all()
            assert torch.equal(full,f.canonical_full_ids(split,a,b))
            count+=b-a
            pop.dump(OUT/'VALIDATION_PROGRESS.json',dict(split=split,queries=count,total=N,at=pop.now(),pid=os.getpid()))
            print('E96_FULL_VALIDATION',split,count,N,'C7 N8449 ordered mismatch0',flush=True)
        assert count==N;counts[split]=count
    pop.dump(OUT/'VALIDATION_GATE.json',dict(status='PASS_ALL_QUERY_NATIVE_AND_CONTROLLED_ORDERED_IDS',queries=counts,
        channels=d.c,bank_count=len(d.starts),checkpoint=row,checkpoint_unchanged=pop.sha(row['checkpoint'])==row['checkpoint_sha256'],
        patch=patch,all_channels_single_batch=True,source_sha256=pop.sha(__file__),
        seconds=time.perf_counter()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_before=before))


if __name__=='__main__':
    lock=(pop.BASE/'00_Audit/d4_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main()
    except BaseException as e:pop.dump(OUT/'VALIDATION_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
