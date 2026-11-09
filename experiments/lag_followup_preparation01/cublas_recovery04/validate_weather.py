"""Validate every approved sampled query before any recovery output is written."""
import sys,time,json,traceback,fcntl
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from weather_patch import prior,bridge,torch,OUT,ROOT,install

@torch.no_grad()
def main():
    lock=(OUT.parent.parent/'00_Audit/d4_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=bridge.start_gpu();install();d=bridge.Data('Weather_336');records=[];begin=time.perf_counter()
    for model in ['V0','V1']:
        row=next(r for r in bridge.inventory(d.cell) if r['architecture']==model and r['training_support']=='P100')
        f=bridge.Frozen(d,row);f.precision();f.batch.build_keys();keys=prior.canonical_keys(f)
        assert torch.equal(keys,f.batch.keys),'key mismatch'
        for split in ['val','test']:
            indices=bridge.sample_indices(len(d.datasets[split]),32)
            for step,i in enumerate(indices):
                i=int(i);a=i//32*32;b=min(a+32,len(d.datasets[split]));j=i-a
                got,mask=f.full_logits(split,a,b);q=prior.canonical_query(f,split,a,b);ref=prior.reference_scores(f,q,keys)
                rm=d.mask(d.datasets[split].starts[a:b])[:,None].expand(b-a,d.c,-1)
                assert torch.equal(mask,rm),'mask changed'
                assert torch.equal(got,ref),dict(model=model,split=split,i=i,batch=b-a,max_delta=float((got-ref).abs().max()))
                gi=bridge.topk(got[j,:,0].masked_fill(~mask[j],-float('inf')),500)
                ri=bridge.topk(ref[j,:,0].masked_fill(~rm[j],-float('inf')),500)
                assert torch.equal(gi,ri) and bool(mask[j].gather(-1,gi).all())
                assert bool((gi.sort(-1).values[:,1:]!=gi.sort(-1).values[:,:-1]).all())
                records.append(dict(model=model,split=split,query_index=i,original_batch=b-a,channels=21,bank_count=36216,scores_bitwise_equal=True,ordered_top500_equal=True))
                bridge.dump(OUT/'VALIDATION_PROGRESS.json',dict(at=time.time(),records=records))
                if step%8==0 or step==31:print(model,split,step+1,32,'EXACT',flush=True)
                del got,ref,q,mask,rm,gi,ri
        assert bridge.sha(row['checkpoint'])==row['checkpoint_sha256'];del f,keys;torch.cuda.empty_cache()
    bridge.dump(OUT/'WEATHER_VALIDATION_GATE.json',dict(status='PASS_ALL_APPROVED_SAMPLE_QUERIES_EXACT',records=records,
        cases=2,queries_per_split=32,channels=21,full_bank_count=36216,all_channels_single_batch=True,production_activated=False,
        seconds=time.perf_counter()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        gpu_before=before,gpu_after=bridge.snapshot(),source_hashes={p.name:bridge.sha(p) for p in HERE.glob('*.py')}))

if __name__=='__main__':
    try:main()
    except BaseException as e:bridge.dump(OUT/'VALIDATION_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),production_activated=False));raise
