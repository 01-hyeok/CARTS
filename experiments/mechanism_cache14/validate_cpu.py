"""Read-only completed-block replay and CPU-only evaluator equivalence.

Writes only to the explicit new audit directory. Never starts a GPU context.
"""
import argparse
import copy
import importlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
import numpy as np
from cache_store import ExactBlockCache, array_info, atomic_json, digest, exact_equal, file_sha, scope
from oracle_reuse import build_cached_evaluate, validate_oracle_inputs


def main(repo, out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU audit must hide CUDA'
    repo, out = Path(repo), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base=repo/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02'
    store=ExactBlockCache(out/'compact_cache')
    records=[]
    for cell, batch in [('ETTh1_96',0),('ETTh1_96',87),('Weather_96',0)]:
        values_by_model={}
        for model in ['V0_P100','V1_P100','V5_P100','V0_FULL','V1_FULL','V5_FULL']:
            p=base/'checkpoint_runs'/cell/model/'test'/('batch%05d.npz'%batch)
            side=json.loads(p.with_suffix('.json').read_text())
            assert side['status']=='PASS' and file_sha(p)==side['sha256']
            start=time.perf_counter()
            with np.load(p,allow_pickle=False) as z:
                names=['query_start','pool_ids']
                for support in ['FULL','P100_CONTROLLED']:
                    names += [support+'__'+k for k in ['oracle100_ids','oracle100_mse']]
                    for size in [10,30]:
                        for group in ['oracle','random','past_nearest']:
                            names += [support+'__geometry__%d_%s_'%(size,group)+k for k in ['ids','future_dispersion']]
                values={k:z[k].copy() for k in names}
                # Only the unconditioned random group is model independent.
                # Missed/false/correct Oracle sums are explicitly excluded.
                for support in ['FULL','P100_CONTROLLED']:
                    for suffix in ['sum','count']:
                        values[support+'__random_feature_'+suffix]=z[support+'__group_feature_'+suffix][:,:,3,:].copy()
            read_seconds=time.perf_counter()-start
            values_by_model[model]=values
            info=side['identity'];data=info['data'];b,c=values['pool_ids'].shape[:2]
            # Completed payload replay is NOT a fresh GPU numerical proof.
            # Source-specific scope prevents unproven cross-runtime reuse.
            proof=scope(cell=cell,split='test',query_ids_sha256=array_info(values['query_start'])['sha256'],
                channel_ids_sha256=digest(data['channel_names']),candidate_ids_sha256=digest([0,data['candidate_count']]),
                pool_ids_sha256=array_info(values['pool_ids'])['sha256'],mask_sha256=digest({'source':side['sha256'],'scope':'completed block validated mask'}),
                data_sha256=digest(data),normalization_sha256=digest([data['scaler_mean'],data['scaler_scale']]),
                code_sha256=digest(info['source_hashes']),runtime={'mode':'immutable saved payload CPU replay','numpy':np.__version__},
                seed=0,batch_shape=[b,c,data['candidate_count']],operation='compact_completed_block')
            t=time.perf_counter();state=store.put(proof,values);write=time.perf_counter()-t
            times=[]
            for _ in range(3):
                t=time.perf_counter();got=store.load(proof,expected=values);times.append(time.perf_counter()-t)
                assert got is not None and all(exact_equal(values[k],got[k]) for k in values)
            records.append(dict(cell=cell,batch=batch,model=model,source=str(p),source_sha256=side['sha256'],
                queries=b,channels=c,array_count=len(values),bytes=sum(a.nbytes for a in values.values()),
                source_file_bytes=p.stat().st_size,source_extract_seconds=read_seconds,cache_write_seconds=write,
                cache_load_exact_compare_median_seconds=statistics.median(times),state=state,bit_exact=True))
        reference=values_by_model['V0_P100'];cross={}
        for model,vals in values_by_model.items():
            differences=[k for k in reference if not exact_equal(reference[k],vals[k])]
            cross[model]=dict(all_exact=not differences,different_arrays=differences)
        atomic_json(out/('%s_batch%05d_cross_model.json'%(cell,batch)),cross)
    atomic_json(out/'COMPLETED_BLOCK_REPLAY.json',dict(status='PASS',records=records,
        limitation='Saved payload replay only; dense future-MSE/full score and new GPU cache path remain unvalidated.',
        production_activated=False))

    # Load the actual evaluator and actual canonical stable_topk implementation.
    sys.path.insert(0,str(repo/'experiments/retrieval_failure_mechanism02'))
    import torch
    original=importlib.import_module('gpu_metrics')
    optimized=build_cached_evaluate(original)
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    results=[]
    for b in [1,12,32]:
        for tied in [False,True]:
            torch.manual_seed(0);c,n,h=21,128,16
            x=torch.randn(b,c,h);y=torch.randn(b,c,h);d=torch.rand(b,c,n)
            if tied:d[:,:,:12]=.125
            score=torch.randn(b,c,n);mask=torch.ones(b,c,n,dtype=torch.bool);mask[:,:,120:]=False
            ids=torch.arange(n).expand(b,c,n)
            bank=torch.randn(c,n,h)
            class Data:
                def values_all(self,selected):
                    return bank[torch.arange(c)[None,:,None],selected]
            data=Data();data.h=h
            oracle=original.topk(-d.masked_fill(~mask,float('inf')),100)
            validate_oracle_inputs(d,mask,ids,oracle)
            reference=original.evaluate(data,x,y,d,score,mask,ids)
            result=optimized(data,x,y,d,score,mask,ids,cached_oracle=oracle)
            assert reference.keys()==result.keys()
            for k in reference:
                assert exact_equal(reference[k].numpy(),result[k].numpy()),(b,tied,k)
            fallback=optimized(data,x,y,d,score,mask,ids)
            assert all(exact_equal(reference[k].numpy(),fallback[k].numpy()) for k in reference)
            results.append(dict(batch=b,channels=c,candidates=n,ties=tied,fields=len(result),bit_exact=True))
    assert not torch.cuda.is_initialized()
    atomic_json(out/'CPU_EVALUATOR_GATE.json',dict(status='PASS',torch=torch.__version__,
        gpu_context_created=False,cases=results,source_sha256=file_sha(repo/'experiments/retrieval_failure_mechanism02/gpu_metrics.py'),
        gpu_equivalence='PENDING_SAFE_BOUNDARY',production_activated=False))
    print(json.dumps(dict(replay_blocks=len(records),replay_exact=all(x['bit_exact'] for x in records),
                          evaluator_cases=len(results),gpu_context_created=False,out=str(out))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();main(a.repo,a.out)
