"""Run once only after the bounded W192/W336 queue has exited.

Reference calls may be per-channel. Every candidate production backend under
test is one real all-channel GPU batch. No rollout or tolerance change here.
"""
import os, sys, json, time, hashlib, importlib.util
from pathlib import Path
ROOT = Path('/data/pjh_workspace/CARTS')
HERE = Path(__file__).resolve().parent
OLD = ROOT/'experiments/retrieval_all_channel03'
TAIL = ROOT/'experiments/retrieval_all_channel_tail05'
sys.path[:0] = [str(HERE), str(TAIL), str(ROOT/'experiments/retrieval_full_precision_repair02'), str(ROOT)]
import full_precision_runtime as canonical
import diagnostic_adapter as a
import torch
import torch.nn.functional as F
from score_backends import BorrowedCublas12, packed_all_pairs
OUT = ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/TAIL06'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adapter(name, folder):
    module = load(name, folder/'packed_batch.py')
    module.CublasLtBatch = load(name+'_lt', folder/'cublaslt_batch.py').CublasLtBatch
    ops = load(name+'_ops', folder/'all_channel_ops.py')
    module.score_all, module.global_mixture = ops.score_all, ops.global_mixture
    return module.PackedChannelFull


def compare(got, reference):
    assert got.shape == reference.shape
    c,b,n = got.shape
    pick = lambda x: a.rt.stable_topk_indices(x.reshape(c*b,n),10,largest=True).reshape(c,b,10)
    x,y = pick(got),pick(reference)
    finite = torch.isfinite(got) & torch.isfinite(reference)
    return dict(score_bitwise_equal=torch.equal(got,reference),
                score_maxdiff=float((got[finite]-reference[finite]).abs().max()),
                ordered_mismatch_units=int((x!=y).any(-1).sum()),
                set_mismatch_units=int((x.sort(-1).values!=y.sort(-1).values).any(-1).sum()),
                canonical_id_sha256=hashlib.sha256(y.cpu().numpy().tobytes()).hexdigest(),
                batched_id_sha256=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest())


@torch.no_grad()
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == a.GPU
    assert int(torch.__version__.split('.')[0]) >= 2
    assert not (OUT/'WEATHER720_DIAGNOSIS.json').exists(), 'Preserve existing attempt'
    a.rt.gpu_guard()
    d = a.Data('Weather_720')
    row = next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V0' and r['status']=='VALID')
    f = a.Frozen(d,row,'FULL'); f.precision()
    assert not torch.backends.cuda.matmul.allow_tf32
    assert d.c == 21 and d.h == 720 and len(d.starts) == 35448
    sources = [HERE/'score_backends.py',Path(__file__)]
    sources += [folder/name for folder in (OLD,TAIL) for name in ['packed_batch.py','cublaslt_batch.py','all_channel_ops.py']]
    provenance = dict(cell=d.cell,model='V0',torch=torch.__version__,tf32=False,seed=0,
                      checkpoint_sha256=row['checkpoint_sha256'],original_generation_runtime='UNKNOWN',
                      source_hashes={str(p):a.rt.sha256_file(p) for p in sources},
                      only_frozen_inference=True,production_activated=False,
                      old_gate_scope='V5 only; V0 not previously covered',
                      candidate_reduction=False,channel_reduction=False,tolerance_change=False)
    old = adapter('old03',OLD)(f); new = adapter('new05',TAIL)(f)
    old.build_keys();new.build_keys()
    blas = BorrowedCublas12()
    torch.cuda.reset_peak_memory_stats(); begin=time.perf_counter(); records=[]
    cases = [('val',0,32),('val',4544,4551),('test',9792,9820)]
    for split,start,end in cases:
        x,y,starts=d.query(split,start,end);b=end-start
        ref=canonical.original_probability(f,split,start,end).permute(1,0,2).contiguous()
        keys=torch.stack([f.keys[c] for c in range(d.c)])
        q=torch.stack([F.normalize(a.encode_raw_channel_first(f.model,x[:,c,:,None].contiguous(),0),dim=-1)
                       for c in range(d.c)])
        mask=d.mask(starts)[None].expand(d.c,b,len(d.starts))
        base=dict(split=split,start=start,end=end,queries=b,channels=d.c,candidates=len(d.starts),
                  query_tensor=list(q.shape),key_tensor=list(keys.shape),
                  old_key_maxdiff=float((old.keys-keys).abs().max()),
                  new_key_maxdiff=float((new.keys-keys).abs().max()))
        if b<d.c:
            base['old_query_maxdiff']=float((old.short_query(x).squeeze(2).permute(1,0,2)-q).abs().max())
            base['new_query_maxdiff']=float((new.short_query(x).squeeze(2).permute(1,0,2)-q).abs().max())
        methods=[('old03_end_to_end',lambda:old.probability(split,start,end).permute(1,0,2)),
                 ('tail05_end_to_end',lambda:new.probability(split,start,end).permute(1,0,2)),
                 ('same_q_key_torch_bmm',lambda:torch.bmm(q,keys.transpose(1,2))),
                 ('same_q_key_cublas12_strided',lambda:blas.score(q,keys)),
                 ('same_q_key_cublas12_pointer',lambda:blas.score(q,keys,pointer=True)),
                 ('same_q_key_cublas12_transposed_strided',lambda:blas.score(q,keys,transpose_output=True)),
                 ('same_q_key_cublas12_transposed_pointer',lambda:blas.score(q,keys,pointer=True,transpose_output=True)),
                 ('same_q_key_one_packed_gemm',lambda:packed_all_pairs(q,keys))]
        for name,fn in methods:
            torch.cuda.synchronize();t=time.perf_counter()
            try:
                got=fn().masked_fill(~mask,-float('inf'));torch.cuda.synchronize()
                result=dict(base,method=name,seconds=time.perf_counter()-t,**compare(got,ref))
                del got
            except RuntimeError as error:
                result=dict(base,method=name,status='BACKEND_FAILED',error=str(error))
                if 'out of memory' in str(error).lower():torch.cuda.empty_cache()
                else:raise
            records.append(result);print(json.dumps(result),flush=True)
            a.rt.write_json(OUT/'NUMERIC_CURRENT.json',dict(status='DIAGNOSTIC_ONLY',pid=os.getpid(),rows=records))
    f.immutable_check()
    assert provenance['source_hashes']=={str(p):a.rt.sha256_file(p) for p in sources}
    result=dict(status='DIAGNOSIS_COMPLETE_NOT_A_RELEASE_GATE',provenance=provenance,rows=records,
                seconds=time.perf_counter()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                full_weather720_release=False,solar_full_training_release=False,
                remaining='Fix both query encoder and score path, then complete model/runtime/split release matrix')
    a.rt.write_json(OUT/'WEATHER720_DIAGNOSIS.json',result)
    print('DIAGNOSIS_COMPLETE_NOT_RELEASED',flush=True)


if __name__=='__main__':main()
