"""Frozen V0 inference in its historical Torch2 runtime; no training or pruning."""
import os,sys,json,time,hashlib,subprocess
from pathlib import Path
import numpy as np
sys.path[:0]=[str(Path('/data/pjh_workspace/CARTS/experiments/retrieval_full_precision_repair02')),str(Path('/data/pjh_workspace/CARTS'))]
import full_precision_runtime as repair
from packed_batch import PackedChannelFull
from all_channel_ops import distance_all
from diagnostic_adapter import Data,Frozen,inventory,rt,OUT,GPU,ROOT
import torch
from scripts.train_factorial_e2e01 import individual_utility_memsafe
from scripts.train_patch_retrieval_expert01 import ndcg_at_k

@torch.no_grad()
def main(cell):
    assert int(torch.__version__.split('.')[0])>=2,'Original V0 cache requires Torch2'
    assert GPU not in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    assert json.loads((ROOT/'results/TRACK-RETRIEVAL-DIAGNOSTIC-SUPPLEMENT01/P100_DELIVERABLE_GATE.json').read_text())['status']=='PASS'
    rt.gpu_guard();data=Data(cell);rows=[r for r in inventory(cell,'FULL') if r['architecture']=='V0' and r['status']=='VALID']
    dest=OUT/'full_v0_canonical_scores'/cell;dest.mkdir(parents=True,exist_ok=True)
    if not rows:
        rt.write_json(dest/'MISSING.json',dict(status='MISSING_MODEL_NOT_IMPUTED',cell=cell));return
    m=Frozen(data,rows[0],'FULL');fast=PackedChannelFull(m)
    assert repair.uses_score_cache(cell), 'This cell uses native legacy inference; no bridge export'
    if (dest/'COMPLETE.json').exists():
        old=json.loads((dest/'COMPLETE.json').read_text());assert old['status']=='PASS'
        assert old['provenance']['checkpoint_sha256']==m.row['checkpoint_sha256']
        for rel,digest in old['file_hashes'].items():assert rt.sha256_file(dest/rel)==digest
        print('EXISTING IMMUTABLE V0 SCORE CACHE VERIFIED',cell,flush=True);return
    prov=dict(cell=cell,checkpoint_sha256=m.row['checkpoint_sha256'],torch=torch.__version__,precision=repair.precision_policy('FULL','V0',cell),data=data.data_info,
              all_channel_tensor_batch=True,channel_loops=False,original_generation_runtime='UNKNOWN',reference_only_channel_loop=True,
              code_hashes={p.name:rt.sha256_file(p) for p in Path(__file__).parent.iterdir() if p.suffix in ['.py','.cu','.so']})
    if (dest/'PROVENANCE.json').exists():assert json.loads((dest/'PROVENANCE.json').read_text())==prov,'Changed score-export provenance'
    else:rt.write_json(dest/'PROVENANCE.json',prov)
    if (dest/'COMPLETE.json').exists():
        old=json.loads((dest/'COMPLETE.json').read_text());assert old['provenance']==prov
        for rel,digest in old['file_hashes'].items():assert rt.sha256_file(dest/rel)==digest
        print('V0 SCORE VERIFIED RESUME',cell,flush=True);return
    begin=time.perf_counter();gates=[];file_hashes={}
    for split in ['val','test']:
        expected,canonical=m.reference(split);maxdiff=0.;sums=np.zeros(4);count=0;ids_hash=hashlib.sha256();canonical_ids_hash=hashlib.sha256();score_maxdiff=0.
        for a in range(0,len(data.datasets[split]),32):
            b=min(a+32,len(data.datasets[split]));p=dest/split/f'batch{a//32:05d}.npz';p.parent.mkdir(exist_ok=True)
            if p.exists():
                with np.load(p) as z:
                    assert np.array_equal(z['query_start'],data.datasets[split].starts[a:b])
                    scores=torch.as_tensor(z['scores'].copy(),device=data.device)
            else:scores=fast.probability(split,a,b)
            assert tuple(scores.shape)==(b-a,data.c,len(data.starts))
            B=b-a;C=data.c;N=len(data.starts)
            x,y,starts=data.query(split,a,b);mask=data.mask(starts)[:,None].expand(B,C,N)
            picks=rt.stable_topk_indices(scores.reshape(B*C,N),10,largest=True)
            # Full split, all models: independently call the actual original
            # runtime as a validation reference, never a production fallback.
            reference_scores=repair.original_probability(m,split,a,b)
            reference_ids=rt.stable_topk_indices(reference_scores.reshape(B*C,N),10,largest=True)
            assert torch.equal(picks,reference_ids),(cell,split,a,'ordered ID mismatch; production blocked')
            finite=torch.isfinite(scores)&torch.isfinite(reference_scores)
            score_maxdiff=max(score_maxdiff,float((scores[finite]-reference_scores[finite]).abs().max()))
            canonical_ids_hash.update(reference_ids.reshape(B,C,10).cpu().numpy().transpose(1,0,2).copy().tobytes())
            assert bool(mask.reshape(B*C,N).gather(1,picks).all())
            actual=picks.reshape(B,C,10)
            values=data.values_all(actual)+x[:,:,-1,None,None]
            pred=values.reshape(B*C,10,data.h).mean(1).reshape(B,C,data.h)
            new=pred.cpu().numpy();ref=expected[a:b].transpose(0,2,1)
            maxdiff=max(maxdiff,float(np.abs(new-ref).max()))
            assert np.allclose(new,ref,atol=1e-5,rtol=1e-5),(cell,split,a,'unchanged V0 canonical prediction gate',maxdiff)
            distances=torch.empty(B,C,N,device=data.device)
            for k in range(0,N,128):
                ids=torch.arange(k,min(k+128,N),device=data.device).expand(B,C,-1)
                distances[...,k:k+128]=distance_all(data.values_all(ids),x[:,:,-1],y)
            d=distances.reshape(B*C,N);valid=mask.reshape(B*C,N)
            oracle=rt.stable_topk_indices(d.masked_fill(~valid,float('inf')),10,largest=False)
            ret=d.gather(1,picks).mean(-1);agg=(pred-y).square().mean(-1).reshape(B*C)
            rec=(picks[:,:,None]==oracle[:,None,:]).any(-1).float().mean(-1)
            sums+=torch.stack([ret,agg,rec,ndcg_at_k(picks,d,valid,10)],-1).double().sum(0).cpu().numpy();count+=B*C
            ids_hash.update(actual.cpu().numpy().transpose(1,0,2).copy().tobytes())
            if not p.exists():
                tmp=p.with_suffix('.tmp.npz');np.savez_compressed(tmp,scores=scores.cpu().numpy(),query_start=data.datasets[split].starts[a:b]);os.replace(tmp,p)
            file_hashes[str(p.relative_to(dest))]=rt.sha256_file(p)
            rt.write_json(dest/'CURRENT.json',dict(status='RUNNING',cell=cell,split=split,completed_queries=b,total_queries=len(data.datasets[split]),channels=data.c,candidates=len(data.starts),pid=os.getpid()))
            if a%512==0:print('V0 SCORE EXPORT',cell,split,b,'/',len(data.datasets[split]),flush=True)
        assert count==len(data.datasets[split])*data.c
        got=sums/count
        if split=='test':
            for j,k in enumerate(['retmse10','agg_mse10','recall10','ndcg10']):assert np.isclose(got[j],canonical[k],atol=1e-5,rtol=1e-4),(k,got[j],canonical[k])
        assert canonical_ids_hash.hexdigest()==ids_hash.hexdigest()
        gates.append(dict(model='V0_FullTrained',split=split,queries=len(data.datasets[split]),channels=data.c,candidates=len(data.starts),units=count,metrics=got.tolist(),max_prediction_diff=maxdiff,ordered_id_sha256=ids_hash.hexdigest(),canonical_ordered_id_sha256=canonical_ids_hash.hexdigest(),ordered_mismatches=0,max_score_difference=score_maxdiff))
    rt.write_json(dest/'COMPLETE.json',dict(status='PASS',provenance=prov,splits=gates,file_hashes=file_hashes,seconds=time.perf_counter()-begin,stage1_training=False,stage2=False))
    print('V0 CANONICAL SCORE EXPORT PASS',cell,flush=True)
if __name__=='__main__':main(sys.argv[1])
