import sys,os,json,time
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS')
sys.path[:0]=[str(ROOT/'experiments/retrieval_full_precision_repair02'),str(ROOT)]
import full_precision_runtime
import diagnostic_adapter as a
from all_channel_ops import encode_all,score_all,global_mixture
import torch
import torch.nn.functional as F

@torch.no_grad()
def main():
    dest=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/ALL_CHANNEL03'
    pause=json.loads((dest/'PAUSE.json').read_text())
    for pid in [pause['worker_pid'],pause['controller_pid']]:assert Path('/proc/%d/stat'%pid).read_text().split(') ')[1].split()[0]=='T'
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.GPU
    a.rt.gpu_guard();d=a.Data('Weather_96');row=next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V5');f=a.Frozen(d,row,'FULL');f.precision()
    ref=f.probability('val',0,32);x,_,starts=d.query('val',0,32)
    canonical_z=torch.stack([a.encode_raw_channel_first(f.model,x[:,c,:,None].contiguous(),0) for c in range(d.c)],1)
    canonical_q=torch.stack([f.heads(canonical_z[:,c]) for c in range(d.c)],1)
    canonical_keys=torch.stack([f.keys[c] for c in range(d.c)])
    mask=d.mask(starts)[:,None,:].expand(32,d.c,-1)
    expected=a.rt.stable_topk_indices(ref.reshape(-1,ref.shape[-1]),10,largest=True)
    rows=[]
    def measure(name,prob):
        ids=a.rt.stable_topk_indices(prob.reshape(-1,prob.shape[-1]),10,largest=True)
        diff=(prob-ref).abs();finite=torch.isfinite(diff)
        rows.append(dict(name=name,ordered_mismatch_units=int((ids!=expected).any(-1).sum()),max_probability_diff=float(diff[finite].max())))
    # Exact embeddings and heads isolate the batched score and softmax kernels.
    canonical_sc=torch.stack([torch.einsum('bsd,nd->bsn',canonical_q[:,c],canonical_keys[c]) for c in range(d.c)],1)
    measure('canonical_scores_batched_softmax',global_mixture(canonical_sc,mask))
    measure('canonical_inputs_batched_score_and_softmax',global_mixture(score_all(canonical_q,canonical_keys),mask))
    z=encode_all(f.model,x,'grouped')
    qflat=f.heads(z.reshape(32*d.c,-1)).reshape(32,d.c,5,-1)
    weights=f.heads.W.reshape(5*128,128).t()[None].expand(d.c,-1,-1)
    qgroup=F.normalize(torch.bmm(z.permute(1,0,2).contiguous(),weights).reshape(d.c,32,5,128),dim=-1).permute(1,0,2,3).contiguous()
    for name,q in [('flat_heads',qflat),('grouped_heads',qgroup)]:measure(name,global_mixture(score_all(q,canonical_keys),mask))
    report=dict(rows=rows,query_embedding_maxdiff=float((z-canonical_z).abs().max()),flat_head_maxdiff=float((qflat-canonical_q).abs().max()),grouped_head_maxdiff=float((qgroup-canonical_q).abs().max()),actual_candidate_count=len(d.starts),channels=d.c)
    a.rt.write_json(dest/'COMPONENT_LOCALIZATION.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
