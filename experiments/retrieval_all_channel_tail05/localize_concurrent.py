import sys,os,json
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS')
sys.path[:0]=[str(ROOT/'experiments/retrieval_full_precision_repair02'),str(ROOT)]
import full_precision_runtime
import diagnostic_adapter as a
import torch
import torch.nn.functional as F
from concurrent_canonical import ConcurrentCanonicalFull
from all_channel_ops import score_all,global_mixture

@torch.no_grad()
def main():
    dest=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/ALL_CHANNEL03'
    pause=json.loads((dest/'PAUSE.json').read_text())
    for pid in [pause['worker_pid'],pause['controller_pid']]:assert Path('/proc/%d/stat'%pid).read_text().split(') ')[1].split()[0]=='T'
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.GPU
    a.rt.gpu_guard();d=a.Data('Weather_96');row=next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V5');f=a.Frozen(d,row,'FULL');f.precision();f.probability('val',0,32)
    fast=ConcurrentCanonicalFull(f);fast.build_keys();canonical_keys=torch.stack([f.keys[c] for c in range(d.c)])
    keydiff=float((fast.keys-canonical_keys).abs().max());rows=[]
    for start in [0,512]:
        ref=f.probability('val',start,start+32);got=fast.probability('val',start,start+32);torch.cuda.synchronize()
        x,_,starts=d.query('val',start,start+32)
        canonical_q=torch.stack([f.heads(a.encode_raw_channel_first(f.model,x[:,c,:,None].contiguous(),0)) for c in range(d.c)],1)
        actual_q=fast.graphs[32][2].clone();mask=d.mask(starts)[:,None,:].expand(32,d.c,-1)
        current=torch.cuda.current_stream();streams=[torch.cuda.Stream() for _ in range(d.c)];outputs=[]
        for c,stream in enumerate(streams):
            stream.wait_stream(current)
            with torch.cuda.stream(stream):outputs.append(f.heads(a.encode_raw_channel_first(f.model,x[:,c,:,None].contiguous(),0)))
        for stream in streams:current.wait_stream(stream)
        noncaptured_q=torch.stack(outputs,1)
        probs={'graph_queries_canonical_keys':global_mixture(score_all(actual_q,canonical_keys),mask),
               'canonical_queries_concurrent_keys':global_mixture(score_all(canonical_q,fast.keys),mask),
               'noncaptured_queries_canonical_keys':global_mixture(score_all(noncaptured_q,canonical_keys),mask)}
        top=lambda v:a.rt.stable_topk_indices(v.reshape(-1,v.shape[-1]),10,largest=True)
        rows.append(dict(start=start,graph_query_maxdiff=float((actual_q-canonical_q).abs().max()),noncaptured_query_maxdiff=float((noncaptured_q-canonical_q).abs().max()),
                    variants=[dict(name=n,ordered_mismatch_units=int((top(p)!=top(ref)).any(-1).sum()),maxprob=float((p-ref).abs()[torch.isfinite(p-ref)].max())) for n,p in probs.items()]))
    r=dict(key_maxdiff=keydiff,rows=rows);a.rt.write_json(dest/'CONCURRENT_COMPONENT_LOCALIZATION.json',r);print(json.dumps(r))

if __name__=='__main__':main()
