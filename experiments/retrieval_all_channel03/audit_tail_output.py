import sys,os,json,argparse
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS')
sys.path[:0]=[str(ROOT/'experiments/retrieval_full_precision_repair02'),str(ROOT)]
import full_precision_runtime
import diagnostic_adapter as a
import torch
from packed_batch import PackedChannelFull
from all_channel_ops import score_all,global_mixture

@torch.no_grad()
def main():
    dest=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/ALL_CHANNEL03'
    pause=json.loads((dest/'PAUSE.json').read_text())
    for pid in [pause['worker_pid'],pause['controller_pid']]:assert Path('/proc/%d/stat'%pid).read_text().split(') ')[1].split()[0]=='T'
    ap=argparse.ArgumentParser();ap.add_argument('--cell',default='Weather_96');args=ap.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.GPU
    a.rt.gpu_guard();d=a.Data(args.cell);row=next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V5');f=a.Frozen(d,row,'FULL');f.precision();fast=PackedChannelFull(f);fast.build_keys();report=[]
    for split in ['val','test']:
        end=len(d.datasets[split]);start=end//32*32;b=end-start
        if not b:continue
        x,_,starts=d.query(split,start,end);qref=[]
        for c in range(d.c):
            xx,_,_=d.channel_batch(split,start,end,c,future=False);qref.append(f.heads(a.encode_raw_channel_first(f.model,xx,0)))
        qref=torch.stack(qref,1);ref=f.probability(split,start,end)
        top=lambda v:a.rt.stable_topk_indices(v.reshape(-1,v.shape[-1]),10,largest=True).reshape(b,d.c,10)
        ids=top(ref);mask=d.mask(starts)[:,None].expand(b,d.c,len(d.starts))
        for rows in [1024,2048,4096]:
            q=fast.encode(x,rows,project=True);got=global_mixture(score_all(q,fast.keys),mask);newids=top(got)
            diff=(got-ref).abs();diff=diff[torch.isfinite(diff)]
            r=dict(split=split,queries=b,channels=d.c,rows=rows,embedding_maxdiff=float((q-qref).abs().max()),embedding_different_units=int((q!=qref).reshape(b,d.c,-1).any(-1).sum()),ordered_mismatch=int((ids!=newids).any(-1).sum()),set_mismatch=int((ids.sort(-1).values!=newids.sort(-1).values).any(-1).sum()),max_probability_diff=float(diff.max()))
            report.append(r);print(json.dumps(r),flush=True)
    a.rt.write_json(dest/(args.cell+'_TAIL_OUTPUT_AUDIT.json'),dict(rows=report,future_used=False))
if __name__=='__main__':main()
