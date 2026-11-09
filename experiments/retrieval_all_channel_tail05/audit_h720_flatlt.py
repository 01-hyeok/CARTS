"""Numerical layer audit using past inputs only, no future/outcome tuning."""
import sys,os,json
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS')
sys.path[:0]=[str(ROOT/'experiments/retrieval_full_precision_repair02'),str(ROOT)]
import full_precision_runtime
import diagnostic_adapter as a
import torch
from cublaslt_batch import CublasLtBatch

@torch.no_grad()
def main():
    dest=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/ALL_CHANNEL03'
    pause=json.loads((dest/'PAUSE.json').read_text())
    for pid in [pause['worker_pid'],pause['controller_pid']]: assert Path('/proc/%d/stat'%pid).read_text().split(') ')[1].split()[0]=='T'
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.GPU
    a.rt.gpu_guard();d=a.Data('Weather_720');row=next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V5');f=a.Frozen(d,row,'FULL');f.precision()
    backend=CublasLtBatch();reports=[]
    for start,end in [(4544,4551)]:
        layers={n:m for n,m in f.model.encoder.named_modules() if isinstance(m,torch.nn.Linear)}
        storage={n:[] for n in layers};hooks=[]
        for n,m in layers.items():
            def hook(m,inputs,output,n=n): storage[n].append((inputs[0].clone(),output.clone()))
            hooks.append(m.register_forward_hook(hook))
        for c in range(d.c):
            xx,_,_=d.channel_batch('val',start,end,c,future=False);a.encode_raw_channel_first(f.model,xx,0)
        for h in hooks:h.remove()
        for n,m in layers.items():
            inp=torch.stack([v[0] for v in storage[n]]);expected=torch.stack([v[1] for v in storage[n]])
            rows=[]
            for compute in [77]:
                for algo in range(8):
                    try:
                        out=backend.linear(inp.reshape(1,-1,inp.shape[-1]),m,compute,1,algo,inp.shape[1]).reshape_as(expected)
                        rows.append(dict(compute=compute,algo=algo,maxdiff=float((out-expected).abs().max()),bit_exact=torch.equal(out,expected),different_elements=int((out!=expected).sum())))
                    except RuntimeError as e: rows.append(dict(compute=compute,algo=algo,error=str(e)))
            rec=dict(start=start,end=end,layer=n,input_shape=list(inp.shape),variants=rows);reports.append(rec)
            print(json.dumps(rec),flush=True)
    a.rt.write_json(dest/'Weather_720_B7_FLAT_CANONICAL_ALGO_AUDIT.json',dict(reports=reports,future_used=False,production_activation=False))
if __name__=='__main__':main()
