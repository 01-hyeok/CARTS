"""Past-only kernel-shape equivalence audit. No future/metric-based selection."""
import sys,os,json,argparse
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS')
sys.path[:0]=[str(ROOT/'experiments/retrieval_full_precision_repair02'),str(ROOT)]
import full_precision_runtime
import diagnostic_adapter as a
import torch
from packed_batch import PackedChannelFull

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cell',default='Weather_96');ap.add_argument('--split',default='test');args=ap.parse_args()
    dest=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/ALL_CHANNEL03';pause=json.loads((dest/'PAUSE.json').read_text())
    for pid in [pause['worker_pid'],pause['controller_pid']]:assert Path('/proc/%d/stat'%pid).read_text().split(') ')[1].split()[0]=='T'
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.GPU
    a.rt.gpu_guard();d=a.Data(args.cell);row=next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V5');f=a.Frozen(d,row,'FULL');f.precision()
    end=len(d.datasets[args.split]);start=(end//32)*32;assert start<end
    x,_,_=d.query(args.split,start,end)
    canonical=torch.stack([f.heads(a.encode_raw_channel_first(f.model,x[:,c,:,None].contiguous(),0)) for c in range(d.c)],1)
    packed=PackedChannelFull(f);rows=[]
    for n in [21,24,28,32,36,40,48,56,64,80,96,112,128,160,192,224,252,256,288,320,384,448,512,672,768,1024]:
        if n<d.c:continue
        chunk=n//d.c;out=[]
        for first in range(0,len(x),chunk):out.append(packed.encode(x[first:min(first+chunk,len(x))],n,True))
        q=torch.cat(out);rows.append(dict(gemm_rows=n,all_channels=d.c,query_chunk=chunk,maxdiff=float((q-canonical).abs().max()),bit_exact=torch.equal(q,canonical),different_elements=int((q!=canonical).sum())))
    result=dict(cell=args.cell,split=args.split,queries=end-start,query_range=[start,end],rows=rows,criterion='bit exact frozen query/head tensors only; no future or performance labels; unchanged TF32=True')
    a.rt.write_json(dest/(args.cell+'_'+args.split+'_TAIL_PACKING_AUDIT.json'),result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
