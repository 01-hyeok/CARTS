"""Inspect true channel-batched arithmetic on identical canonical layer inputs."""
import sys,os,json
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS')
sys.path[:0]=[str(ROOT/'experiments/retrieval_full_precision_repair02'),str(ROOT)]
import full_precision_runtime
import diagnostic_adapter as a
import torch
from all_channel_ops import channel_linear,encode_all
from models.RelationStage1 import transform_relation_features

@torch.no_grad()
def main():
    dest=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/ALL_CHANNEL03';pause=json.loads((dest/'PAUSE.json').read_text())
    for pid in [pause['worker_pid'],pause['controller_pid']]:assert Path('/proc/%d/stat'%pid).read_text().split(') ')[1].split()[0]=='T'
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.GPU
    a.rt.gpu_guard();d=a.Data('Weather_96');row=next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V5');f=a.Frozen(d,row,'FULL');f.precision()
    split='test'; end=len(d.datasets[split]); start=(end//32)*32; x,_,_=d.query(split,start,end);layers={n:m for n,m in f.model.encoder.named_modules() if isinstance(m,(torch.nn.Linear,torch.nn.LayerNorm,torch.nn.GELU))}
    storage={n:[] for n in layers};hooks=[]
    for n,m in layers.items():
        def hook(m,inputs,output,n=n):storage[n].append((inputs[0].clone(),output.clone()))
        hooks.append(m.register_forward_hook(hook))
    canonical=[];legacy_input=[]
    for c in range(d.c):
        xx,_,_=d.channel_batch(split,start,end,c,future=False);legacy_input.append(xx[:,:,0]);canonical.append(a.encode_raw_channel_first(f.model,xx,0))
    for h in hooks:h.remove()
    canon=torch.stack(canonical,1);legacy_input=torch.stack(legacy_input,1)
    views=transform_relation_features(x.permute(0,2,1),f.model.relation_input_space)
    combined=torch.stack(views,dim=2).permute(0,3,2,1).contiguous()
    separate=torch.stack([torch.stack([v[...,0] for v in transform_relation_features(x[:,c,:,None].contiguous(),f.model.relation_input_space)],1) for c in range(d.c)],1)
    rows=[]
    for n,m in layers.items():
        inp=torch.stack([v[0] for v in storage[n]],1);expected=torch.stack([v[1] for v in storage[n]],1)
        b,c,dim=inp.shape
        variants=[]
        for precision in [True,False]:
            torch.backends.cuda.matmul.allow_tf32=precision
            out=m(inp.reshape(b*c,dim)).reshape(b,c,-1)
            variants.append(dict(mode='flattened',tf32=precision,maxdiff=float((out-expected).abs().max()),bit_exact=torch.equal(out,expected),different_elements=int((out!=expected).sum())))
            if isinstance(m,torch.nn.Linear):
                out=channel_linear(inp.permute(1,0,2).contiguous(),m).permute(1,0,2).contiguous()
                variants.append(dict(mode='batched_gemm',tf32=precision,maxdiff=float((out-expected).abs().max()),bit_exact=torch.equal(out,expected),different_elements=int((out!=expected).sum())))
        rows.append(dict(layer=n,type=type(m).__name__,canonical_per_channel_shape=[b,dim],batched_input=list(inp.shape),variants=variants))
    f.precision()
    report=dict(status='DIAGNOSED',raw_input_bit_exact=torch.equal(x,legacy_input),feature_transform_bit_exact=torch.equal(combined,separate),
        channel_axis_order='query,channel,feature,time',all_models_eval=not f.model.training and not f.heads.training,
        dropout_training=[m.training for m in f.model.modules() if isinstance(m,torch.nn.Dropout)],
        relation_input_space=f.model.relation_input_space,encoder_type=f.model.encoder.encoder_type,
        query_shape=list(x.shape),feature_shape=list(combined.shape),weight_sha256=row['checkpoint_sha256'],layers=rows,
        production_stream_activation=False,original_generation_runtime='UNKNOWN')
    a.rt.write_json(dest/'BATCHED_TAIL_LAYER_AUDIT.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
