"""Isolated past-input numerical diagnosis. No production or training release."""
import os, sys, json, time, fcntl, subprocess
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments/retrieval_all_channel_tail06')]
import diagnose_weather720 as previous
from diagnose_weather720 import a, canonical, torch, F, adapter, compare, TAIL
from models.RelationStage1 import transform_relation_features
OUT=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/TAIL08'


def delta(x,y):
    return dict(shape=list(x.shape),exact=torch.equal(x,y),maxdiff=float((x-y).abs().max()),different=int((x!=y).sum()))


def partial_score(q,k,parts):
    """All channels in each bmm; chunk the embedding reduction only."""
    d=q.shape[-1];assert d%parts==0
    result=None
    for start in range(0,d,d//parts):
        piece=torch.bmm(q[:,:,start:start+d//parts].contiguous(),k[:,:,start:start+d//parts].contiguous().transpose(1,2))
        result=piece if result is None else result+piece
    return result


def fma_score(q,k,parts):
    """Explicit FP32 FMA chains with all channels batched; diagnostic only."""
    c,b,d=q.shape;n=k.shape[1];assert d%parts==0
    result=None
    for start in range(0,d,d//parts):
        piece=q.new_zeros(c,b,n)
        for j in range(start,start+d//parts):
            piece=torch.addcmul(piece,q[:,:,j,None],k[:,None,:,j])
        result=piece if result is None else result+piece
    return result


@torch.no_grad()
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.GPU
    assert a.GPU not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    lock=(a.OUT/'diagnostic.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir(exist_ok=False)
    a.rt.gpu_guard();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    d=a.Data('Weather_720');row=next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V0' and r['status']=='VALID')
    frozen=a.Frozen(d,row,'FULL');frozen.precision();assert not torch.backends.cuda.matmul.allow_tf32
    packed=adapter('tail05_trace',TAIL)(frozen);packed.build_keys()
    split,start,end='val',4544,4551
    x,_,starts=d.query(split,start,end);c,b=d.c,end-start
    reference=canonical.original_probability(frozen,split,start,end).permute(1,0,2).contiguous()
    keys=torch.stack([frozen.keys[ch] for ch in range(c)])
    q=torch.stack([F.normalize(a.encode_raw_channel_first(frozen.model,x[:,ch,:,None].contiguous(),0),dim=-1) for ch in range(c)])
    mask=d.mask(starts)[None].expand(c,b,len(d.starts))
    records=[]
    def save():
        a.rt.write_json(OUT/'CURRENT.json',dict(pid=os.getpid(),status='DIAGNOSTIC_ONLY',rows=records,production_release=False))
    for mode,fn in [('bmm',lambda:torch.bmm(q,keys.transpose(1,2)))]+[(f'partial_bmm_{p}',lambda p=p:partial_score(q,keys,p)) for p in [2,4,8,16]]+[(f'fma_chains_{p}',lambda p=p:fma_score(q,keys,p)) for p in [1,2,4,8]]:
        torch.cuda.synchronize();t=time.perf_counter();got=fn().masked_fill(~mask,-float('inf'));torch.cuda.synchronize()
        record=dict(method=mode,seconds=time.perf_counter()-t,**compare(got,reference));records.append(record);print(json.dumps(record),flush=True);save();del got
    enc=frozen.model.encoder
    packed.short_query(x)  # Initialize the same Lt backend before layer tracing.
    views=transform_relation_features(x.permute(0,2,1),frozen.model.relation_input_space)
    z=torch.stack(views,2).permute(3,0,2,1).contiguous().reshape(c,b,-1)
    reference_z=z.clone();layers=[]
    for seqname,seq in [('encoder',enc.encoder),('norm',enc.norm),('proj',enc.proj)]:
        items=list(seq) if isinstance(seq,torch.nn.Sequential) else [seq]
        for i,layer in enumerate(items):
            reference_z=torch.stack([layer(reference_z[ch]) for ch in range(c)])
            z=packed.lt.linear(z,layer,68,1,0) if isinstance(layer,torch.nn.Linear) else layer(z)
            record=dict(layer=f'{seqname}.{i}',**delta(z,reference_z));layers.append(record);print(json.dumps(record),flush=True)
    norm_records=[]
    # Separate canonical row-wise normalization from input propagation differences.
    for source_name,source in [('canonical_pre_normalize',reference_z),('batched_pre_normalize',z)]:
        norm_ref=torch.stack([F.normalize(source[ch],dim=-1) for ch in range(c)])
        variants={'three_dim':F.normalize(source,dim=-1),'flat':F.normalize(source.reshape(c*b,-1),dim=-1).reshape(c,b,-1)}
        for name,v in variants.items():
            record=dict(source=source_name,normalization=name,**delta(v,norm_ref));norm_records.append(record);print(json.dumps(record),flush=True)
    first=torch.stack([F.normalize(reference_z[ch],dim=-1) for ch in range(c)])
    print('reference_encoder_complete',delta(first,torch.stack([a.encode_raw_channel_first(frozen.model,x[:,ch,:,None].contiguous(),0) for ch in range(c)])),flush=True)
    for name,v in [('canonical_CBD_flat',F.normalize(first.reshape(c*b,-1),dim=-1).reshape(c,b,-1)),('BC_flat',F.normalize(first.permute(1,0,2).contiguous().reshape(b*c,-1),dim=-1).reshape(b,c,-1).permute(1,0,2))]:
        record=dict(second_normalization=name,**delta(v,q));norm_records.append(record);print(json.dumps(record),flush=True)
    # Base immutable check validates weights, not an output cache absent by design.
    integrity=canonical.old_immutable(frozen)
    result=dict(status='DIAGNOSIS_COMPLETE_NOT_RELEASED',rows=records,layers=layers,normalizations=norm_records,
                checkpoint_sha256=row['checkpoint_sha256'],source_sha256=a.rt.sha256_file(Path(__file__)),
                integrity=integrity,torch=torch.__version__,tf32=False,all_channels=c,candidates=len(d.starts),
                query_shape=list(q.shape),key_shape=list(keys.shape),seconds=time.perf_counter()-begin,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                historical_runtime='UNKNOWN',production_release=False,solar_training_release=False)
    a.rt.write_json(OUT/'DIAGNOSIS.json',result)


if __name__=='__main__':main()
