"""Read-only decomposition of a short-batch mismatch; never a fallback."""
import os,json,argparse,inspect,textwrap,types
import torch
import torch.nn.functional as F
import numpy as np
from adapter import Data,Frozen,inventory,rt,normalize,topk
from worker import guard,ROOT,OUT
from io_utils import dump,now,asnumpy

@torch.no_grad()
def main(cell,model):
    os.chdir(ROOT);guard();torch.set_num_threads(1);d=Data(cell)
    row=next(r for r in inventory(cell) if r['architecture']==model and r['training_support']=='P100')
    f=Frozen(d,row);f.precision();split='test';b=len(d.datasets[split]);a=b-b%32
    x,_,starts=d.query(split,a,b);B,C,H=x.shape;ids=d.pool_all(split,a,b)
    nq=f.batch.short_query(x)
    hist=d.histories_all(ids).permute(0,2,1,3).reshape(B*100,C,H);rows=B*100;step=max(1,rows//C)
    nk=torch.cat([f.batch.encode(hist[i:i+step],rows) for i in range(0,B*100,step)]).reshape(B,100,C,-1).permute(0,2,1,3).contiguous()
    cq=[];ck=[];cs=[]
    for c in range(C):
        xx,_,_=d.channel_batch(split,a,b,c,future=False);z=rt.encode_raw(f.model,xx,0)
        cq.append(F.normalize(z,dim=-1)[:,None] if f.heads is None else f.heads(z))
        hh=d.histories(ids[:,c],c);kk=F.normalize(rt.encode_raw(f.model,hh.reshape(B*100,H,1),0).reshape(B,100,-1),dim=-1)
        ck.append(kk);cs.append(rt.scores(f.model,f.heads,xx,d,c,ids[:,c],query_channel=0))
    cq=torch.stack(cq,1);ck=torch.stack(ck,1);cs=torch.stack(cs,1);ns,_=f.native_p100_logits(split,a,b)
    cases={'canonical':cs,'production':ns,
           'canonical_encoded_batched':torch.einsum('bcsd,bcpd->bcsp',cq,ck),
           'native_encoded_reduce':(nq[:,:,:,None,:]*nk[:,:,None,:,:]).sum(-1),
           'canonical_encoded_reduce':(cq[:,:,:,None,:]*ck[:,:,None,:,:]).sum(-1)}
    # One Lt operation across B*C query/channel rows, preserving the canonical
    # per-query matrix dimensions and original B-count heuristic.
    from cublaslt_batch import CublasLtBatch
    lt=CublasLtBatch()
    import packed_batch
    original_source=textwrap.dedent(inspect.getsource(packed_batch.PackedChannelFull.short_query))
    source=original_source.replace("exact_torch2_tail=(compute==68 and h==720 and b==7 and c==21 and self.frozen.heads is None)","exact_torch2_tail=(compute==68 and self.frozen.heads is None)")
    namespace=dict(packed_batch.__dict__);exec(source,namespace)
    norm_q=namespace['short_query'](f.batch,x)
    cases['canonical_norm_batched']=torch.einsum('bcsd,bcpd->bcsp',norm_q,nk)
    queries=[('native',nq,nk),('canonical',cq,ck),('canonical_norm',norm_q,nk)]
    for label,source in [('first_lt',original_source),('first_lt_norm',source)]:
        source=source.replace('(compute==68 and layer.in_features==720)','(compute==68)')
        namespace=dict(packed_batch.__dict__);exec(source,namespace)
        z=namespace['short_query'](f.batch,x);queries.append((label,z,nk))
    qdiff={label:float((q-cq).abs().max()) for label,q,k in queries}
    for source,q,k in queries:
        try:cases[source+'_lt']=lt.score(q.reshape(B*C,q.shape[2],-1),k.reshape(B*C,100,-1),68,B,0).reshape(B,C,q.shape[2],100)
        except Exception as exc:print('LT_ERROR',repr(exc),flush=True)
    from cublas_batch import CublasBatch
    cb=CublasBatch();q=queries[-1][1];k=nk
    qq=q.reshape(B*C,q.shape[2],-1);kk=k.reshape(B*C,100,-1)
    cases['exact_query_reduce']=(q[:,:,:,None,:]*k[:,:,None,:,:]).sum(-1)
    cases['bmm_qk']=torch.bmm(qq,kk.transpose(1,2)).reshape(B,C,q.shape[2],100)
    cases['bmm_kq']=torch.bmm(kk,qq.transpose(1,2)).transpose(1,2).reshape(B,C,q.shape[2],100)
    for algo in [-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]:
        try:cases['blas_'+str(algo)]=cb.score(qq,kk,68,algo).reshape(B,C,q.shape[2],100)
        except Exception as exc:print('ALGO_UNAVAILABLE',algo,str(exc),flush=True)
    for choice in range(1,8):
        try:cases['lt_choice_'+str(choice)]=lt.score(qq,kk,68,B,choice).reshape(B,C,q.shape[2],100)
        except Exception as exc:print('LT_UNAVAILABLE',choice,str(exc),flush=True)
    for library in ['libcublasLt.so.12']:
        try:
            other=CublasLtBatch(library)
            for choice in range(5):
                try:cases['lt12_'+str(choice)]=other.score(qq,kk,68,B,choice).reshape(B,C,q.shape[2],100)
                except Exception as exc:print('LT12_ALGO_UNAVAILABLE',choice,str(exc),flush=True)
        except Exception as exc:print('LT12_UNAVAILABLE',str(exc),flush=True)
    from torch.profiler import profile,ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
        torch.einsum('bd,bpd->bp',cq[:,0,0].contiguous(),ck[:,0].contiguous())
        torch.cuda.synchronize()
    print('CANONICAL_KERNELS',[(e.name,e.input_shapes) for e in prof.events() if str(e.device_type).endswith('CUDA')],flush=True)
    expected=f.reference(split)['ids'][a:b]
    result=dict(at=now(),cell=cell,model=model,start=a,end=b,channels=C,q_max=float((nq-cq).abs().max()),query_max_by_case=qdiff,key_max=float((nk-ck).abs().max()),cases={})
    for name,s in cases.items():
        p=normalize(s,torch.ones(B,C,100,device=d.device,dtype=torch.bool));actual=asnumpy(ids.gather(-1,topk(p,10)))
        mism=np.argwhere(np.any(actual!=expected,axis=-1));details=[]
        for q,c in mism[:10]:
            details.append(dict(query=int(starts[q]),channel=int(c),got=actual[q,c].tolist(),expected=expected[q,c].tolist(),score_max=float((s-cs)[q,c].abs().max())))
        result['cases'][name]=dict(score_max=float((s-cs).abs().max()),mismatch_qc=len(mism),details=details)
    dump(OUT/'00_Audit'/f'P100_SCORE_KERNEL12_{cell}_{model}_AUDIT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cell');p.add_argument('model');a=p.parse_args();main(a.cell,a.model)
