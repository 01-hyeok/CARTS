"""Separate E96 tail encoder, head projection and score kernels using past only."""
import sys,inspect,textwrap,json,traceback,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
import d123_full_population as pop
from utils.full_candidate_bank import encode_raw_channel_first
from cublaslt_batch import CublasLtBatch
from cublas_batch import CublasBatch
import packed_batch,torch
import torch.nn.functional as F
OUT=pop.BASE/'full_population_completion/population_recovery05'

@torch.no_grad()
def main():
    before=pop.start_gpu();d=pop.Data('ETTh1_96');row=next(r for r in pop.inventory(d.cell) if r['architecture']=='V5' and r['training_support']=='FULL')
    f=pop.Frozen(d,row);f.precision();f.batch.build_keys();lt=CublasLtBatch();blas=CublasBatch()
    source=textwrap.dedent(inspect.getsource(packed_batch.PackedChannelFull.short_query))
    source=source[:source.index('    if self.frozen.heads is None:')]+'    return z\n'
    ns=dict(packed_batch.__dict__);exec(compile(source,__file__+'::encoder_prefix','exec'),ns);prefix=ns['short_query'];rows=[]
    for split in ['val','test']:
        a=len(d.datasets[split])-1;b=a+1;x,_,starts=d.query(split,a,b);pool=d.pool_all(split,a,b)
        pop.canonical_cross_ids(f,split,a,b,'FULL');keyref=torch.stack([f.extension_reference_keys[c] for c in range(d.c)])
        zz=[];qq=[]
        for c in range(d.c):
            xx,_,_=d.channel_batch(split,a,b,c,future=False);z=encode_raw_channel_first(f.model,xx,0);zz.append(z);qq.append(f.heads(z))
        zr=torch.stack(zz,1);qr=torch.stack(qq,1);zp=prefix(f.batch,x);qp=f.batch.short_query(x)
        ref=torch.stack([torch.einsum('bsd,nd->bsn',qr[:,c],keyref[c]) for c in range(d.c)],1)
        mask=d.mask(starts)[:,None].expand(1,d.c,-1);pm=torch.ones_like(pool,dtype=torch.bool)
        rid=pop.canonical_cross_ids(f,split,a,b,'FULL')
        rpid=pop.canonical_cross_ids(f,split,a,b,'P100')
        base=dict(split=split,query_index=a,batch=1,channels=d.c,bank_count=len(d.starts),keys_exact=torch.equal(keyref,f.batch.keys),key_delta=float((keyref-f.batch.keys).abs().max()),encoder_exact=torch.equal(zp,zr),encoder_delta=float((zp-zr).abs().max()),original_query_exact=torch.equal(qp,qr),original_query_delta=float((qp-qr).abs().max()))
        weight=f.heads.W.reshape(-1,128)[None].expand(d.c,-1,-1).contiguous();zpc=zp.permute(1,0,2).contiguous()
        queries={'original':qp,'flat_torch_heads':f.heads(zp.reshape(d.c,128)).reshape(1,d.c,5,128),'REFERENCE_ONLY':qr}
        for mode in ['lt11_fp32','lt11_tf32','blas11_fp32','blas11_tf32']:
            try:
                comp=68 if mode.endswith('fp32') else 77
                v=lt.score(zpc,weight,comp,1,0) if mode.startswith('lt') else blas.score(zpc,weight,comp,-1)
                queries[mode]=F.normalize(v.reshape(d.c,1,5,128),dim=-1).permute(1,0,2,3).contiguous()
            except Exception as e:rows.append(dict(**base,query_mode=mode,error=repr(e)))
        for qmode,q in queries.items():
            for smode in ['lt11_tf32','lt11_fp32','blas11_tf32','blas11_fp32','torch_bmm']:
                rec=dict(**base,query_mode=qmode,score_mode=smode,reference_only=qmode=='REFERENCE_ONLY',query_exact=torch.equal(q,qr),query_delta=float((q-qr).abs().max()))
                try:
                    qc=q.permute(1,0,2,3).reshape(d.c,5,128).contiguous();comp=68 if smode.endswith('fp32') else 77
                    if smode=='torch_bmm':v=torch.bmm(qc,f.batch.keys.transpose(1,2))
                    elif smode.startswith('lt'):v=lt.score(qc,f.batch.keys,comp,1,0)
                    else:v=blas.score(qc,f.batch.keys,comp,-1)
                    v=v[:,None].permute(1,0,2,3).contiguous();ids=pop.topk(pop.normalize(v,mask),10)
                    pg=v.gather(-1,pool[:,:,None].expand(-1,-1,5,-1));pids=pool.gather(-1,pop.topk(pop.normalize(pg,pm),10))
                    rec.update(scores_exact=torch.equal(v,ref),score_delta=float((v-ref).abs().max()),full_ordered_exact=torch.equal(ids,rid),controlled_ordered_exact=torch.equal(pids,rpid))
                except Exception as e:rec['error']=repr(e)
                rows.append(rec)
        pop.dump(OUT/'PROBE_PROGRESS.json',dict(rows=rows));print('E96_KERNEL_PROBE',json.dumps(base),flush=True)
    exact=[v for v in rows if not v.get('reference_only',True) and v.get('keys_exact') and v.get('encoder_exact') and v.get('query_exact') and v.get('scores_exact')]
    pairsets=[{(v['query_mode'],v['score_mode']) for v in exact if v['split']==s} for s in ['val','test']]
    candidates=sorted(pairsets[0]&pairsets[1])
    pop.dump(OUT/'PROBE_GATE.json',dict(status='EXACT_PAST_ONLY_KERNEL_CANDIDATE_FOUND' if candidates else 'DIAGNOSED_MORE_WORK_REQUIRED',rows=rows,candidates=candidates,production_activated=False,
        checkpoint_unchanged=pop.sha(row['checkpoint'])==row['checkpoint_sha256'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_before=before,source_sha256=pop.sha(__file__)))
if __name__=='__main__':
    try:main()
    except BaseException as e:pop.dump(OUT/'ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),production_activated=False));raise
