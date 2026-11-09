"""Past-only localization of canonical head projection and norm reductions."""
import sys,inspect,textwrap,json,fcntl,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
import d123_full_population as pop
import packed_batch,torch
import torch.nn.functional as F
from cublaslt_batch import CublasLtBatch
from cublas_batch import CublasBatch
from exact_short import canonical_norm128
from utils.full_candidate_bank import encode_raw_channel_first
OUT=pop.OUT/'population_recovery06'


@torch.no_grad()
def main():
    before=pop.start_gpu();d=pop.Data('ETTh1_96');row=next(r for r in pop.inventory(d.cell) if r['architecture']=='V5' and r['training_support']=='FULL')
    f=pop.Frozen(d,row);f.precision();f.batch.build_keys();lt=CublasLtBatch();blas=CublasBatch();rows=[]
    source=textwrap.dedent(inspect.getsource(packed_batch.PackedChannelFull.short_query));source=source[:source.index('    if self.frozen.heads is None:')]+'    return z\n'
    ns=dict(packed_batch.__dict__);exec(compile(source,__file__+'::prefix','exec'),ns);prefix=ns['short_query']
    for split in ['val','test']:
        a=len(d.datasets[split])-1;b=a+1;x,_,starts=d.query(split,a,b);pool=d.pool_all(split,a,b)
        rid=pop.canonical_cross_ids(f,split,a,b,'FULL');rpid=pop.canonical_cross_ids(f,split,a,b,'P100')
        zz=[];raw=[];q=[];strides=[]
        for c in range(d.c):
            xx,_,_=d.channel_batch(split,a,b,c,future=False);z=encode_raw_channel_first(f.model,xx,0)
            v=torch.einsum('bd,sod->bso',z,f.heads.W);zz.append(z);raw.append(v);q.append(F.normalize(v,dim=-1));strides.append(v.stride())
        zr=torch.stack(zz,1);rr=torch.stack(raw,1);qr=torch.stack(q,1);zp=prefix(f.batch,x)
        ref=torch.stack([torch.einsum('bsd,nd->bsn',qr[:,c],f.extension_reference_keys[c]) for c in range(d.c)],1)
        weight=f.heads.W.reshape(-1,128)[None].expand(d.c,-1,-1).contiguous();zpc=zp.permute(1,0,2).contiguous()
        mask=d.mask(starts)[:,None].expand(1,d.c,-1);pm=torch.ones_like(pool,dtype=torch.bool)
        for mode in ['lt11_fp32','lt11_tf32','blas11_fp32','blas11_tf32','flat_torch']:
            comp=68 if mode.endswith('fp32') else 77
            if mode=='flat_torch':v=torch.einsum('bd,sod->bso',zp.reshape(d.c,128),f.heads.W).reshape(d.c,5,128)
            else:
                v=lt.score(zpc,weight,comp,1,0) if mode.startswith('lt') else blas.score(zpc,weight,comp,-1)
                v=v.reshape(d.c,5,128)
            normalizers={'torch':lambda v:F.normalize(v,dim=-1),
                         'canonical5':lambda v:canonical_norm128(v,5),
                         'canonical1':lambda v:canonical_norm128(v,1),
                         'canonical35':lambda v:canonical_norm128(v,35),
                         'transposed_torch':lambda v:F.normalize(v.transpose(1,2).contiguous().transpose(1,2),dim=-1)}
            for nm,fn in normalizers.items():
                qp=fn(v)[None];score=torch.bmm(qp[0],f.batch.keys.transpose(1,2))[None]
                full=pop.topk(pop.normalize(score,mask),10);pg=score.gather(-1,pool[:,:,None].expand(-1,-1,5,-1));pid=pool.gather(-1,pop.topk(pop.normalize(pg,pm),10))
                rec=dict(split=split,projection=mode,norm=nm,canonical_raw_strides=strides,batched_raw_stride=v.stride(),encoder_exact=torch.equal(zp,zr),
                         raw_exact=torch.equal(v[None],rr),raw_delta=float((v[None]-rr).abs().max()),query_exact=torch.equal(qp,qr),query_delta=float((qp-qr).abs().max()),
                         scores_exact=torch.equal(score,ref),score_delta=float((score-ref).abs().max()),full_ordered_exact=torch.equal(full,rid),controlled_ordered_exact=torch.equal(pid,rpid),channels=d.c,bank_count=len(d.starts))
                rows.append(rec)
        print('NORM_PROBE',split,'exact',[(r['projection'],r['norm']) for r in rows if r['split']==split and r['query_exact'] and r['scores_exact']],flush=True)
    keys=lambda s:{(r['projection'],r['norm']) for r in rows if r['split']==s and all(r[k] for k in ['encoder_exact','raw_exact','query_exact','scores_exact','full_ordered_exact','controlled_ordered_exact'])}
    candidates=sorted(keys('val')&keys('test'))
    pop.dump(OUT/'NORM_PROBE_GATE.json',dict(status='PASS_EXACT_ALL_CHANNEL_CANDIDATE' if candidates else 'MORE_DIAGNOSIS_REQUIRED',rows=rows,candidates=candidates,
        checkpoint=row,checkpoint_unchanged=pop.sha(row['checkpoint'])==row['checkpoint_sha256'],source_sha256=pop.sha(__file__),gpu_before=before,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),production_activated=False))


if __name__=='__main__':
    boundary=(pop.BASE/'00_Audit/d4_controller.lock').open('a');fcntl.flock(boundary,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=(pop.BASE/'00_Audit/d4_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main()
    except BaseException as e:pop.dump(OUT/'NORM_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
