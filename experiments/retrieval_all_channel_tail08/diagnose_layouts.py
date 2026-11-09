"""All-channel numerical layout diagnosis, not an inference release."""
from diagnose_splitk import *


@torch.no_grad()
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.GPU
    assert a.GPU not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    lock=(a.OUT/'diagnostic.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dest=OUT/'LAYOUT_DIAGNOSIS.json';assert not dest.exists()
    a.rt.gpu_guard();torch.cuda.reset_peak_memory_stats()
    d=a.Data('Weather_720');row=next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V0' and r['status']=='VALID')
    f=a.Frozen(d,row,'FULL');f.precision();packed=adapter('layout_trace',TAIL)(f);packed.build_keys()
    x,_,starts=d.query('val',4544,4551);c,b,h=x.shape
    reference=canonical.original_probability(f,'val',4544,4551).permute(1,0,2).contiguous()
    keys=torch.stack([f.keys[ch] for ch in range(c)])
    q=torch.stack([F.normalize(a.encode_raw_channel_first(f.model,x[:,ch,:,None].contiguous(),0),dim=-1) for ch in range(c)])
    mask=d.mask(starts)[None].expand(c,b,len(d.starts))
    n,dim=keys.shape[1:];records=[]
    # Preserve GEMM's short N=B, expose all channels via a block-diagonal K axis.
    # All banks remain present; zeros carry no extra candidate or query.
    for pad in [128,256]:
        t=time.perf_counter()
        block=keys.new_zeros(c,n,c,pad)
        idx=torch.arange(c,device=keys.device)
        block[idx,:,idx,:dim]=keys
        query=q.new_zeros(b,c,pad);query[:,:,:dim]=q.permute(1,0,2)
        got=(query.reshape(b,c*pad)@block.reshape(c*n,c*pad).T).reshape(b,c,n).permute(1,0,2).contiguous()
        torch.cuda.synchronize();result=dict(method='block_diagonal_all_channels',pad_dimension=pad,seconds=time.perf_counter()-t,
               key_matrix=list(block.reshape(c*n,c*pad).shape),**compare(got.masked_fill(~mask,-float('inf')),reference))
        records.append(result);print(json.dumps(result),flush=True);del block,query,got;torch.cuda.empty_cache()
    # Trace actual norm and divide kernels and intermediates independently.
    saved=[]
    hook=f.model.encoder.proj.register_forward_hook(lambda m,i,o:saved.append(o.clone()))
    for ch in range(c):a.encode_raw_channel_first(f.model,x[:,ch,:,None].contiguous(),0)
    hook.remove();z=torch.stack(saved)
    norm_ref=torch.stack([zz.norm(2,dim=-1,keepdim=True) for zz in saved])
    norm_batch=z.norm(2,dim=-1,keepdim=True)
    norm_records=[dict(stage='norm',**delta(norm_batch,norm_ref))]
    normed_ref=torch.stack([zz/nn for zz,nn in zip(saved,norm_ref)])
    norm_records.append(dict(stage='divide_same_norm',**delta(z/norm_ref,normed_ref)))
    for name,fn in [('sum_square',lambda:(z*z).sum(-1,keepdim=True).sqrt()),
                    ('vector_norm_transposed',lambda:z.permute(2,0,1).contiguous().norm(2,dim=0).unsqueeze(-1)),
                    ('float64',lambda:z.double().norm(2,dim=-1,keepdim=True).float())]:
        norm_records.append(dict(stage=name,**delta(fn(),norm_ref)))
    for name,fn in [('norm_reference',lambda:saved[0].norm(2,dim=-1,keepdim=True)),('norm_batch',lambda:z.norm(2,dim=-1,keepdim=True))]:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn();torch.cuda.synchronize()
        prof.export_chrome_trace(str(OUT/(name+'.json')))
    for r in norm_records:print(json.dumps(r),flush=True)
    canonical.old_immutable(f)
    a.rt.write_json(dest,dict(status='DIAGNOSTIC_ONLY_NO_RELEASE',scores=records,norms=norm_records,
        source_sha256=a.rt.sha256_file(Path(__file__)),checkpoint_sha256=row['checkpoint_sha256'],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))


if __name__=='__main__':main()
