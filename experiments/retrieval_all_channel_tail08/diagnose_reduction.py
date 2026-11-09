from diagnose_splitk import *


def norm128_canonical(z):
    """Torch2 Reduce.cuh canonical B7 block_x=128 reduction, all C at once."""
    assert z.shape[-1]==128
    v=z*z
    v=v[...,:64]+v[...,64:]
    v=v[...,:32]+v[...,32:]
    for offset in [1,2,4,8,16]:v=v[...,:-offset]+v[...,offset:]
    return v.sqrt()


def interleaved_fma(q,k,tile,parts=2,reverse=False):
    c,b,d=q.shape;n=k.shape[1];acc=[]
    assert d%(tile*parts)==0
    for part in range(parts):
        v=q.new_zeros(c,b,n)
        order=[j for start in range(part*tile,d,parts*tile) for j in range(start,start+tile)]
        if reverse:order=order[::-1]
        for j in order:v=torch.addcmul(v,q[:,:,j,None],k[:,None,:,j])
        acc.append(v)
    out=acc[0]
    for v in acc[1:]:out=out+v
    return out


@torch.no_grad()
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.GPU
    assert a.GPU not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    lock=(a.OUT/'diagnostic.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dest=OUT/'REDUCTION_DIAGNOSIS.json';assert not dest.exists()
    a.rt.gpu_guard();torch.cuda.reset_peak_memory_stats()
    d=a.Data('Weather_720');row=next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V0' and r['status']=='VALID')
    f=a.Frozen(d,row,'FULL');f.precision()
    x,_,starts=d.query('val',4544,4551);b,c,h=x.shape
    reference=canonical.original_probability(f,'val',4544,4551).permute(1,0,2).contiguous()
    keys=torch.stack([f.keys[ch] for ch in range(c)])
    q=torch.stack([F.normalize(a.encode_raw_channel_first(f.model,x[:,ch,:,None].contiguous(),0),dim=-1) for ch in range(c)])
    mask=d.mask(starts)[None].expand(c,b,len(d.starts));records=[]
    for tile in [1,2,4,8,16,32,64]:
        for reverse in [False,True]:
            t=time.perf_counter();got=interleaved_fma(q,keys,tile,2,reverse);torch.cuda.synchronize()
            result=dict(tile=tile,parts=2,reverse=reverse,seconds=time.perf_counter()-t,**compare(got.masked_fill(~mask,-float('inf')),reference))
            records.append(result);print(json.dumps(result),flush=True);del got
    saved=[];hook=f.model.encoder.proj.register_forward_hook(lambda m,i,o:saved.append(o.clone()))
    for ch in range(c):a.encode_raw_channel_first(f.model,x[:,ch,:,None].contiguous(),0)
    hook.remove();z=torch.stack(saved)
    norms=[]
    for stage in range(2):
        nr=torch.stack([F.normalize(z[ch],dim=-1) for ch in range(c)])
        got=z/norm128_canonical(z).clamp_min(1e-12)
        result=dict(stage=stage,**delta(got,nr));norms.append(result);print(json.dumps(result),flush=True)
        z=nr
    canonical.old_immutable(f)
    a.rt.write_json(dest,dict(status='DIAGNOSTIC_ONLY_NO_RELEASE',scores=records,norms=norms,
        source_sha256=a.rt.sha256_file(Path(__file__)),checkpoint_sha256=row['checkpoint_sha256'],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))


if __name__=='__main__':main()
