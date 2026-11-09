from diagnose_reduction import *
import itertools


@torch.no_grad()
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.GPU
    assert a.GPU not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    lock=(a.OUT/'diagnostic.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dest=OUT/'ACCUMULATION_DIAGNOSIS.json';assert not dest.exists()
    a.rt.gpu_guard();d=a.Data('Weather_720');row=next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V0' and r['status']=='VALID')
    f=a.Frozen(d,row,'FULL');f.precision();x,_,starts=d.query('val',4544,4551);b,c,h=x.shape
    ref=canonical.original_probability(f,'val',4544,4551).permute(1,0,2).contiguous()
    keys=torch.stack([f.keys[ch] for ch in range(c)])
    q=torch.stack([F.normalize(a.encode_raw_channel_first(f.model,x[:,ch,:,None].contiguous(),0),dim=-1) for ch in range(c)])
    # Candidate subsampling is ONLY a reduction-order diagnostic, never inference.
    idx=torch.arange(0,len(d.starts),max(1,len(d.starts)//128),device=q.device)[:128]
    ks=keys[:,idx,:];rs=ref[:,:,idx];records=[]
    shape=(c,b,len(idx));zero=q.new_zeros(shape)
    def score(order,double=False):
        parts=[]
        for start in [0,64]:
            v=zero.clone()
            for j in order:
                left,right=q[:,:,start+j,None],ks[:,None,:,start+j]
                v=(v.double()+left.double()*right.double()).float() if double else torch.addcmul(v,left,right)
            parts.append(v)
        return parts[0]+parts[1]
    standard=list(range(64));fma=score(standard);dbl=score(standard,True)
    print('FMA_EMULATION',delta(fma,dbl),flush=True)
    records.append(dict(method='float64_fma_reference',**delta(dbl,rs)))
    orders={}
    for width in [2,4,8,16,32,64]:
        for stride in [1,2,4,8,16,32]:
            if stride>width:continue
            perm=[j for offset in range(stride) for j in range(offset,width,stride)]
            for rev in [False,True]:
                if rev:perm=perm[::-1]
                order=tuple(base+j for base in range(0,64,width) for j in perm)
                orders[order]=dict(width=width,stride=stride,reverse=rev)
    # All 4-item permutations cover register unrolling without cherry-picking data.
    for perm in itertools.permutations(range(4)):
        orders[tuple(base+j for base in range(0,64,4) for j in perm)]=dict(permutation4=list(perm))
    for order,label in orders.items():
        got=score(order);r=dict(label,**delta(got,rs));records.append(r)
        if r['exact']:print('EXACT_ORDER',label,flush=True)
    records.sort(key=lambda r:(r['different'],r['maxdiff']))
    print('BEST',json.dumps(records[:8]),flush=True)
    canonical.old_immutable(f)
    a.rt.write_json(dest,dict(status='DIAGNOSTIC_ONLY_NO_RELEASE',rows=records,
        fma_emulation=delta(fma,dbl),diagnostic_candidate_indices=idx.cpu().tolist(),
        note='Only arithmetic diagnostic sampled; production and release still full support and blocked',
        source_sha256=a.rt.sha256_file(Path(__file__)),checkpoint_sha256=row['checkpoint_sha256']))


if __name__=='__main__':main()
