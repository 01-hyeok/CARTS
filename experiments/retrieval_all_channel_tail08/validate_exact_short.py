from diagnose_splitk import *
from exact_short import Split64,canonical_norm128


@torch.no_grad()
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.GPU
    assert a.GPU not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    lock=(a.OUT/'diagnostic.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dest=OUT/'EXACT_SHORT_GATE.json';assert not dest.exists()
    a.rt.gpu_guard();d=a.Data('Weather_720');row=next(r for r in a.inventory(d.cell,'FULL') if r['architecture']=='V0' and r['status']=='VALID')
    f=a.Frozen(d,row,'FULL');f.precision();packed=adapter('exact_short_trace',TAIL)(f);packed.build_keys();kernel=Split64()
    rows=[]
    for split,start,end in [('val',4544,4551),('test',9792,9820),('val',0,32)]:
        x,_,starts=d.query(split,start,end);b,c,h=x.shape
        ref=canonical.original_probability(f,split,start,end).permute(1,0,2).contiguous()
        qref=torch.stack([F.normalize(a.encode_raw_channel_first(f.model,x[:,ch,:,None].contiguous(),0),dim=-1) for ch in range(c)])
        packed.short_query(x[:min(b,c-1)]) # initialize Lt for independent layer trace
        enc=f.model.encoder;views=transform_relation_features(x.permute(0,2,1),f.model.relation_input_space)
        z=torch.stack(views,2).permute(3,0,2,1).contiguous().reshape(c,b,-1)
        for seq in (enc.encoder,enc.norm,enc.proj):
            for layer in (list(seq) if isinstance(seq,torch.nn.Sequential) else [seq]):
                z=packed.lt.linear(z,layer,68,1,0) if isinstance(layer,torch.nn.Linear) else layer(z)
        q=canonical_norm128(canonical_norm128(z,b),b)
        mask=d.mask(starts)[None].expand(c,b,len(d.starts))
        for name,got in [('split64_same_query',kernel(qref,packed.keys)),('split64_end_to_end',kernel(q,packed.keys))]:
            r=dict(split=split,start=start,end=end,method=name,query=delta(q,qref),**compare(got.masked_fill(~mask,-float('inf')),ref));rows.append(r);print(json.dumps(r),flush=True)
    canonical.old_immutable(f)
    a.rt.write_json(dest,dict(status='ACTUAL_TAIL_DIAGNOSIS_NOT_FULL_RELEASE',rows=rows,
       source_hashes={p.name:a.rt.sha256_file(p) for p in Path(__file__).parent.glob('*') if p.suffix in ['.py','.cu','.so']},
       checkpoint_sha256=row['checkpoint_sha256'],production_release=False))


if __name__=='__main__':main()
