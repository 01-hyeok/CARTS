import sys,json,torch,inspect,textwrap,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from evaluate import setup,canonical_logits,normalize,topk,now,dump,OUT
import packed_batch

def patch(batch):
    src=textwrap.dedent(inspect.getsource(packed_batch.PackedChannelFull.short_query))
    needle='canonical_norm128(z,b) if exact_torch2_tail else F.normalize(z.reshape(c*b,-1),dim=-1).reshape(c,b,-1)'
    assert src.count(needle)==1;src=src.replace(needle,'canonical_norm128(z,b)')
    ns=dict(packed_batch.__dict__);exec(compile(src,__file__+'::canonical_encoder_norm','exec'),ns)
    batch.short_query=types.MethodType(ns['short_query'],batch)

@torch.no_grad()
def main(cell):
    d,f,prior,identity,before=setup(cell,'FULL');patch(f.batch);rows=[]
    for split in ('val','test'):
        n=len(d.datasets[split]);a=(n-1)//32*32;b=n
        g,m=f.full_logits(split,a,b);ref=canonical_logits(f,split,a,b,'FULL');pool=d.pool_all(split,a,b);raw=prior.batch(split,a,b)
        for support in ('FULL','P100'):
            gg=g if support=='FULL' else g.gather(-1,pool[:,:,None].expand(-1,-1,5,-1))
            rr=ref if support=='FULL' else ref.gather(-1,pool[:,:,None].expand(-1,-1,5,-1))
            mm=m if support=='FULL' else torch.ones_like(pool,dtype=torch.bool)
            ids=torch.arange(len(d.starts),device=d.device).expand(b-a,d.c,-1) if support=='FULL' else pool
            p=normalize(gg,mm);rp=normalize(rr,mm);pick=ids.gather(-1,topk(p,10));vn='FULL' if support=='FULL' else 'P100_CONTROLLED'
            heads=ids[:,:,None].expand(-1,-1,5,-1).gather(-1,topk(gg.masked_fill(~mm[:,:,None],-float('inf')),10))
            row=dict(split=split,start=a,end=b,support=support,raw_exact=bool(torch.equal(gg,rr)),raw_max=float((gg-rr).abs().max()),prob_exact=bool(torch.equal(p,rp)),prob_max=float((p-rp).abs().max()),ids_match_old=bool((pick.cpu().numpy()==raw[vn+'__selected_ids']).all()),head_id_differences=int((heads.cpu().numpy()!=raw[vn+'__head_ids']).sum()))
            print(json.dumps(row),flush=True);rows.append(row)
    dump(OUT/'recovery02'/(cell+'_NORM_PROBE.json'),dict(at=now(),rows=rows,source_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()))
if __name__=='__main__':main(sys.argv[1])
