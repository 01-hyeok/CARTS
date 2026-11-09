"""Canonical P100 head cache, only after legacy queues release GPU1."""
import os,sys,json,time
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments/score_fusion_ablation01'),str(ROOT/'experiments/learned_dynamics_cross01')]
import evaluate as ev
from learned_scores import log_v5_score,masked_log_probability
import numpy as np
import torch
OUT=ROOT/'results/TRACK-PREDICTIVE-SCORE-CALIBRATION01';OLD=ROOT/'results/TRACK-LEARNED-DYNAMICS-CROSS01'
CELLS=['ETTh1_96','ETTh1_336','ETTh1_720','Weather_96','Weather_336','Weather_720','ETTh1_192','Weather_192']

def main(cell):
    d,f,prior,identity,before=ev.setup(cell,'P100')
    dest=OUT/'canonical_cache/P100'/cell;dest.mkdir(parents=True,exist_ok=True)
    old=OLD/'P100'/cell
    previous=json.loads((old/'IDENTITY.json').read_text())
    assert identity['checkpoint']==previous['checkpoint'] and identity['pool_hashes']==previous['pool_hashes'] and identity['data']==previous['data']
    ip=dest/'IDENTITY.json'
    if ip.exists():assert json.loads(ip.read_text())==identity
    else:ev.dump(ip,identity)
    hashes={};beg=time.time();counts={};fresh=[]
    for split in ('train','val','test'):
        N=len(d.datasets[split]);count=0
        for a in range(0,N,32):
            b=min(a+32,N);path=dest/split/('batch%05d.npz'%(a//32));path.parent.mkdir(exist_ok=True)
            if path.exists():
                meta=json.loads(path.with_suffix('.json').read_text());assert ev.sha(path)==meta['sha256'];hashes[str(path.relative_to(dest))]=meta['sha256'];count+=b-a;continue
            cache=old/'frozen_scalar_cache'/split/path.name;meta=json.loads(cache.with_suffix('.json').read_text());assert ev.sha(cache)==meta['sha256']
            assert meta['identity_checkpoint']==f.row['checkpoint_sha256'] and meta['pool_hash']==d.pool_hashes[split]
            with np.load(cache) as z:
                assert np.array_equal(z['starts'],d.datasets[split].starts[a:b]);pold=torch.from_numpy(z['p'].copy()).to(d.device);dist=torch.from_numpy(z['distance'].copy()).to(d.device)
            with torch.no_grad():
                g,ids=f.native_p100_logits(split,a,b);mask=torch.ones_like(ids,dtype=torch.bool);p=ev.normalize(g,mask)
                assert torch.equal(ids,d.pool_all(split,a,b)) and d.mask(d.datasets[split].starts[a:b])[:,None].expand(b-a,d.c,-1).gather(-1,ids).all()
                assert torch.equal(ev.topk(p,10),ev.topk(pold,10)),'Cached/new ordered top10 mismatch'
                assert torch.allclose(p,pold,rtol=2e-6,atol=2e-8),'Cached/new probability mismatch'
                if split!='train':
                    oldraw=prior.batch(split,a,b);vn=ev.variant_name('P100','P100')
                    assert np.array_equal(ids.gather(-1,ev.topk(p,10)).cpu().numpy(),oldraw[vn+'__selected_ids'])
                    assert np.array_equal(dist.gather(-1,ev.topk(p,10)).cpu().numpy(),oldraw[vn+'__selected_mse'])
                if a in {0,(N-1)//32*32}:
                    ref=ev.canonical_logits(f,split,a,b,'P100');rp=ev.normalize(ref,mask)
                    assert torch.equal(ev.topk(p,10),ev.topk(rp,10))
                    assert torch.allclose(p,rp,rtol=2e-6,atol=2e-8)
                    xx,yy,_=d.query(split,a,b);actual=ev.candidate_distances(d,xx,yy,ids)
                    assert torch.equal(actual,dist),'Future reconstruction differs from original scalar cache'
                    fresh.append(dict(split=split,start=a,raw_cosine_max_delta=float((g-ref).abs().max()),p_max_delta=float((p-rp).abs().max()),candidate_mse_exact=True,ordered_ids_exact=True))
                assert torch.allclose(masked_log_probability(log_v5_score(p,mask)/.1,mask).exp(),p.double(),atol=2e-8,rtol=2e-6)
            tmp=path.with_suffix('.tmp')
            with tmp.open('xb') as stream:np.savez(stream,g=g.cpu().numpy(),p=p.cpu().numpy(),ids=ids.cpu().numpy(),starts=d.datasets[split].starts[a:b])
            os.replace(tmp,path);digest=ev.sha(path);hashes[str(path.relative_to(dest))]=digest
            ev.dump(path.with_suffix('.json'),dict(sha256=digest,old_scalar_cache=str(cache),old_scalar_sha256=meta['sha256'],all_ordered_ids_exact=True,all_masks_valid=True))
            count+=b-a
            if a%1024==0:
                ev.dump(OUT/'CURRENT.json',dict(phase='CANONICAL_HEAD_CACHE',cell=cell,split=split,end=b,total=N,pid=os.getpid()));print('HEAD_CACHE',cell,split,b,N,flush=True)
        assert count==N;counts[split]=count
    ev.dump(dest/'COMPLETE.json',dict(status='PASS_ALL_QUERY_P100_HEAD_CACHE',counts=counts,channels=d.c,files=hashes,identity=identity,fresh_first_tail=fresh,seconds=time.time()-beg,gpu_peak_allocated=torch.cuda.max_memory_allocated(),gpu_peak_reserved=torch.cuda.max_memory_reserved()))
if __name__=='__main__':main(sys.argv[1])
