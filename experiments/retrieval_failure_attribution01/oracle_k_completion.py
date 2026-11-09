"""Complete missing per-K Oracle outputs from frozen values; no model/optimizer.

GPU1 only, entire existing P100 and every val/test query/channel. Batched-channel
arithmetic is checked against the original channel path and saved K10 metrics.
"""
import argparse,csv,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from diagnostic_adapter import ROOT,HERE,OLD,GPU,rt,Data

OUT=ROOT/'results/TRACK-P100-HEAD-RETRIEVAL-QUALITY01-ORACLE-K-SUPPLEMENT01'
KS=(1,5,10,20)

def read(path):return json.loads(Path(path).read_text())
def dump(path,obj):rt.write_json(path,obj)
def sync():torch.cuda.synchronize()
def manifest(cell):
    return next(x for x in read(OLD/'configs/run_manifest.json') if x['run_id']==cell+'_V0_seed0')

def calculate(values,last,target):
    """Shape [rows,100,H]; canonical functions, including alignment addition order."""
    d=rt.pooled_future_mse(values,last,target)
    order=rt.stable_topk_indices(d,100,largest=False)
    lm=(last[:,None]-target).square().mean(1)
    rows=[rt.metric_rows(d,values,last,target,order[:,:k],order[:,:k],lm) for k in KS]
    return torch.stack(rows,1),order[:,:20]

@torch.no_grad()
def joint(data,split,start,end):
    x,y,starts=data.query(split,start,end);ids=data.pool_all(split,start,end)
    b,c,p=ids.shape
    assert p==100 and c==data.c
    assert bool((ids.sort(-1).values[...,1:]!=ids.sort(-1).values[...,:-1]).all())
    valid=data.mask(starts)[:,None].expand(-1,c,-1)
    assert bool(valid.gather(-1,ids).all())
    values=data.values_all(ids)
    metrics,order=calculate(values.reshape(b*c,p,data.h),x[...,-1].reshape(-1),y.reshape(b*c,data.h))
    picks=ids.gather(-1,order.reshape(b,c,20))
    return metrics.reshape(b,c,4,7),picks

@torch.no_grad()
def canonical(data,split,start,end,channel):
    x,y,starts=data.channel_batch(split,start,end,channel)
    ids=data.pool(split,starts,channel)
    metric,order=calculate(data.values(ids,channel),x[:,-1,0],y[:,:,0])
    return metric,ids.gather(1,order)

@torch.no_grad()
def smoke(cell):
    data=Data(cell);n=min(32,len(data.datasets['val']))
    torch.cuda.reset_peak_memory_stats();sync();t=time.perf_counter()
    got,ids=joint(data,'val',0,n);sync();joint_seconds=time.perf_counter()-t
    t=time.perf_counter();expected=[];eids=[]
    for channel in range(data.c):
        m,i=canonical(data,'val',0,n,channel);expected.append(m);eids.append(i)
    sync();canonical_seconds=time.perf_counter()-t
    expected=torch.stack(expected,1);eids=torch.stack(eids,1)
    assert torch.equal(ids,eids),'Joint-channel Oracle IDs differ from canonical path'
    assert torch.equal(got,expected),'Joint-channel Oracle metrics differ from canonical path'
    # K1 AggMSE=RetMSE and Oracle recall=1 / regret=0, exact by this path.
    assert torch.equal(got[:,:,0,0],got[:,:,0,1])
    assert torch.equal(got[:,:,:,3],torch.ones_like(got[:,:,:,3]))
    assert torch.equal(got[:,:,:,4],torch.zeros_like(got[:,:,:,4]))
    assert torch.isfinite(got).all()
    result=dict(status='PASS',cell=cell,queries=n,channels=data.c,p100=100,k=list(KS),
        ordered_candidate_ids_exact=True,metrics_exact=True,joint_seconds=joint_seconds,canonical_seconds=canonical_seconds,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_uuid=GPU)
    dump(OUT/'smoke'/f'{cell}.json',result);print('SMOKE',json.dumps(result),flush=True)

def synthetic():
    torch.manual_seed(0)
    for h in [96,192,336,720]:
        values=torch.randn(6,100,h,device='cuda');last=torch.randn(6,device='cuda');target=torch.randn(6,h,device='cuda')
        got,picks=calculate(values,last,target)
        d=((values+last[:,None,None]-target[:,None])**2).mean(-1)
        assert torch.equal(picks,rt.stable_topk_indices(d,20,largest=False))
        for j,k in enumerate(KS):
            aligned=values.gather(1,picks[:,:k,None].expand(-1,-1,h))+last[:,None,None]
            error=aligned.mean(1)-target
            assert torch.equal(got[:,j,0],d.gather(1,picks[:,:k]).mean(1))
            assert torch.equal(got[:,j,1],error.square().mean(1))
            assert torch.equal(got[:,j,2],error.abs().mean(1))
    tied=torch.zeros(3,100,96,device='cuda');zero=torch.zeros(3,device='cuda');target=torch.zeros(3,96,device='cuda')
    met,ids=calculate(tied,zero,target)
    assert torch.equal(ids,torch.arange(20,device='cuda').expand(3,20))
    assert not bool(met[:,:,:3].any()) and bool((met[:,:,3]==1).all())
    dump(OUT/'smoke/SYNTHETIC.json',dict(status='PASS',horizons=4,tie_policy='stable stored P100 order',cases=5))

@torch.no_grad()
def run_cell(cell):
    target=OUT/cell
    if (target/'COMPLETE.json').exists():
        done=read(target/'COMPLETE.json');assert done['status']=='COMPLETE'
        assert done['source_sha256']==rt.sha256_file(__file__),'Source changed after completed Oracle cell'
        print('RESUME COMPLETE',cell,flush=True);return
    begin=time.perf_counter();torch.cuda.reset_peak_memory_stats();data=Data(cell)
    original=ROOT/manifest(cell)['output_path'];results=[];gates={};counts={};fallbacks=0
    target.mkdir(parents=True,exist_ok=True)
    for split in ['val','test']:
        old=read(original/f'{split}_metrics.json');n=len(data.datasets[split]);sums=np.zeros((4,7),np.float64)
        saved=[]
        for c in range(data.c):
            with np.load(original/'queries'/split/f'channel{c:03d}.npz',allow_pickle=False) as a:
                assert np.array_equal(a['query_start'],data.datasets[split].starts)
                assert int(a['channel'])==c and a['baselines'].shape==(n,22,7)
                saved.append(a['baselines'][:,0].copy())
        saved=np.stack(saved,1);maxdiff=0.;total=0;id_hash=hashlib.sha256()
        for a in range(0,n,32):
            b=min(a+32,n);met,ids=joint(data,split,a,b)
            # Canonical check at every batch against all saved K10 query/channel metrics.
            observed=met.cpu().numpy();bad=np.any(observed[:,:,2]!=saved[a:b],axis=(0,2))
            for c in np.flatnonzero(bad):
                cm,ci=canonical(data,split,a,b,int(c));met[:,c]=cm;ids[:,c]=ci;fallbacks+=1
            observed=met.cpu().numpy()
            assert np.array_equal(observed[:,:,2],saved[a:b]),(cell,split,a,'Saved K10 exact metric gate')
            assert np.isfinite(observed).all()
            id_hash.update(ids.cpu().numpy().tobytes());sums+=observed.astype(np.float64).sum(axis=(0,1));total+=(b-a)*data.c
            rt.write_json(target/'CURRENT.json',dict(split=split,completed_queries=b,total_queries=n,channels=data.c,pid=os.getpid(),at=rt.now()))
            # Compact query metrics and the top20 Oracle IDs; never save candidate futures.
            dest=target/split;dest.mkdir(exist_ok=True)
            np.savez_compressed(dest/f'batch{a//32:05d}.npz',metrics=observed,oracle_top20_ids=ids.cpu().numpy().astype(np.int32),
                query_start=data.datasets[split].starts[a:b],k=np.asarray(KS),metric_names=np.asarray(rt.FIELDS))
            if a%512==0:print('ORACLE',cell,split,b,'/',n,'channels',data.c,flush=True)
        assert total==n*data.c;means=sums/total
        for j,k in enumerate(KS):
            prev=next(z for z in old['rows'] if z['name']==('All' if k==10 else f'All_K{k}'))
            derived=prev['retmse']-prev['regret']
            assert abs(means[j,0]-derived)<1e-7,(cell,split,k,'derived Ret mismatch')
            results.append(dict(cell=cell,split=split,k=k,**dict(zip(rt.FIELDS,means[j].tolist())),unit_count=total,status='OBSERVED_FROZEN_VALUE_EVALUATION'))
        gates[split]=dict(status='PASS',saved_k10_metrics_exact=True,query_ids_exact=True,queries=n,channels=data.c,units=total,
            oracle_top20_ordered_sha256=id_hash.hexdigest(),source_metrics_sha256=rt.sha256_file(original/f'{split}_metrics.json'))
        counts[split]=n
    with (target/'ORACLE_K.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(results[0]));w.writeheader();w.writerows(results)
    result=dict(status='COMPLETE',cell=cell,completed_at=rt.now(),source_sha256=rt.sha256_file(__file__),pool_hashes=data.pool_hashes,
        data=data.data_info,query_counts=counts,channels=data.c,k=list(KS),gates=gates,canonical_fallback_channel_batches=fallbacks,
        wall_seconds=time.perf_counter()-begin,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        gpu_uuid=GPU,stage1_training=False,stage2=False,original_results_unchanged=True)
    dump(target/'COMPLETE.json',result);print('ORACLE CELL COMPLETE',json.dumps(result),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('cell',nargs='?');ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    assert GPU not in apps,'GPU1 already occupied; do not compete'
    rt.gpu_guard();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.smoke:
        synthetic();smoke('ETTh1_96');torch.cuda.empty_cache();smoke('Solar_720')
        dump(OUT/'SMOKE_GATE.json',dict(status='PASS',source_sha256=rt.sha256_file(__file__),at=rt.now(),gpu_uuid=GPU))
    else:
        gate=read(OUT/'SMOKE_GATE.json');assert gate['status']=='PASS' and gate['source_sha256']==rt.sha256_file(__file__)
        assert args.cell in read(HERE/'PROTOCOL.json')['cell_order'];run_cell(args.cell)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        dump(OUT/'FAILED.json',dict(at=rt.now(),error=repr(exc),traceback=traceback.format_exc()))
        raise
