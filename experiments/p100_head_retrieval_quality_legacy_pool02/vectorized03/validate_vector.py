import sys,json,copy,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
import runtime as rt
import torch,numpy as np
from vector_ops import *
from vector_eval import evaluate
from benchmark_vector import compare
from runtime_vector import source_hashes


class Limited:
    def __init__(self,ds,n):self.ds=ds;self.n=n;self.starts=ds.starts[:n]
    def __len__(self):return self.n
    def __getitem__(self,i):assert i<self.n;return self.ds[i]
    def __getattr__(self,k):return getattr(self.ds,k)


if __name__=='__main__':
    rt.gpu_guard();report=dict(status='RUNNING',started_at=rt.now(),eval=[],h720=[])
    data=rt.CellData('Solar_96','cuda:0');ds=data.datasets['val'];data.datasets['val']=Limited(ds,65)
    for nh in (0,5):
        m,h,_,_=rt.model_init('Solar_96',0,nh,'cuda:0')
        if nh==0:
            path=rt.OUT/'runs/Solar_96/seed0/V0/COMPLETE.json'
            completed=json.loads(path.read_text());payload=torch.load(completed['selected_checkpoint'],map_location='cpu',weights_only=False)
            m.load_state_dict(payload['model'])
        a=HERE/f'eval_smoke/V{nh}/legacy';b=HERE/f'eval_smoke/V{nh}/vector'
        torch.cuda.synchronize();begin=time.perf_counter()
        old=rt.evaluate(data,m,h,'val',True,a,baselines=nh==0);torch.cuda.synchronize();oldsec=time.perf_counter()-begin
        install_dropout(m)
        begin=time.perf_counter();new=rt.evaluate(data,m,h,'val',True,b,baselines=nh==0);torch.cuda.synchronize();newsec=time.perf_counter()-begin
        plain=rt.evaluate(data,m,h,'val',False)
        assert plain['rows'][0]==new['rows'][0],'Detailed/nondetailed checkpoint selection mismatch'
        diffs={k:max(abs(r[k]-s[k]) for r,s in zip(old['rows'],new['rows'])) for k in rt.FIELDS}
        changed=0;changed_sets=0;total=0;maxquerymse=0.;maxqueryret=0.;baseline_max_abs=0.
        for c in range(137):
            x=np.load(a/f'channel{c:03d}.npz');y=np.load(b/f'channel{c:03d}.npz')
            assert np.array_equal(x['query_start'],y['query_start'])
            assert np.array_equal(x['last'],y['last'])
            assert np.array_equal(x['input_features'],y['input_features'])
            ia=x['candidate_ids'];ib=y['candidate_ids']
            changed+=int((ia!=ib).any(-1).sum());changed_sets+=int((np.sort(ia,-1)!=np.sort(ib,-1)).any(-1).sum());total+=ia.shape[0]*ia.shape[1]
            maxquerymse=max(maxquerymse,float(abs(x['metrics'][:,:,1]-y['metrics'][:,:,1]).max()))
            maxqueryret=max(maxqueryret,float(abs(x['metrics'][:,:,0]-y['metrics'][:,:,0]).max()))
            if nh==0:
                baseline_max_abs=max(baseline_max_abs,float(abs(x['baselines']-y['baselines']).max()))
                assert np.allclose(x['baselines'],y['baselines'],rtol=1e-6,atol=5e-7),'Oracle/raw/random exceeded FP32 reduction tolerance'
        item=dict(heads=nh,queries=65,channels=137,units=8905,metric_abs_differences=diffs,
            ordered_selection_changed=changed,set_selection_changed=changed_sets,selection_units=total,
            max_query_mse_difference=maxquerymse,max_query_retmse_difference=maxqueryret,
            legacy_seconds=oldsec,vector_seconds=newsec,checkpoint_selection_detailed_exact=True,
            baseline_random_oracle_max_abs=baseline_max_abs if nh==0 else None)
        print('EVAL_PARITY',json.dumps(item),flush=True);report['eval'].append(item)
        assert max(diffs.values())==0 and changed==0 and baseline_max_abs==0,item
    del data,m,h
    torch.cuda.empty_cache()
    data=rt.CellData('Solar_720','cuda:0');x,y,starts=data.batch('train',range(32))
    ids=rt.canonical_sample_pool(data,'train',np.arange(32)).permute(1,0,2).contiguous()
    report['h720']=[compare(data,x,y,ids,False),compare(data,x,y,ids,True)]
    benchmark=json.loads((HERE/'BENCHMARK.json').read_text());assert benchmark['status']=='PASS'
    report.update(status='PASS',finished_at=rt.now(),source_hashes=source_hashes(),
        benchmark_sha256=rt.sha256_file(HERE/'BENCHMARK.json'),evaluation_policy='original canonical channelwise evaluation retained exactly; vector evaluation rejected after near-tie mismatch')
    rt.write_json(HERE/'VECTOR_GATES.json',report)
