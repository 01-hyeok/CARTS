"""Missing full-query V5 D1/D2 traces and exact D3 normalization control.

Frozen weights; score each Full bank once, then derive common-P100 control.
Native P100 is computed and labeled separately. Previously verified Oracle IDs
are reused, never selected from a different support. No future values in scores.
"""
import argparse,fcntl,time
from server_bridge import *
from adapter import distance_all
from gpu_metrics import normalization_control
from d56_complete_run import reproduction,save,read_npz,now,canonical_cross_ids

BASE=ROOT/'results/TRACK-HEAD-SHIFT-DIAGNOSTIC01';OUT=BASE/'full_population_completion'

def head_probability(g,mask):
    z=g.masked_fill(~mask[:,:,None],-float('inf'))
    if g.shape[-1]>100:
        B,C,S,N=z.shape;v=z.permute(1,0,2,3).contiguous();pad=(-B)%4
        if pad:v=torch.cat([v,v.new_zeros(C,pad,S,N)],1)
        return torch.softmax(v/.1,-1)[:,:B].permute(1,0,2,3).contiguous()
    return torch.softmax(z/.1,-1)

class OracleSource:
    def __init__(self,data,support,split):
        self.data=data;self.support=support;self.split=split
        self.path=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01'/support/data.cell
        self.prov=json.loads((self.path/'PROVENANCE.json').read_text());assert self.prov['data']==data.data_info and self.prov['pool_hashes']==data.pool_hashes
        complete=json.loads((self.path/'COMPLETE.json').read_text());assert complete['status']=='COMPLETE'
        gate=json.loads((self.path/split/'SPLIT_GATE.json').read_text());assert gate['status']=='PASS' and gate['query_count']==len(data.datasets[split]) and gate['channels']==data.c
        supplement=ROOT/'results/TRACK-RETRIEVAL-DIAGNOSTIC-SUPPLEMENT01'/support/data.cell
        sg=json.loads((supplement/'INFERENCE_GATE.json').read_text());assert sg['status']=='PASS' and sg['source_complete_sha256']==sha(self.path/'COMPLETE.json')
        row=next(r for r in sg['splits'] if r['split']==split);assert row['all_methods_ordered_top10_and_oracle_ids_exact'] and row['queries']==len(data.datasets[split])
        self.identity=dict(complete_sha256=sha(self.path/'COMPLETE.json'),provenance_sha256=sha(self.path/'PROVENANCE.json'),split_gate_sha256=sha(self.path/split/'SPLIT_GATE.json'),independent_oracle_reproduction_gate_sha256=sha(supplement/'INFERENCE_GATE.json'))
    def batch(self,a,b):
        p=self.path/self.split/('batch%05d.npz'%(a//32));z=read_npz(p)
        assert np.array_equal(z['query_start'],self.data.datasets[self.split].starts[a:b])
        ids=torch.as_tensor(z['oracle_ids'],device=self.data.device)
        assert ids.shape==(b-a,self.data.c,10)
        methods=[str(x) for x in z['method_names']];metrics=[str(x) for x in z['metric_names']]
        oracle_index=next(i for i,s in enumerate(methods) if s in ['IndividualMSE_Oracle','Oracle'])
        ret_index=next(i for i,s in enumerate(metrics) if s.lower() in ['retmse','retmse@10','ret'])
        return ids,z['metrics'][:,:,oracle_index,ret_index],dict(path=str(p),sha256=sha(p),**self.identity)

def candidate_distances(data,x,y,ids):
    # Fixed 128-candidate chunks preserve canonical per-row MSE reduction.
    B,C,P=ids.shape;parts=[]
    for a in range(0,P,128):
        block=ids[...,a:a+128];n=block.shape[-1]
        if n<128:block=torch.cat([block,block.new_zeros(B,C,128-n)],-1)
        parts.append(distance_all(data.values_all(block),x[:,:,-1],y)[...,:n])
    return torch.cat(parts,-1)

def selections(g,mask,allids):
    ph=head_probability(g,mask);score=ph.mean(2);assert torch.equal(score,normalize(g,mask))
    sel=topk(score,10);head=topk(g.masked_fill(~mask[:,:,None],-float('inf')),10)
    loo=torch.stack([topk(ph[:,:,[k for k in range(5) if k!=h]].mean(2),10) for h in range(5)],2)
    ranked=topk(score,score.shape[-1]);rank=torch.empty_like(ranked)
    rank.scatter_(-1,ranked,torch.arange(1,score.shape[-1]+1,device=score.device).expand_as(ranked))
    return dict(ph=ph,score=score,sel=sel,head=head,loo=loo,rank=rank,ids=allids,g=g)

def export_variant(v,oracle,data,x,y):
    ids=v['ids'];B,C,N=ids.shape;head=v['head'].reshape(B,C,50);loo=v['loo'].reshape(B,C,50)
    locations=torch.cat([v['sel'],head,loo],-1);selected_ids=ids.gather(-1,locations)
    needed=torch.cat([selected_ids,oracle],-1);ds=candidate_distances(data,x,y,needed)
    out=dict(selected_ids=selected_ids[...,:10].cpu().numpy(),selected_mse=ds[...,:10].cpu().numpy(),
        head_ids=selected_ids[...,10:60].reshape(B,C,5,10).cpu().numpy(),head_mse=ds[...,10:60].reshape(B,C,5,10).cpu().numpy(),
        loo_ids=selected_ids[...,60:110].reshape(B,C,5,10).cpu().numpy(),loo_mse=ds[...,60:110].reshape(B,C,5,10).cpu().numpy(),
        oracle_ids=oracle.cpu().numpy(),oracle_mse=ds[...,110:].cpu().numpy(),
        head_candidate_mixture_rank=v['rank'].gather(-1,head).reshape(B,C,5,10).cpu().numpy(),
        head_candidate_mixture_score=v['score'].gather(-1,head).reshape(B,C,5,10).cpu().numpy(),
        head_candidate_all_head_logits=v['g'].gather(-1,head[:,:,None].expand(-1,-1,5,-1)).cpu().numpy(),
        head_candidate_all_head_probabilities=v['ph'].gather(-1,head[:,:,None].expand(-1,-1,5,-1)).cpu().numpy())
    return out

@torch.no_grad()
def run(cell,training,smoke):
    os.chdir(ROOT);before=start_gpu();data=Data(cell);row=next(r for r in inventory(cell) if r['architecture']=='V5' and r['training_support']==training)
    prior=reproduction(data,row);assert prior is not None,'full native reproduction required before follow-up'
    f=Frozen(data,row);f.precision()
    dest=OUT/('smoke' if smoke else 'raw')/cell/('V5_'+training);dest.mkdir(parents=True,exist_ok=True)
    identity=dict(checkpoint=row,pool_hashes=data.pool_hashes,data=data.data_info,reproduction_gate_sha256=sha(prior),
        source_sha256=sha(HERE/'d123_full_population.py'),adapter_sha256=sha(ROOT/'experiments/retrieval_failure_mechanism02/adapter.py'),
        original_runtime='UNKNOWN',current_torch=torch.__version__,all_channels_single_batch=True)
    if (dest/'PROVENANCE.json').exists():assert json.loads((dest/'PROVENANCE.json').read_text())==identity
    else:dump(dest/'PROVENANCE.json',identity)
    if not smoke:
        gate=OUT/'smoke'/cell/('V5_'+training)/'GATE.json';sg=json.loads(gate.read_text());assert sg['status']=='PASS_FIRST_TAIL_FULL_POPULATION_SMOKE' and sg['identity']==identity
        for k,h in sg['files'].items():assert sha(gate.parent/k)==h
    begin=time.perf_counter();hashes={};counts={};source_hashes={};max_distance_difference=0.
    for split in ['val','test']:
        N=len(data.datasets[split]);sources={s:OracleSource(data,s,split) for s in ['FULL','P100']}
        batches=list(range(0,N,32));batches=sorted(set([0,batches[-1]])) if smoke else batches
        count=0
        for a in batches:
            b=min(a+32,N);p=dest/split/('batch%05d.npz'%(a//32));side=p.with_suffix('.json')
            if p.exists():
                meta=json.loads(side.read_text());assert meta['identity']==identity and sha(p)==meta['sha256'];hashes[str(p.relative_to(dest))]=meta['sha256'];count+=b-a;continue
            dump(dest/'CURRENT.json',dict(at=now(),pid=os.getpid(),split=split,query_start=a,total_queries=N,phase='ALL_CHANNEL_FULL_SCORE'))
            x,y,starts=data.query(split,a,b);pool=data.pool_all(split,a,b);g,mask=f.full_logits(split,a,b)
            allids=torch.arange(len(data.starts),device=data.device).expand(b-a,data.c,-1);assert mask.gather(-1,pool).all()
            full=selections(g,mask,allids);gg=g.gather(-1,pool[:,:,None].expand(-1,-1,5,-1));pmask=torch.ones_like(pool,dtype=torch.bool)
            controlled=selections(gg,pmask,pool);variants={'FULL':full,'P100_CONTROLLED':controlled}
            if training=='P100':
                gn,pn=f.native_p100_logits(split,a,b);assert torch.equal(pn,pool);variants['P100_NATIVE']=selections(gn,pmask,pn)
            oracles={};priorret={};proofs={}
            for s in sources:oracles[s],priorret[s],proofs[s]=sources[s].batch(a,b)
            out={'query_start':np.asarray(starts),'pool_ids':pool.cpu().numpy()}
            for name,v in variants.items():
                support='FULL' if name=='FULL' else 'P100';r=export_variant(v,oracles[support],data,x,y)
                # Old Oracle average used FP32/FP64 reductions depending on method;
                # exact candidate values are separately tested against canonical full
                # distance vectors in first/tail smoke and sampled context below.
                np.testing.assert_allclose(r['oracle_mse'].astype(float).mean(-1),priorret[support],atol=2e-6,rtol=2e-6)
                if name==('FULL' if training=='FULL' else 'P100_NATIVE'):
                    if training=='P100':assert np.array_equal(r['selected_ids'],f.reference(split)['ids'][a:b])
                    else:
                        # Existing canonical full IDs, reference only; never fallback.
                        assert np.array_equal(r['selected_ids'],f.canonical_full_ids(split,a,b).cpu().numpy())
                if smoke:
                    expected=canonical_cross_ids(f,split,a,b,support) if name not in ['P100_NATIVE'] else torch.as_tensor(f.reference(split)['ids'][a:b],device=data.device)
                    assert np.array_equal(r['selected_ids'],expected.cpu().numpy())
                for k,vv in r.items():out[name+'__'+k]=vv
            # Same logits, same P100 IDs, only per-head denominator differs.
            n=normalization_control(g,mask,pool);A=topk(n['local'],10);B=topk(n['restricted'],10)
            dcids=torch.cat([pool.gather(-1,A),pool.gather(-1,B),out_ids(full),oracles['P100'],oracles['FULL']],-1)
            dcm=candidate_distances(data,x,y,dcids).double();RA=dcm[...,:10].mean(-1);RB=dcm[...,10:20].mean(-1);RC=dcm[...,20:30].mean(-1)
            assert torch.allclose(RC-RA,(RB-RA)+(RC-RB),atol=1e-12,rtol=1e-12)
            out.update(D3_A_ids=pool.gather(-1,A).cpu().numpy(),D3_B_ids=pool.gather(-1,B).cpu().numpy(),D3_C_ids=out_ids(full).cpu().numpy(),
                D3_A_mse=dcm[...,:10].cpu().numpy(),D3_B_mse=dcm[...,10:20].cpu().numpy(),D3_C_mse=dcm[...,20:30].cpu().numpy(),
                D3_local_scores=n['local'].cpu().numpy(),D3_restricted_scores=n['restricted'].cpu().numpy(),D3_head_mass=n['mass'].cpu().numpy(),D3_implicit_weights=n['weights'].cpu().numpy(),
                D3_logZ_full=n['logZ_full'].cpu().numpy(),D3_logZ_pool=n['logZ_pool'].cpu().numpy())
            if smoke:
                _,_,_,d=distances(data,split,a,b)
                assert torch.equal(topk(-d.masked_fill(~mask,float('inf')),10),oracles['FULL'])
                pd=d.gather(-1,pool);assert torch.equal(pool.gather(-1,topk(-pd,10)),oracles['P100'])
                for name in variants:
                    for idkey,msekey in [('selected_ids','selected_mse'),('head_ids','head_mse'),('loo_ids','loo_mse'),('oracle_ids','oracle_mse')]:
                        ids=torch.as_tensor(out[name+'__'+idkey],device=data.device);expected=d.gather(-1,ids.reshape(b-a,data.c,-1)).reshape(ids.shape).cpu().numpy()
                        actual=out[name+'__'+msekey];diff=float(np.max(np.abs(expected-actual)));max_distance_difference=max(max_distance_difference,diff)
                        assert np.array_equal(expected,actual),'candidate MSE packing changes canonical values'
                assert np.array_equal(d.gather(-1,dcids).cpu().numpy(),dcm.float().cpu().numpy())
            save(p,out);dump(side,dict(identity=identity,sha256=sha(p),oracle_sources=proofs,query_start=a,query_end=b,at=now()))
            hashes[str(p.relative_to(dest))]=sha(p);count+=b-a
            source_hashes[split+':'+str(a)]=proofs
            print(cell,training,split,b,N,flush=True)
            del g,full,controlled,variants,n,out,dcm
        counts[split]=count
        if not smoke:assert count==N
    dump(dest/'GATE.json',dict(status='PASS_FIRST_TAIL_FULL_POPULATION_SMOKE' if smoke else 'PASS_FULL_QUERY_D123_RAW',at=now(),identity=identity,
        queries=counts,channels=data.c,full_bank_count=len(data.starts),all_channels_single_batch=True,future_free_scores=True,seed=0,
        max_canonical_candidate_mse_difference=max_distance_difference,files=hashes,seconds=time.perf_counter()-begin,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),before=before,
        canonical_native_ordered_ids_checked=True,normalizer_identity_verified=True,negative_regret_clamp=False,source_oracle_blocks=source_hashes))

def out_ids(v):return v['ids'].gather(-1,v['sel'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cell');p.add_argument('training');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);lock=(BASE/'00_Audit/d4_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run(a.cell,a.training,a.smoke)
