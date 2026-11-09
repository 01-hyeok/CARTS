"""Streaming all-query Full P0 with exact old selected-scorer replay."""
import argparse,csv,time
from native_runtime import *
from train_models import Arm
from ranking_metrics import metrics,aggregate,pairs

def csv_write(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def run(cell):
    source=OLD/'FULL'/cell;assert (source/'COMPLETE.json').exists()
    dest=OUT/'full_extension01/P0'/cell;assert not dest.exists(),'Preserve previous Full P0 attempt'
    ctx=NativeContext(cell,'FULL','FULL');d=ctx.d;dest.mkdir(parents=True);started=time.time();selection=json.loads((source/'SELECTION.json').read_text())['arms'];arms={}
    for n in ('D2_MLP','C2_MLP'):
        a=Arm(n).to(d.device);a.load_state_dict(load(source/'checkpoints'/n/('epoch%02d.pt'%selection[n]['best_epoch']))['model']);a.eval();arms[n]=a
    sample_rows=np.unique(np.linspace(0,len(d.datasets['train'])-1,256,dtype=int));sample=[]
    for block in np.unique(sample_rows//32):
        batch=ctx.batch('train',int(block*32),False)
        for row in sample_rows[sample_rows//32==block]:
            j=int(row%32);valid=torch.nonzero(batch['mask'][j,0],as_tuple=False).flatten();pos=valid[torch.as_tensor(np.linspace(0,len(valid)-1,128,dtype=int),device=d.device)]
            sample.append(batch['g'][j:j+1,:,:,pos].permute(0,1,3,2).cpu().numpy().reshape(-1,5))
    thresholds=np.quantile(np.concatenate(sample),np.arange(1,10)/10,axis=0)
    dump(dest/'TRAIN_HEAD_THRESHOLDS.json',dict(rows=sample_rows.tolist(),candidates='128 evenly spaced eligible candidates/query/channel; thresholds fitted on Train only',head_deciles=thresholds.tolist()))
    regime=json.loads((OUT/'regime_features'/cell/'TRAIN_THRESHOLDS.json').read_text());regrows=[];ranking=[];calibration=[];headquality=[];swaps=[];proofs=[];pairsets={}
    for split in ('val','test'):
        total={};heads={};headbins=np.zeros((5,10,3));delta_parts={n:[] for n in arms};swap_tot={n:dict(added_sum=0.,removed_sum=0.,count=0,delta_sum=0.,oracle_added=0,oracle_removed=0,near_oracle_added=0,near_oracle_removed=0,added_original_rank_sum=0.,added_new_rank_sum=0.,removed_original_rank_sum=0.,removed_new_rank_sum=0.,added_original_boundary_count=0,boundary_correct=0.,boundary_pair_count=0) for n in arms};units=0;baseline=0.
        for start in range(0,len(d.datasets[split]),32):
            batch=ctx.batch(split,start);oldpath=source/'final_raw'/split/('batch%05d.npz'%(start//32));assert sha(oldpath)==json.loads(oldpath.with_suffix('.json').read_text())['sha256']
            with np.load(oldpath) as z:old={k:z[k].copy() for k in z.files if k=='query_start' or k.startswith(('S0__','D2_MLP__','C2_MLP__'))}
            assert np.array_equal(old['query_start'],d.datasets[split].starts[start:batch['b']]);block_delta={n:[] for n in arms}
            with torch.no_grad():
                for j in range(0,batch['b']-start,2):
                    stop=min(j+2,batch['b']-start);ids=batch['ids'][j:stop];mask=batch['mask'][j:stop];p=batch['p'][j:stop];distance=batch['distance'][j:stop].double();rows=np.arange(start+j,start+stop)
                    scores={'S0':p.masked_fill(~mask,-torch.inf)}
                    for n,arm in arms.items():scores[n]=arm.score(p,arm.raw_chunks(ctx.features,split,rows,ids),mask,ranking=True)
                    basepos=stable_topk(scores['S0']);baseids=ids.gather(-1,basepos);basemse=distance.gather(-1,basepos);baseret=basemse.mean(-1);baseline+=float(baseret.sum());units+=baseret.numel()
                    oraclepos=stable_topk((-distance).masked_fill(~mask,-torch.inf));oracle=ids.gather(-1,oraclepos)
                    for n in arms:
                        pos=stable_topk(scores[n]);chosen=ids.gather(-1,pos);sm=distance.gather(-1,pos)
                        assert np.array_equal(chosen.cpu().numpy(),old[n+'__selected_ids'][j:stop]),'Old Full scorer ordered IDs differ: '+n
                        assert np.array_equal(sm.cpu().numpy(),old[n+'__selected_mse'][j:stop])
                        removed=~(baseids[...,None]==chosen[...,None,:]).any(-1);added=~(chosen[...,None]==baseids[...,None,:]).any(-1)
                        delta=((sm*added).sum(-1)-(basemse*removed).sum(-1))/10
                        assert torch.allclose(delta,sm.mean(-1)-baseret,rtol=0,atol=1e-12)
                        st=swap_tot[n];st['added_sum']+=float((sm*added).sum());st['removed_sum']+=float((basemse*removed).sum());st['count']+=int(added.sum());st['delta_sum']+=float(delta.sum())
                        st['oracle_added']+=int((added&(chosen[...,None]==oracle[...,None,:]).any(-1)).sum());st['oracle_removed']+=int((removed&(baseids[...,None]==oracle[...,None,:]).any(-1)).sum());block_delta[n].append(delta.cpu().numpy())
                    gg=batch['g'][j:stop].cpu().numpy();dd=distance.cpu().numpy();valid=mask.cpu().numpy();sc={n:s.cpu().numpy() for n,s in scores.items()}
                    for q in range(stop-j):
                        for c in range(d.c):
                            ix=np.flatnonzero(valid[q,c]);dist=dd[q,c,ix][None];base_order=np.argsort(-sc['S0'][q,c,ix],kind='stable');base_rank=np.argsort(base_order,kind='stable')+1;near_oracle=np.argsort(dist[0],kind='stable')[:20]
                            if len(ix) not in pairsets:pairsets[len(ix)]=pairs(len(ix))
                            for h in range(5):
                                hs=gg[q,c,h,ix];bins=np.searchsorted(thresholds[:,h],hs,side='left')
                                value=metrics(hs[None],dist,pairsets[len(ix)]);acc=heads.setdefault(h,{k:0. for k in value})
                                for key,v in value.items():acc[key]+=v
                                for k in range(10):
                                    m=bins==k;headbins[h,k]+=[m.sum(),hs[m].sum(),dist[0,m].sum()]
                            for n in arms:
                                score=sc[n][q,c,ix];order=np.argsort(-score,kind='stable');new_rank=np.argsort(order,kind='stable')+1;added=order[:10][~np.isin(order[:10],base_order[:10])];removed=base_order[:10][~np.isin(base_order[:10],order[:10])];st=swap_tot[n]
                                st['near_oracle_added']+=int(np.isin(added,near_oracle).sum());st['near_oracle_removed']+=int(np.isin(removed,near_oracle).sum())
                                st['added_original_rank_sum']+=float(base_rank[added].sum());st['added_new_rank_sum']+=float(new_rank[added].sum());st['removed_original_rank_sum']+=float(base_rank[removed].sum());st['removed_new_rank_sum']+=float(new_rank[removed].sum());st['added_original_boundary_count']+=int((base_rank[added]<=20).sum())
                                for left in (8,9):
                                    for right in (10,11):
                                        li,ri=base_order[left],base_order[right];truth=dist[0,ri]-dist[0,li];pred=score[li]-score[ri]
                                        if abs(truth)>1e-10:st['boundary_pair_count']+=1;st['boundary_correct']+=.5 if pred==0 else float(pred*truth>0)
                            for size in (20,50,100,len(ix)):
                                subset=base_order[:size] if size!=len(ix) else np.arange(len(ix));label='ALL' if size==len(ix) else 'S0_TOP%d'%size
                                if size not in pairsets:pairsets[size]=pairs(size)
                                for n in scores:
                                    value=metrics(sc[n][q,c,ix][None,subset],dist[:,subset],pairsets[size]);key=(n,label);acc=total.setdefault(key,{k:0. for k in value})
                                    for k,v in value.items():acc[k]+=v
            for n in arms:delta_parts[n].append(np.concatenate(block_delta[n]))
            if start%512==0:dump(dest/'CURRENT.json',dict(split=split,end=batch['b'],total=len(d.datasets[split]),at=time.time()));print('FULL_P0',cell,split,batch['b'],flush=True)
        reference=json.loads((source/'FULL_RESULTS.json').read_text());official=next(r['retmse'] for r in reference if r['method']=='S0' and r['split']==split);assert np.isclose(baseline/units,official,atol=2e-6,rtol=2e-6)
        for (n,label),v in total.items():ranking.append(dict(cell=cell,split=split,method=n,subset=label,**aggregate(v)))
        for h,v in heads.items():headquality.append(dict(cell=cell,split=split,head=h,subset='ALL',**aggregate(v)))
        for h in range(5):
            for k in range(10):
                count,ss,ds=headbins[h,k];calibration.append(dict(cell=cell,split=split,head=h,train_decile=k,count=int(count),mean_score=ss/count if count else None,mean_future_mse=ds/count if count else None))
        with np.load(OUT/'regime_features'/cell/(split+'.npz')) as z:features=z['features'];assert np.array_equal(z['starts'],d.datasets[split].starts)
        th=np.array(regime['thresholds'])
        for n,parts in delta_parts.items():
            delta=np.concatenate(parts);assert delta.shape==features.shape[:2];st=swap_tot[n];timestamps=delta.mean(-1);ref=next(r['retmse'] for r in reference if r['method']==n and r['split']==split)-official
            assert np.isclose(st['delta_sum']/units,ref,atol=2e-6,rtol=2e-6)
            row=dict(cell=cell,split=split,method=n,delta_retmse=st['delta_sum']/units,added_mean=st['added_sum']/st['count'] if st['count'] else None,removed_mean=st['removed_sum']/st['count'] if st['count'] else None,swap_count=st['count'],oracle_added=st['oracle_added'],oracle_removed=st['oracle_removed'])
            row.update(near_oracle_added=st['near_oracle_added'],near_oracle_removed=st['near_oracle_removed'],boundary_pair_accuracy=st['boundary_correct']/st['boundary_pair_count'] if st['boundary_pair_count'] else None,boundary_pair_count=st['boundary_pair_count'],added_original_boundary_fraction=st['added_original_boundary_count']/st['count'] if st['count'] else None,win_timestamps=int((timestamps<0).sum()),tie_timestamps=int((timestamps==0).sum()),loss_timestamps=int((timestamps>0).sum()))
            for key in ('added_original_rank','added_new_rank','removed_original_rank','removed_new_rank'):row[key]=st[key+'_sum']/st['count'] if st['count'] else None
            for pct in (1,5,10):
                tail=np.sort(timestamps)[-max(1,int(np.ceil(len(timestamps)*pct/100))):];row['worst_%d_mean_delta'%pct]=float(tail.mean());row['worst_%d_total_mean_contribution'%pct]=float(tail.sum()/len(timestamps))
            swaps.append(row)
            with (dest/('%s_%s_DELTA.npz'%(split,n))).open('xb') as f:np.savez(f,query_start=d.datasets[split].starts,delta=delta)
            for h,name in enumerate(regime['feature_names']):
                x=features[...,h];bins=(x>th[0,h]).astype(int)+(x>th[1,h]).astype(int)
                for k in range(3):
                    y=delta[bins==k];regrows.append(dict(cell=cell,split=split,method=n,feature=name,bin=k,count=len(y),mean_delta=float(y.mean()) if len(y) else None,win=int((y<0).sum()),tie=int((y==0).sum()),loss=int((y>0).sum())))
        proofs.append(dict(split=split,queries=len(d.datasets[split]),channels=d.c,baseline_retmse=baseline/units,baseline_reference=official,canonical_baseline_and_scorer_ids_exact=True))
    for name,rows in [('LOCAL_RANKING.csv',ranking),('HEAD_CALIBRATION.csv',calibration),('HEAD_QUALITY.csv',headquality),('CANDIDATE_SWAP.csv',swaps),('REGIME_ANALYSIS.csv',regrows)]:csv_write(dest/name,rows)
    dump(dest/'PAIR_SAMPLING.json',{str(n):[a.tolist(),b.tolist()] for n,(a,b) in pairsets.items()})
    dump(dest/'COMPLETE.json',dict(status='PASS_FULL_P0_ALL_QUERY',cell=cell,proofs=proofs,source_identity=ctx.identity,full_candidates=len(d.starts),candidate_reduction=False,seconds=time.time()-started,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),pair_policy='all pairs up to512 else fixed RandomState(1000+N) 512 unique pairs; same comparisons across methods',near_oracle='prespecified same-support FutureMSE Top20',boundary='original S0 ranks9/10 versus11/12; added original rank11..20 fraction',test_selection=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('cell');args=parser.parse_args();os.chdir(ROOT)
    try:run(args.cell)
    except BaseException as e:
        import traceback
        dump(OUT/'full_extension01'/('P0_FAILURE_%s_%d.json'%(args.cell,int(time.time()))),dict(error=repr(e),traceback=traceback.format_exc(),preserved=True));raise
