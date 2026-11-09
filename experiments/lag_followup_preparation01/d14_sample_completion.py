"""CPU-only missing D1/D2/D4 statistics on the declared D4 timestamp sample.

This does not substitute for full-population D1/D2/D3 diagnostics. Source gates,
numeric strings, checkpoint identity and scope are retained in separate outputs.
"""
import argparse,csv,gzip,json,time
from pathlib import Path
import numpy as np
from d56_aggregate import OUT,sha,dump,csvout,load,positions,group_masks,now
from fusion import analyze_unit

FIELDS=['zero','best','gain','best_lag','recent_difference','candidate_std','mse']
GROUPS=['CorrectOracle','MissedOracle','FalseSelected','Random10','NearOracle','FullOracleOutsideP100']

def mean_or_none(x):
    a=np.asarray(x,float);a=a[np.isfinite(a)]
    return float(a.mean()) if len(a) else None

def correlation(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float);m=np.isfinite(x)&np.isfinite(y);x,y=x[m],y[m]
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:return None
    return float(np.corrcoef(x,y)[0,1])

def paired_ci(delta,starts,length):
    # Missing groups remain undefined; pair only same-query nonempty groups.
    delta=np.asarray(delta,float);n=np.isfinite(delta).sum(1)
    keep=n>0;v=np.nansum(delta[keep],axis=1)/n[keep];starts=np.asarray(starts)[keep]
    if not len(v):return dict(delta=None,ci_low=None,ci_high=None,blocks=0,sufficient=False,paired_timestamps=0)
    g=(starts-starts.min())//length;chunks=[v[g==k] for k in np.unique(g)]
    rng=np.random.RandomState(0);est=[]
    for _ in range(2000):est.append(np.concatenate([chunks[i] for i in rng.randint(0,len(chunks),len(chunks))]).mean())
    return dict(delta=float(v.mean()),ci_low=float(np.quantile(est,.025)),ci_high=float(np.quantile(est,.975)),blocks=len(chunks),
        sufficient=len(chunks)>=4,paired_timestamps=len(v),block_length=length,repeats=2000,confidence=.95,
        resampling='timestamp blocks; all available channels jointly; absent groups remain undefined')

def rerank_metrics(selected,original,oracle,mse_lookup):
    s,o,b=map(lambda x:set(map(int,x)),(selected,oracle,original));assert len(s)==len(o)==len(b)==10
    missed=o-b;new=s-b;false_new=new-o
    d=np.array([mse_lookup[int(i)] for i in selected]);od=np.sort([mse_lookup[int(i)] for i in oracle])
    discount=1/np.log2(np.arange(2,12));idcg=np.sum(discount/(1+od))
    return dict(ndcg10=float(np.sum(discount/(1+d))/idcg),
        missed_oracle_recovery_rate=len(s&missed)/len(missed) if missed else None,
        missed_oracle_recovered_count=len(s&missed),initial_missed_count=len(missed),
        false_selected_promotion_rate=len(false_new)/len(new) if new else None,
        false_selected_promoted_count=len(false_new),newly_promoted_count=len(new),
        false_selected_promoted_fraction_of_k=len(false_new)/10)

def main(cell,model,training,evaluation):
    src=OUT/'complete_features'/cell/(model+'_'+training+'_to_'+evaluation)
    ana=OUT/'complete_analysis'/cell/src.name;fg=json.loads((src/'FEATURE_GATE.json').read_text());ag=json.loads((ana/'ANALYSIS_GATE.json').read_text())
    assert ag['status']=='PASS_D5_D6_CASE' and ag['source_feature_gate_sha256']==sha(src/'FEATURE_GATE.json')
    dest=OUT/'d14_statistics'/cell/src.name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'SAMPLE_STATISTICS_GATE.json').exists():
        old=json.loads((dest/'SAMPLE_STATISTICS_GATE.json').read_text());assert old['source_feature_gate_sha256']==sha(src/'FEATURE_GATE.json')
        for name,h in old['files'].items():assert sha(dest/name)==h
        return
    tick=time.perf_counter();tags=dict(**ag['identity'],coverage='32 fixed timestamps per split; all channels; NOT full-query D1/D2')
    C=fg['channels'];H=int(cell.rsplit('_',1)[1]);data={};summary=[];features=[];relevance=[];headrows=[];fusion=[];loo=[];traces=[];cis=[]
    groups_by_split={};starts_by_split={}
    for split in ['val','test']:
        rows=[]
        for p in sorted((src/split).glob('query*.npz')):
            assert sha(p)==fg['files'][str(p.relative_to(src))];rows.append(load(p))
        assert len(rows)==32;data[split]=rows;starts_by_split[split]=np.concatenate([z['query_start'] for z in rows])
        tables={name:{k:np.full((32,C),np.nan) for k in FIELDS} for name in GROUPS}
        for q,z in enumerate(rows):
            for c in range(C):
                ids=z['global_ids'][c];valid=z['valid'][c];masks=group_masks(z,c,model)
                unit=dict(**tags,split=split,query_index=int(z['query_index'][0]),query_start=int(z['query_start'][0]),channel=c)
                for name in GROUPS:
                    m=masks[name];r=dict(**unit,group=name,count=int(m.sum()),query_std=float(z['query_std'][c]))
                    for k in FIELDS:
                        v=z[k][c,m].astype(float);r[k+'_mean']=mean_or_none(v);r[k+'_median']=float(np.median(v)) if len(v) else None
                        if len(v):tables[name][k][q,c]=v.mean()
                    r['abs_lag_mean']=mean_or_none(np.abs(z['best_lag'][c,m]));features.append(r)
                # Candidate-level correlations are descriptive and aggregated by Q,C.
                for name in ['TopR','MissedOracle','FalseSelected','NearOracle']:
                    m=(np.isin(ids,z['ordered_ids'][c])&valid) if name=='TopR' else masks[name]
                    relevance.append(dict(**unit,group=name,count=int(m.sum()),gain_vs_future_mse_pearson=correlation(z['gain'][c,m],z['mse'][c,m]),
                        zero_vs_future_mse_pearson=correlation(z['zero'][c,m],z['mse'][c,m]),recent_vs_future_mse_pearson=correlation(z['recent_difference'][c,m],z['mse'][c,m])))
                if model=='V5':
                    ix=lambda x:positions(z,c,x)
                    hd=z['head_ids'][c];sd=z['ordered_ids'][c,:10];od=z['oracle_ids'][c,:10]
                    r=analyze_unit(hd,z['mse'][c,ix(hd.ravel())].reshape(5,10).astype(float),sd,z['mse'][c,ix(sd)].astype(float),od,z['mse'][c,ix(od)].astype(float))
                    for h in r.pop('heads'):headrows.append(dict(**unit,**h))
                    r.pop('missing_fields');r['lost_near_counts']=json.dumps(r['lost_near_counts']);r['lost_oracle_ids']=json.dumps(r['lost_oracle_ids']);r['oracle_C_ids']=json.dumps(r['oracle_C_ids'])
                    fusion.append(dict(**unit,**r))
                    uset=set(hd.ravel());mix=set(sd);d10=float(z['oracle_mse'][c,9]);lost=sorted(uset-mix)
                    for gid in lost:
                        j=int(ix([gid])[0]);ds=float(z['mse'][c,j]);oracle_hit=gid in set(od)
                        if not oracle_hit and ds>d10+max(1e-8,.10*d10):continue
                        traces.append(dict(**unit,candidate_id=int(gid),oracle10=oracle_hit,near1=ds<=d10+max(1e-8,.01*d10),near5=ds<=d10+max(1e-8,.05*d10),near10=ds<=d10+max(1e-8,.10*d10),
                            future_mse=ds,mixture_rank=int(z['candidate_mixture_rank'][c,j]),mixture_score=float(z['candidate_mixture_scores'][c,j]),
                            all_head_logits=json.dumps(z['candidate_head_logits'][c,:,j].tolist()),all_head_probabilities=json.dumps(z['candidate_head_probabilities'][c,:,j].tolist()),
                            heads_containing_id=json.dumps((np.flatnonzero((hd==gid).any(1))+1).tolist())))
                    for h in range(5):
                        without=float(z['loo_mse'][c,h].astype(float).mean())
                        loo.append(dict(**unit,removed_head=h+1,all5_retmse=r['r_mix'],without_head_retmse=without,delta_all_minus_without=r['r_mix']-without,
                            meaning='positive means removal improves; inference ablation only'))
        groups_by_split[split]=tables
        for name in GROUPS:
            selected=[r for r in features if r['split']==split and r['group']==name]
            r=dict(**tags,split=split,group=name,query_channels=32*C,nonempty_groups=sum(x['count']>0 for x in selected),candidate_count=sum(x['count'] for x in selected))
            for k in FIELDS:r[k+'_mean_over_nonempty_qc']=mean_or_none([x[k+'_mean'] if x[k+'_mean'] is not None else np.nan for x in selected])
            summary.append(r)
    for a,b in [('MissedOracle','CorrectOracle'),('MissedOracle','FalseSelected')]:
        for k in ['zero','best','gain','recent_difference','candidate_std','mse']:
            delta=groups_by_split['test'][a][k]-groups_by_split['test'][b][k]
            for scale in [1,2,4]:cis.append(dict(**tags,comparison=a+' minus '+b,metric=k,split='test',**paired_ci(delta,starts_by_split['test'],H*scale)))
    # Thresholds are selected from validation past-only TopR features, independently
    # of all test labels. Explicit median and upper-quartile cutoffs, no search.
    thresholds={};strata=[]
    for key in ['gain','recent_difference']:
        vv=np.stack([[np.mean(z[key][c,positions(z,c,z['ordered_ids'][c])]) for c in range(C)] for z in data['val']])
        thresholds[key]=np.quantile(vv,[.5,.75],axis=0).tolist()
    thresholds['high_gain_candidate_cutoff']=[]
    for c in range(C):thresholds['high_gain_candidate_cutoff'].append(float(np.quantile(np.concatenate([z['gain'][c,positions(z,c,z['ordered_ids'][c])] for z in data['val']]),.75)))
    for q,z in enumerate(data['test']):
        for c in range(C):
            ix=positions(z,c,z['ordered_ids'][c]);high=z['gain'][c,ix]>=thresholds['high_gain_candidate_cutoff'][c]
            strata.append(dict(**tags,query_index=int(z['query_index'][0]),channel=c,analysis='candidate_high_gain',count=int(high.sum()),
                oracle10_fraction=mean_or_none(np.isin(z['ordered_ids'][c,high],z['oracle_ids'][c,:10])),near_oracle_fraction=mean_or_none(z['near_mask'][c,ix][high])))
    metrics=[];stratified={}
    assert sha(ana/'D6_QUERY_CHANNEL_REGRET.jsonl.gz')==ag['files']['D6_QUERY_CHANNEL_REGRET.jsonl.gz']
    qlookup={int(z['query_index'][0]):z for z in data['test']}
    with gzip.open(ana/'D6_QUERY_CHANNEL_REGRET.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);z=qlookup[r['query_index']];c=r['channel'];look={int(i):float(d) for i,d,v in zip(z['global_ids'][c],z['mse'][c],z['valid'][c]) if v}
            add=rerank_metrics(r['selected_ids'],z['ordered_ids'][c,:10],z['oracle_ids'][c,:10],look)
            metrics.append(dict(**tags,query_index=r['query_index'],channel=c,R=r['R'],method=r['method'],retmse=r['reranked_retmse'],recall=r['recall'],**add))
            for key in ['gain','recent_difference']:
                value=float(z[key][c,positions(z,c,z['ordered_ids'][c])].mean());lo,hi=np.asarray(thresholds[key])[:,c]
                label='low_le_valmedian' if value<=lo else 'middle' if value<hi else 'high_ge_valq75'
                stratified.setdefault((r['R'],r['method'],key,label),[]).append(r)
    for (R,method,key,label),rows in stratified.items():
        strata.append(dict(**tags,analysis='query_channel_stratum',R=R,method=method,feature=key,stratum=label,count=len(rows),
            original_retmse=float(np.mean([x['original_retmse'] for x in rows])),reranked_retmse=float(np.mean([x['reranked_retmse'] for x in rows])),
            recall=float(np.mean([x['recall'] for x in rows])),win_rate=float(np.mean([x['win'] for x in rows]))))
    aggregate=[]
    for R,method in sorted({(r['R'],r['method']) for r in metrics}):
        rows=[r for r in metrics if r['R']==R and r['method']==method];r=dict(**tags,R=R,method=method,query_channels=len(rows))
        for k in ['retmse','recall','ndcg10','missed_oracle_recovery_rate','false_selected_promotion_rate','false_selected_promoted_fraction_of_k']:
            vals=[x[k] for x in rows if x[k] is not None];r[k]=mean_or_none(vals);r[k+'_defined_qc']=len(vals)
        r['pooled_recovery_rate']=sum(x['missed_oracle_recovered_count'] for x in rows)/sum(x['initial_missed_count'] for x in rows) if sum(x['initial_missed_count'] for x in rows) else None
        aggregate.append(r)
    # D5 matched-size controls: paired blocks on each timestamp with all channels.
    d5ci=[];p=ana/'D5_SAME_SIZE_PAIRED_CONTROLS.csv'
    if p.exists():
        assert sha(p)==ag['files'][p.name]
        with p.open() as f:rr=list(csv.DictReader(f))
        qmap={int(z['query_index'][0]):i for i,z in enumerate(data['test'])}
        for group in ['Oracle10','Oracle30','NearOracle']:
            for field in ['unique_lag_delta','entropy_delta','shift_gain_delta']:
                delta=np.full((32,C),np.nan)
                for r in rr:
                    if r['split']=='test' and r['group']==group:delta[qmap[int(r['query_index'])],int(r['channel'])]=float(r[field])
                for scale in [1,2,4]:d5ci.append(dict(**tags,group=group,metric=field,**paired_ci(delta,starts_by_split['test'],H*scale)))
    families={'D4_SHIFT_GROUP_COMPARISON':summary,'D4_SHIFT_FEATURE_QUERY_CHANNEL':features,'D4_SHIFT_RELEVANCE_ANALYSIS':relevance,
        'D4_GROUP_PAIRED_CI':cis,'D4_SHIFT_RERANKING_PROBE':aggregate,'D4_RERANK_QUERY_CHANNEL':metrics,'D4_VALIDATION_FIXED_STRATA':strata,
        'D5_SAME_SIZE_BLOCK_CI':d5ci,'D1_HEAD_UNIQUE_USEFUL_CANDIDATES':headrows,'D2_FUSION_DISCOVERY_REGRET':fusion,
        'D2_LOST_GOOD_CANDIDATE_TRACES':traces,'D2_HEAD_LEAVE_ONE_OUT':loo}
    for name,rows in families.items():
        if rows:csvout(dest/(name+'.csv'),rows)
    dump(dest/'D4_STRATA_THRESHOLDS.json',dict(source='validation only; all 32 pre-fixed timestamps/all channels; past features only',thresholds=thresholds))
    dump(dest/'SAMPLE_STATISTICS_GATE.json',dict(status='PASS_D4_SAMPLE_STATISTICS_AND_D12_SAMPLE_ADDITIONS',at=now(),identity=tags,
        source_feature_gate_sha256=sha(src/'FEATURE_GATE.json'),source_analysis_gate_sha256=sha(ana/'ANALYSIS_GATE.json'),queries={'val':32,'test':32},channels=C,
        D1_D2_full_population_complete=False,D3_normalizer_not_computed_here=True,D4_NDCG_gain='1/(1+MSE), direct gain; log2 discount; full evaluation-support ideal top10',
        recovery_definition='recovered original missed Oracle10 / original missed count; undefined when none',
        promotion_definition='newly admitted non-Oracle candidates / all newly admitted candidates; undefined when none; fraction of K also saved',
        paired_CI='H/2H/4H, seed0,2000 bootstrap,95% percentile; less than4 blocks insufficient',
        group_empty_not_zero=True,seconds=time.perf_counter()-tick,files={p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='SAMPLE_STATISTICS_GATE.json'}))
    print('D14 SAMPLE STATISTICS PASS',cell,model,training,evaluation,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cell');p.add_argument('model');p.add_argument('training');p.add_argument('evaluation');a=p.parse_args();main(a.cell,a.model,a.training,a.evaluation)
