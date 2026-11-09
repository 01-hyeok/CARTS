"""Full D5/D6 CPU aggregation from immutable, identity-gated feature extension."""
import argparse,csv,gzip,json,os,time,datetime,hashlib
from pathlib import Path
import numpy as np
from d56_metrics import *
from d56_cpu_index_cache import positions_cached
INDEX_LOOKUP_SHA256='e4341eac96721657a67d7edc28b42e67caaff10f38ba20bc19623672f9201ee4'
assert hashlib.sha256(Path(__file__).with_name('d56_cpu_index_cache.py').read_bytes()).hexdigest()==INDEX_LOOKUP_SHA256
OUT=Path('/data/pjh_workspace/CARTS/results/TRACK-HEAD-SHIFT-DIAGNOSTIC01/extensions_d5_d6')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(p)
def csvout(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r));t=p.with_suffix('.tmp')
    with t.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    t.replace(p)

def load(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}

def positions(row,c,ids):
    return positions_cached(row,c,ids)

def group_masks(row,c,model):
    ids=row['global_ids'][c];valid=row['valid'][c];eligible=row['eligible'][c];near=row['near_mask'][c]
    o=row['oracle_ids'][c];s=row['ordered_ids'][c,:10];rnd=row['random_ids'][c]
    assert int(near.sum())==int(row['near_support_count'][c])
    groups={'Oracle10':np.isin(ids,o[:10])&valid,'Oracle30':np.isin(ids,o)&valid,'NearOracle':near,
       'CorrectOracle':np.isin(ids,o[:10])&np.isin(ids,s)&valid,
       'MissedOracle':np.isin(ids,o[:10])&~np.isin(ids,s)&valid,
       'FalseSelected':np.isin(ids,s)&~np.isin(ids,o[:10])&valid,
       'Random10':np.isin(ids,rnd[:10])&valid,'Random30':np.isin(ids,rnd[:30])&valid,
       'RandomNearCount':np.isin(ids,rnd[:int(near.sum())])&valid,
       'FullOracleOutsideP100':np.isin(ids,row['full_oracle_ids'][c])&~eligible&valid}
    assert groups['Oracle10'].sum()==10 and groups['Oracle30'].sum()==30
    assert groups['MissedOracle'].sum()==groups['FalseSelected'].sum()
    if model=='V5':
        heads=row['head_ids'][c];assert heads.shape==(5,10)
        for h in range(5):
            hs=np.isin(ids,heads[h])&valid;others=np.delete(heads,h,0).ravel()
            groups['H%d_all'%(h+1)]=hs;groups['H%d_oracle'%(h+1)]=hs&groups['Oracle10']
            groups['H%d_near'%(h+1)]=hs&near
            groups['H%d_unique_useful'%(h+1)]=hs&near&~np.isin(ids,others)
            # Same support/size, greedy nearest Future-MSE diagnostic control.
            # This control is never a retrieval score or candidate preselector.
            target=np.flatnonzero(hs&near);available=list(np.flatnonzero(eligible&~hs));chosen=[]
            for k in target:
                if not available:break
                j=min(available,key=lambda j:(abs(float(row['mse'][c,j])-float(row['mse'][c,k])),int(ids[j])))
                chosen.append(j);available.remove(j)
            match=np.zeros(len(ids),bool);match[chosen]=True;groups['H%d_mse_matched_control'%(h+1)]=match
    return groups

def block_ci(delta,starts,length):
    v=np.asarray(delta).mean(1);g=(starts-starts.min())//length;chunks=[v[g==k] for k in np.unique(g)]
    rng=np.random.RandomState(0);est=[]
    for _ in range(2000):est.append(np.concatenate([chunks[i] for i in rng.randint(0,len(chunks),len(chunks))]).mean())
    return dict(delta=float(v.mean()),ci_low=float(np.quantile(est,.025)),ci_high=float(np.quantile(est,.975)),blocks=len(chunks),sufficient=len(chunks)>=4,
       block_length=length,repeats=2000,confidence=.95,sampling='paired timestamps/all channels jointly')

def main(cell,model,training,evaluation):
    src=OUT/'complete_features'/cell/(model+'_'+training+'_to_'+evaluation)
    gate=json.loads((src/'FEATURE_GATE.json').read_text());assert gate['status']=='PASS_COMPLETE_D5_D6_FEATURES_AGGREGATION_PENDING'
    dest=OUT/'complete_analysis'/cell/(model+'_'+training+'_to_'+evaluation);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'ANALYSIS_GATE.json').exists():
        g=json.loads((dest/'ANALYSIS_GATE.json').read_text());assert g['source_feature_gate_sha256']==sha(src/'FEATURE_GATE.json')
        for p,h in g['files'].items():assert sha(dest/p)==h
        print('REUSED_VALID',cell,model,training,evaluation,flush=True);return
    begin=time.perf_counter();splits={};provenance=gate['identity'];C=gate['channels'];horizon=int(cell.rsplit('_',1)[1])
    for split in ['val','test']:
        rows=[]
        for p in sorted((src/split).glob('query*.npz')):
            assert sha(p)==gate['files'][str(p.relative_to(src))]
            meta=json.loads(p.with_suffix('.json').read_text());assert meta['identity']==provenance and meta['all_channels_single_batch']
            z=load(p);assert z['global_ids'].shape[0]==C and np.array_equal(z['valid'].sum(1),[len(np.unique(z['global_ids'][c,z['valid'][c]])) for c in range(C)])
            for shard,hh in meta['lag_shard_hashes'].items():assert sha(OUT/shard)==hh
            rows.append(z)
        assert len(rows)==gate['queries'][split]==32;splits[split]=rows
    tags=dict(cell=cell,model=model,training_support=training,evaluation_support=evaluation,variant='NATIVE' if training==evaluation else 'CROSS_SUPPORT',checkpoint_sha256=provenance['checkpoint']['checkpoint_sha256'])
    allstats={};hist={};headret=[];paired_controls=[]
    p=dest/'D5_QUERY_CHANNEL_GROUPS.jsonl.gz';tmp=p.with_suffix('.tmp')
    with gzip.open(tmp,'wt') as stream:
        for split,rows in splits.items():
            for row in rows:
                for c in range(C):
                    masks=group_masks(row,c,model);summaries={}
                    for name,m in masks.items():
                        stat=lag_summary(*[row[k][c,m] for k in ['best_lag','gain','margin','entropy','candidate_std']])
                        if stat['count']:
                            stat['near_tie_lag_fraction_mean']=float(row['near_tie_lag_fraction'][c,m].mean())
                            stat['group_future_mse']=float(row['mse'][c,m].astype(float).mean())
                        summaries[name]=stat;allstats.setdefault((split,name),[]).append(stat)
                        hs=hist.setdefault((split,name),{})
                        for k,v in stat.get('best_lag_histogram',{}).items():hs[k]=hs.get(k,0)+v
                        stream.write(json.dumps(dict(**tags,split=split,query_index=int(row['query_index'][0]),query_start=int(row['query_start'][0]),channel=c,group=name,**stat),allow_nan=False)+'\n')
                    for a,b in [('Oracle10','Random10'),('Oracle30','Random30'),('NearOracle','RandomNearCount')]:
                        x,y=summaries[a],summaries[b];assert x['count']==y['count']
                        if x['count']:paired_controls.append(dict(**tags,split=split,query_index=int(row['query_index'][0]),channel=c,group=a,control=b,count=x['count'],
                            unique_lag_delta=x['unique_best_lags']-y['unique_best_lags'],entropy_delta=x['useful_candidate_lag_entropy']-y['useful_candidate_lag_entropy'],shift_gain_delta=x['shift_gain_mean']-y['shift_gain_mean']))
                    if model=='V5':
                        mix=float(row['mse'][c,positions(row,c,row['ordered_ids'][c,:10])].astype(float).mean())
                        for head in range(5):
                            hs=positions(row,c,row['head_ids'][c,head]);mse=float(row['mse'][c,hs].astype(float).mean())
                            headret.append(dict(**tags,split=split,query_index=int(row['query_index'][0]),channel=c,head=head+1,all5_retmse=mix,head_retmse=mse,
                                unique_useful_count=int(masks['H%d_unique_useful'%(head+1)].sum()),
                                unique_useful_lag_entropy=summaries['H%d_unique_useful'%(head+1)].get('useful_candidate_lag_entropy'),
                                mean_shift_gain=float(row['gain'][c,hs].mean())))
    tmp.replace(p)
    grouped=[];heads=[];stability=[]
    for (split,name),stats in allstats.items():
        valid=[s for s in stats if s['count']];numeric=[k for k,v in valid[0].items() if isinstance(v,(int,float))] if valid else []
        row=dict(**tags,split=split,group=name,queries=32,channels=C,query_channel_groups=len(stats),nonempty_groups=len(valid),
                 metric_aggregation='mean over nonempty query-channel groups; not candidate-pooled',
                 histogram=json.dumps(hist[(split,name)],sort_keys=True),sampling='32 fixed timestamps/all channels')
        for k in numeric:row[k]=float(np.mean([s[k] for s in valid]))
        (heads if name.startswith('H') else grouped).append(row)
    for name in {k[1] for k in hist}:
        stability.append(dict(**tags,group=name,comparison='validation-test',**hist_distance(hist[('val',name)],hist[('test',name)])))
    if model=='V5':
        for split in ['val','test']:
            for suffix in ['all','oracle','near','unique_useful','mse_matched_control']:
                for a in range(1,6):
                    for b in range(a+1,6):stability.append(dict(**tags,split=split,group=suffix,comparison='H%d-H%d'%(a,b),**hist_distance(hist[(split,'H%d_%s'%(a,suffix))],hist[(split,'H%d_%s'%(b,suffix))])))
    for filename,values in [('D5_ORACLE_LAG_DIVERSITY',grouped),('D5_HEAD_LAG_SPECIALIZATION',heads),('D5_LAG_AMBIGUITY',grouped+heads),('D5_LAG_DISTRIBUTION_STABILITY',stability),('D5_SAME_SIZE_PAIRED_CONTROLS',paired_controls),('D5_HEAD_UTILITY_RELATION',headret)]:
        if values:csvout(dest/(filename+'.csv'),values)
    val,test=splits['val'],splits['test'];Rlist=[10,20,50,100,200,500] if evaluation=='FULL' else [10,20,50,100]
    Rlist=[r for r in Rlist if r<=min(z['ordered_ids'].shape[-1] for z in val+test)]
    def stacked(rows,key,R):
        return np.stack([np.stack([r[key][c,positions(r,c,r['ordered_ids'][c,:R])] for c in range(C)]) for r in rows])
    results=[];selection=[];cis=[];curves={};rankrows=[]
    for split,rows in splits.items():
        ranks=np.stack([r['oracle_initial_rank'][:,:10] for r in rows])
        for c in range(C):
            x=ranks[:,c].ravel();rankrows.append(dict(**tags,split=split,channel=c,oracle_count=len(x),mean=float(x.mean()),median=float(np.median(x)),p90=float(np.quantile(x,.9)),
                rank1_10=float(np.mean(x<=10)),rank11_20=float(np.mean((x>10)&(x<=20))),rank21_50=float(np.mean((x>20)&(x<=50))),rank51_100=float(np.mean((x>50)&(x<=100))),rank101_500=float(np.mean((x>100)&(x<=500))),rank501_plus=float(np.mean(x>500))))
    p=dest/'D6_QUERY_CHANNEL_REGRET.jsonl.gz';tmp=p.with_suffix('.tmp')
    with gzip.open(tmp,'wt') as stream:
        for R in Rlist:
            vv={k:stacked(val,k,R) for k in ['mse','gain','recent_difference']};vv['score']=np.stack([r['ordered_scores'][:,:R] for r in val])
            chosen={name:select_validation(vv['score'],vv['gain'],vv['recent_difference'],vv['mse'],name) for name in ['ShiftOnly','ShiftRecent']}
            for name,sel in chosen.items():selection.append(dict(R=R,method=name,**sel))
            for name in ['Original','ShiftOnly','ShiftRecent']:
                tick=time.perf_counter();unit=[];delta=np.zeros((32,C));channel_rows={c:[] for c in range(C)}
                for q,row in enumerate(test):
                    for c in range(C):
                        ids=row['ordered_ids'][c,:R];ix=positions(row,c,ids)
                        sel=dict(alpha=0.,beta=0.,score_scale=1.,recent_scale=1.) if name=='Original' else dict(**{k:chosen[name]['selected'][k] for k in ['alpha','beta']},score_scale=chosen[name]['score_scale'][c],recent_scale=chosen[name]['recent_scale'][c])
                        r=d6_unit(ids,row['mse'][c,ix],row['ordered_scores'][c,:R],row['gain'][c,ix],row['recent_difference'][c,ix],row['oracle_ids'][c,:10],row['oracle_mse'][c,:10],sel,int(row['near_support_count'][c]))
                        r['near_oracle_coverage']=float(row['near_mask'][c,ix].sum()/row['near_support_count'][c])
                        r.update(**tags,R=R,method=name,query_index=int(row['query_index'][0]),query_start=int(row['query_start'][0]),channel=c,
                            random_retmse=float(row['mse'][c,positions(row,c,row['random_ids'][c,:10])].astype(float).mean()))
                        if R==10:assert set(r['selected_ids'])==set(ids.tolist())
                        if name=='Original':curves.setdefault((q,c),[]).append(r)
                        unit.append(r);channel_rows[c].append(r);delta[q,c]=r['reranked_retmse']-r['original_retmse']
                        stream.write(json.dumps(r,allow_nan=False)+'\n')
                elapsed=time.perf_counter()-tick
                fields=['original_retmse','reranked_retmse','oracle_at_r','support_oracle_retmse','random_retmse','oracle10_coverage','near_oracle_coverage','candidate_limitation_regret','reranking_regret','total_regret','recoverable_headroom','recall','win','tie','loss']
                for c,values in [('ALL',unit)]+list(channel_rows.items()):
                    r=dict(**tags,R=R,method=name,split='test',channel=c,queries=32,channels=C if c=='ALL' else 1,
                        metric_aggregation='equal query-channel mean',sampling='32 fixed timestamps/all channels',runtime_cpu_reranking_seconds=elapsed if c=='ALL' else None,
                        peak_gpu_allocated_bytes=gate['peak_allocated_bytes'],peak_gpu_reserved_bytes=gate['peak_reserved_bytes'])
                    for k in fields:r[k]=float(np.mean([v[k] for v in values]))
                    if c=='ALL':
                        r['worst1pct_positive_deterioration_share']=loss_tail_contribution(delta,.01);r['worst5pct_positive_deterioration_share']=loss_tail_contribution(delta,.05)
                        r['timestamp_win_rate']=float(np.mean(delta.mean(1)<0));r['timestamp_tie_rate']=float(np.mean(delta.mean(1)==0))
                        # Retain the original query-channel rates separately.
                    results.append(r)
                if name!='Original':
                    starts=np.concatenate([z['query_start'] for z in test])
                    for scale in [1,2,4]:cis.append(dict(**tags,R=R,method=name,metric='reranked-original RetMSE',**block_ci(delta,starts,horizon*scale)))
    tmp.replace(p)
    for rows in curves.values():validate_r_curve(rows)
    for filename in ['D6_TOPR_ORACLE_COVERAGE','D6_TOPR_RESTRICTED_ORACLE','D6_SHIFT_RERANKING','D6_CANDIDATE_VS_RERANKING_REGRET']:csvout(dest/(filename+'.csv'),results)
    csvout(dest/'D6_ORACLE_INITIAL_RANK.csv',rankrows);csvout(dest/'D6_PAIRED_BLOCK_CI.csv',cis);dump(dest/'D6_VALIDATION_SELECTION.json',selection)
    dump(dest/'ANALYSIS_GATE.json',dict(status='PASS_D5_D6_CASE',at=now(),identity=tags,source_feature_gate_sha256=sha(src/'FEATURE_GATE.json'),
        aggregation_source_sha256=sha(__file__),integer_lookup_source_sha256=INDEX_LOOKUP_SHA256,
        integer_lookup_cache_only=True,floating_point_operations_changed=False,
        queries={'val':32,'test':32},channels=C,R=Rlist,all_named_groups_complete_on_sample=True,
        exact_near_support_count=True,Oracle30=True,topR_future_blind=True,validation_only_selection=True,
        query_channel_regret_identity=True,R10_selected_set_and_ret_exact=True,coverage_monotonic=True,oracle_R_monotonic=True,
        matched_mse_control_scope='all cached union candidates in same support, nearest future MSE; diagnostic only, not global optimal matching',
        low_activity_definition='candidate std<=1e-8; margin<=0.01 cosine ambiguity',
        limited_sample=True,negative_regret_clamp=False,seconds=time.perf_counter()-begin,
        files={p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='ANALYSIS_GATE.json'}))
    print('D56 CASE PASS',cell,model,training,evaluation,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cell');p.add_argument('model');p.add_argument('training');p.add_argument('evaluation');a=p.parse_args();main(a.cell,a.model,a.training,a.evaluation)
