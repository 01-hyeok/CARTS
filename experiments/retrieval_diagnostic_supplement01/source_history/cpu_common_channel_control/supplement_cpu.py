"""Complete input-only strata and explicitly qualify all existing intervals."""
import argparse, itertools
import numpy as np
from supplement_common import *

FEATURES=['query_std','query_trend_slope','query_volatility','query_local_change','query_low_frequency_power_ratio','query_dominant_period']
METRICS=['retmse','ranking_regret','recall','agg_mse','agg_mae','spearman','past10_future_dispersion']

def run(support,cell):
    src=SOURCE/support/cell;dest=OUT/support/cell
    complete=read(src/'COMPLETE.json');assert complete['status'].startswith('COMPLETE')
    schemas={s:read(src/'aggregate'/f'{s}_schema.json') for s in ['val','test']}
    groups={s:np.load(src/'aggregate'/f'{s}_groups.npy',mmap_mode='r') for s in schemas}
    metrics={s:np.load(src/'aggregate'/f'{s}_metrics.npy',mmap_mode='r') for s in schemas}
    rows=[];macro=[];cutrows=[];overlap=[];cirows=[]
    for feature in FEATURES:
        # The random-candidate group is always nonempty. These query features
        # depend only on history, independent of oracle/false-selection labels.
        values={s:np.asarray(groups[s][:,:,0,3,schemas[s]['features'].index(feature)],np.float64) for s in schemas}
        cuts=thresholds(values['val'])
        for c,cut in enumerate(cuts):cutrows.append(dict(cell=cell,support=support,feature=feature,channel=c,lower=float(cut[0]),upper=float(cut[1]),source='validation history only',tie_rule='<=lower low; <=upper middle; else high; NaN undefined'))
        for split,schema in schemas.items():
            labels=strata(values[split],cuts);m=metrics[split]
            assert labels.shape==(complete['query_counts'][split],complete['channels'])
            for mi,name in enumerate(schema['methods']):
                for label,value in [('low',0),('middle',1),('high',2),('undefined',-1)]:
                    channel_means={k:[] for k in METRICS}
                    for c in range(schema['channels']):
                        take=labels[:,c]==value;row=dict(cell=cell,support=support,split=split,feature=feature,stratum=label,method=name,channel=c,query_count=int(take.sum()))
                        for key in METRICS:
                            a=np.asarray(m[:,c,mi,schema['metrics'].index(key)])[take];a=a[np.isfinite(a)]
                            row[key]=float(a.mean(dtype=np.float64)) if len(a) else None
                            if len(a):channel_means[key].append(row[key])
                        rows.append(row)
                    macro.append(dict(cell=cell,support=support,split=split,feature=feature,stratum=label,method=name,aggregation='equal channel mean; nonempty channels only, counts explicit',channels_represented=len(channel_means['retmse']),total_channels=schema['channels'],query_channel_units=int((labels==value).sum()),**{k:float(np.mean(v)) if v else None for k,v in channel_means.items()},ci_status='DESCRIPTIVE_ONLY_NO_SIGNIFICANCE_CLAIM'))
    for split,schema in schemas.items():
        sums={};count=0;prev=None
        for f in sorted((src/split).glob('batch[0-9]*.npz')):
            with np.load(f) as a:
                ids=a['selected_ids'];q=a['query_start'];assert ids.shape[1]==complete['channels']
                assert np.all(np.diff(q)==1) and (prev is None or q[0]==prev+1);prev=int(q[-1]);count+=len(q)
                for i,j in itertools.combinations(range(len(schema['methods'])),2):
                    z=(ids[:,:,i,:,None]==ids[:,:,j,None,:]).any(-1).mean(-1)
                    sums[i,j]=sums.get((i,j),0.)+z.sum(dtype=np.float64)
        assert count==schema['queries']==complete['query_counts'][split]
        for (i,j),s in sums.items():overlap.append(dict(cell=cell,support=support,split=split,method_a=schema['methods'][i],method_b=schema['methods'][j],top10_overlap=s/(count*schema['channels']),queries=count,channels=schema['channels']))
    for filename in ['BASELINE_WIN_RATE.csv','FEATURE_GROUP_COMPARISON.csv']:
        for row in csvread(src/filename):
            status,effective=interval_status(int(row['query_times']),int(row['block']))
            row.update(source_file=str(src/filename),original_ci_low=row.get('ci_low'),original_ci_high=row.get('ci_high'),ci_status=status,effective_nonoverlap_blocks=effective,usable_for_significance=False)
            if status=='INSUFFICIENT_TIME_BLOCKS':row['ci_low']=row['ci_high']=None
            cirows.append(row)
    probe=read(src/'probe_train.json');selection=read(src/'probe_selection.json')
    assert not probe['test_used_for_fit'] and not selection['test_used_for_selection']
    assert probe['channels']==complete['channels'] and probe['pairs']==probe['train_query_count']*complete['channels']*100
    assert len(set(probe['train_query_rows']))==probe['train_query_count']<=256
    csvwrite(dest/'INPUT_STRATA_THRESHOLDS.csv',cutrows);csvwrite(dest/'INPUT_STRATA_BY_CHANNEL.csv',rows)
    csvwrite(dest/'INPUT_STRATA_EQUAL_CHANNEL_MACRO.csv',macro);csvwrite(dest/'TOP10_METHOD_OVERLAP.csv',overlap)
    csvwrite(dest/'CI_VALIDITY_REVIEW.csv',cirows)
    dump(dest/'PROBE_SAMPLING_DISCLOSURE.json',dict(status='VERIFIED',train_query_sampling='fixed evenly spaced train-only indices; no future-based choice',train_query_count=probe['train_query_count'],all_channels=True,channels=probe['channels'],training_candidate_pairs=probe['pairs'],train_candidate_policy=probe['train_candidate_policy'],full_training_split_used=False,validation_test='all queries and channels',hyperparameters='Ridge alpha 0.1,1,10; validation regression MSE selection',test_selection=False,source=str(src/'probe_train.json'),sha256=sha(src/'probe_train.json')))
    dump(dest/'CPU_GATE.json',dict(status='PASS',source_complete_sha256=sha(src/'COMPLETE.json'),input_strata=True,all_query_channels=True,channel_composition_control='per-channel thresholds and channel-specific rows; equal nonempty-channel macro',bootstrap_policy='retain original 4H block, clipped to query count; effective blocks<2 means insufficient CI; no block chosen using outcomes',inference_details_status='PENDING_GPU_SUPPLEMENT',full_dependency='DEFERRED_FULL_DEPENDENCY' if support=='P100' else 'AVAILABLE'))
    print('CPU COMPLETED',support,cell,'strata rows',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('support',choices=['P100','FULL']);p.add_argument('cell',nargs='?');a=p.parse_args()
    for c in ([a.cell] if a.cell else CELLS):
        if (SOURCE/a.support/c/'COMPLETE.json').exists():run(a.support,c)
