"""CPU-only, artifact-verified Mechanism02 cell aggregation.

Native comparisons and same-checkpoint support interventions are separate rows.
Core completion never implies that missing supplementary analyses are complete.
"""
import argparse,csv,json,os,sys,time
from pathlib import Path
import numpy as np
from protocol import ROOT,MODELS
from io_utils import read,dump,sha,now,source_hashes
from artifact_checks import validated_blocks,concatenate,strict_head_choice,geometry_choice,finite_float
from mechanism_statistics import sensitivity

OUT=ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02'

def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def csv_rows(path):return list(csv.DictReader(Path(path).open()))
def means(x):return np.nanmean(x,axis=(0,1))
def match_fraction(a,b):return (a[...,None]==b[...,None,:]).any(-1).mean(-1)

def aggregate(cell,smoke=False):
    start=time.perf_counter();ds,h=cell.rsplit('_',1);h=int(h)
    parent=OUT/('analysis_smoke' if smoke else 'checkpoint_runs')/cell
    dest=OUT/('summary_smoke' if smoke else 'cells')/cell;dest.mkdir(parents=True,exist_ok=True)
    tables={k:[] for k in ['CORE_COMPARISON','CHANNEL_METRICS','CROSS_SUPPORT','REGRET_DECOMPOSITION','REGRET_TAILS',
      'MISSED_ORACLE','NORMALIZER_CONTROL','RANKING_QUALITY','HEAD_UTILITY','HEAD_UNIQUE_RELEVANCE',
      'GEOMETRY_VALIDATION_SELECTION','GEOMETRY_TEST','HEAD_CLUSTER_COVERAGE','FEATURE_PAIRED',
      'PROBE_COMPARISON','PAIRED_TIME_BLOCK_CI','METHOD_OVERLAP','WEATHER_RECALL_DIAGNOSTIC','HEAD_PROBABILITY_MASS']}
    sources={};native={};all_checkpoint_metrics={};common_queries={};pending=[];resources=[]
    reused=csv_rows(OUT/'00_Audit/REUSED_BASELINE_ROWS.csv')
    reused=[r for r in reused if r['cell']==cell and r['split']=='test']
    baseline=[]
    for support in ('P100','FULL'):
        group=[r for r in reused if r['support']==support];assert len(group)==6
        for r in group:
            assert sha(r['source_csv'])==r['source_sha256']
            baseline.append(dict(cell=cell,support=support,method=r['method'],retmse=r['retmse'],agg_mse=r['agg_mse'],agg_mae=r['agg_mae'],
                                 recall10=r['recall'],selection='deterministic/fixed prior baseline',source=r['source_csv']))
        r=group[0]
        baseline.append(dict(cell=cell,support=support,method='Last-repeat',retmse=None,agg_mse=r['last_mse'],agg_mae=r['last_mae'],recall10=None,
                             selection='last observed input repeated over all horizon',source=r['source_csv']))
    for support in ('P100','FULL'):
      for model in MODELS:
        base=parent/(model+'_'+support);gate=read(base/'REPRODUCTION_GATE.json')
        assert gate['status']=='PASS' and gate['identity']['source_hashes']==source_hashes()
        proof=gate['identity'];source=gate['source'];sources[model+'_'+support]=dict(checkpoint=source['checkpoint'],sha256=proof['checkpoint_sha256'],
          reproduction_sha256=sha(base/'REPRODUCTION_GATE.json'),selection=source.get('selection_criterion',source.get('selection','original validation checkpoint; unchanged')),
          original_runtime=source['original_runtime'])
        identity=dict(cell=cell,model=model,training_support=support,checkpoint_sha256=proof['checkpoint_sha256'])
        splits={s:validated_blocks(base,s,smoke) for s in ('val','test')}
        for s,(_,q,g) in splits.items():
            if s in common_queries:np.testing.assert_array_equal(q,common_queries[s])
            else:common_queries[s]=q
        paths,q,sg=splits['test'];vpaths,vq,vg=splits['val'];C=sg['channels'];Q=len(q)
        with np.load(paths[0]) as z:
            metric_names=z['metric_names'].tolist();keys=z.files
            feature_names=z['feature_names'].tolist();geom_names=z['geometry_fields'].tolist()
        variants=[k[:-len('__metrics')] for k in keys if k.endswith('__metrics')]
        assert {'FULL','P100_CONTROLLED','P100_LOCAL_NORMALIZER','P100_FULL_NORMALIZER'}<=set(variants)
        sm={};ids={}
        for variant in variants:
            m=concatenate(paths,variant+'__metrics');sm[variant]=m
            selected=concatenate(paths,variant+'__selected_ids');ids[variant]=selected
            assert np.isfinite(m[...,:7]).all() and np.all(m[...,5]>=-2e-6)
            np.testing.assert_allclose(m[...,5],m[...,6],atol=1e-10,rtol=1e-10)
            row=dict(**identity,variant=variant,queries=Q,channels=C,units=Q*C,**dict(zip(metric_names,means(m))))
            tables['CROSS_SUPPORT'].append(row)
            for c in range(C):tables['CHANNEL_METRICS'].append(dict(**identity,variant=variant,channel=c,queries=Q,**dict(zip(metric_names,np.nanmean(m[:,c],axis=0)))))
            miss=concatenate(paths,variant+'__missed_ranks');ranks=miss[miss>0]
            r=dict(**identity,variant=variant,missed_count=int(len(ranks)),pooled_mean=finite_float(ranks.mean()) if len(ranks) else None,
              pooled_median=finite_float(np.median(ranks)) if len(ranks) else None,pooled_p90=finite_float(np.quantile(ranks,.9)) if len(ranks) else None)
            for lo,hi in [(11,20),(21,50),(51,100),(101,500),(501,1000),(1001,10**9)]:r[f'rank_{lo}_{hi}_fraction']=float(((ranks>=lo)&(ranks<=hi)).mean()) if len(ranks) else None
            tables['MISSED_ORACLE'].append(r)
            if variant!='FULL':
                reg=concatenate(paths,variant+'__regret_decomposition');np.testing.assert_allclose(reg[...,0]+reg[...,1],reg[...,2],atol=1e-10,rtol=1e-10)
                av=means(reg);den=av[2]
                tables['REGRET_DECOMPOSITION'].append(dict(**identity,variant=variant,pool=av[0],ranking=av[1],total=av[2],
                  pool_share=av[0]/den if abs(den)>1e-12 else None,ranking_share=av[1]/den if abs(den)>1e-12 else None,
                  full_oracle_coverage100=match_fraction(concatenate(paths,'FULL__oracle_ids'),concatenate(paths,'pool_ids')).mean(),
                  recall_to_full_oracle10=match_fraction(selected,concatenate(paths,'FULL__oracle_ids')).mean()))
                t=reg.mean(1)
                for j,name in enumerate(['pool','ranking','total']):
                    a=t[:,j];total=a.sum();r=dict(**identity,variant=variant,regret=name,unit='timestamp channel-macro',mean=a.mean(),median=np.median(a),p90=np.quantile(a,.9),p95=np.quantile(a,.95),p99=np.quantile(a,.99))
                    for frac in (.01,.05,.1):
                        count=int(np.ceil(Q*frac));r[f'worst_{frac}_share']=np.sort(a)[-count:].sum()/total if abs(total)>1e-12 else None
                    tables['REGRET_TAILS'].append(r)
        nat='P100_NATIVE' if support=='P100' else 'FULL';m=sm[nat];native[(support,model)]=m
        all_checkpoint_metrics[(support,model)]={v:a[...,:7] for v,a in sm.items()}
        row=dict(**identity,support=support,method=model,queries=Q,channels=C,retmse=m[...,0].mean(),agg_mse=m[...,1].mean(),agg_mae=m[...,2].mean(),recall10=m[...,3].mean(),
                 selection='original validation-selected checkpoint; not reselected',checkpoint=source['checkpoint'])
        if not smoke:
            expected=next(r for r in gate['rows'] if r['split']=='test')['metrics'];np.testing.assert_allclose([row[k] for k in ['retmse','agg_mse','agg_mae','recall10']],expected,atol=2e-6,rtol=1e-5)
        tables['CORE_COMPARISON'].append(row)
        # Strict same-logit, same-FP64-arithmetic denominator intervention.
        local=sm['P100_LOCAL_NORMALIZER'];restricted=sm['P100_FULL_NORMALIZER']
        a=ids['P100_LOCAL_NORMALIZER'];b=ids['P100_FULL_NORMALIZER'];changed=(a!=b).any(-1)
        if model!='V5':assert not changed.any(),'single-head denominator-only ranking must not change'
        mass=concatenate(paths,'head_mass');weights=concatenate(paths,'implicit_head_weight')
        for head in range(mass.shape[-1]):
            tables['HEAD_PROBABILITY_MASS'].append(dict(**identity,head=head+1,mass_mean=mass[...,head].mean(),mass_p05=np.quantile(mass[...,head],.05),
              mass_p95=np.quantile(mass[...,head],.95),implicit_weight_mean=weights[...,head].mean(),definition='Z_P100/Z_Full; exact same logits'))
        tables['NORMALIZER_CONTROL'].append(dict(**identity,changed_order_fraction=changed.mean(),same_set_fraction=(match_fraction(a,b)==1).mean(),
           restricted_minus_local_retmse=(restricted[...,0]-local[...,0]).mean(),mass_min=mass.min(),mass_mean=mass.mean(),mass_max=mass.max(),
           implicit_weight_min=weights.min(),implicit_weight_max=weights.max(),arithmetic='both FP64 logsumexp; identical logits/mask/tau/IDs'))
        for variant in ('FULL','P100_CONTROLLED'):
            if support=='P100' and variant=='P100_CONTROLLED':
                tables['METHOD_OVERLAP'].append(dict(**identity,comparison='controlled Full-logit restriction vs native P100 path',overlap10=match_fraction(ids[variant],ids['P100_NATIVE']).mean(),ordered_equal_fraction=(ids[variant]==ids['P100_NATIVE']).all(-1).mean()))
            if variant+'__group_feature_sum' in keys:
                sums=concatenate(paths,variant+'__group_feature_sum');counts=concatenate(paths,variant+'__group_feature_count')
                avg=np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>0)
                delta=avg[:,:,1]-avg[:,:,2]
                for label,st in sensitivity(delta,h):
                    for j,name in enumerate(feature_names):
                        tables['FEATURE_PAIRED'].append(dict(**identity,variant=variant,feature=name,comparison='MissedOracle minus FalseSelected per Q/C',block_rule=label,
                          mean=st['mean'][j],ci_low=st['ci_low'][j],ci_high=st['ci_high'][j],standardized_paired_effect=st['mean'][j]/np.nanstd(delta[...,j]) if np.nanstd(delta[...,j])>1e-20 else None,sufficient_blocks=st['sufficient'],effective_blocks=st['effective_blocks'],
                          queries=Q,channels=C,finite_units=int(np.isfinite(delta[...,j]).sum()),exploratory=True))
            # Geometry rules learned on validation, never selected by test.
            for key in keys:
                prefix=variant+'__geometry__'
                if not key.startswith(prefix) or '_oracle_' not in key or not key.endswith('_stats'):continue
                valpos=concatenate(vpaths,key);valrandom=concatenate(vpaths,key.replace('_oracle_','_random_'));chosen=geometry_choice(valpos,valrandom)
                label=key[len(prefix):-len('_stats')];tables['GEOMETRY_VALIDATION_SELECTION'].append(dict(**identity,variant=variant,space=label,selected_k=chosen,
                  validation_only=True,rule='silhouette .25, stability .8, min size3, advantage .10; else null'))
                for group in ('oracle','random','past_nearest'):
                    values=concatenate(paths,key.replace('_oracle_',f'_{group}_'))
                    labs=concatenate(paths,key.replace('_oracle_',f'_{group}_')[:-len('_stats')]+'_labels')
                    sizes={} if chosen==1 else {f'cluster_{j}_mean_count':float((labs[:,:,chosen-2]==j).sum(-1).mean()) for j in range(chosen)}
                    tables['GEOMETRY_TEST'].append(dict(**identity,variant=variant,space=label,group=group,selected_k=chosen,
                       verdict='NO_CLUSTER' if chosen==1 else 'DESCRIPTIVE_VALIDATION_SELECTED_GEOMETRY',**sizes,**dict(zip(geom_names,means(values)))))
                if model=='V5' and chosen>1:
                    size=int(label.split('_')[0]);labels=concatenate(paths,key[:-len('_stats')]+'_labels')[:,:,chosen-2]
                    oracleids=concatenate(paths,prefix+f'{size}_oracle_ids');heads=concatenate(paths,variant+'__head_ids')
                    hit=(heads[...,None]==oracleids[:,:,None,None,:]).any(-2)
                    for head in range(5):
                        cov=np.stack([((labels==k)&hit[:,:,head]).any(-1) for k in range(chosen)],-1).mean(-1)
                        tables['HEAD_CLUSTER_COVERAGE'].append(dict(**identity,variant=variant,space=label,head=head+1,selected_k=chosen,mean_fraction=cov.mean(),
                          definition='oracle-positive clusters containing at least one retrieved exact oracle-positive ID; outside-positive candidates unassigned'))
                    ah=ids[variant];best=strict_head_choice(concatenate(vpaths,variant+'__head_metrics'))
                    for label,ch in [('All5',ah),('ValidationBestSingle',heads[:,:,best])]:
                        got=(ch[...,None]==oracleids[:,:,None,:]).any(-2)
                        cov=np.stack([((labels==k)&got).any(-1) for k in range(chosen)],-1).mean(-1)
                        tables['HEAD_CLUSTER_COVERAGE'].append(dict(**identity,variant=variant,space=key[len(prefix):-len('_stats')],head=label,selected_k=chosen,mean_fraction=cov.mean(),
                          definition='same positive set; fixed K10; best head from validation only'))
            if model=='V5':
                hm=concatenate(paths,variant+'__head_metrics');vh=concatenate(vpaths,variant+'__head_metrics');best=strict_head_choice(vh)
                loo=concatenate(paths,variant+'__loo_metrics');union=concatenate(paths,variant+'__union_deployable_metrics');uo=concatenate(paths,variant+'__union_oracle_metrics')
                posthoc=int(hm[...,0].mean((0,1)).argmin())
                choices=[('All5',sm[variant]),('ValidationBestSingle',hm[:,:,best]),('POSTHOC_ORACLE_HEAD',hm[:,:,posthoc]),('DeployableUnionK10',union),('OracleUnionK10_DIAGNOSTIC',uo)]
                choices += [(f'H{i+1}',hm[:,:,i]) for i in range(5)]+[(f'WithoutH{i+1}',loo[:,:,i]) for i in range(5)]
                for name,values in choices:
                    tables['HEAD_UTILITY'].append(dict(**identity,variant=variant,method=name,validation_selected_head=best+1,retmse=values[...,0].mean(),agg_mse=values[...,1].mean(),
                     agg_mae=values[...,2].mean(),recall10=values[...,3].mean(),ndcg10=values[...,7].mean(),oracle_regret=values[...,5].mean(),
                     delta_ret_vs_all5=(values[...,0]-sm[variant][...,0]).mean(),equal_k=10,equal_compute=False))
                heads=concatenate(paths,variant+'__head_ids');mse=concatenate(paths,variant+'__head_selected_mse');oi=concatenate(paths,variant+'__oracle_ids')
                th=concatenate(paths,variant+'__near_optimal_threshold')
                for head in range(5):
                    others=np.concatenate([heads[:,:,j] for j in range(5) if j!=head],-1)
                    unique=~(heads[:,:,head,:,None]==others[:,:,None]).any(-1);oraclehit=(heads[:,:,head,:,None]==oi[:,:,None]).any(-1)
                    tables['HEAD_UNIQUE_RELEVANCE'].append(dict(**identity,variant=variant,head=head+1,unique_count=unique.sum(-1).mean(),
                     unique_near_optimal_count=(unique&(mse[:,:,head]<=th[...,None])).sum(-1).mean(),exclusive_oracle_count=(unique&oraclehit).sum(-1).mean(),
                     pair_overlap=concatenate(paths,variant+'__head_overlap').mean(),union_size=concatenate(paths,variant+'__union_size').mean(),
                     signed_error_cosine=concatenate(paths,variant+'__head_error_cosine')[:,:,head,[j for j in range(5) if j!=head]].mean()))
        for split,(ps,_,_) in splits.items():
            for p in ps:
                rp=p.with_name(p.stem+'_ranking.json');assert rp.exists()
                for r in read(rp):tables['RANKING_QUALITY'].append({**identity,**r})
        complete=read(base/('ANALYSIS_SMOKE_GATE.json' if smoke else 'CORE_COMPLETE.json'))
        resources.append(dict(**identity,wall_seconds=complete['seconds'],peak_allocated_bytes=complete['peak_allocated'],peak_reserved_bytes=complete['peak_reserved'],
          shared_gpu=True,scope=complete.get('timing_scope','diagnostic worker, excludes preceding reproduction and controller waits'),
          **{'phase_'+k+'_wall_seconds':v for k,v in complete.get('phase_wall_seconds',{}).items()}))
    # Paired architecture comparisons retain all channels at a timestamp together.
    for support in ('P100','FULL'):
        for a,b in [('V1','V0'),('V5','V0'),('V5','V1')]:
            delta=native[(support,a)][...,:4]-native[(support,b)][...,:4]
            for label,st in sensitivity(delta,h):
                for i,metric in enumerate(['RetMSE','AggMSE','AggMAE','Recall']):
                    tables['PAIRED_TIME_BLOCK_CI'].append(dict(cell=cell,support=support,comparison=a+' minus '+b,metric=metric,block_rule=label,
                      mean=st['mean'][i],ci_low=st['ci_low'][i],ci_high=st['ci_high'][i],effective_blocks=st['effective_blocks'],sufficient_blocks=st['sufficient'],
                      queries=st['queries'],channels=st['channels'],repeats=2000,seed=0,exploratory=True,seed_uncertainty_estimated=False))
    tables['CORE_COMPARISON'].extend(baseline if not smoke else [])
    pg=read(OUT/'00_Audit/FROZEN_PROBE_REUSE_GATE.json');assert pg['status']=='PASS'
    for p in [x for x in pg['rows'] if x['cell']==cell]:
        f=p['files']['PAST_ONLY_PROBE_RESULTS.csv'];assert sha(f['path'])==f['sha256']
        test=next(r for r in csv_rows(f['path']) if r['split']=='test')
        for model in MODELS:
            tables['PROBE_COMPARISON'].append(dict(cell=cell,support=p['support'],model=model,probe_ret=float(test['retmse']),
              model_ret=native[(p['support'],model)][...,0].mean(),probe_minus_model=float(test['retmse'])-native[(p['support'],model)][...,0].mean(),
              train_queries=p['train_queries'],channels=p['channels'],train_only=True,selection='frozen validation-selected ridge; reused; no refit',source=f['path']))
    # Remaining feature strata/sentinel inference is explicitly closed separately.
    pending=['input-only validation-fixed strata with new V1 outcomes',
             'Weather sentinel/channel exclusion sensitivity (official metrics unchanged)',
             'cross-model same-support disagreement/near-oracle overlap interpretation',
             'dataset-level attribution/recommendations and complete figure inventory']
    if smoke:pending=['SMOKE_ONLY_NOT_OFFICIAL_RESULTS']
    for name,rows in tables.items():write_csv(dest/(name+'.csv'),rows)
    write_csv(dest/'RESOURCE_SUMMARY.csv',resources)
    # CPU supplement consumes only existing saved IDs/metrics and past inputs.
    # This call applies to already-running controllers as well as future ones.
    from supplement import run as supplement_run
    supplement_run(cell,smoke)
    assert read(dest/'CPU_SUPPLEMENT_GATE.json')['status']==('SMOKE_PASS' if smoke else 'PASS')
    pending=['dataset-level attribution/recommendations and complete figure inventory'] if not smoke else ['SMOKE_ONLY_NOT_OFFICIAL_RESULTS']
    dump(dest/'SOURCES.json',dict(at=now(),checkpoints=sources,source_hashes=source_hashes(),queries={k:len(v) for k,v in common_queries.items()},smoke=smoke))
    dump(dest/'AGGREGATION_GATE.json',dict(status='SMOKE_PASS' if smoke else 'CORE_AGGREGATION_PASS_SUPPLEMENT_PENDING',at=now(),cell=cell,source_hashes=source_hashes(),
      files={p.name:sha(p) for p in dest.glob('*.csv')},checkpoint_count=6,all_v1_present=True,core_full_population=not smoke,pending=pending,seconds=time.perf_counter()-start))
    from deliverables import finish
    finish(cell,smoke)
    print('AGGREGATION',cell,'SMOKE_PASS' if smoke else 'CORE_PASS_SUPPLEMENT_PENDING',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cell');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='';aggregate(a.cell,a.smoke)
