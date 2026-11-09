"""CPU-only validation-frozen strata and Weather sensitivity from saved results."""
import argparse,os,csv
from pathlib import Path
import numpy as np
from protocol import ROOT,MODELS
from io_utils import read,dump,sha,now
from artifact_checks import validated_blocks,concatenate
from mechanism_statistics import sensitivity as block_sensitivity
OUT=ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02'
def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def match_fraction(a,b):return (a[...,None]==b[...,None,:]).any(-1).mean(-1)

FEATURES=['input_std','abs_trend_slope','diff_std','acf1','last_relative_change']

def input_features(x):
    x=np.asarray(x,np.float64);h=x.shape[-1];t=np.arange(h,dtype=float);t-=t.mean()
    z=x-x.mean(-1,keepdims=True);left=z[...,:-1];right=z[...,1:]
    den=np.sqrt((left*left).sum(-1)*(right*right).sum(-1))
    acf=np.divide((left*right).sum(-1),den,out=np.zeros_like(den),where=den>1e-20)
    return np.stack([x.std(-1),abs((z*t).sum(-1)/(t*t).sum()),np.diff(x,axis=-1).std(-1),acf,abs(x[...,-1]-x[...,0])],-1)

def freeze_strata(val):return np.quantile(val,[1/3,2/3],axis=0)
def apply_strata(test,cut):return (test>cut[0]).astype(np.int8)+(test>cut[1]).astype(np.int8)

def run(cell,smoke=False):
    os.chdir(ROOT)
    from adapter import Data,rt
    d=Data(cell,'cpu');dest=OUT/('summary_smoke' if smoke else 'cells')/cell
    parent=OUT/('analysis_smoke' if smoke else 'checkpoint_runs')/cell
    all_features={};qids={};last={}
    example=parent/'V0_P100'
    for split in ('val','test'):
        _,qids[split],g=validated_blocks(example,split,smoke);pieces=[];lm=[]
        local=qids[split]-d.datasets[split].starts[0]
        for a in range(0,len(local),128):
            # Use exact original split arrays. Noncontiguous smoke rows are explicit.
            ix=local[a:a+128,None]+np.arange(d.h)[None]
            x=np.asarray(d.datasets[split].data_x[ix],np.float32).transpose(0,2,1)
            y=np.asarray(d.datasets[split].data_y[ix+d.h],np.float32).transpose(0,2,1)
            pieces.append(input_features(x));err=x[:,:,-1,None]-y;lm.append(np.stack([(err**2).mean(-1),abs(err).mean(-1)],-1))
        all_features[split]=np.concatenate(pieces);last[split]=np.concatenate(lm)
    cuts=freeze_strata(all_features['val']);labels=apply_strata(all_features['test'],cuts)
    dump(dest/'INPUT_STRATA_FROZEN.json',dict(source_split='val',test_used=False,names=FEATURES,quantiles=[1/3,2/3],thresholds=cuts.tolist(),
          per_channel=True,seed=0,smoke=smoke))
    rows=[];near=[];ambiguity=[];sensitivity=[];overlap=[];calendar=[];baseline_wins=[];baseline_sources=[];metrics={};selected={};allpaths={}
    import pandas as pd
    source=rt.config()['data'][cell.rsplit('_',1)[0]];assert sha(source['path'])==source['sha256']
    dates=pd.to_datetime(pd.read_csv(source['path'],usecols=['date'])['date'])
    cal=np.stack([dates.dt.hour.to_numpy()+dates.dt.minute.to_numpy()/60,dates.dt.dayofweek.to_numpy(),dates.dt.month.to_numpy()],-1)
    qt=cal[qids['test']+d.h-1]
    # Train sentinel detection is diagnostic only; no data/scaler modification.
    train=np.asarray(d.datasets['train'].data_x,dtype=np.float32)
    means=np.asarray(d.data_info['scaler_mean']);scales=np.asarray(d.data_info['scaler_scale']);sentinel=np.asarray((-9999-means)/scales,np.float32)
    affected=(train==sentinel).any(0) if cell.startswith('Weather') else np.zeros(d.c,bool)
    dump(dest/'CHANNEL_SENSITIVITY_DEFINITION.json',dict(policy='train exact normalized -9999 match; diagnostic only, official all-channel metrics unchanged',
      channels=d.data_info.get('channel_names'),affected_indices=np.flatnonzero(affected).tolist(),train_counts=(train==sentinel).sum(0).tolist() if cell.startswith('Weather') else [],
      scaler_mean=means.tolist(),scaler_scale=scales.tolist()))
    for support in ('P100','FULL'):
      for model in MODELS:
        base=parent/(model+'_'+support);paths,q,_=validated_blocks(base,'test',smoke);vpaths,_,_=validated_blocks(base,'val',smoke)
        np.testing.assert_array_equal(q,qids['test']);allpaths[(support,model)]=paths
        for variant in ('FULL','P100_CONTROLLED')+ (('P100_NATIVE',) if support=='P100' else ()):
            m=concatenate(paths,variant+'__metrics');metrics[(support,model,variant)]=m
            ids=concatenate(paths,variant+'__selected_ids');selected[(support,model,variant)]=ids
            oi=concatenate(paths,variant+'__oracle_ids');hit=(ids[...,None]==oi[:,:,None]).any(-1);found=(oi[...,None]==ids[:,:,None]).any(-1)
            group_values=[]
            for group,candidate,valid in [('CorrectOracle',oi,found),('MissedOracle',oi,~found),('FalseSelected',ids,~hit)]:
                ct=cal[candidate+d.h-1];qcal=qt[:,None,None,:]
                f=np.stack([np.cos((ct[...,0]-qcal[...,0])*2*np.pi/24),(ct[...,1]==qcal[...,1]),((ct[...,1]>=5)==(qcal[...,1]>=5)),(ct[...,2]==qcal[...,2])],-1).astype(float)
                count=valid.sum(-1);avg=np.divide((f*valid[...,None]).sum(-2),count[...,None],out=np.full(f.shape[:2]+(4,),np.nan),where=count[...,None]>0)
                group_values.append(avg)
                for j,name in enumerate(['hour_cyclic_similarity','weekday_equal','weekend_equal','month_equal']):
                    calendar.append(dict(cell=cell,training_support=support,model=model,variant=variant,group=group,feature=name,mean=np.nanmean(avg[...,j]),
                      finite_units=int(np.isfinite(avg[...,j]).sum()),source_file_sha256=source['sha256'],timestamp='history final observed point',future_input=False))
            delta=group_values[1]-group_values[2]
            for label,st in block_sensitivity(delta,d.h):
                for j,name in enumerate(['hour_cyclic_similarity','weekday_equal','weekend_equal','month_equal']):
                    calendar.append(dict(cell=cell,training_support=support,model=model,variant=variant,group='paired Missed minus FalseSelected',feature=name,mean=st['mean'][j],
                      ci_low=st['ci_low'][j],ci_high=st['ci_high'][j],block_rule=label,sufficient_blocks=st['sufficient'],source_file_sha256=source['sha256'],future_input=False))
            if variant in ('FULL','P100_CONTROLLED'):
                od=concatenate(paths,variant+'__oracle100_mse');sd=concatenate(paths,variant+'__selected_mse');nc=concatenate(paths,variant+'__near_optimal_count')
                threshold=concatenate(paths,variant+'__near_optimal_threshold')
                relation=np.corrcoef(m[...,0].ravel(),m[...,3].ravel())[0,1]
                near.append(dict(cell=cell,training_support=support,model=model,variant=variant,near_optimal_count=nc.mean(),selected_near_fraction=(sd<=threshold[...,None]).mean(),
                  exact_recall10=m[...,3].mean(),oracle_kth_mse=od[...,9].mean(),selected_mse_p50=np.median(sd),selected_mse_p90=np.quantile(sd,.9),
                  selected_minus_oracle_kth=(sd-od[...,9,None]).mean(),query_channel_recall_ret_correlation=relation,oracle_boundary_gap=m[...,16].mean()))
                vr=concatenate(vpaths,variant+'__metrics')[...,5];regcut=np.quantile(vr,[1/3,2/3],axis=0)
                strata=(m[...,5]>regcut[0]).astype(int)+(m[...,5]>regcut[1]).astype(int)
                dispersion=concatenate(paths,variant+'__geometry__10_past_nearest_future_dispersion')
                for group in range(3):
                    mask=strata==group
                    ambiguity.append(dict(cell=cell,training_support=support,model=model,variant=variant,regret_group=group,selection='per-channel validation regret terciles; diagnostic future-defined strata',
                       units=int(mask.sum()),ranking_regret=m[...,5][mask].mean(),past_nearest_future_variance=dispersion[mask].mean(),
                       description='observed ambiguity, no irreducibility or causal claim'))
            for label,keep in [('all_channels',np.ones(d.c,bool)),('train_sentinel_channels',affected),('other_channels',~affected)]:
                if label!='all_channels' and not cell.startswith('Weather'):continue
                if not keep.any():continue
                avg=m[:,keep].mean((0,1));sensitivity.append(dict(cell=cell,training_support=support,model=model,variant=variant,group=label,channels=int(keep.sum()),
                   retmse=avg[0],agg_mse=avg[1],agg_mae=avg[2],recall10=avg[3],ranking_regret=avg[5],official=label=='all_channels'))
        native='P100_NATIVE' if support=='P100' else 'FULL';m=metrics[(support,model,native)]
        for fi,feature in enumerate(FEATURES):
            for group in range(3):
                channelmeans=[]
                for c in range(d.c):
                    keep=labels[:,c,fi]==group
                    if not keep.any():continue
                    avg=m[keep,c].mean(0);channelmeans.append(avg[:4])
                    rows.append(dict(cell=cell,support=support,model=model,feature=feature,stratum=group,channel=c,queries=int(keep.sum()),
                      retmse=avg[0],agg_mse=avg[1],agg_mae=avg[2],recall10=avg[3],aggregation='within-channel'))
                if channelmeans:
                    av=np.asarray(channelmeans).mean(0);rows.append(dict(cell=cell,support=support,model=model,feature=feature,stratum=group,channel='equal-channel-macro',queries=None,
                      retmse=av[0],agg_mse=av[1],agg_mae=av[2],recall10=av[3],aggregation='equal channels with nonempty stratum'))
    for support in ('P100','FULL'):
        for variant in ('FULL','P100_CONTROLLED'):
            for a,b in [('V1','V0'),('V5','V0'),('V5','V1')]:
                ma=metrics[(support,a,variant)];mb=metrics[(support,b,variant)]
                overlap.append(dict(cell=cell,training_support=support,variant=variant,comparison=a+' vs '+b,
                  top10_overlap=match_fraction(selected[(support,a,variant)],selected[(support,b,variant)]).mean(),
                  ret_win_fraction=(ma[...,0]<mb[...,0]).mean(),ret_tie_fraction=(ma[...,0]==mb[...,0]).mean(),
                  mean_ret_difference=(ma[...,0]-mb[...,0]).mean(),difference_p05=np.quantile(ma[...,0]-mb[...,0],.05),difference_p95=np.quantile(ma[...,0]-mb[...,0],.95)))
        old=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01'/support/cell
        schema=read(old/'aggregate/test_schema.json');oldq=np.load(old/'aggregate/test_starts.npy')
        arr=np.load(old/'aggregate/test_metrics.npy',mmap_mode='r');idx=qids['test']-oldq[0]
        np.testing.assert_array_equal(oldq[idx],qids['test']);assert arr.shape[:2]==(len(oldq),d.c)
        with (old/'RETRIEVAL_BASELINE_COMPARISON.csv').open() as f:oldtable=list(csv.DictReader(f))
        indices=[schema['metrics'].index(k) for k in ['retmse','agg_mse','agg_mae','recall']]
        for method in ['Raw Cosine','Delta-last Cosine','Pearson','PAA-16 Cosine','Random','IndividualMSE_Oracle']:
            mi=schema['methods'].index(method);bm=np.take(np.asarray(arr[:,:,mi]),indices,axis=-1)
            target=next(r for r in oldtable if r['split']=='test' and r['method']==method)
            np.testing.assert_allclose(bm.mean((0,1),dtype=np.float64),[float(target[k]) for k in ['retmse','agg_mse','agg_mae','recall']],atol=2e-6,rtol=1e-5)
            for model in MODELS:
                native='P100_NATIVE' if support=='P100' else 'FULL';mm=metrics[(support,model,native)][...,:4];difference=mm-bm[idx]
                baseline_wins.append(dict(cell=cell,support=support,model=model,baseline=method,queries=len(idx),channels=d.c,
                  ret_win_fraction=(difference[...,0]<0).mean(),ret_tie_fraction=(difference[...,0]==0).mean(),ret_mean_difference=difference[...,0].mean(),
                  agg_win_fraction=(difference[...,1]<0).mean(),agg_mean_difference=difference[...,1].mean(),smoke_only=smoke))
        baseline_sources.append(dict(support=support,files={n:sha(old/n) for n in ['aggregate/test_schema.json','aggregate/test_starts.npy','aggregate/test_metrics.npy','RETRIEVAL_BASELINE_COMPARISON.csv']}))
    for name,value in [('INPUT_STRATA_BY_CHANNEL',rows),('NEAR_ORACLE_CANDIDATE_ANALYSIS',near),('WEATHER_CHANNEL_SENSITIVITY',sensitivity),
                       ('AMBIGUITY_NEW_MODELS',ambiguity),('MODEL_DISAGREEMENT',overlap),('CALENDAR_FEATURES',calendar),('BASELINE_QUERY_WIN_RATE',baseline_wins)]:write_csv(dest/(name+'.csv'),value)
    dump(dest/'BASELINE_QUERY_REUSE_GATE.json',dict(status='PASS',sources=baseline_sources,query_ids_aligned=True,full_array_means_reproduce_csv=True,smoke_only=smoke))
    # Last-repeat is explicitly the same population and horizon, not a retrieval arm.
    dump(dest/'LAST_REPEAT_RECHECK.json',dict(status='SMOKE_PASS' if smoke else 'PASS',queries=len(qids['test']),channels=d.c,
      agg_mse=float(last['test'][...,0].mean()),agg_mae=float(last['test'][...,1].mean()),normalized_space=True,all_horizon=True))
    dump(dest/'CPU_SUPPLEMENT_GATE.json',dict(status='SMOKE_PASS' if smoke else 'PASS',at=now(),cell=cell,full_population=not smoke,
      v1_included_both_training_supports=True,queries=len(qids['test']),channels=d.c,files={p.name:sha(p) for p in dest.glob('*.csv')}))
    print('CPU SUPPLEMENT',cell,'SMOKE_PASS' if smoke else 'PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cell');p.add_argument('--smoke',action='store_true');a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']='';run(a.cell,a.smoke)
