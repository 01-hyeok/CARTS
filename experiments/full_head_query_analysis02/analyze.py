import os,sys,json,csv,itertools,time,argparse,gc
from pathlib import Path
import numpy as np,pandas as pd
from core import NAMES,SUBSETS,FEATURES,features,bins,block_ci,delta_decomposition,dump,sha,comparison_events
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/'results/TRACK-V5-FULL-HEAD-ANALYSIS01';OUT=ROOT/'results/TRACK-V5-FULL-HEAD-QUERY-ANALYSIS02';PAIRS=list(itertools.combinations(range(5),2))
def table(p,rows):pd.DataFrame(rows).to_csv(p,index=False,float_format='%.9f')
def macro(a,mask=None):
 if mask is None:return float(np.mean(a,dtype=np.float64))
 mask=mask & np.isfinite(a);n=mask.sum(0);v=n>0
 return float((np.where(mask,a,0).sum(0)[v]/n[v]).mean()) if v.any() else float('nan')
def run(cell):
 p=OUT/cell;loc=p/'local';assert json.loads((p/'EXTRACTION_PASS.json').read_text())['status']=='PASS';t0=time.time();h=int(cell.split('_')[1]);old=pd.read_csv(OLD/cell/'HEAD_STAGE2_MSE_MAE.csv',dtype={'subset':str});prov=json.loads((p/'PROVENANCE.json').read_text());lam=prov['fixed_lambda'];rows=[]
 for split in ['val','test']:
  ids=np.load(loc/(split+'_query_ids.npy'));B=np.load(loc/(split+'_base.npy'),mmap_mode='r');Y=np.load(loc/(split+'_truth.npy'),mmap_mode='r');R=[np.load(OLD/cell/'cache'/name/(split+'.npy'),mmap_mode='r') for name in NAMES];n,_,c=B.shape
  metrics=np.lib.format.open_memmap(loc/(split+'_forecast_metrics.npy'),mode='w+',dtype='float64',shape=(n,c,11,4));baseline=np.zeros((n,c,2));average=np.zeros((n,c,4));pair=np.lib.format.open_memmap(loc/(split+'_signed_pairs.npy'),mode='w+',dtype='float32',shape=(n,c,10,6));bias=np.lib.format.open_memmap(loc/(split+'_head_signed_bias.npy'),mode='w+',dtype='float32',shape=(n,c,5,2));decomp=np.lib.format.open_memmap(loc/(split+'_loo_decomposition.npy'),mode='w+',dtype='float64',shape=(n,c,5,3))
  lams=np.array([float(old[old['subset']==name].iloc[0].primary_lambda) for name in NAMES])
  for lo in range(0,n,32):
   hi=min(lo+32,n);b=np.asarray(B[lo:hi],dtype='float64').transpose(0,2,1);y=np.asarray(Y[lo:hi],dtype='float64').transpose(0,2,1);r=np.stack([np.asarray(z[lo:hi],dtype='float64').transpose(0,2,1) for z in R],2);baseerr=b-y;fixed=b[:,:,None]+lam*(r-b[:,:,None]);err=fixed-y[:,:,None];refit=b[:,:,None]+lams[None,None,:,None]*(r-b[:,:,None]);referr=refit-y[:,:,None]
   metrics[lo:hi]=np.stack([(err**2).mean(-1),abs(err).mean(-1),(referr**2).mean(-1),abs(referr).mean(-1)],-1);baseline[lo:hi]=np.stack([(baseerr**2).mean(-1),abs(baseerr).mean(-1)],-1)
   avg=r[:,:,:5].mean(2);ae=avg-y;afe=b+lam*(avg-b)-y;average[lo:hi]=np.stack([(ae**2).mean(-1),abs(ae).mean(-1),(afe**2).mean(-1),abs(afe).mean(-1)],-1)
   s1=r[:,:,:5]-y[:,:,None];s2=err[:,:,:5];bias[lo:hi]=np.stack([s1.mean(-1),s2.mean(-1)],-1)
   for k,(i,j) in enumerate(PAIRS):
    vals=[]
    for e in (s1,s2):
     dot=(e[:,:,i]*e[:,:,j]).mean(-1);den=np.sqrt((e[:,:,i]**2).mean(-1)*(e[:,:,j]**2).mean(-1));cos=np.divide(dot,den,out=np.zeros_like(dot),where=den>1e-12);opp=(e[:,:,i]*e[:,:,j]<0).mean(-1);vals.extend([dot,cos,opp])
    pair[lo:hi,:,k]=np.stack(vals,-1)
   for k in range(5):
    change=r[:,:,6+k]-r[:,:,5];cross,energy=delta_decomposition(err[:,:,5],change,lam);observed=(err[:,:,6+k]**2).mean(-1)-(err[:,:,5]**2).mean(-1);assert np.max(abs(cross+energy-observed))<1e-9;decomp[lo:hi,:,k]=np.stack([cross,energy,observed],-1)
  metrics.flush();pair.flush();bias.flush();decomp.flush();np.save(loc/(split+'_base_metrics.npy'),baseline);np.save(loc/(split+'_simple_prediction_average_metrics.npy'),average)
  for j,name in enumerate(NAMES):
   z=old[old['subset']==name].iloc[0];a=metrics[:,:,j].mean((0,1));rows.append(dict(cell=cell,split=split,subset=name,fixed_mse=a[0],fixed_mae=a[1],existing_refit_mse=a[2],existing_refit_mae=a[3],fixed_lambda=lam,existing_refit_lambda=lams[j],n_queries=n,n_channels=c))
   if split=='test':assert abs(a[0]-z.secondary_mse)<1e-5 and abs(a[1]-z.secondary_mae)<1e-5 and abs(a[2]-z.primary_mse)<1e-5 and abs(a[3]-z.primary_mae)<1e-5
  del metrics,R,B,Y,pair,bias,decomp;gc.collect();print('[CPU METRICS]',cell,split,flush=True)
 table(p/'FORECAST_SUMMARY.csv',rows)
 val=np.load(loc/'val_forecast_metrics.npy',mmap_mode='r');test=np.load(loc/'test_forecast_metrics.npy',mmap_mode='r');base=np.load(loc/'test_base_metrics.npy');n,c=test.shape[:2];qids=np.load(loc/'test_query_ids.npy');vfeatures=np.load(loc/'val_input_features.npy');tfeatures=np.load(loc/'test_input_features.npy');cuts,tbins=bins(vfeatures,tfeatures);np.save(loc/'test_strata.npy',tbins);np.save(loc/'validation_feature_cuts.npy',cuts)
 table(p/'FEATURE_CUTS.csv',[dict(cell=cell,channel=ch,feature=f,validation_q33=cuts[ch,k,0],validation_q67=cuts[ch,k,1],fit_split='validation_history_only') for ch in range(c) for k,f in enumerate(FEATURES)])
 globalrows=[];stratarows=[];reference=test[:,:,5,0];vbins=(vfeatures[...,None]>cuts[None]).sum(-1);sorted_ids=np.sort(np.load(loc/'test_top10_ids.npy',mmap_mode='r'),axis=-1)
 for j,name in enumerate(NAMES):
  if j==5:continue
  delta=test[:,:,j,0]-reference;vdelta=val[:,:,j,0]-val[:,:,5,0];ci=block_ci(delta,block=2*h);cih=block_ci(delta,block=h);changed_set,gain,loss=comparison_events(delta,sorted_ids[:,:,j],sorted_ids[:,:,5])
  if j<6:ci['family_low']=None;ci['family_high']=None
  globalrows.append(dict(cell=cell,subset=name,comparison='configuration minus All5; positive favors All5',fixed_mse_delta=ci.pop('mean'),fixed_mae_delta=macro(test[:,:,j,1]-test[:,:,5,1]),refit_mse_delta=macro(test[:,:,j,2]-test[:,:,5,2]),unchanged_set_fraction=macro(~changed_set),unchanged_set_max_abs_mse_delta=float(abs(delta[~changed_set]).max()) if (~changed_set).any() else 0.,all5_query_macro_improvement_fraction=float(((delta.mean(1)>0)&changed_set.any(1)).mean()),all5_improvement_fraction=macro(gain),all5_worsening_fraction=macro(loss),all5_improvement_size_if_improved=macro(delta,gain),all5_worsening_size_if_worse=-macro(delta,loss),h_block_ci_low=cih['ci_low'],h_block_ci_high=cih['ci_high'],multiplicity_family='20 global LOO MSE contrasts' if j>=6 else 'descriptive single-head contrast',**ci))
  for fi,feat in enumerate(FEATURES):
   for level in range(3):
    mask=tbins[:,:,fi]==level;vmask=vbins[:,:,fi]==level;stats=block_ci(delta,mask,block=2*h,reps=1000,family=1);stats.pop('family_low');stats.pop('family_high');stratarows.append(dict(cell=cell,subset=name,feature=feat,stratum=level,validation_equal_channel_macro_delta=macro(vdelta,vmask),validation_test_direction_agrees=bool(np.sign(macro(vdelta,vmask))==np.sign(macro(delta,mask))),equal_channel_macro_delta=stats.pop('mean'),all5_improvement_fraction=macro(gain,mask),all5_mse=macro(reference,mask),configuration_mse=macro(test[:,:,j,0],mask),**stats))
 table(p/'GLOBAL_HEAD_CONTRIBUTION.csv',globalrows);table(p/'INPUT_STRATA_EXPLORATORY.csv',stratarows)
 chosen=[]
 for fi,feat in enumerate(FEATURES):
  for level in range(3):
   vm=vbins[:,:,fi]==level;tm=tbins[:,:,fi]==level;best=int(np.nanargmin([macro(val[:,:,j,0],vm) for j in range(5)]));chosen.append(dict(cell=cell,feature=feat,stratum=level,validation_selected_head=best+1,validation_head_mse=macro(val[:,:,best,0],vm),test_head_mse=macro(test[:,:,best,0],tm),test_all5_mse=macro(reference,tm),test_head_minus_all5=macro(test[:,:,best,0]-reference,tm),n_test_query_channels=int(tm.sum()),n_channels=int(tm.any(0).sum()),selection='validation only; exploratory strata, not a new routing model'))
 table(p/'STRATUM_VALIDATION_HEAD_SELECTION.csv',chosen)
 # Fixed-lambda failure/success overlap; thresholds fixed on validation per channel/head.
 threshold=np.quantile(val[:,:,:5,0],.75,axis=0);fail=test[:,:,:5,0]>threshold[None];success=test[:,:,:5,0]<base[:,:,None,0];signed=np.load(loc/'test_signed_pairs.npy',mmap_mode='r');pairs=[]
 for k,(i,j) in enumerate(PAIRS):
  for kind,events in [('failure_above_val_q75',fail),('success_vs_base',success)]:
   both=events[:,:,i]&events[:,:,j];union=events[:,:,i]|events[:,:,j];a=signed[:,:,k]
   pairs.append(dict(cell=cell,head1=i+1,head2=j+1,event=kind,joint_fraction=macro(both),jaccard=macro(both,union),union_query_channels=int(union.sum()),n_queries=n,n_channels=c,stage1_mean_signed_dot=macro(a[:,:,0]),stage1_error_cosine=macro(a[:,:,1]),stage1_opposite_sign_fraction=macro(a[:,:,2]),fixed_stage2_mean_signed_dot=macro(a[:,:,3]),fixed_stage2_error_cosine=macro(a[:,:,4]),fixed_stage2_opposite_sign_fraction=macro(a[:,:,5])))
 np.save(loc/'failure_thresholds_val_q75.npy',threshold);table(p/'HEAD_FAILURE_SUCCESS_SIGNED.csv',pairs)
 # Candidate replacements, quality and exact delta decomposition by changed-set size.
 swap=np.load(loc/'test_candidate_swaps.npy',mmap_mode='r');dec=np.load(loc/'test_loo_decomposition.npy',mmap_mode='r');swrows=[]
 for j in range(5):
  count=swap[:,:,j,0];quality=swap[:,:,j,1]-swap[:,:,j,2];delta=test[:,:,6+j,0]-reference
  for group,mask in [('all',np.ones_like(count,dtype=bool)),('unchanged',count==0),('1_to_3', (count>=1)&(count<=3)),('4_to_10',count>=4)]:
   swrows.append(dict(cell=cell,removed_head=j+1,change_group=group,n_query_channels=int(mask.sum()),n_channels=int(mask.any(0).sum()),mean_replaced_count=macro(count,mask),incoming_minus_outgoing_individual_mse=macro(quality,mask),incoming_quality_better_fraction=macro(quality<0,mask&(count>0)),fixed_stage2_mse_delta=macro(delta,mask),fixed_stage2_mae_delta=macro(test[:,:,6+j,1]-test[:,:,5,1],mask),existing_refit_mse_delta=macro(test[:,:,6+j,2]-test[:,:,5,2],mask),linear_alignment_term=macro(dec[:,:,j,0],mask),quadratic_change_term=macro(dec[:,:,j,1],mask)))
 table(p/'LOO_CANDIDATE_REPLACEMENTS.csv',swrows)
 # 31-subset saturation, all criteria distinct; no new Stage2 fits.
 vq=np.load(OLD/cell/'val_subset_query_metrics.npy',mmap_mode='r');tq=np.load(OLD/cell/'test_subset_query_metrics.npy',mmap_mode='r');assert tq.shape[:2]==(n,c) and tq.shape[2]==31
 sat=[]
 for size in range(1,6):
  idx=[k for k,s in enumerate(SUBSETS) if len(s)==size];winner=idx[int(np.argmin(vq[:,:,idx,0].mean((0,1))))];post=idx[int(np.argmin(tq[:,:,idx,0].mean((0,1))))];oracle=tq[:,:,idx,0].min(-1);ci=block_ci(tq[:,:,winner,0]-tq[:,:,30,0],block=2*h,family=1)
  sat.append(dict(cell=cell,head_count=size,n_combinations=len(idx),test_mean_over_combinations=float(tq[:,:,idx,0].mean(dtype='float64')),validation_selected_subset=SUBSETS[winner],validation_selected_test_mse=float(tq[:,:,winner,0].mean(dtype='float64')),validation_selected_test_mae=float(tq[:,:,winner,1].mean(dtype='float64')),test_posthoc_best_fixed_subset=SUBSETS[post],test_posthoc_fixed_mse=float(tq[:,:,post,0].mean(dtype='float64')),test_query_channel_oracle_mse=float(oracle.mean(dtype='float64')),test_query_channel_oracle_mae_independent=float(tq[:,:,idx,1].min(-1).mean(dtype='float64')),selected_minus_all5_mse=ci.pop('mean'),**ci))
 table(p/'HEAD_COUNT_SATURATION.csv',sat)
 avg=np.load(loc/'test_simple_prediction_average_metrics.npy');ac=block_ci(avg[:,:,2]-reference,block=2*h,family=1);table(p/'MIXTURE_VS_PREDICTION_AVERAGE.csv',[dict(cell=cell,actual_meanprob_top10_stage1_mse=float(tq[:,:,30,0].mean(dtype='float64')),simple_head_prediction_average_stage1_mse=macro(avg[:,:,0]),actual_meanprob_fixed_stage2_mse=macro(reference),simple_head_prediction_average_fixed_stage2_mse=macro(avg[:,:,2]),simple_average_minus_actual_fixed_mse=ac.pop('mean'),**ac)])
 query=dict(query_start_idx=qids,base_mse=base[:,:,0].mean(1),all5_fixed_mse=reference.mean(1))
 for j,name in enumerate(NAMES):query['subset_'+name+'_fixed_mse']=test[:,:,j,0].mean(1)
 pd.DataFrame(query).to_csv(loc/'test_query_channel_macro_errors.csv',index=False)
 dump(p/'QUERY_ARRAY_SCHEMA.json',dict(subset_order=NAMES,features=FEATURES,forecast_metrics=['fixed_mse','fixed_mae','existing_refit_mse','existing_refit_mae'],candidate_swap_metrics=['replaced_count','incoming_candidate_individual_mse','outgoing_candidate_individual_mse'],pairs=[[i+1,j+1] for i,j in PAIRS],signed_pairs=['stage1_mean_dot','stage1_cosine','stage1_opposite_sign_fraction','fixed_stage2_mean_dot','fixed_stage2_cosine','fixed_stage2_opposite_sign_fraction'],id_unit='top10 IDs index the unchanged full memory; candidate_start_ids.npy maps to global train starts',local_arrays_only=True,strata_bootstrap_reps=1000,global_bootstrap_reps=5000,win_rule='identical candidate sets count as ties despite float32 summation order residual; absolute metrics retain canonical predictions',aggregation='equal channel macro within each stratum; time blocks sampled jointly across channels'))
 dump(p/'ANALYSIS_PASS.json',dict(status='PASS',cell=cell,queries=n,channels=c,block_length=2*h,n_time_blocks=int(np.ceil(n/(2*h))),seed=0,existing_stage2_refit_only=True,new_training=False,wall_seconds=time.time()-t0,val_subset_hash=sha(OLD/cell/'val_subset_query_metrics.npy'),test_subset_hash=sha(OLD/cell/'test_subset_query_metrics.npy')));print('[ANALYSIS COMPLETE]',cell,flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--cell',required=True);run(a.parse_args().cell)
