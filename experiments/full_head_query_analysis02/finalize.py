import json,time,subprocess
from pathlib import Path
import numpy as np,pandas as pd
from core import NAMES,SUBSETS,FEATURES,block_ci,sha,dump
from analyze import macro,table
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/TRACK-V5-FULL-HEAD-QUERY-ANALYSIS02';OLD=ROOT/'results/TRACK-V5-FULL-HEAD-ANALYSIS01';CELLS=['ETTh1_96','ETTh1_720','Weather_96','Weather_720']
TABLES=['FORECAST_SUMMARY','GLOBAL_HEAD_CONTRIBUTION','INPUT_STRATA_EXPLORATORY','STRATUM_VALIDATION_HEAD_SELECTION','HEAD_FAILURE_SUCCESS_SIGNED','LOO_CANDIDATE_REPLACEMENTS','HEAD_COUNT_SATURATION','MIXTURE_VS_PREDICTION_AVERAGE','CHANNEL_STANDARDIZED_STRATA','CHANNEL_STANDARDIZED_VALIDATION_SELECTION','TIME_BLOCKS','CHANNEL_CONTRIBUTION','SIGNED_ERROR_ZERO_NORMS']
def read(cell,name):return pd.read_csv(OUT/cell/(name+'.csv'),dtype={'subset':str,'validation_selected_subset':str,'test_posthoc_best_fixed_subset':str})
def md(df):
 df=df.copy()
 def fmt(v):
  if isinstance(v,(float,np.floating)):return 'NA' if not np.isfinite(v) else f'{v:.6f}'
  return str(v)
 rows=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
 rows += ['| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)]
 return '\n'.join(rows)
def postcell(cell):
 p=OUT/cell;loc=p/'local';h=int(cell.split('_')[1]);bundle={name:sha(Path(__file__).parent/name) for name in ['core.py','analyze.py','finalize.py']};apsha=sha(p/'ANALYSIS_PASS.json')
 if (p/'POST_VERIFICATION.json').exists():
  prior=json.loads((p/'POST_VERIFICATION.json').read_text())
  if prior.get('source_bundle_sha256')==bundle and prior.get('analysis_pass_sha256')==apsha:
   print('[REUSE POST VERIFY]',cell,flush=True);return
 prov=json.loads((p/'PROVENANCE.json').read_text());ext=json.loads((p/'EXTRACTION_PASS.json').read_text());a=json.loads((p/'ANALYSIS_PASS.json').read_text());assert ext['status']==a['status']=='PASS'
 for key,filekey in [('checkpoint_sha256','checkpoint'),('base_sha256','base'),('dataset_sha256','dataset')]:assert sha(ROOT/prov[filekey])==prov[key]
 val=np.load(loc/'val_forecast_metrics.npy',mmap_mode='r');test=np.load(loc/'test_forecast_metrics.npy',mmap_mode='r');cut=np.load(loc/'validation_feature_cuts.npy');vf=np.load(loc/'val_input_features.npy');tf=np.load(loc/'test_input_features.npy');vb=(vf[...,None]>cut[None]).sum(-1);tb=(tf[...,None]>cut[None]).sum(-1);n,c=test.shape[:2];standard=[];chosen=[]
 for fi,feature in enumerate(FEATURES):
  common=np.logical_and.reduce([(vb[:,:,fi]==level).any(0)&(tb[:,:,fi]==level).any(0) for level in range(3)])
  for level in range(3):
   vm=(vb[:,:,fi]==level)&common;tm=(tb[:,:,fi]==level)&common
   if not common.any():continue
   for j,name in enumerate(NAMES):
    if j==5:continue
    d=test[:,:,j,0]-test[:,:,5,0];vd=val[:,:,j,0]-val[:,:,5,0];ci=block_ci(d,tm,block=2*h,reps=1000,family=1);ci.pop('family_low');ci.pop('family_high');vmean=macro(vd,vm);tmean=ci.pop('mean')
    standard.append(dict(cell=cell,feature=feature,stratum=level,subset=name,common_channels=';'.join(map(str,np.flatnonzero(common))),validation_mse_delta=vmean,test_mse_delta=tmean,direction_agrees=bool(np.sign(vmean)==np.sign(tmean)),**ci))
   j=int(np.nanargmin([macro(val[:,:,k,0],vm) for k in range(5)]));chosen.append(dict(cell=cell,feature=feature,stratum=level,common_channels=';'.join(map(str,np.flatnonzero(common))),n_channels=int(common.sum()),n_test_query_channels=int(tm.sum()),validation_selected_head=j+1,validation_head_minus_all5=macro(val[:,:,j,0]-val[:,:,5,0],vm),test_head_minus_all5=macro(test[:,:,j,0]-test[:,:,5,0],tm),selection='validation only, same channel set across all three strata'))
 table(p/'CHANNEL_STANDARDIZED_STRATA.csv',standard);table(p/'CHANNEL_STANDARDIZED_VALIDATION_SELECTION.csv',chosen)
 raw=pd.read_csv(prov['dataset']);dates=pd.to_datetime(raw['date']);channels=list(raw.columns[1:]);assert len(channels)==c
 ids=np.load(loc/'test_query_ids.npy');assert np.all(np.diff(ids)==1);blocks=[]
 for st in range(0,n,2*h):
  en=min(n,st+2*h);first=int(ids[st]);last=int(ids[en-1]);blocks.append(dict(cell=cell,block_index=len(blocks),n_queries=en-st,n_channels=c,query_start_first=first,query_start_last=last,history_start=str(dates.iloc[first]),first_forecast_start=str(dates.iloc[first+h]),last_forecast_end=str(dates.iloc[last+2*h-1]),full_block=en-st==2*h))
 table(p/'TIME_BLOCKS.csv',blocks)
 channelrows=[];channel_ids=np.sort(np.load(loc/'test_top10_ids.npy',mmap_mode='r'),axis=-1)
 for ch in range(c):
  for j,name in enumerate(NAMES):
   channelrows.append(dict(cell=cell,channel=ch,channel_name=channels[ch],subset=name,n_queries=n,fixed_mse=float(test[:,ch,j,0].mean()),fixed_mae=float(test[:,ch,j,1].mean()),mse_delta_vs_all5=float((test[:,ch,j,0]-test[:,ch,5,0]).mean()),query_win_fraction_vs_all5=float(((test[:,ch,j,0]<test[:,ch,5,0])&(channel_ids[:,ch,j]!=channel_ids[:,ch,5]).any(-1)).mean()),unchanged_candidate_set_fraction=float((channel_ids[:,ch,j]==channel_ids[:,ch,5]).all(-1).mean())))
 table(p/'CHANNEL_CONTRIBUTION.csv',channelrows)
 checks=[];zero=[]
 for split in ['val','test']:
  qids=np.load(loc/(split+'_query_ids.npy'));top=np.load(loc/(split+'_top10_ids.npy'),mmap_mode='r');sw=np.load(loc/(split+'_candidate_swaps.npy'),mmap_mode='r');Y=np.load(loc/(split+'_truth.npy'),mmap_mode='r');B=np.load(loc/(split+'_base.npy'),mmap_mode='r');oldq=np.load(OLD/cell/(split+'_subset_query_metrics.npy'),mmap_mode='r');fm=np.load(loc/(split+'_forecast_metrics.npy'),mmap_mode='r');dec=np.load(loc/(split+'_loo_decomposition.npy'),mmap_mode='r');countmem=len(np.load(loc/'candidate_start_ids.npy'));maxerr=0.;unchanged_residual=0.;zero_counts=np.zeros((5,2),dtype='int64');num=0
  assert top.shape==(len(qids),c,11,10) and oldq.shape[:3]==(len(qids),c,31)
  entering=np.lib.format.open_memmap(loc/(split+'_loo_entering_ids.npy'),mode='w+',dtype='int32',shape=(len(qids),c,5,10));leaving=np.lib.format.open_memmap(loc/(split+'_loo_leaving_ids.npy'),mode='w+',dtype='int32',shape=(len(qids),c,5,10))
  for st in range(0,len(qids),32):
   en=min(st+32,len(qids));z=top[st:en];assert (z>=0).all() and (z<countmem).all() and (np.diff(np.sort(z,axis=-1),axis=-1)>0).all();allids=z[:,:,5]
   for j in range(5):
    new=z[:,:,6+j];changed=~(new[...,None]==allids[...,None,:]).any(-1);assert np.array_equal(changed.sum(-1),sw[st:en,:,j,0]);entering[st:en,:,j]=np.where(changed,new,-1);left=~(allids[...,None]==new[...,None,:]).any(-1);leaving[st:en,:,j]=np.where(left,allids,-1);same=changed.sum(-1)==0;unchanged_residual=max(unchanged_residual,float(abs(fm[st:en,:,6+j,0][same]-fm[st:en,:,5,0][same]).max()) if same.any() else 0.)
   assert np.allclose(dec[st:en,:,:,0]+dec[st:en,:,:,1],dec[st:en,:,:,2],atol=1e-9,rtol=0)
   for j,name in enumerate(NAMES):
    R=np.load(OLD/cell/'cache'/name/(split+'.npy'),mmap_mode='r');r=np.asarray(R[st:en],dtype='float64');y=np.asarray(Y[st:en],dtype='float64');e=r-y;agg=(e*e).mean(1);expected=oldq[st:en,:,SUBSETS.index(name),0];maxerr=max(maxerr,float(abs(agg-expected).max()));assert np.allclose(agg,expected,rtol=2e-5,atol=1e-6)
    if j<5:
     b=np.asarray(B[st:en],dtype='float64');fe=b+prov['fixed_lambda']*(r-b)-y;zero_counts[j,0]+=int(((e*e).mean(1)<=1e-12).sum());zero_counts[j,1]+=int(((fe*fe).mean(1)<=1e-12).sum())
   num+=(en-st)*c
  entering.flush();leaving.flush()
  for j in range(5):zero.append(dict(cell=cell,split=split,head=j+1,n_query_channels=num,stage1_near_zero_mse_count=int(zero_counts[j,0]),fixed_stage2_near_zero_mse_count=int(zero_counts[j,1]),threshold_mse=1e-12,cosine_rule='pair norm product <= 1e-12 returns 0; counts use per-head MSE threshold as a conservative diagnostic'))
  checks.append(dict(split=split,queries=len(qids),channels=c,all11_stage1_vs_saved31_max_abs_difference=maxerr,ids_cardinality_unique_and_swap_count='PASS',unchanged_set_max_abs_mse_rounding_residual=unchanged_residual,unchanged_set_check='canonical prediction parity PASS; same-set order rounding measured and excluded from win counts',delta_decomposition='PASS'))
 table(p/'SIGNED_ERROR_ZERO_NORMS.csv',zero)
 if cell=='ETTh1_720':
  enter=np.load(loc/'test_loo_entering_ids.npy',mmap_mode='r');leave=np.load(loc/'test_loo_leaving_ids.npy',mmap_mode='r');candidate_starts=np.load(loc/'candidate_start_ids.npy');sw=np.load(loc/'test_candidate_swaps.npy',mmap_mode='r');dc=np.load(loc/'test_loo_decomposition.npy',mmap_mode='r');delta=test[:,:,8,0]-test[:,:,5,0];examples=[]
  for label,mask in [('improves',(delta<0)&(sw[:,:,2,0]>0)),('worsens',(delta>0)&(sw[:,:,2,0]>0)),('unchanged',sw[:,:,2,0]==0)]:
   ix=np.argwhere(mask)
   if len(ix)==0:continue
   i,ch=map(int,ix[0]);ei=enter[i,ch,2];ei=ei[ei>=0];li=leave[i,ch,2];li=li[li>=0];examples.append(dict(case=label,selection='first chronological query then channel; illustrative only',query_start_idx=int(ids[i]),channel=ch,channel_name=channels[ch],entering_candidate_ids=';'.join(map(str,ei)),leaving_candidate_ids=';'.join(map(str,li)),entering_train_starts=';'.join(map(str,candidate_starts[ei])),leaving_train_starts=';'.join(map(str,candidate_starts[li])),fixed_mse_delta=float(delta[i,ch]),fixed_mae_delta=float(test[i,ch,8,1]-test[i,ch,5,1]),linear_term=float(dc[i,ch,2,0]),quadratic_term=float(dc[i,ch,2,1])))
  table(p/'H3_REPLACEMENT_EXAMPLES.csv',examples)
 dump(p/'POST_VERIFICATION.json',dict(status='PASS',cell=cell,checks=checks,source_bundle_sha256=bundle,analysis_pass_sha256=apsha,original_checkpoints_unchanged=True,all5_exact_gate=next(g for g in ext['gates'] if 'All5' in g['gate']),common_channel_strata='all levels share same channel set present in both validation and test',time=time.time()));print('[POST VERIFY PASS]',cell,flush=True)
def finish():
 for cell in CELLS:
  assert (OUT/cell/'ANALYSIS_PASS.json').exists(),cell
  postcell(cell)
 resources=[]
 for cell in CELLS:
  e=json.loads((OUT/cell/'EXTRACTION_PASS.json').read_text());a=json.loads((OUT/cell/'ANALYSIS_PASS.json').read_text());resources.append(dict(cell=cell,extraction_seconds=e['wall_seconds'],latest_cpu_analysis_seconds=a['wall_seconds'],peak_gpu_allocated_mib=e['peak_gpu_allocated_mib'],peak_gpu_reserved_mib='NOT_RECORDED',gpu_physical_index=1,shared_gpu_with_p100=True,speed_comparison='not an independent benchmark'))
 table(OUT/'RESOURCE_SUMMARY.csv',resources)
 data={}
 for name in TABLES:
  data[name]=pd.concat([read(c,name) for c in CELLS],ignore_index=True);data[name].to_csv(OUT/(name+'.csv'),index=False,float_format='%.9f')
 g=data['GLOBAL_HEAD_CONTRIBUTION'];loo=g[g['subset'].str.len()==4];fore=data['FORECAST_SUMMARY'];sat=data['HEAD_COUNT_SATURATION'];mix=data['MIXTURE_VS_PREDICTION_AVERAGE'];signed=data['HEAD_FAILURE_SUCCESS_SIGNED'];repl=data['LOO_CANDIDATE_REPLACEMENTS'];std=data['CHANNEL_STANDARDIZED_STRATA'];chosen=data['CHANNEL_STANDARDIZED_VALIDATION_SELECTION'];focus=repl[(repl.cell=='ETTh1_720')&(repl.removed_head==3)];s=focus[focus.change_group=='all'].iloc[0]
 overview=[];roles=[]
 for cell in CELLS:
  row=fore[(fore.cell==cell)&(fore.split=='test')&(fore.subset=='12345')].iloc[0];a=json.loads((OUT/cell/'ANALYSIS_PASS.json').read_text());v=fore[(fore.cell==cell)&(fore.split=='val')&(fore.subset.str.len()==1)];best=str(v.loc[v.fixed_mse.idxmin(),'subset']);bt=fore[(fore.cell==cell)&(fore.split=='test')&(fore.subset==best)].iloc[0];bestloo=loo[loo.cell==cell].sort_values('fixed_mse_delta').iloc[0]
  bm=np.load(OUT/cell/'local/test_base_metrics.npy').mean((0,1));overview.append(dict(Cell=cell,BaseMSE=bm[0],BaseMAE=bm[1],All5固定MSE=row.fixed_mse,All5固定MAE=row.fixed_mae,Validation選択単独Head=best,同HeadTestMSE=bt.fixed_mse,Test事後最良LOO=bestloo.subset,LOO差=bestloo.fixed_mse_delta,Queries=a['queries'],Channels=a['channels'],TimeBlocks=a['n_time_blocks']))
  t=std[std.cell==cell];ch=chosen[chosen.cell==cell];f=signed[(signed.cell==cell)&(signed.event=='failure_above_val_q75')];u=signed[(signed.cell==cell)&(signed.event=='success_vs_base')]
  roles.append(dict(Cell=cell,標準化層数=len(ch),ValTest符号一致率=float(t.direction_agrees.mean()),Val選択単独がTestAll5を改善した層数=int((ch.test_head_minus_all5<0).sum()),失敗Jaccard平均=float(f.jaccard.mean()),成功Jaccard平均=float(u.jaccard.mean()),Stage1ErrorCosine平均=float(f.stage1_error_cosine.mean()),Stage2ErrorCosine平均=float(f.fixed_stage2_error_cosine.mean()),Stage1逆符号割合=float(f.stage1_opposite_sign_fraction.mean())))
 ov=pd.DataFrame(overview);ov.columns=['Cell','Base MSE','Base MAE','All5 fixed MSE','All5 fixed MAE','Val-selected single head','Its test MSE','Test-posthoc best LOO subset','LOO delta MSE','Queries','Channels','2H time blocks'];rr=pd.DataFrame(roles);rr.columns=['Cell','Common-channel strata','Val/test sign agreement','Val-selected single wins / strata','Failure Jaccard mean','Success Jaccard mean','Stage1 error cosine','Stage2 error cosine','Stage1 opposite-sign fraction'];table(OUT/'PRESENTATION_OVERVIEW.csv',ov.to_dict('records'));table(OUT/'ROLE_EVIDENCE_SUMMARY.csv',rr.to_dict('records'))
 paragraphs=['# Full Candidate V5 query별 head 역할 추가 분석\n\n확인 시각(UTC): '+time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())+'\n\n상태: COMPLETE (4 cells, seed0). 새 encoder/head/λ 학습 없이 기존 예측을 재사용했다. GPU 물리1만 사용했으며 P100 작업은 유지했다. GitHub 업로드는 수행하지 않았다.',
 '## 1. 관측과 비교 기준\n\nPrimary diagnostic은 기존 All5의 λ를 모든 구성에 고정한 결과다. 기존 구성별 refit λ 결과는 별도 열로 기록했다. 앞선 ANALYSIS01에서 primary라고 부른 refit 결과와 이번 고정-λ 진단을 혼동하면 안 된다. 표의 test 사후 최선 LOO는 기전 분석용이며 배포 모델 선택이 아니다. MSE/MAE 전체 11구성 val/test는 FORECAST_SUMMARY.csv에 있다.\n\n'+md(ov),
 '## 2. 어느 query에서 도움이 되는가\n\n아래 Δ는 구성−All5이며 양수일 때 All5가 유리하다. Query-channel 비율과 모든 채널을 평균한 query 비율을 분리했다. 동일 후보 집합의 순서 반올림은 승패로 세지 않았다. 동일 조건에서 positive/negative 오류 크기를 함께 계산했으므로 승률만으로 전체 이득을 주장하지 않는다.\n\n'+md(loo[['cell','subset','fixed_mse_delta','fixed_mae_delta','all5_query_macro_improvement_fraction','all5_improvement_fraction','all5_worsening_size_if_worse','family_low','family_high']]),
 '## 3. 입력 특성·채널 구성 통제\n\n추세, 차분 변동성, detrended FFT 최대 power 비중, 정규화 최대 급변의 구간 경계는 validation history의 채널별 삼분위에서 고정했다. 미래값으로 구간이나 head를 선택하지 않았다. 원래 표는 존재하는 채널을 동일 가중 평균하고, CHANNEL_STANDARDIZED_STRATA.csv에서는 validation/test의 세 구간 모두 존재하는 공통 채널만 동일 가중 비교했다. feature 간 공통 채널 집합은 다를 수 있으며 채널 ID와 표본수를 명시했다. 원시 query-channel 결과와 채널별 결과도 보존했다.\n\n'+md(rr)+'\n\nVal/test 부호 일치율은 10 head 대비×12구간의 기술 통계이며 독립 검정 성공률이 아니다. 한 구간의 특정 head가 좋아 보인다고 추세/주기성에 특화된 의미적 역할을 학습했다고 결론내릴 수 없다. Validation에서 선택한 head가 test에서 개선되는지는 공통 채널로 다시 평가했다.',
 '## 4. ETTh1 H720 H3 제거 기전\n\n고정 λ ΔMSE = '+f'{s.fixed_stage2_mse_delta:.6f}'+', ΔMAE = '+f'{s.fixed_stage2_mae_delta:.6f}'+'. 평균 '+f'{s.mean_replaced_count:.6f}'+'/10 후보가 바뀌며, 교체된 후보의 incoming−outgoing 개별 미래 MSE는 '+f'{s.incoming_minus_outgoing_individual_mse:.6f}'+'. 기존 All5 오류와 예측 변화의 정렬 항 '+f'{s.linear_alignment_term:.6f}'+' + 예측 변화 크기 항 '+f'{s.quadratic_change_term:.6f}'+' = ΔMSE이다. 따라서 실제 Top10 재선택이 All5 오류를 줄이는 방향으로 작용했다는 산술적 근거가 있다. H3의 학습 목적 자체가 틀렸거나 모든 데이터에서 불필요하다는 증거는 아니다.\n\n'+md(std[(std.cell=='ETTh1_720')&(std.subset=='1245')][['feature','stratum','common_channels','validation_mse_delta','test_mse_delta','direction_agrees']])+'\n\n'+md(focus[['change_group','n_query_channels','mean_replaced_count','incoming_minus_outgoing_individual_mse','fixed_stage2_mse_delta','linear_alignment_term','quadratic_change_term']])+'\n\n각 교체 수 그룹도 채널별 동일 가중 macro이므로 그룹 평균을 단순 표본비율로 합쳐 전체 평균을 복원하면 안 된다. 모든 실제 entering/leaving candidate는 local top10_ids와 candidate_start_ids로 추적 가능하다. 동일 후보 집합도 topk 정렬 순서가 달라 float32 합산 반올림이 미세하게 달랐다. 원래 prediction/지표는 그대로 보존하고, 동일 집합은 승패 비율에서 tie로 처리했다. 최대 residual은 POST_VERIFICATION.json에 기록했다.',
 '## 5. 오류 보완성과 mean-mixture\n\n실패는 head/채널별 validation MSE 상위25% 임계값 초과, 성공은 같은 query-channel에서 Base보다 낮은 MSE로 정의했다. 실패/성공 Jaccard, raw dot, cosine, horizon 원소의 반대 부호 비율을 별도 저장했다. 양의 큰 cosine과 실패 중복은 서로 다른 후보가 강한 오류 독립성을 보장하지 않음을 보여준다. 실제 모델은 head 확률 평균→Top10 재선택이다. 아래 단순 head 예측 평균은 최대50개의 후보를 섞을 수 있어 동일 Top10 예산의 공정한 대안 모델로 해석하지 않는다.\n\n'+md(mix[['cell','actual_meanprob_top10_stage1_mse','simple_head_prediction_average_stage1_mse','actual_meanprob_fixed_stage2_mse','simple_head_prediction_average_fixed_stage2_mse','simple_average_minus_actual_fixed_mse']]),
 '## 6. Head 수별 포화\n\n기존 31 subset의 Stage1 query 지표를 재사용했다. Validation 고정 subset 선택, test 사후 최선 고정 subset, test query-channel oracle은 서로 다른 프로토콜이다. Oracle은 배포 성능이 아니며 MSE와 MAE oracle도 각각 최소화했다. 31개 Stage2를 새로 학습하지 않았다.\n\n'+md(sat[['cell','head_count','validation_selected_subset','validation_selected_test_mse','test_mean_over_combinations','test_query_channel_oracle_mse']]),
 '## 7. 검증·불확실성\n\n4셀의 모든 val/test 11구성 retrieval prediction이 기존 캐시와 bit-exact다. Query ID, candidate mask, 중복 없는 Top10, checkpoint/base/data hash와 All5의 canonical 고정-λ GPU MSE/MAE exact gate를 통과했다. CPU float64로 계산한 query별 집계는 원래 GPU float32 reduction과 미세한 차이가 있으므로 원본 수치와 별도 추적했다. 31subset와 새 오차의 정렬, 후보 교체 수, ΔMSE 분해도 검증했다.\n\nCI는 2H 길이의 연속 시간 블록을 모든 채널 함께 재표본화한 paired bootstrap이다. 전역5000회, 층별1000회이며 H 길이 민감도 결과도 저장했다. 20개 전역 LOO MSE 비교의 Bonferroni 구간은 최소8블록일 때만 제시한다. H720처럼 블록이 적으면 95% 구간이 좁아도 통계적 확증으로 쓰지 않는다. 블록 경계를 넘는 입력/target 중첩과 장기 의존이 남을 수 있다. Strata bootstrap에서도 원래 공통 채널 전부가 남는 draw만 사용하고 유효 draw 수를 기록했다. Strata와 subset/평균 예측 진단은 탐색적 다중비교다. Seed0 단일 checkpoint의 조건부 분석으로 학습 seed 변동을 평가하지 않았다.\n\nWeather는 원본 -9999 값이 train scaler를 왜곡한 기존 프로토콜을 그대로 사용한다. 이번 분석은 해당 정규화 공간에서 조건부로 유효하며 결측치 정제 후 재학습 성능으로 일반화하지 않는다. Base감사 경로: results/BASE-WEATHER-AUDIT01-20261028/.',
 '## 8. 연구 판단과 최소 후속 과제\n\n새로 확인한 것은 query별 손익의 불균일성, 실제 후보 교체가 고정 λ 예측 오차에 미치는 정확한 분해, 채널 구성을 통제한 입력 구간별 차이, 고정 subset 선택과 oracle의 간격이다. 기존의 단순 평균/overlap 표보다 강한 조건부 기전 증거다. 그러나 각 head가 안정적인 의미 역할을 학습했다는 주장, 모든5개 head 필요성, 여러 seed에 걸친 일반성은 NOT ESTABLISHED다. 후보 발견, 후보 선택, 최종 예측 기여를 분리해야 한다.\n\n경쟁 가설 H1: 특정 입력 영역에서 H3의 후보 선택이 안정적으로 해롭다. H2: 단일 checkpoint·시간 구간과 mean-mixture의 우연한 결합 효과다. 가장 작은 후속 과제는 이미 관측한 E720 H3 규칙과 validation에서만 결정한 고정 선택을 사전 등록하고 추가 독립 기간 또는 승인된 추가 seed에서 확인하는 것이다. 현재 요청에는 포함하지 않았으므로 실행하지 않았다. Weather scaler 정제 비교는 별도 통제 실험으로 다루어야 한다.',
 '## 9. 파일과 실행\n\n실행: controller.py가 E96→E720→W96→W720 순서로 extract.py, analyze.py를 실행했고 finalize.py가 CPU 검증/집계를 수행했다. 구현: experiments/full_head_query_analysis02/. 원본: results/TRACK-V5-FULL-HEAD-ANALYSIS01/. 새 산출물: results/TRACK-V5-FULL-HEAD-QUERY-ANALYSIS02/. 상세 코드·체크포인트·데이터 hash는 SOURCE_MANIFEST.json, 셀별 PROVENANCE.json, POST_VERIFICATION.json에 있다. 대용량 query prediction/ID 배열은 각 셀 local/에만 보관했다. RESOURCE_SUMMARY.csv의 시간은 P100과 GPU를 공유한 경합 환경 값이며 독립 속도 비교가 아니다. Allocated peak는 기록했고 reserved peak는 이번 실행에서 미기록이다.']
 (OUT/'FINAL_REPORT_KO.md').write_text('\n\n'.join(paragraphs),encoding='utf-8')
 manifests={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/full_head_query_analysis02').glob('*.py')};dump(OUT/'SOURCE_MANIFEST.json',dict(source_sha256=manifests,live_repo_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),new_training=False,new_seeds=False,pushed=False))
 checks={c:json.loads((OUT/c/'POST_VERIFICATION.json').read_text()) for c in CELLS};dump(OUT/'COMPLETION_VERIFICATION.json',dict(status='COMPLETE',cells=checks,synthetic_tests=10,new_training=False,gpu_physical_index=1,large_arrays_local_only=True,github_push=False))
 dump(OUT/'STATUS.json',dict(status='COMPLETE',cells=CELLS,report=str(OUT/'FINAL_REPORT_KO.md'),time=time.time()));print('[COMPLETE] Four cells, query diagnostics and report',flush=True)
if __name__=='__main__':finish()
