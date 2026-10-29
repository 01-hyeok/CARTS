import json,io,zipfile,urllib.request,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
torch.set_num_threads(2)
r=Path('/data/pjh_workspace/CARTS');o=r/'results/BASE-WEATHER-AUDIT01-20261028'
a=json.loads((o/'Weather_96/AUDIT.json').read_text());df=pd.read_csv(a['dataset']);raw=df[a['channel_names']].to_numpy(float);nt=36887;tb=42157
mean=raw[:nt].mean(0);scale=raw[:nt].std(0);z=((raw-mean)/scale).astype('float32');bad=(raw[:nt]==-9999).any(0);rows=[];summary={}
for h in (96,720):
 p=o/f'Weather_{h}';B=np.load(p/'fresh_test_predictions.npy',mmap_mode='r');ids=np.load(p/'test_window_ids.npy');f=r/f'results/TRACK-V5-FULL-HEAD-ANALYSIS01/Weather_{h}'
 cache=torch.load(f/'cache/12345/test.pt',map_location='cpu');assert np.array_equal(ids,cache['query_start_idx'].numpy());R=cache['relation_outputs'].numpy();m=json.loads((f/'stage2/12345/metrics.json').read_text());lam=np.float32(m['lambda']);s=np.zeros((2,2,21));missing=np.zeros(21,dtype='int64');missse=np.zeros((2,21));missae=missse.copy()
 for i in range(0,len(ids),32):
  ix=ids[i:i+32,None]+h+np.arange(h);Y=z[ix];mask=raw[ix]==-9999;missing+=mask.sum((0,1));b=B[i:i+32];full=b+lam*(R[i:i+32]-b)
  for k,pred in enumerate((b,full)):
   e=pred.astype('float64')-Y.astype('float64');s[k,0]+=(e*e).sum((0,1));s[k,1]+=abs(e).sum((0,1));missse[k]+=((e*e)*mask).sum((0,1));missae[k]+=(abs(e)*mask).sum((0,1))
 s/=len(ids)*h
 assert abs(s[0,0].mean()-json.loads((p/'AUDIT.json').read_text())['numpy_float64_mse'])<1e-10
 assert abs(s[1,0].mean()-m['test_mse'])<1e-5
 for c,name in enumerate(a['channel_names']):
  q=dict(horizon=h,channel=name,train_scale=scale[c],train_minus9999=int((raw[:nt,c]==-9999).sum()),test_minus9999=int((raw[tb:,c]==-9999).sum()),target_minus9999_elements=int(missing[c]),train_scale_without_minus9999=float(raw[:nt,c][raw[:nt,c]!=-9999].std()))
  for k,label in enumerate(('base','full_all5')):
   q.update({label+'_mse':s[k,0,c],label+'_mae':s[k,1,c],label+'_mse_contribution':s[k,0,c]/21,label+'_missing_target_mse_contribution':missse[k,c]/(len(ids)*h*21),label+'_missing_target_mae_contribution':missae[k,c]/(len(ids)*h*21)})
  q['full_minus_base_mse']=s[1,0,c]-s[0,0,c];rows.append(q)
 gap=s[1,0]-s[0,0];pos=np.maximum(gap,0);order=np.argsort(-pos)
 summary[str(h)]=dict(all21=dict(base_mse=s[0,0].mean(),base_mae=s[0,1].mean(),full_mse=s[1,0].mean(),full_mae=s[1,1].mean()),exclude_train_sentinel_channels18=dict(base_mse=s[0,0,~bad].mean(),base_mae=s[0,1,~bad].mean(),full_mse=s[1,0,~bad].mean(),full_mae=s[1,1,~bad].mean()),base_better_mse_channels=int((gap>0).sum()),base_better_mae_channels=int((s[1,1]>s[0,1]).sum()),top3_base_advantage=[dict(channel=a['channel_names'][j],gap=float(gap[j]),overall_gap_contribution=float(gap[j]/21)) for j in order[:3]],top3_share_of_positive_base_advantage=float(pos[order[:3]].sum()/pos.sum()),lambda_value=float(lam),full_metric_absdiff=float(abs(s[1,0].mean()-m['test_mse'])))
pd.DataFrame(rows).to_csv(o/'FOLLOWUP_CHANNEL_DIAGNOSTICS.csv',index=False)
summary['excluded_channels']=np.array(a['channel_names'])[bad].tolist();summary['note']='Diagnostic channel exclusion only, unchanged model/scaler/predictions. Not a cleaned-data performance or replacement official score.'
(o/'FOLLOWUP_SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
frames=[];origin=[]
for half in ('a','b'):
 url=f'https://www.bgc-jena.mpg.de/wetter/data/mpi_roof_2020{half}.zip';data=urllib.request.urlopen(url,timeout=30).read();(o/f'mpi_roof_2020{half}.zip').write_bytes(data);zz=zipfile.ZipFile(io.BytesIO(data));name=zz.namelist()[0];v=pd.read_csv(zz.open(name),encoding='latin1');frames.append(v);origin.append(dict(url=url,sha256=hashlib.sha256(data).hexdigest(),files=zz.namelist(),shape=list(v.shape)))
v=pd.concat(frames,ignore_index=True);origin.append(dict(columns=v.columns.tolist(),shape=list(v.shape),numeric_exact=bool(v.shape==df.shape and np.array_equal(v.iloc[:,1:].to_numpy(float),raw)),timestamp_exact=bool(len(v)==len(df) and np.array_equal(pd.to_datetime(v.iloc[:,0],dayfirst=True).to_numpy(),pd.to_datetime(df.date).to_numpy()))))
(o/'OFFICIAL_SOURCE_COMPARISON.json').write_text(json.dumps(origin,indent=2));print(json.dumps(origin,indent=2))
