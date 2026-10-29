import os,sys,json,csv,hashlib,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd,torch
ROOT=Path('/data/pjh_workspace/CARTS');sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from models.RelationStage2 import BaseForecastHead
from data_provider.data_factory import data_provider
from scripts.verify_v5_head_loss_stage2_correctness01 import CELLS
from scripts.train_r_stage2_lambda01 import load_tensors
OUT=ROOT/'results/BASE-WEATHER-AUDIT01-20261028';OUT.mkdir(parents=True,exist_ok=True)
GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
assert os.environ['CUDA_VISIBLE_DEVICES']==GPU
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str))
class OriginalLoader:
 def __init__(self,args):self.args=args
 def _get_data(self,flag,shuffle=False):return data_provider(self.args,flag,shuffle=shuffle)
for cell in ['Weather_96','Weather_720']:
 t0=time.time();cfg=CELLS[cell];h=cfg['pred_len'];p=OUT/cell;p.mkdir(exist_ok=True)
 ckpath=ROOT/cfg['base_checkpoint'];before=sha(ckpath)
 ck=torch.load(ckpath,map_location='cpu');ref=torch.load(ROOT/cfg['reference_ckpt'],map_location='cpu')
 ad=dict(ref['args']);ad.update(pred_len=h,seq_len=h,batch_size=32,seed=0)
 args=SimpleNamespace(**ad);assert args.features=='M' and args.enc_in==21 and getattr(args,'teacher_mse_space','normalized')=='normalized'
 assert ck['config']['seq_len']==h and ck['config']['pred_len']==h and ck['config']['channels']==21
 path=(ROOT/args.root_path/args.data_path).resolve();df=pd.read_csv(path);dates=pd.to_datetime(df.date)
 cols=[c for c in df.columns if c not in ['date',args.target]]+[args.target];raw=df[cols].to_numpy(dtype=np.float64)
 n=len(df);nt=int(n*.7);nte=int(n*.2);nv=n-nt-nte;test_boundary=n-nte
 mean=raw[:nt].mean(0);scale=raw[:nt].std(0);scale[scale==0]=1
 normalized=((raw-mean)/scale).astype(np.float32)
 splits={};datasets={}
 for split,lo,hi,future_lo in [('train',0,nt,h),('val',nt-h,test_boundary,nt),('test',test_boundary-h,n,test_boundary)]:
  ds,loader=data_provider(args,split,shuffle=False);bd=ds.base_dataset;datasets[split]=ds
  assert bd.border1==lo and len(bd.data_x)==hi-lo
  assert np.array_equal(ds.starts,np.arange(lo,hi-2*h+1))
  assert np.max(abs(bd.scaler.mean_-mean))<1e-10 and np.max(abs(bd.scaler.scale_-scale))<1e-10
  maxnorm=float(np.max(abs(bd.data_x-normalized[lo:hi])));assert maxnorm<2e-6
  splits[split]=dict(context_row_start=lo,row_end_exclusive=hi,future_rows=[future_lo,hi-1],context_time_start=str(dates.iloc[lo]),future_time_start=str(dates.iloc[future_lo]),future_time_end=str(dates.iloc[hi-1]),windows=len(ds),first_window_id=int(ds.starts[0]),last_window_id=int(ds.starts[-1]),scaler_matches_independent_train_fit=True,max_normalized_data_difference=maxnorm,drop_last=loader.drop_last)
 ds,loader=data_provider(args,'test',shuffle=False)
 model=BaseForecastHead(h,h,21,mode='per_channel_linear').cuda();model.load_state_dict(ck['model_state_dict']);model.eval()
 for par in model.parameters():par.requires_grad_(False)
 pred=np.lib.format.open_memmap(p/'fresh_test_predictions.npy',mode='w+',dtype='float32',shape=(len(ds),h,21))
 ids=np.zeros(len(ds),dtype=np.int64);se=ae=cnt=0;ptse=ptae=0.;persist_se=persist_ae=0.;channel_se=np.zeros(21);channel_ae=np.zeros(21);horizon_se=np.zeros(h);horizon_ae=np.zeros(h)
 manualdiff=0.;inputdiff=targetdiff=0.;offset=0;batchsizes=[]
 weights=[ck['model_state_dict'][f'heads.{c}.weight'].numpy().astype('float64') for c in range(21)]
 biases=[ck['model_state_dict'][f'heads.{c}.bias'].numpy().astype('float64') for c in range(21)]
 torch.cuda.reset_peak_memory_stats()
 with torch.no_grad():
  for bi,(x,y,start) in enumerate(loader):
   xx=x.numpy();yy=y.numpy();starts=start.numpy();b=len(x)
   ix=starts[:,None]+np.arange(h);iy=starts[:,None]+h+np.arange(h)
   inputdiff=max(inputdiff,float(np.max(abs(xx-normalized[ix]))));targetdiff=max(targetdiff,float(np.max(abs(yy-normalized[iy]))))
   assert inputdiff<2e-6 and targetdiff<2e-6
   assert iy.min()>=test_boundary and iy.max()<n
   x=x.float().cuda();y=y.float().cuda();bpred=model(x)+x[:,-1:,:]
   arr=bpred.cpu().numpy();pred[offset:offset+b]=arr;ids[offset:offset+b]=starts
   err=arr.astype(np.float64)-yy.astype(np.float64)
   se+=float(np.square(err).sum());ae+=float(np.abs(err).sum());cnt+=err.size
   ptse+=float((bpred-y).square().sum());ptae+=float((bpred-y).abs().sum())
   channel_se+=np.square(err).sum((0,1));channel_ae+=np.abs(err).sum((0,1))
   horizon_se+=np.square(err).sum((0,2));horizon_ae+=np.abs(err).sum((0,2))
   pe=xx[:,-1:,:].astype(np.float64)-yy
   persist_se+=float(np.square(pe).sum());persist_ae+=float(np.abs(pe).sum())
   if bi in [0,len(loader)//2,len(loader)-1]:
    z=xx[:1].astype('float64');last=z[:,-1:,:];z=z-last
    direct=np.stack([z[:,:,c]@weights[c].T+biases[c] for c in range(21)],-1)+last
    manualdiff=max(manualdiff,float(np.max(abs(direct-arr[:1]))))
   offset+=b;batchsizes.append(b)
 pred.flush();np.save(p/'test_window_ids.npy',ids)
 assert offset==len(ds) and np.array_equal(ids,ds.starts)
 assert cnt==len(ds)*h*21 and manualdiff<1e-4
 orig=(ROOT/f'results/TRACK-R-FINAL-METHOD-GENERALIZATION01/Weather/H96/seed0/base/metrics.json') if h==96 else ROOT/'results/TRACK-V-MULTIQUERY-GENERALIZATION01/Weather/H720/seed0/base/metrics.json'
 original=json.loads(orig.read_text());assert original['best_epoch']==ck['epoch']
 mse,mae=se/cnt,ae/cnt;assert abs(mse-original['test_mse'])<1e-5 and abs(mae-original['test_mae'])<1e-5
 with (p/'per_channel_metrics.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['channel','mse','mae'])
  for c,name in enumerate(cols):w.writerow([name,channel_se[c]/(len(ds)*h),channel_ae[c]/(len(ds)*h)])
 with (p/'per_horizon_metrics.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['step','mse','mae'])
  for k in range(h):w.writerow([k+1,horizon_se[k]/(len(ds)*21),horizon_ae[k]/(len(ds)*21)])
 # Re-execute original Stage2 loader independently; R is never used in Base-only inference.
 exp=OriginalLoader(args);tracks=[]
 caches=[('Full',ROOT/f'results/TRACK-V5-FULL-HEAD-ANALYSIS01/{cell}/cache/12345')]
 p100root=ROOT/f'experiments/p100_preselection_comparison/results/runs/{cell}'
 existing=sorted(p100root.glob('*/*/cache/test.pt')) if p100root.exists() else []
 if existing:caches.append(('P100',existing[0].parent))
 for track,cache in caches:
  st,B,R,Y=load_tensors(exp,model,cache,'test','cuda',21)
  assert torch.equal(st,torch.from_numpy(ids))
  difference=float(np.max(abs(B.cpu().numpy()-np.asarray(pred))))
  assert difference==0.0
  fused=B+0*(R-B);assert torch.equal(fused,B)
  err=fused.double()-Y.double()
  smse=float(err.square().mean());smae=float(err.abs().mean())
  assert abs(smse-mse)<1e-10 and abs(smae-mae)<1e-10
  tracks.append(dict(track=track,cache=str(cache),fresh_base_max_diff=difference,lambda0_mse=smse,lambda0_mae=smae,status='PASS'))
  del B,R,Y,fused,err;torch.cuda.empty_cache()
 completed=[]
 for complete in sorted(p100root.glob('*/*/COMPLETE.json')) if p100root.exists() else []:
  d=json.loads(complete.read_text());r=d.get('independent_stage2',{})
  bsha=r.get('base_sha256');assert bsha==before,(complete,bsha,before)
  completed.append(dict(path=str(complete),base_sha256=bsha))
 # Every available P100 cache uses the same exact window ID universe.
 cacheids=[]
 for cp in existing:
  ca=torch.load(cp,map_location='cpu');assert np.array_equal(ca['query_start_idx'].numpy(),ids)
  cacheids.append(dict(path=str(cp),query_ids_exact=True));del ca
 fullprov=json.loads((ROOT/f'results/TRACK-V5-FULL-HEAD-ANALYSIS01/{cell}/CHECKPOINT_PROVENANCE.json').read_text());assert fullprov['base_sha256']==before
 assert sha(ckpath)==before
 report=dict(status='PASS',cell=cell,base_class='models.RelationStage2.BaseForecastHead',mode='per_channel_linear',formula='B_c(x)=W_c (x_c-x_last,c)+b_c+x_last,c',parameter_count=sum(z.numel() for z in model.parameters()),checkpoint=str(ckpath),checkpoint_sha256=before,checkpoint_config=ck['config'],checkpoint_epoch=ck['epoch'],checkpoint_val_mse=ck.get('val_mse'),args={k:getattr(args,k,None) for k in ['data','root_path','data_path','features','target','enc_in','seq_len','pred_len','label_len','task_name','teacher_mse_space','augmentation_ratio','freq','num_workers']},dataset=str(path),dataset_sha256=sha(path),dataset_rows=n,channel_names=cols,timestamps_monotonic=bool(dates.is_monotonic_increasing),timestamp_step_counts=dates.diff().astype(str).value_counts().head().to_dict(),split=splits,scaler_fit_rows=[0,nt],scaler_mean=mean.tolist(),scaler_scale=scale.tolist(),evaluation_space='train-standardized; no inverse transform',normalization_and_targets_match_raw_csv=True,input_max_diff=inputdiff,target_max_diff=targetdiff,test_elements=cnt,test_windows=len(ds),batch_size=32,last_batch_size=batchsizes[-1],aggregation='sum over every window,horizon,channel / total elements; overlapping test windows intentionally counted',numpy_float64_mse=mse,numpy_float64_mae=mae,original_batchwise_float32_mse=ptse/cnt,original_batchwise_float32_mae=ptae/cnt,original_metrics=original,original_metrics_path=str(orig),direct_numpy_formula_sample_max_diff=manualdiff,persistence_mse=persist_se/cnt,persistence_mae=persist_ae/cnt,stage2_fresh_paths=tracks,p100_completed_checkpoint_checks=completed,p100_cache_id_checks=cacheids,p100_h720_status='available' if existing else 'NOT_YET_AVAILABLE; code wiring shares CELLS and original Stage2 trainer, no current-run cache to compare',historical_base_prediction_array='NOT_SAVED in original base directory; exact comparison is fresh Base vs rerun original Stage2 path, not historical B bytes',runtime=dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,tf32=torch.backends.cuda.matmul.allow_tf32,gpu_uuid=GPU),wall_seconds=time.time()-t0,peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
 dump(p/'AUDIT.json',report)
 print(cell,'PASS',mse,mae,'manual maxdiff',manualdiff,flush=True)
 del model,ck,ref,pred;torch.cuda.empty_cache()
dump(OUT/'STATUS.json',dict(status='PASS',cells=['Weather_96','Weather_720'],github='NOT_REQUESTED_DEFERRED',training=False))
