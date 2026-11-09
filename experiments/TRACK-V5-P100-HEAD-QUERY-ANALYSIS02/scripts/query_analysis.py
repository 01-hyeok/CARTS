#!/usr/bin/env python3
"""Frozen P100 query/head diagnostics. No training; All5 lambda held fixed."""
import argparse,csv,json,os,sys,time,subprocess,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/data/pjh_workspace/CARTS');sys.path.insert(0,str(ROOT))
from experiments.p100_preselection_comparison import pipeline as p
from scripts.build_v_meanmix_cache_pool01 import compute_scores_pool_channel_first
R=ROOT/'experiments/TRACK-V5-P100-HEAD-QUERY-ANALYSIS02'
SUBSETS=[tuple(range(5))]+[(i,) for i in range(5)]+[tuple(j for j in range(5) if j!=i) for i in range(5)]
NAMES=['All5']+[f'Head{i+1}' for i in range(5)]+[f'Without{i+1}' for i in range(5)]
FEATURES=['trend','volatility','periodicity','jump']
def read(f):return json.loads(Path(f).read_text())
def dump(f,x):p.dump(f,x)
def csvout(f,rows):
 if not rows:return
 f=Path(f);f.parent.mkdir(parents=True,exist_ok=True)
 with f.open('w') as w:
  dw=csv.DictWriter(w,list(dict.fromkeys(k for row in rows for k in row)));dw.writeheader();dw.writerows(rows)
def features(x):
 # Input-only, per query/channel; no future observations.
 z=x-x.mean(1,keepdims=True);scale=z.square().mean(1).sqrt().clamp_min(1e-6)
 t=torch.linspace(-1,1,x.shape[1],device=x.device);trend=(z*t[None,:,None]).mean(1)/t.square().mean()/scale
 diff=x[:,1:]-x[:,:-1];vol=diff.square().mean(1).sqrt()/scale
 detrended=z-(z*t[None,:,None]).mean(1)[:,None,:]/t.square().mean()*t[None,:,None]
 power=torch.fft.rfft(detrended,dim=1).abs().square()[:,1:];period=power.max(1).values/power.sum(1).clamp_min(1e-12)
 jump=diff.abs().max(1).values/scale
 return torch.stack([trend,vol,period,jump],-1)
def block_ci(values,starts,length):
 # Channels stay clustered; consecutive non-overlapping start-index blocks span L+H.
 vals=np.asarray(values,dtype=np.float64)
 if vals.ndim>1:vals=vals.mean(tuple(range(1,vals.ndim)))
 groups=(np.asarray(starts)-np.min(starts))//length
 sums=np.array([vals[groups==g].sum() for g in np.unique(groups)])
 counts=np.array([(groups==g).sum() for g in np.unique(groups)])
 if len(counts)<2:return dict(mean=float(vals.mean()),ci95=None,blocks=len(counts))
 rng=np.random.RandomState(0);ix=rng.randint(len(counts),size=(1000,len(counts)));boot=sums[ix].sum(1)/counts[ix].sum(1)
 return dict(mean=float(vals.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),blocks=len(counts))
def selections(prob,valid):
 return [p.stable_topk_indices((prob if s==SUBSETS[0] else prob[:,list(s)]).mean(1).masked_fill(~valid,float('-inf')),10,largest=True) for s in SUBSETS]
def masked_block_ci(values,mask,starts,length):
 groups=(starts-starts.min())//length;ss=[];nn=[]
 for g in np.unique(groups):
  v=values[groups==g];m=mask[groups==g]
  if m.any():ss.append(float(v[m].sum()));nn.append(int(m.sum()))
 if len(nn)<2:return None
 ss=np.asarray(ss);nn=np.asarray(nn);ix=np.random.RandomState(0).randint(len(nn),size=(1000,len(nn)))
 return np.quantile(ss[ix].sum(1)/nn[ix].sum(1),[.025,.975]).tolist()
def preflight():
 rows=[]
 for cell in p.ORDER:
  aligned=None
  for method in p.METHODS:
   out=p.HERE/'results/runs'/cell/method/'V5';done=out/'COMPLETE.json'
   if not done.exists():rows.append(dict(cell=cell,method=method,status='Pending Stage2'));continue
   d=read(done);m=read(out/'PROVENANCE.json');s=read(out/'stage2/metrics.json')
   assert d['provenance']==m and m['seed']==s['seed']==0 and m['method']==method
   assert p.sha(m['checkpoint'])==m['checkpoint_sha256'] and d['independent_stage2']['passed']
   for split,h in m['pool_sha256'].items():assert p.sha(p.HERE/'results/pools'/cell/method/f'{split}.pt')==h
   ca=torch.load(out/'cache/test.pt',map_location='cpu');ids=ca['query_start_idx'].numpy();assert len(np.unique(ids))==len(ids)
   if aligned is None:aligned=ids
   else:assert np.array_equal(aligned,ids)
   rows.append(dict(cell=cell,method=method,status='PASS',queries=len(ids),channels=ca['channels'],checkpoint_sha=m['checkpoint_sha256'],mse=s['test_mse'],mae=s['test_mae']))
 csvout(R/'reports/CACHE_PREFLIGHT.csv',rows);dump(R/'reports/PREFLIGHT.json',rows);print('PREFLIGHT',len(rows),flush=True)
@torch.no_grad()
def analyze(cell,method,smoke=False):
 out=p.HERE/'results/runs'/cell/method/'V5';dest=R/'results'/cell/method
 if not smoke and (dest/'COMPLETE.json').exists():return
 done=read(out/'COMPLETE.json');manifest=read(out/'PROVENANCE.json');metrics=read(out/'stage2/metrics.json');lam=metrics['lambda']
 assert manifest==done['provenance'] and manifest['seed']==0 and manifest['method']==method
 assert p.sha(manifest['checkpoint'])==manifest['checkpoint_sha256']
 assert p.sha(p.CELLS[cell]['base_checkpoint'])==done['independent_stage2']['base_sha256']
 exp,args,model=p.build_model(p.common(cell),torch.device('cuda'));exp.memory_x=exp.memory_x.cuda();model.eval()
 model.load_state_dict(torch.load(manifest['checkpoint'],map_location='cuda')['model_state_dict'])
 heads=p.SlotHeads(args.d_model,n_slots=5).cuda().eval();heads.load_state_dict(torch.load(manifest['checkpoint'],map_location='cuda')['slot_heads_state_dict'])
 base=p.BaseForecastHead(seq_len=args.seq_len,pred_len=args.pred_len,channels=args.enc_in,mode='per_channel_linear').cuda().eval()
 base.load_state_dict(torch.load(p.CELLS[cell]['base_checkpoint'],map_location='cuda')['model_state_dict'])
 valfeatures=[]
 _,vl=exp._get_data(flag='val',shuffle=False)
 for x,_,_ in vl:valfeatures.append(features(x.float()).numpy())
 vf=np.concatenate(valfeatures);threshold=np.quantile(vf,[1/3,2/3],axis=0)
 threshold_path=R/'results'/cell/'STRATA_THRESHOLDS.json'
 threshold_record=dict(features=FEATURES,split='val',quantiles=[1/3,2/3],thresholds=threshold.tolist(),n_val=len(vf),definition='per-channel input-only terciles; fixed before test; trend signed normalized OLS; volatility RMS first differences/std; periodicity detrended max non-DC FFT power/total; jump max absolute first difference/std')
 if threshold_path.exists():assert read(threshold_path)==threshold_record
 else:dump(threshold_path,threshold_record)
 cache=torch.load(out/'cache/test.pt',map_location='cpu');starts_expected=cache['query_start_idx'];C=args.enc_in;H=args.pred_len
 assert cache['channels']==list(range(C))
 pc=p.CandidatePoolCache(p.HERE/'results/pools'/cell/method/'test.pt',dict(seq_len=args.seq_len,pred_len=H,candidate_mask='raft',candidate_pool_size=100,candidate_pool_metric=p.METRICS[method]))
 _,loader=exp._get_data(flag='test',shuffle=False);N=min(32,len(starts_expected)) if smoke else len(starts_expected)
 dest.mkdir(parents=True,exist_ok=True)
 arrays={k:np.zeros(shape,dtype=np.float32) for k,shape in dict(stage2=(N,C,11,2),retrieval=(N,C,11,2),base=(N,C,2),kl=(N,C,2),replacement=(N,C,5),error_cos=(N,C,5,5),head_overlap=(N,C,5,5),features=(N,C,4),quality=(N,C,7)).items()}
 klout=p.HERE/'results/runs'/cell/method/'V0';klcache=torch.load(klout/'cache/test.pt',map_location='cpu');klmetrics=read(klout/'stage2/metrics.json')
 assert torch.equal(klcache['query_start_idx'],starts_expected)
 ids=np.zeros((N,C,11,10),np.int32);starts=np.zeros(N,np.int64);position=0;maxdiff=0.
 for bi,(x,y,s) in enumerate(loader):
  if position>=N:break
  x=x.float().cuda();y=y.float().cuda();b=len(x);end=position+b;assert torch.equal(s,starts_expected[position:end])
  starts[position:end]=s.numpy();B=base(x)+x[:,-1:];arrays['base'][position:end,:,0]=(B-y).square().mean(1).cpu();arrays['base'][position:end,:,1]=(B-y).abs().mean(1).cpu();arrays['features'][position:end]=features(x).cpu()
  kpred=B+klmetrics['lambda']*(klcache['relation_outputs'][position:end].cuda()-B);arrays['kl'][position:end,:,0]=(kpred-y).square().mean(1).cpu();arrays['kl'][position:end,:,1]=(kpred-y).abs().mean(1).cpu()
  for c in range(C):
   scores,valid,pool=compute_scores_pool_channel_first(model,heads,x,exp,c,pc,s,torch.device('cuda'))
   canonical,prob,_=p.mean_mixture_topk_selection(scores,valid,.1,10);ix=selections(prob,valid);assert torch.equal(ix[0],canonical)
   mem,off=p.memory_value(args,x,exp.memory_y,exp.memory_x_last,c);values=p.gather_candidate_values(mem,pool)+off[:,None,None]
   predictions=torch.stack([values.gather(1,i[:,:,None].expand(-1,-1,H)).mean(1) for i in ix],1)
   old=cache['relation_outputs'][position:end,:,c].cuda();err=float((old-predictions[:,0]).abs().max());maxdiff=max(maxdiff,err)
   assert err<1e-5,('All5 cache reproduction',cell,method,bi,c,err)
   truth=y[:,:,c];E=predictions-truth[:,None];fused=B[:,:,c][:,None]+lam*(predictions-B[:,:,c][:,None]);F=fused-truth[:,None]
   arrays['retrieval'][position:end,c]=torch.stack([E.square().mean(-1),E.abs().mean(-1)],-1).cpu()
   arrays['stage2'][position:end,c]=torch.stack([F.square().mean(-1),F.abs().mean(-1)],-1).cpu()
   for j,i in enumerate(ix):ids[position:end,c,j]=pool.gather(1,i).cpu()
   for j in range(5):arrays['replacement'][position:end,c,j]=1-(ix[0][:,:,None]==ix[6+j][:,None,:]).any(-1).float().mean(-1).cpu()
   he=E[:,1:6];norm=he.norm(dim=-1);den=norm[:,:,None]*norm[:,None,:];cos=torch.where(den>1e-12,he@he.transpose(1,2)/den.clamp_min(1e-12),torch.zeros_like(den));arrays['error_cos'][position:end,c]=cos.cpu()
   for i in range(5):
    for j in range(5):arrays['head_overlap'][position:end,c,i,j]=(ix[1+i][:,:,None]==ix[1+j][:,None,:]).any(-1).float().mean(-1).cpu()
   d=(values-truth[:,None]).square().mean(-1);best=p.stable_topk_indices(d,10,largest=False)
   histories=exp.memory_x[:,:,c][pool];dx=histories-histories[:,:,-1:];qx=x[:,:,c]-x[:,-1:,c]
   rawcos=torch.nn.functional.cosine_similarity(dx,qx[:,None,:],dim=-1,eps=1e-8)
   rc=rawcos-rawcos.mean(-1,keepdim=True);util=-d+d.mean(-1,keepdim=True);den=rc.norm(dim=-1)*util.norm(dim=-1);corr=torch.where(den>1e-12,(rc*util).sum(-1)/den.clamp_min(1e-12),torch.zeros_like(den))
   arrays['quality'][position:end,c]=torch.stack([d.min(-1).values,d.gather(1,best).mean(-1),d.gather(1,ix[0]).mean(-1),d.mean(-1),rawcos.mean(-1),rawcos.gather(1,ix[0]).mean(-1),corr],-1).cpu()
  position=end
  if bi%50==0:print('PROGRESS',cell,method,position,N,flush=True)
 assert position==N
 summary=dict(cell=cell,method=method,seed=0,lambda_fixed=lam,checkpoint_sha256=manifest['checkpoint_sha256'],base_sha256=p.sha(p.CELLS[cell]['base_checkpoint']),pool_sha256=manifest['pool_sha256'],analysis_source_sha256=p.sha(__file__),query_order='dataset start index then channel; full test',n=N,channels=C,all5_cache_maxdiff=maxdiff,subsets=NAMES,python=sys.executable,torch=torch.__version__,matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
 if smoke:
  dump(R/'reports/SMOKE.json',dict(status='PASS',**summary));return
 actual=arrays['stage2'][:,:,0].mean((0,1),dtype=np.float64)
 assert abs(actual[0]-metrics['test_mse'])<1e-5 and abs(actual[1]-metrics['test_mae'])<1e-5
 klactual=arrays['kl'].mean((0,1),dtype=np.float64);assert abs(klactual[0]-klmetrics['test_mse'])<1e-5 and abs(klactual[1]-klmetrics['test_mae'])<1e-5
 summary.update(all5_mse=float(actual[0]),all5_mae=float(actual[1]),stage2_reproduction=True,kl_stage2_reproduction=True,kl_mse=float(klactual[0]),kl_mae=float(klactual[1]),base_mse=float(arrays['base'][:,:,0].mean(dtype=np.float64)))
 np.savez(dest/'query_metrics.npz',starts=starts,top10_ids=ids,**arrays)
 report(dest,summary,arrays,starts,threshold,H)
 dump(dest/'COMPLETE.json',summary);print('COMPLETED',cell,method,actual,flush=True)
def report(dest,summary,a,starts,threshold,H):
 S=a['stage2'];rows=[]
 for j,name in enumerate(NAMES):
  delta=S[:,:,j,0]-S[:,:,0,0];ci=block_ci(delta,starts,2*H)
  rows.append(dict(subset=name,mse=float(S[:,:,j,0].mean(dtype=np.float64)),mae=float(S[:,:,j,1].mean(dtype=np.float64)),delta_mse_vs_All5=ci['mean'],ci95=str(ci['ci95']),blocks=ci['blocks'],improved_query_channel_fraction=float((delta<0).mean()),worsened_query_channel_fraction=float((delta>0).mean()),improved_query_fraction=float((delta.mean(1)<0).mean()),worsened_query_fraction=float((delta.mean(1)>0).mean()),mean_positive_harm=float(np.maximum(delta,0).mean()),retrieval_mse=float(a['retrieval'][:,:,j,0].mean()),top10_replacement=float(a['replacement'][:,:,j-6].mean()) if j>=6 else ''))
 csvout(dest/'HEAD_FIXED_LAMBDA.csv',rows)
 pairs=[];success=S[:,:,1:6,0]<a['base'][:,:,None,0]
 for i in range(5):
  for j in range(i+1,5):
   pairs.append(dict(head_i=i+1,head_j=j+1,both_success=float((success[:,:,i]&success[:,:,j]).mean()),both_failure=float((~success[:,:,i]&~success[:,:,j]).mean()),i_success_j_failure=float((success[:,:,i]&~success[:,:,j]).mean()),j_success_i_failure=float((success[:,:,j]&~success[:,:,i]).mean()),error_cosine=float(a['error_cos'][:,:,i,j].mean()),top10_overlap=float(a['head_overlap'][:,:,i,j].mean())))
 csvout(dest/'HEAD_SUCCESS_FAILURE.csv',pairs)
 strata=[]
 for f,name in enumerate(FEATURES):
  labels=(a['features'][:,:,f]>threshold[0,:,f]).astype(int)+(a['features'][:,:,f]>threshold[1,:,f]).astype(int)
  for group in range(3):
   mask=labels==group
   for h in range(5):
    if not mask.any():
     strata.append(dict(feature=name,stratum=group,removed_head=h+1,n_query_channels=0,delta_mse=None,ci95=None,all5_mse=None,removed_mse=None,replacement=None));continue
    d=S[:,:,6+h,0]-S[:,:,0,0]
    strata.append(dict(feature=name,stratum=group,removed_head=h+1,n_query_channels=int(mask.sum()),delta_mse=float(d[mask].mean()),ci95=str(masked_block_ci(d,mask,starts,2*H)),all5_mse=float(S[:,:,0,0][mask].mean()),removed_mse=float(S[:,:,6+h,0][mask].mean()),replacement=float(a['replacement'][:,:,h][mask].mean())))
 csvout(dest/'INPUT_STRATA.csv',strata)
 best=S[:,:,1:6,0].min(-1);gain=best-S[:,:,0,0]
 query_s=S[:,:,:,0].mean(1);query_gain=query_s[:,1:6].min(-1)-query_s[:,0]
 dump(dest/'QUERY_AGGREGATE_COMPLEMENTARITY.json',dict(unit='query averaged across channels before selecting best single head',all5_beats_best_single_fraction=float((query_gain>0).mean()),all5_worse_fraction=float((query_gain<0).mean()),mean_positive_harm=float(np.maximum(-query_gain,0).mean()),block_ci=block_ci(query_gain,starts,2*H)))
 dump(dest/'COMPLEMENTARITY.json',dict(all5_beats_best_single_fraction=float((gain>0).mean()),all5_worse_than_best_single_fraction=float((gain<0).mean()),mean_gain_vs_best_single=float(gain.mean()),mean_harm_vs_best_single=float(np.maximum(-gain,0).mean()),block_ci_gain=block_ci(gain,starts,2*H),success_definition='fixed-lambda single-head Stage2 MSE below same-query Base',oracle_best_single='future-based diagnostic only, not deployable',quality_columns=['best_individual_P100_mse','oracle_individual_top10_mean_mse','selected_All5_individual_mean_mse','P100_individual_mean_mse','P100_input_delta_last_cosine_eps1e-8','selected_All5_input_delta_last_cosine_eps1e-8','within_P100_input_cosine_vs_negative_future_mse_correlation_zero_if_degenerate'],quality_mean=a['quality'].mean((0,1)).tolist()))
def tests():
 prob=torch.zeros(2,5,12);prob[:,:,0:10]=.1;valid=torch.ones(2,12,dtype=torch.bool);ix=selections(prob,valid)
 assert all(torch.equal(ix[0],x) for x in ix)
 prob.zero_();prob[:,0,11]=1;prob[:,1:,0]=1;ix=selections(prob,valid)
 assert bool((ix[0]==11).any()) and not bool((ix[6]==11).any())
 x=torch.randn(4,96,3);assert features(x).shape==(4,3,4);assert torch.isfinite(features(torch.ones_like(x))).all()
 ci=block_ci(np.ones((100,3)),np.arange(100),20);assert ci['mean']==1 and ci['ci95']==[1.,1.]
 # Fixed lambda endpoints and candidate replacement identity.
 B=torch.randn(4,96);Y=torch.randn(4,96);assert torch.equal(B+0*(Y-B),B);assert torch.allclose(B+(Y-B),Y,atol=1e-6)
 dump(R/'reports/UNIT_TESTS.json',dict(status='PASS',checks=['identical-head LOO identity','distinct-head removal changes selected candidate','input feature shape','constant input finite','block CI identity','lambda endpoints']));print('UNIT PASS')
def controller():
 tests();preflight()
 while not (p.HERE/'results/runs/Weather_720/saraf_time_aligned/V5/head_metrics_test.json').exists():
  if not Path('/proc/3587985').exists():raise RuntimeError('P100 main pipeline stopped before final head report')
  time.sleep(30)
 # Do not overlap with the final original pipeline inference.
 while Path('/proc/3587985').exists():
  stat=Path('/proc/3587985/stat').read_text().split()[2]
  if stat=='Z':break
  time.sleep(10)
 preflight()
 for cell in p.ORDER:
  for method in p.METHODS:
   log=R/'logs'/f'{cell}_{method}.log';log.parent.mkdir(parents=True,exist_ok=True)
   dump(R/'STATUS.json',dict(status='RUNNING',cell=cell,method=method,controller_pid=os.getpid()))
   with log.open('a') as f:
    if cell==p.ORDER[0] and method==p.METHODS[0]:subprocess.run([sys.executable,__file__,'smoke','--cell',cell,'--method',method],stdout=f,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,__file__,'analyze','--cell',cell,'--method',method],stdout=f,stderr=subprocess.STDOUT,check=True)
 dump(R/'STATUS.json',dict(status='COVERAGE_RUNNING',controller_pid=os.getpid()))
 with (R/'logs/coverage.log').open('a') as f:subprocess.run(['/usr/bin/python3',str(R/'scripts/full_coverage.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
 subprocess.run([sys.executable,str(R/'scripts/final_report.py')],check=True)
 dump(R/'STATUS.json',dict(status='COMPLETED',controller_pid=os.getpid()))
if __name__=='__main__':
 torch.set_num_threads(4)
 ap=argparse.ArgumentParser();ap.add_argument('mode');ap.add_argument('--cell');ap.add_argument('--method');args=ap.parse_args()
 R.mkdir(parents=True,exist_ok=True)
 try:
  if args.mode=='preflight':preflight()
  elif args.mode=='tests':tests()
  elif args.mode=='controller':controller()
  else:analyze(args.cell,args.method,args.mode=='smoke')
 except Exception:
  dump(R/'FAILED.json',dict(time=time.time(),traceback=traceback.format_exc()));raise

