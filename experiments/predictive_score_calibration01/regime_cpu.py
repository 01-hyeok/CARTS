"""All-history regime features and Train-fixed bins; no CUDA or learned fit."""
import os,sys,json,csv,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/data/pjh_workspace/CARTS');HERE=ROOT/'experiments/predictive_score_calibration01'
sys.path[:0]=[str(ROOT/'experiments/p100_head_retrieval_quality_legacy_pool02'),str(ROOT/'experiments/learned_dynamics_cross01')]
import runtime as rt
import numpy as np
import torch
from learned_scores import fit_ar
torch.set_num_threads(1)
OLD=ROOT/'results/TRACK-LEARNED-DYNAMICS-CROSS01';OUT=ROOT/'results/TRACK-PREDICTIVE-SCORE-CALIBRATION01'
NAMES=['trend_slope','recent_volatility','long_short_vol_ratio','trend_strength','dominant_frequency','spectral_entropy','autocorrelation_strength','ar_residual_variance']

def features(past):
    x=past.double();h=x.shape[-1];t=torch.arange(h,dtype=x.dtype);t=t-t.mean()
    centered=x-x.mean(-1,keepdim=True);slope=(centered*t).sum(-1)/t.square().sum()
    residual=centered-slope[...,None]*t
    var=centered.square().mean(-1);strength=(1-residual.square().mean(-1)/var.clamp_min(1e-12)).clamp(0,1)
    diff=x.diff(dim=-1);short=diff[...,-min(24,h-1):].std(-1,unbiased=False);long=diff.std(-1,unbiased=False)
    spec=torch.fft.rfft(centered,dim=-1).abs().square()[...,1:]
    probs=spec/spec.sum(-1,keepdim=True).clamp_min(1e-12)
    entropy=-(probs*probs.clamp_min(1e-12).log()).sum(-1)/np.log(spec.shape[-1])
    dominant=(spec.argmax(-1).double()+1)/h;dominant=torch.where(var>1e-12,dominant,torch.zeros_like(dominant))
    # Maximum absolute normalized correlation over prespecified lags <= min(24,H/4).
    ac=[]
    for lag in range(1,min(24,h//4)+1):
        a,b=centered[...,:-lag],centered[...,lag:]
        ac.append((a*b).sum(-1)/(a.square().sum(-1)*b.square().sum(-1)).sqrt().clamp_min(1e-12))
    ar=fit_ar(x)['residual']
    return torch.stack([slope,short,long/short.clamp_min(1e-12),strength,dominant,entropy,torch.stack(ac,-1).abs().max(-1).values,ar],-1).numpy()

def main():
    began=time.time();os.chdir(ROOT);outrows=[];manifests=[]
    for cell in sorted(p.name for p in (OLD/'features').iterdir() if (p/'COMPLETE.json').exists()):
        d=rt.CellData(cell,'cpu');folder=OUT/'regime_features'/cell;folder.mkdir(parents=True,exist_ok=True)
        arrays={}
        for split in ('train','val','test'):
            path=folder/(split+'.npz')
            if not path.exists():
                ds=d.datasets[split];parts=[]
                for a in range(0,len(ds),64):
                    idx=np.arange(a,min(a+64,len(ds)))[:,None]+np.arange(d.h)
                    past=torch.from_numpy(np.asarray(ds.data_x[idx],np.float32)).permute(0,2,1)
                    parts.append(features(past))
                values=np.concatenate(parts);assert np.isfinite(values).all()
                with path.open('xb') as f:np.savez(f,features=values,starts=ds.starts,names=np.array(NAMES))
            with np.load(path) as z:
                assert np.array_equal(z['starts'],d.datasets[split].starts);arrays[split]=z['features'].copy()
        thresholds=np.quantile(arrays['train'].reshape(-1,len(NAMES)),[1/3,2/3],axis=0)
        meta=dict(cell=cell,seq_len=d.h,feature_names=NAMES,thresholds=thresholds.tolist(),fit_split='train',bins='low <= q33; mid q33 < x <= q67; high > q67; equal quantiles may yield empty bins',data=d.data_info,future_inputs=False)
        p=folder/'TRAIN_THRESHOLDS.json'
        if not p.exists():p.write_text(json.dumps(meta,indent=2))
        else:assert json.loads(p.read_text())==meta
        manifests.append(meta)
        for support in ('P100','FULL'):
            source=OLD/support/cell
            if not (source/'COMPLETE.json').exists():continue
            for split in ('val','test'):
                deltas={m:[] for m in ('D2_MLP','C2_MLP')}
                for p in sorted((source/'final_raw'/split).glob('batch*.npz')):
                    with np.load(p) as z:
                        for m in deltas:deltas[m].append(z[m+'__delta_retmse_vs_s0'].copy())
                for m,parts in deltas.items():
                    delta=np.concatenate(parts);assert delta.shape==arrays[split].shape[:2]
                    for j,name in enumerate(NAMES):
                        x=arrays[split][...,j];bins=(x>thresholds[0,j]).astype(int)+(x>thresholds[1,j]).astype(int)
                        for binid in range(3):
                            mask=bins==binid;y=delta[mask];n=len(y)
                            outrows.append(dict(cell=cell,support=support,split=split,method=m,feature=name,bin=binid,count=n,mean_delta=float(y.mean()) if n else None,win=int((y<0).sum()),tie=int((y==0).sum()),loss=int((y>0).sum()),worst_delta=float(y.max()) if n else None,selection_use=False))
        print('REGIME_COMPLETE',cell,flush=True)
    with (OUT/'P0_REGIME_ANALYSIS.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,list(outrows[0]));w.writeheader();w.writerows(outrows)
    (OUT/'REGIME_CPU_COMPLETE.json').write_text(json.dumps(dict(status='PASS_ALL_AVAILABLE_COMPLETED_CONDITIONS',cells=len(manifests),rows=len(outrows),seconds=time.time()-began,train_thresholds_only=True,gpu_used=False),indent=2))
if __name__=='__main__':main()
