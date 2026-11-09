"""Complete P100 head calibration and boundary diagnostics from canonical caches."""
import os,sys,json,csv,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/data/pjh_workspace/CARTS')
sys.path[:0]=[str(ROOT/'experiments/p100_head_retrieval_quality_legacy_pool02'),str(ROOT/'experiments/learned_dynamics_cross01')]
import runtime as rt
from train_models import Features,Arm
from experiment_common import sha,load,dump
import numpy as np,torch
from scipy.stats import rankdata
torch.set_num_threads(1)
OUT=ROOT/'results/TRACK-PREDICTIVE-SCORE-CALIBRATION01';OLD=ROOT/'results/TRACK-LEARNED-DYNAMICS-CROSS01'
CELLS=['ETTh1_96','ETTh1_336','ETTh1_720','Weather_96','Weather_336','Weather_720','ETTh1_192','Weather_192']

def main():
    os.chdir(ROOT);beg=time.time();calrows=[];summary=[];boundary=[];manifests=[]
    for cell in CELLS:
        cache=OUT/'canonical_cache/P100'/cell;proof=json.loads((cache/'COMPLETE.json').read_text());d=rt.CellData(cell,'cpu');feat=Features(d,OLD/'features'/cell)
        selection=json.loads((OLD/'P100'/cell/'SELECTION.json').read_text())['arms'];arms={}
        for name in ('D2_MLP','C2_MLP'):
            arm=Arm(name);arm.load_state_dict(load(OLD/'P100'/cell/'checkpoints'/name/('epoch%02d.pt'%selection[name]['best_epoch']))['model']);arm.eval();arms[name]=arm
        samples=[];trainrows=np.unique(np.linspace(0,len(d.datasets['train'])-1,256,dtype=int))
        for block in np.unique(trainrows//32):
            path=cache/'train'/('batch%05d.npz'%block);assert sha(path)==proof['files'][str(path.relative_to(cache))]
            with np.load(path) as z:samples.append(z['g'][trainrows[trainrows//32==block]%32].transpose(0,1,3,2).reshape(-1,5))
        fitted=np.concatenate(samples);threshold=np.quantile(fitted,np.arange(1,10)/10,axis=0)
        manifests.append(dict(cell=cell,trainrows=trainrows.tolist(),head_deciles=threshold.tolist()))
        for split in ('val','test'):
            sums=np.zeros((5,10,3));stats=np.zeros((5,5));bs={m:dict(correct=0.,n=0,swaps=0.,added_base_rank=0.,removed_new_rank=0.,boundary_swaps=0.) for m in arms};mismatch={m:0 for m in arms}
            for path in sorted((cache/split).glob('batch*.npz')):
                assert sha(path)==proof['files'][str(path.relative_to(cache))];a=int(path.stem[5:])*32
                with np.load(path) as z:g=z['g'].copy();p=z['p'].copy();ids=z['ids'].copy()
                side=json.loads(path.with_suffix('.json').read_text());dp=Path(side['old_scalar_cache']);assert sha(dp)==side['old_scalar_sha256']
                with np.load(dp) as z:dist=z['distance'].copy().reshape(-1,100).astype(float)
                b=a+len(g);gg=g.reshape(-1,5,100);oracle=np.argsort(dist,axis=-1,kind='stable')[:,:10]
                for h in range(5):
                    scores=gg[:,h].astype(float);bins=np.searchsorted(threshold[:,h],scores,side='left')
                    for k in range(10):
                        m=bins==k;sums[h,k]+=[m.sum(),scores[m].sum(),dist[m].sum()]
                    sr=rankdata(scores,axis=-1);dr=rankdata(-dist,axis=-1);sr-=sr.mean(-1,keepdims=True);dr-=dr.mean(-1,keepdims=True)
                    den=np.sqrt((sr*sr).sum(-1)*(dr*dr).sum(-1));rho=np.divide((sr*dr).sum(-1),den,out=np.zeros(len(den)),where=den>0)
                    chosen=np.argsort(-scores,axis=-1,kind='stable')[:,:10];recall=(chosen[:,:,None]==oracle[:,None,:]).any(-1).mean(-1)
                    rank=np.argsort(np.argsort(-scores,axis=-1,kind='stable'),axis=-1,kind='stable')+1
                    stats[h]+=[rho.sum(),(den>0).sum(),recall.sum(),np.take_along_axis(rank,oracle,-1).mean(-1).sum(),len(scores)]
                base_order=np.argsort(-p.reshape(-1,100),axis=-1,kind='stable');base_rank=np.argsort(base_order,axis=-1,kind='stable')+1;base_top=base_order[:,:10]
                ti=torch.from_numpy(ids);tp=torch.from_numpy(p);mask=torch.ones_like(tp,dtype=torch.bool)
                with np.load(OLD/'P100'/cell/'final_raw'/split/path.name) as old:
                    for name,arm in arms.items():
                        with torch.no_grad():s=arm.score(tp,arm.raw_chunks(feat,split,np.arange(a,b),ti),mask,ranking=True).numpy().reshape(-1,100)
                        order=np.argsort(-s,axis=-1,kind='stable');rank=np.argsort(order,axis=-1,kind='stable')+1;top=order[:,:10]
                        actual=np.take_along_axis(ids.reshape(-1,100),top,-1).reshape(b-a,d.c,10);mismatch[name]+=int((actual!=old[name+'__selected_ids']).any(-1).sum())
                        added=~(top[:,:,None]==base_top[:,None,:]).any(-1);removed=~(base_top[:,:,None]==top[:,None,:]).any(-1)
                        ar=np.take_along_axis(base_rank,top,-1);rr=np.take_along_axis(rank,base_top,-1)
                        bs[name]['swaps']+=added.sum();bs[name]['added_base_rank']+=ar[added].sum();bs[name]['removed_new_rank']+=rr[removed].sum();bs[name]['boundary_swaps']+=((ar>=11)&(ar<=20)&added).sum()
                        for left in (8,9):
                            for right in (10,11):
                                li=base_order[:,left];ri=base_order[:,right];r=np.arange(len(s));truth=dist[r,ri]-dist[r,li];pred=s[r,li]-s[r,ri];valid=np.abs(truth)>1e-10
                                bs[name]['n']+=valid.sum();bs[name]['correct']+=np.where(pred[valid]==0,.5,(pred[valid]*truth[valid]>0)).sum()
            for h in range(5):
                for k in range(10):
                    n,ss,dd=sums[h,k];calrows.append(dict(cell=cell,split=split,head=h,train_decile=k,count=int(n),mean_score=ss/n if n else None,mean_future_mse=dd/n if n else None))
                rs,rn,rec,orr,n=stats[h];summary.append(dict(cell=cell,split=split,head=h,spearman=rs/rn if rn else None,spearman_defined=int(rn),precision10=rec/n,recall10=rec/n,oracle_mean_rank=orr/n,query_channels=int(n)))
            for name,v in bs.items():
                boundary.append(dict(cell=cell,split=split,method=name,pair_accuracy=v['correct']/v['n'] if v['n'] else None,pair_count=int(v['n']),swaps=int(v['swaps']),added_original_rank=v['added_base_rank']/v['swaps'] if v['swaps'] else None,removed_new_rank=v['removed_new_rank']/v['swaps'] if v['swaps'] else None,added_from_original_11_to_20_fraction=v['boundary_swaps']/v['swaps'] if v['swaps'] else None,cpu_id_mismatches=mismatch[name]))
            assert not any(mismatch.values()),'CPU selected-scorer replay differs; require canonical scoring for diagnostics'
        print('HEAD_DIAGNOSTICS_COMPLETE',cell,flush=True)
    for name,rows in [('P0_HEAD_CALIBRATION.csv',calrows),('P0_HEAD_QUALITY.csv',summary),('P0_BOUNDARY_SWAP_RANKS.csv',boundary)]:
        with (OUT/name).open('x',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    dump(OUT/'HEAD_CALIBRATION_THRESHOLDS.json',dict(fit_split='train',bin_policy='left-search against fixed Train deciles; repeated quantiles produce empty bins',cells=manifests))
    dump(OUT/'HEAD_DIAGNOSTICS_COMPLETE.json',dict(status='PASS_P100_ALL8',seconds=time.time()-beg,boundary='Original S0 ranks 9/10 against ranks11/12; future abs delta <=1e-10 excluded, score tie half-credit',precision='Oracle10 membership; precision equals recall because both sets have K=10',near_oracle='Not separately thresholded; Oracle10 counts provided in swap report',full_pending=True))
if __name__=='__main__':main()
