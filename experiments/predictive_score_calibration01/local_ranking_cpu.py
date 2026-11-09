"""P100 full-query local ranking on immutable scalar caches and selected scorers."""
import os,sys,json,csv,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/data/pjh_workspace/CARTS')
sys.path[:0]=[str(ROOT/'experiments/p100_head_retrieval_quality_legacy_pool02'),str(ROOT/'experiments/learned_dynamics_cross01')]
import runtime as rt
from train_models import Features,Arm
from experiment_common import load,sha
import torch,numpy as np
from scipy.stats import rankdata
torch.set_num_threads(1)
OLD=ROOT/'results/TRACK-LEARNED-DYNAMICS-CROSS01';OUT=ROOT/'results/TRACK-PREDICTIVE-SCORE-CALIBRATION01'

def metrics(score,distance,pairs):
    # All arrays are [query-channel, local candidate]. Ties get average ranks.
    sr=rankdata(score,axis=-1);dr=rankdata(-distance,axis=-1)
    sr-=sr.mean(-1,keepdims=True);dr-=dr.mean(-1,keepdims=True)
    den=np.sqrt((sr*sr).sum(-1)*(dr*dr).sum(-1));defined=den>0
    rho=np.divide((sr*dr).sum(-1),den,out=np.zeros(len(den)),where=defined)
    i,j=pairs;truth=distance[:,j]-distance[:,i];pred=score[:,i]-score[:,j]
    valid=np.abs(truth)>1e-10;correct=np.where(pred==0,.5,(pred*truth>0).astype(float));w=np.abs(truth)*valid
    chosen=np.argsort(-score,axis=-1,kind='stable')[:,:10];oracle=np.argsort(distance,axis=-1,kind='stable')[:,:10]
    sm=np.take_along_axis(distance,chosen,-1);om=np.take_along_axis(distance,oracle,-1)
    discount=1/np.log2(np.arange(2,12));ndcg=((1/(1+sm))*discount).sum(-1)/((1/(1+om))*discount).sum(-1)
    recall=(chosen[:,:,None]==oracle[:,None,:]).any(-1).mean(-1)
    return dict(rho_sum=rho.sum(),rho_n=defined.sum(),pair_correct=(correct*valid).sum(),pair_n=valid.sum(),pair_ties=(~valid).sum(),margin_correct=(correct*w).sum(),margin_sum=w.sum(),ndcg_sum=ndcg.sum(),recall_sum=recall.sum(),rows=len(score))

def main():
    os.chdir(ROOT);started=time.time();results=[];replay=[]
    policies={}
    for n in (20,50,100):
        i,j=np.triu_indices(n,1);all_count=len(i)
        if all_count>512:
            ix=np.sort(np.random.RandomState(1000+n).choice(all_count,512,replace=False));i,j=i[ix],j[ix]
        policies[n]=(i,j)
    for source in sorted((OLD/'P100').iterdir()):
        if not (source/'COMPLETE.json').exists():continue
        cell=source.name;d=rt.CellData(cell,'cpu');features=Features(d,OLD/'features'/cell)
        selection=json.loads((source/'SELECTION.json').read_text())['arms'];arms={}
        for name in ('D2_MLP','C2_MLP'):
            arm=Arm(name);ep=selection[name]['best_epoch'];arm.load_state_dict(load(source/'checkpoints'/name/('epoch%02d.pt'%ep))['model']);arm.eval();arms[name]=arm
        for split in ('val','test'):
            totals={};mismatch={m:0 for m in arms};max_diff={m:0. for m in arms};count=0
            pool=np.load(d.pool_dir/(split+'.npy'),mmap_mode='r')
            for path in sorted((source/'frozen_scalar_cache'/split).glob('batch*.npz')):
                assert sha(path)==json.loads(path.with_suffix('.json').read_text())['sha256']
                batch=int(path.stem[5:]);a=batch*32;b=min(a+32,len(d.datasets[split]));rows=np.arange(a,b)
                with np.load(path) as z:
                    assert np.array_equal(z['starts'],d.datasets[split].starts[a:b]);p=torch.from_numpy(z['p'].copy());distance=z['distance'].copy().reshape(-1,100).astype(np.float64)
                ids=torch.from_numpy(np.array(pool[a:b],copy=True));mask=torch.ones_like(p,dtype=torch.bool)
                scores={'S0':p.double().numpy().reshape(-1,100)}
                with torch.no_grad():
                    for m,arm in arms.items():scores[m]=arm.score(p,arm.raw_chunks(features,split,rows,ids),mask,ranking=True).numpy().reshape(-1,100)
                final=source/'final_raw'/split/path.name
                with np.load(final) as z:
                    for m,s in scores.items():
                        ix=np.argsort(-s,axis=-1,kind='stable')[:,:10]
                        actual=np.take_along_axis(ids.numpy().reshape(-1,100),ix,-1).reshape(b-a,d.c,10)
                        if m in arms:
                            mismatch[m]+=int((actual!=z[m+'__selected_ids']).any(-1).sum())
                            delta=np.take_along_axis(distance,ix,-1).mean(-1).reshape(b-a,d.c)-z[m+'__retmse']
                            max_diff[m]=max(max_diff[m],float(np.abs(delta).max()))
                base_order=np.argsort(-scores['S0'],axis=-1,kind='stable')
                for n,pairs in policies.items():
                    ix=base_order[:,:n];dd=np.take_along_axis(distance,ix,-1)
                    for m,s in scores.items():
                        mm=metrics(np.take_along_axis(s,ix,-1),dd,pairs);key=(m,n)
                        total=totals.setdefault(key,{k:0. for k in mm})
                        for k,v in mm.items():total[k]+=float(v)
                count+=b-a
            assert count==len(d.datasets[split])
            # A mismatched CPU scorer is diagnostic-only; never silently label exact replay.
            replay.append(dict(cell=cell,split=split,queries=count,ordered_id_mismatches=mismatch,max_retmse_delta=max_diff))
            for (m,n),v in totals.items():
                results.append(dict(cell=cell,support='P100',split=split,method=m,subset='S0_TOP%d'%n,spearman=v['rho_sum']/v['rho_n'] if v['rho_n'] else None,spearman_defined=int(v['rho_n']),pair_accuracy=v['pair_correct']/v['pair_n'] if v['pair_n'] else None,pair_denominator=int(v['pair_n']),future_ties=int(v['pair_ties']),margin_pair_accuracy=v['margin_correct']/v['margin_sum'] if v['margin_sum'] else None,ndcg10=v['ndcg_sum']/v['rows'],local_oracle_recall10=v['recall_sum']/v['rows'],query_channels=int(v['rows']),cpu_replay_exact=m=='S0' or mismatch[m]==0))
            print('LOCAL_RANKING_COMPLETE',cell,split,mismatch,flush=True)
    with (OUT/'P0_LOCAL_RANKING.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,list(results[0]));w.writeheader();w.writerows(results)
    with (OUT/'LOCAL_RANKING_CPU_COMPLETE.json').open('x') as f:
        json.dump(dict(status='P100_LOCAL_RANKING_COMPLETE',gpu_used=False,seconds=time.time()-started,replay=replay,full_pending=True,sampling='all unordered pairs for top20; 512 without replacement fixed RandomState(1000+n) for top50/100; same positional pairs for every method/query',tie_policy='future abs delta <=1e-10 excluded; score exact tie half credit; Spearman average ties excludes constant rows',ndcg='gain 1/(1+FutureMSE), local IDCG; local Oracle10 from same restricted subset',policies={str(k):[a.tolist(),b.tolist()] for k,(a,b) in policies.items()}),f,indent=2)
if __name__=='__main__':main()
