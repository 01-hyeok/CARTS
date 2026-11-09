"""Inspect CPU/original score differences; no repair or GPU use."""
import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS');sys.path[:0]=[str(ROOT/'experiments/p100_head_retrieval_quality_legacy_pool02'),str(ROOT/'experiments/learned_dynamics_cross01')]
import runtime as rt
from train_models import Features,Arm
from experiment_common import load
import numpy as np,torch
torch.set_num_threads(1)
OLD=ROOT/'results/TRACK-LEARNED-DYNAMICS-CROSS01';OUT=ROOT/'results/TRACK-PREDICTIVE-SCORE-CALIBRATION01/full_extension01'
def main():
    os.chdir(ROOT);examples=[]
    for cell in ('Weather_192','Weather_336','Weather_720','Weather_96'):
        d=rt.CellData(cell,'cpu');feat=Features(d,OLD/'features'/cell);source=OLD/'P100'/cell;sel=json.loads((source/'SELECTION.json').read_text())['arms']['D2_MLP']['best_epoch'];arm=Arm('D2_MLP');arm.load_state_dict(load(source/'checkpoints/D2_MLP'/('epoch%02d.pt'%sel))['model']);arm.eval()
        for split in ('val','test'):
            found=0;pool=np.load(d.pool_dir/(split+'.npy'),mmap_mode='r')
            for path in sorted((source/'frozen_scalar_cache'/split).glob('batch*.npz')):
                a=int(path.stem[5:])*32;b=min(a+32,len(d.datasets[split]));ids=torch.from_numpy(np.array(pool[a:b],dtype=np.int64,copy=True))
                with np.load(path) as z:p=torch.from_numpy(z['p'].copy());distance=z['distance'].copy()
                with torch.no_grad():raw=arm.raw_chunks(feat,split,np.arange(a,b),ids);s=arm.score(p,raw,torch.ones_like(p,dtype=torch.bool),ranking=True).numpy()
                order=np.argsort(-s,axis=-1,kind='stable')[...,:10];actual=np.take_along_axis(ids.numpy(),order,-1)
                with np.load(source/'final_raw'/split/path.name) as old:
                    mismatch=np.argwhere((actual!=old['D2_MLP__selected_ids']).any(-1))
                    for q,c in mismatch:
                        oldids=old['D2_MLP__selected_ids'][q,c];newids=actual[q,c];oldpos=np.array([int(np.flatnonzero(ids[q,c].numpy()==v)[0]) for v in oldids]);newpos=order[q,c]
                        added=np.setdiff1d(newpos,oldpos);removed=np.setdiff1d(oldpos,newpos)
                        examples.append(dict(cell=cell,split=split,query_row=int(a+q),channel=int(c),set_difference_count=len(added),max_abs_score=float(abs(s[q,c]).max()),cpu_score_std=float(s[q,c].std()),original_score_std=float(old['D2_MLP__score_std'][q,c]),score_mean_delta=float(s[q,c].mean()-old['D2_MLP__score_mean'][q,c]),cpu_rank10_rank11_gap=float(np.sort(s[q,c])[-10]-np.sort(s[q,c])[-11]),cpu_retmse_delta=float(distance[q,c,newpos].mean(dtype=np.float64)-old['D2_MLP__retmse'][q,c]),changed_candidate_score_range=float(np.ptp(s[q,c,np.concatenate([added,removed])])) if len(added) else 0.,beta=float(arm.fusion.beta.detach())))
                        found+=1
                        if found>=5:break
                if found>=5:break
            print('CPU_ORDER_DIAG',cell,split,found,flush=True)
    with (OUT/'CPU_ORDER_DIAGNOSIS.json').open('x') as f:json.dump(dict(status='DIAGNOSTIC_ONLY_NOT_GPU_REPRODUCTION',examples=examples,selection_policy_changed=False,results_repaired=False),f,indent=2)
if __name__=='__main__':main()
