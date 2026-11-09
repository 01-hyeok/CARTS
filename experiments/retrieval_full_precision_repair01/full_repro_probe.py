import sys,json,os,time
from pathlib import Path
ROOT=Path('/data/pjh_workspace/CARTS')
sys.path.insert(0,str(ROOT/'experiments/retrieval_failure_attribution01'))
import diagnostic_adapter as ad
import torch,numpy as np
from utils.full_candidate_bank import compute_scores_full_grad_channel_first
from scripts.build_t2_true_original_kl_cache01 import compute_scores_true_original_kl
from scripts.train_t_pure_multislot01 import round_robin_topk_selection
from scripts.train_factorial_e2e01 import encode_raw

@torch.no_grad()
def main():
    ad.rt.gpu_guard();torch.set_num_threads(2)
    data=ad.Data('ETTh1_96','cuda:0')
    arch=sys.argv[1] if len(sys.argv)>1 else 'V0'
    row=[r for r in ad.inventory(data.cell,'FULL') if r['architecture']==arch][0]
    split=sys.argv[2] if len(sys.argv)>2 else 'val'
    start=int(sys.argv[3]) if len(sys.argv)>3 else 0
    m=ad.Frozen(data,row,'FULL');x,y,starts=data.batch(split,np.arange(start,start+32))
    bank=data.series[torch.arange(len(data.starts),device=data.device)[:,None]+data.offsets]
    mask=data.mask(starts);ref=m.reference(split)[0][start:start+32]
    out={'torch':torch.__version__,'data':data.data_info,'checkpoint':row,'variants':[]}
    for tf32 in [True,False]:
        torch.backends.cuda.matmul.allow_tf32=tf32;torch.backends.cudnn.allow_tf32=tf32
        for c in range(data.c):
            one,_,_=data.channel_batch(split,start,start+32,c)
            fullinput=m.model._relation_tensor(x,c,c)
            oneinput=m.model._relation_tensor(one,0,0)
            if arch=='V0':
                sc=compute_scores_true_original_kl(m.model,x,bank,c)
                compact=compute_scores_true_original_kl(m.model,one,data.bank_history(c),0)
            else:
                sc=compute_scores_full_grad_channel_first(m.model,m.heads,x,bank,c)
                compact=compute_scores_full_grad_channel_first(m.model,m.heads,one,data.bank_history(c),0)
            for name,s in [('canonical_full',sc),('channel_first',compact)]:
                if arch=='V0':
                    ids=round_robin_topk_selection(s,mask,10)
                    stable=ad.rt.stable_topk_indices(s[:,0].masked_fill(~mask,-float('inf')),10,largest=True)
                else:
                    ids,_,prob=ad.rt.mean_mixture_topk_selection(s,mask,.1,10)
                    stable=ad.rt.stable_topk_indices(prob,10,largest=True)
                vals=data.values(torch.arange(len(data.starts),device=data.device),c)
                pred=(vals[ids]+x[:,-1,c][:,None,None]).mean(1).cpu().numpy()
                diff=np.abs(pred-ref[:,:,c])
                r=dict(tf32=tf32,channel=c,path=name,max_prediction_diff=float(diff.max()),prediction_mismatched_queries=int((~np.isclose(pred,ref[:,:,c],atol=1e-5,rtol=1e-5)).any(-1).sum()),selection_rule_equal=bool(torch.equal(ids,stable)),input_equal=bool(torch.equal(fullinput,oneinput)),score_max_difference=float((sc-compact).abs().max()))
                out['variants'].append(r);print(json.dumps(r),flush=True)
    dst=ROOT/'results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/00_Audit/FULL_REPAIR01'
    dst.mkdir(parents=True,exist_ok=True)
    ad.rt.write_json(dst/('probe_'+arch+'_'+split+'_'+str(start)+'_'+torch.__version__.split('+')[0]+'.json'),out)
if __name__=='__main__':main()
