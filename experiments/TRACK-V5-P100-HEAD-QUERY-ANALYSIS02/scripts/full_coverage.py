#!/usr/bin/env python3
"""Matched-query Full oracle/head retention. Historical Full runtime only."""
import sys,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/data/pjh_workspace/CARTS');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
import query_analysis as q
from experiments.full_head_analysis01 import full_analysis as f
from experiments.p100_preselection_comparison import pipeline as p
@torch.no_grad()
def run(cell):
 out=q.R/'coverage'/cell
 if (out/'COMPLETE.json').exists():return
 assert sys.executable=='/usr/bin/python3'
 torch.backends.cuda.matmul.allow_tf32=True
 cfg,canonical,sel,exp,args,model,heads,base=f.setup(cell,32)
 assert p.sha(sel['selected_checkpoint_path'])==q.read(f.OUT/cell/'CHECKPOINT_PROVENANCE.json')['checkpoint_sha256']
 trainset,_=exp._get_data(flag='train',shuffle=False)
 assert np.array_equal(exp.memory_sampler.starts,trainset.starts)
 ds,loader=exp._get_data(flag='test',shuffle=False);n=len(ds);take=np.linspace(0,n-1,64).round().astype(int);take=set(take.tolist())
 old=torch.load(f.OUT/cell/'cache/12345/test.pt',map_location='cpu')
 pools={m:p.CandidatePoolCache(p.HERE/'results/pools'/cell/m/'test.pt',dict(seq_len=args.seq_len,pred_len=args.pred_len,candidate_mask='raft',candidate_pool_size=100,candidate_pool_metric=p.METRICS[m])) for m in p.METHODS}
 rows=[];fullids=[];maxdiff=0.
 for c in range(args.enc_in):
  keys=f.bank(model,exp.memory_x,c,0);pos=0
  for x,y,s in loader:
   ix=[i for i in range(len(x)) if pos+i in take]
   if not ix:pos+=len(x);continue
   assert torch.equal(s,old['query_start_idx'][pos:pos+len(x)])
   x=x.float().cuda();y=y.float().cuda();mask,_=exp._candidate_mask(s)
   scores=torch.einsum('bsd,nd->bsn',heads(f.encode_raw_channel_first(model,x,c)),keys)
   mix,prob,_=p.mean_mixture_topk_selection(scores,mask,.1,10);hi=p.stable_topk_indices(prob,10,largest=True)
   mem,off=p.memory_value(args,x,exp.memory_y,exp.memory_x_last,c)
   pred=mem[mix]+off[:,None,None];retr=pred.mean(1);err=float((retr[ix]-old['relation_outputs'][pos:pos+len(x),:,c][ix].cuda()).abs().max());maxdiff=max(maxdiff,err)
   assert err<1e-5,('Full canonical cache parity',cell,c,pos,err)
   d=-p.individual_utility_memsafe(mem,off[ix],y[ix,:,c],1024)
   oracle=p.stable_topk_indices(d.masked_fill(~mask[ix],float('inf')),10,largest=False)
   for j,i in enumerate(ix):
    fullids.append(dict(start=int(s[i]),channel=c,full_top10=mix[i].cpu().tolist(),head_top10=hi[i].cpu().tolist(),oracle_top10=oracle[j].cpu().tolist()))
   for method,pc in pools.items():
    ids=pc.lookup(s,c,torch.device('cuda'))[ix]
    assert mask[ix].gather(1,ids).all()
    retained=lambda idx:(idx[:,:,None]==ids[:,None,:]).any(-1).float().mean(-1)
    cols=dict(full_mixture_retention=retained(mix[ix]),full_oracle_retention=retained(oracle))
    for h in range(5):cols[f'full_head{h+1}_retention']=retained(hi[ix,h])
    dp=d.gather(1,ids);cols['pool_best_individual_mse']=dp.min(-1).values;cols['full_best_individual_mse']=d.masked_fill(~mask[ix],float('inf')).min(-1).values
    for j,i in enumerate(ix):rows.append(dict(method=method,start=int(s[i]),channel=c,**{k:float(v[j]) for k,v in cols.items()}))
   pos+=len(x)
  print('COVERAGE',cell,'channel',c,flush=True)
 q.csvout(out/'MATCHED_QUERY_COVERAGE.csv',rows);q.dump(out/'full_top10_ids.json',fullids)
 aggregate=[]
 for method in p.METHODS:
  rr=[r for r in rows if r['method']==method];record=dict(method=method,n_query_channel=len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in rr[0] if k not in ['method','start','channel']})
  starts=np.array(sorted({r['start'] for r in rr}))
  for key in ['full_mixture_retention','full_oracle_retention']+[f'full_head{i+1}_retention' for i in range(5)]:
   arr=np.array([[r[key] for r in rr if r['start']==s] for s in starts]);ci=q.block_ci(arr,starts,2*args.pred_len);record[key+'_ci95']=str(ci['ci95']);record['blocks']=ci['blocks']
  aggregate.append(record)
 q.csvout(out/'COVERAGE_SUMMARY.csv',aggregate)
 q.dump(out/'COMPLETE.json',dict(cell=cell,split='test',sample='64 evenly spaced test row indices, all channels; same query/channel for every P100 method',query_count=64,full_checkpoint=sel['selected_checkpoint_path'],full_checkpoint_sha=p.sha(sel['selected_checkpoint_path']),full_cache_maxdiff=maxdiff,python=sys.executable,torch=torch.__version__,caveat='Full and P100 checkpoints differ; retention is descriptive, not a pure preselection causal effect. Future oracle is diagnostic only.'))
if __name__=='__main__':
 torch.set_num_threads(4)
 try:
  for cell in p.ORDER:run(cell)
 except Exception:
  import traceback
  q.dump(q.R/'COVERAGE_FAILED.json',dict(traceback=traceback.format_exc()));raise

