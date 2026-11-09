#!/usr/bin/env python3
"""CPU-only matched query P100 overlap, independent of live training."""
import sys,itertools
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parent));import query_analysis as q
rows=[]
for cell in q.p.ORDER:
 pools={m:torch.load(q.p.HERE/'results/pools'/cell/m/'test.pt',map_location='cpu') for m in q.p.METHODS}
 starts=pools[q.p.METHODS[0]]['query_start_idx'].numpy();output={}
 for m1,m2 in itertools.combinations(q.p.METHODS,2):
  a,b=pools[m1],pools[m2];assert torch.equal(a['query_start_idx'],b['query_start_idx'])
  aa=a['pool_idx_global'].numpy();bb=b['pool_idx_global'].numpy();shape=aa.shape[:2];aa=aa.reshape(-1,100);bb=bb.reshape(-1,100);over=np.zeros(len(aa))
  stride=max(aa.max(),bb.max())+1
  for i in range(0,len(aa),2048):
   av,bv=aa[i:i+2048],bb[i:i+2048];offset=np.arange(len(av))[:,None]*stride
   over[i:i+len(av)]=np.isin((av+offset).ravel(),(bv+offset).ravel(),assume_unique=True).reshape(-1,100).mean(1)
  over=over.reshape(shape);ci=q.block_ci(over,starts,2*int(cell.split('_')[1]));rows.append(dict(cell=cell,method1=m1,method2=m2,mean_P100_overlap=ci['mean'],ci95=str(ci['ci95']),blocks=ci['blocks'],n_query_channel=int(over.size)))
  output[m1+'__'+m2]=over
 dest=q.R/'cache_analysis'/cell;dest.mkdir(parents=True,exist_ok=True);np.savez(dest/'query_overlap.npz',starts=starts,**output)
 print('CACHE OVERLAP COMPLETE',cell,flush=True)
q.csvout(q.R/'reports/CANDIDATE_POOL_OVERLAP.csv',rows)

