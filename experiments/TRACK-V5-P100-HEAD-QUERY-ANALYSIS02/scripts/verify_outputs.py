#!/usr/bin/env python3
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import query_analysis as q
rows=[];hashes=set()
for cell in q.p.ORDER:
 aligned=None
 assert q.read(q.R/'results'/cell/'STRATA_THRESHOLDS.json')['split']=='val'
 for method in q.p.METHODS:
  out=q.R/'results'/cell/method;m=q.read(out/'COMPLETE.json');a=np.load(out/'query_metrics.npz')
  hashes.add(m['analysis_source_sha256']);assert m['analysis_source_sha256']==q.p.sha(q.R/'scripts/query_analysis.py')
  assert m['stage2_reproduction'] and m['kl_stage2_reproduction'] and m['all5_cache_maxdiff']==0
  assert len(a['starts'])==m['n'] and np.all(np.diff(a['starts'])>0)
  if aligned is None:aligned=a['starts'].copy()
  else:assert np.array_equal(aligned,a['starts'])
  for key in a.files:assert np.isfinite(a[key]).all(),(cell,method,key)
  ids=a['top10_ids'];assert ids.shape==(m['n'],m['channels'],11,10);assert np.all(np.diff(np.sort(ids,axis=-1),axis=-1)>0)
  for h in range(5):
   rep=1-(ids[:,:,0,:,None]==ids[:,:,6+h,None,:]).any(-1).mean(-1)
   assert np.allclose(rep,a['replacement'][:,:,h],atol=1e-7)
  for arm,key in [('V5','stage2'),('V0','kl')]:
   gold=q.read(q.p.HERE/'results/runs'/cell/method/arm/'stage2/metrics.json');values=a[key][:,:,0] if arm=='V5' else a[key];actual=values.mean((0,1),dtype=np.float64)
   assert abs(actual[0]-gold['test_mse'])<1e-5 and abs(actual[1]-gold['test_mae'])<1e-5
  rows.append(dict(cell=cell,method=method,status='PASS',queries=m['n'],channels=m['channels'],all5_exact=True,stage2_and_kl_verified=True,finite=True,unique_top10=True,loo_replacement_recomputed=True))
 cov=q.read(q.R/'coverage'/cell/'COMPLETE.json');assert cov['full_cache_maxdiff']<1e-5
assert len(hashes)==1 and len(rows)==12
assert (q.R/'reports/ANALYSIS_KO.md').exists()
q.dump(q.R/'reports/INDEPENDENT_OUTPUT_VERIFICATION.json',dict(status='PASS',settings=rows,shared_analysis_source_sha=next(iter(hashes)),coverage_cells=4))
print('INDEPENDENT OUTPUT VERIFICATION PASS 12/12',flush=True)

