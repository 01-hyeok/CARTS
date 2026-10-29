import csv,json,sys
from pathlib import Path
p=Path('/data/pjh_workspace/CARTS/results/TRACK-V5-FULL-HEAD-ANALYSIS01')/sys.argv[1]
rows=list(csv.DictReader((p/'INDEPENDENT_MODELS.csv').open()))
for r in rows:
 if r.get('status')=='MISSING':
  r['source_kind']='MISSING_NOT_RECONSTRUCTED';continue
 r['verification_forward_mse']=r['test_mse'];r['verification_forward_mae']=r['test_mae']
 if r.get('source'):
  raw=json.loads(Path(r['source']).read_text())
  assert abs(float(r['test_mse'])-raw['test_mse'])<1e-5
  assert abs(float(r['test_mae'])-raw['test_mae'])<1e-5
  r['test_mse']=raw['test_mse'];r['test_mae']=raw['test_mae']
  r['source_kind']='canonical_raw_metrics_reused_with_separate_forward_verification'
keys=list(dict.fromkeys(k for r in rows for k in r))
with (p/'CANONICAL_COMPARATORS_RAW.csv').open('w') as f:
 w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
print(p/'CANONICAL_COMPARATORS_RAW.csv')
