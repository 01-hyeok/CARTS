import csv,json,hashlib,sys
from pathlib import Path
root=Path('/data/pjh_workspace/CARTS')
cell=sys.argv[1]
out=root/'results/TRACK-V5-FULL-HEAD-ANALYSIS01'/cell
assert (out/'COMPLETE.json').exists(),'Cell not complete'
rows=list(csv.DictReader((out/'HEAD_STAGE2_MSE_MAE.csv').open()))
assert len(rows)==11 and len({r['subset'] for r in rows})==11
for split in ['val','test']:
 data=json.loads((out/f'{split}_metrics.json').read_text())
 assert len(data)==31 and len({r['subset'] for r in data})==31
verified=[]
for r in rows:
 p=out/'stage2'/r['subset'];m=json.loads((p/'metrics.json').read_text())
 epochs=list(csv.DictReader((p/'epoch_metrics.csv').open()))
 chosen=min(epochs,key=lambda e:float(e['val_mse']))
 assert int(chosen['epoch'])==m['best_epoch']
 assert m['seed']==0
 assert float(r['primary_mse'])==m['test_mse'] and float(r['primary_mae'])==m['test_mae']
 assert (p/'checkpoint.pth').exists()
 verified.append(dict(subset=r['subset'],validation_mse=m['val_mse'],best_epoch=m['best_epoch'],test_mse=m['test_mse'],test_mae=m['test_mae'],lambda_value=m['lambda'],training_wall_seconds=m['wall_clock_seconds']))
prov=json.loads((out/'CHECKPOINT_PROVENANCE.json').read_text())
integrity=[]
for name,key in [('checkpoint','checkpoint_sha256'),('base_checkpoint','base_sha256')]:
 p=root/prov[name];actual=hashlib.sha256(p.read_bytes()).hexdigest();assert actual==prov[key]
 integrity.append(dict(path=str(p),sha256=actual,status='UNCHANGED'))
cache=json.loads((out/'newcache_stored_lambda_verification.json').read_text());assert cache['status']=='PASS'
vbest=min(verified,key=lambda r:r['validation_mse'])
single=min((r for r in verified if len(r['subset'])==1),key=lambda r:r['validation_mse'])
all5=next(r for r in verified if r['subset']=='12345')
report=dict(cell=cell,status='VERIFIED',stage2_configurations=11,val_subsets=31,test_subsets=31,seed=0,epoch_selection='minimum validation MSE verified against all epoch records',validation_selected_stage2_configuration=vbest,validation_selected_single_head=single,all5=all5,cache_verification=cache,original_checkpoint_integrity=integrity,resource=json.loads((out/'COMPLETE.json').read_text()),resource_scope='analysis worker peak excludes Stage2 child/P100; stage2 wall excludes data preparation',primary_rows=verified)
(out/'COMPLETION_VERIFICATION.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
