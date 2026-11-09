"""First/tail runtime audit against immutable predictions; not model selection."""
import argparse,os,json,gc
import numpy as np
import torch
from adapter import Data,Frozen,inventory
from worker import guard,ROOT,OUT,distances,native_batch
from io_utils import dump,now,asnumpy

@torch.no_grad()
def main(cell):
    os.chdir(ROOT);guard();torch.set_num_threads(1);d=Data(cell)
    row=next(r for r in inventory(cell) if r['architecture']=='V1' and r['training_support']=='FULL');rows=[]
    for tf32 in [False,True]:
        f=Frozen(d,row);f.tf32=tf32
        for split in ['val','test']:
            total=len(d.datasets[split])
            for a in sorted({0,((total-1)//32)*32}):
                b=min(a+32,total);x,y,q,dist=distances(d,split,a,b);ev=native_batch(f,split,a,b,dist,x,y)
                can=f.canonical_full_ids(split,a,b);vals=d.values_all(can)+x[:,:,-1,None,None]
                pred=vals.reshape((b-a)*d.c,10,d.h).mean(1).reshape(b-a,d.c,d.h)
                old=f.reference(split)['prediction'][a:b];actual=asnumpy(ev['prediction']);canonical=asnumpy(pred)
                rows.append(dict(torch=torch.__version__,tf32=tf32,split=split,start=a,end=b,all_channels=d.c,
                    canonical_prediction_max=float(abs(canonical-old).max()),production_prediction_max=float(abs(actual-old).max()),
                    canonical_stored_match=bool(np.allclose(canonical,old,atol=1e-5,rtol=1e-5)),
                    production_stored_match=bool(np.allclose(actual,old,atol=1e-5,rtol=1e-5)),
                    ordered_id_mismatch=int((can!=ev['selected_ids']).any(-1).sum())))
        del f;gc.collect();torch.cuda.empty_cache()
    result=dict(at=now(),cell=cell,model='V1_FULL',checkpoint_sha256=row['checkpoint_sha256'],rows=rows,
        historical_runtime='UNKNOWN; immutable-cache equivalence does not prove original runtime',no_tolerance_change=True)
    p=OUT/'00_Audit'/f'RUNTIME_AUDIT_{cell}_V1_TORCH{torch.__version__.split(".")[0]}.json';dump(p,result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cell');main(p.parse_args().cell)
