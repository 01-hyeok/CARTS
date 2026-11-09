"""Numerical provenance audit; cache match is not a forecast-quality choice."""
import os,json
import torch
import numpy as np
from adapter import Data,Frozen,inventory,LogitBatch
from worker import ROOT,OUT,guard,distances,native_batch
from io_utils import dump,now,asnumpy

def main():
    os.chdir(ROOT);guard();torch.set_num_threads(1);d=Data('ETTh1_96')
    row=next(r for r in inventory(d.cell) if r['architecture']=='V1' and r['training_support']=='FULL')
    rows=[]
    for tf32 in (False,True):
        f=Frozen(d,row);f.tf32=tf32
        for split in ('val','test'):
            total=len(d.datasets[split])
            for a in (0,total//32*32):
                b=min(a+32,total);x,y,starts,dist=distances(d,split,a,b);ev=native_batch(f,split,a,b,dist,x,y)
                old=f.reference(split)['prediction'][a:b];new=asnumpy(ev['prediction']);ids=asnumpy(ev['selected_ids']);can=asnumpy(f.canonical_full_ids(split,a,b))
                rows.append(dict(torch=torch.__version__,tf32=tf32,split=split,a=a,b=b,
                    max_difference=float(abs(new-old).max()),match=bool(np.allclose(new,old,atol=1e-5,rtol=1e-5)),
                    changed_query_channels=int(np.any(abs(new-old)>1e-5+1e-5*abs(old),axis=-1).sum()),
                    ordered_current_canonical_exact=bool(np.array_equal(ids,can))))
        del f
    out=dict(at=now(),cell=d.cell,model='V1_FULL',checkpoint_sha256=row['checkpoint_sha256'],rows=rows,
             original_runtime='UNKNOWN; source runner uses default torch precision, cache match does not establish original runtime')
    dump(OUT/'00_Audit'/('FULL_V1_RUNTIME_AUDIT_TORCH'+torch.__version__.split('.')[0]+'.json'),out);print(json.dumps(out))
if __name__=='__main__':main()
