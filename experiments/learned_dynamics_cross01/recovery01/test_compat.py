import sys,json,argparse
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
import compat
from experiment_common import OUT,dump,sha,now

def synthetic():
    rng=np.random.RandomState(0)
    v=np.round(rng.normal(size=(20,513)),1);m=rng.rand(20,513)>.2
    v[0]=0;v[1]=np.inf;v[2]=-np.inf;v[3,:4]=[0.,-0.,np.inf,-np.inf];m[4]=False
    return torch.tensor(v),torch.tensor(m)

def capture():
    from resume import restored
    ex,state,cp=restored(OUT/'recovery01/test_only')
    data={};records=[]
    with torch.no_grad():
        for split in ('val','test'):
            for a in (0,(len(ex.d.datasets[split])-1)//32*32):
                b,ids,mask,p,dist,t=ex.batch(split,a);n=min(2,b-a);rows=np.arange(a,a+n);mm=mask[:n]
                values={'distance':dist[:n],'negative_probability':-p[:n],'teacher':t[:n]}
                for name,arm in ex.arms.items():
                    raw=arm.raw_chunks(ex.features,split,rows,ids[:n])
                    values[name]=-arm.score(p[:n],raw,mm,ranking=True)
                for name,v in values.items():
                    key=split+'_'+str(a)+'_'+name
                    data[key+'__value']=v.cpu().numpy();data[key+'__mask']=mm.cpu().numpy()
                    data[key+'__rank']=compat.ranks(v,mm).cpu().numpy()
                    data[key+'__order']=compat.stable_order(v.double().masked_fill(~mm,torch.inf)).cpu().numpy()
                    records.append(key)
        v,m=synthetic();v=v.to(ex.d.device);m=m.to(ex.d.device)
        data['synthetic__value']=v.cpu().numpy();data['synthetic__mask']=m.cpu().numpy()
        data['synthetic__rank']=compat.ranks(v,m).cpu().numpy();data['synthetic__order']=compat.stable_order(v.masked_fill(~m,torch.inf)).cpu().numpy()
    root=OUT/'recovery01';np.savez(root/'ACTUAL_RANK_FIXTURE.npz',**data)
    dump(root/'CAPTURE.json',dict(status='PASS_OLD_RUNTIME_GPU_CAPTURE',torch_version=torch.__version__,checkpoints=cp,records=records,at=now()))

def verify():
    import final_evaluation as original
    root=OUT/'recovery01';records=[]
    with np.load(root/'ACTUAL_RANK_FIXTURE.npz') as z:
        for key in sorted(k[:-7] for k in z.files if k.endswith('__value')):
            v=torch.from_numpy(z[key+'__value']);m=torch.from_numpy(z[key+'__mask'])
            expected=original.ranks(v,m);actual=compat.ranks(v,m)
            order=torch.argsort(v.double().masked_fill(~m,torch.inf),dim=-1,stable=True)
            assert np.array_equal(order.numpy(),z[key+'__order']),key
            assert np.array_equal(expected.numpy(),z[key+'__rank']),key
            assert torch.equal(expected,actual),key
            # Average ranks are also independently checked using NumPy tie groups.
            a=v.double().masked_fill(~m,torch.inf).numpy();ref=np.empty_like(a)
            for row,out in zip(a.reshape(-1,a.shape[-1]),ref.reshape(-1,a.shape[-1])):
                ix=np.argsort(row,kind='mergesort');s=row[ix]
                edges=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1,len(s)]
                for lo,hi in zip(edges[:-1],edges[1:]):out[ix[lo:hi]]=(lo+hi-1)/2
            assert np.array_equal(ref,actual.numpy()),key
            records.append(dict(key=key,shape=list(v.shape),order_exact=True,rank_exact=True))
    dump(root/'COMPATIBILITY_GATE.json',dict(status='PASS',at=now(),compat_sha256=sha(HERE/'compat.py'),
        modern_torch=torch.__version__,old_runtime_capture=json.loads((root/'CAPTURE.json').read_text()),rows=records,
        scope='CPU modern stable argsort versus legacy GPU compatibility; synthetic ties/masks/infinities and actual first/tail Val/Test all six selected scorers; no fitting'))
    print('PASS',len(records),'exact stable order and tie-average rank cases')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['capture','verify']);a=ap.parse_args()
    capture() if a.mode=='capture' else verify()
