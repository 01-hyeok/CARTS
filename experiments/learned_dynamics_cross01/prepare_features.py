"""CPU-only full history features and train-only normalization. No targets used."""
import sys,os,time,traceback
from experiment_common import *
sys.path.insert(0,str(ROOT/'experiments/p100_head_retrieval_quality_legacy_pool02'))
import runtime as rt
import torch,numpy as np
from learned_scores import fit_ar,FixedStandardizer

def prepare(cell):
    assert not os.environ.get('CUDA_VISIBLE_DEVICES')
    torch.set_num_threads(1);d=rt.CellData(cell,'cpu');dest=OUT/'features'/cell
    manifest=dest/'COMPLETE.json'
    if manifest.exists():
        m=json.loads(manifest.read_text())
        assert m['core_sha256']==sha(HERE/'learned_scores.py')
        assert m['data']==d.data_info
        for n,h in m['files'].items():assert sha(dest/n)==h
        return
    assert not dest.exists(),'Partial feature folder requires explicit recovery; preserve it'
    dest.mkdir(parents=True);begin=time.perf_counter();files={};norm=None
    for split in ('train','val','test'):
        ds=d.datasets[split];parts={}
        for a in range(0,len(ds),64):
            b=min(a+64,len(ds));idx=np.arange(a,b)[:,None]+np.arange(d.h)
            past=torch.from_numpy(np.asarray(ds.data_x[idx],np.float32)).permute(0,2,1)
            with torch.no_grad():values=fit_ar(past)
            for k,v in values.items():
                assert torch.isfinite(v).all();parts.setdefault(k,[]).append(v.cpu())
            if a%1024==0:dump(OUT/'CPU_PREPARATION_STATUS.json',dict(status='RUNNING',cell=cell,split=split,queries=b,total=len(ds),at=now(),pid=os.getpid()))
        values={k:torch.cat(v) for k,v in parts.items()};values['starts']=torch.tensor(ds.starts.copy())
        if split=='train':
            norm=FixedStandardizer(6).fit(values['phi'],'train')
            torch_save(dest/'feature_normalizer.pt',norm.state_dict());files['feature_normalizer.pt']=sha(dest/'feature_normalizer.pt')
        torch_save(dest/(split+'.pt'),values);files[split+'.pt']=sha(dest/(split+'.pt'))
        del parts,values
    dump(manifest,dict(status='PASS_ALL_QUERY_PAST_FEATURES',at=now(),cell=cell,data=d.data_info,
        ridge=.001,ar_order=4,tail=16,fit_population='ALL TRAIN timestamps x ALL channels',
        source_future_access=False,core_sha256=sha(HERE/'learned_scores.py'),files=files,seconds=time.perf_counter()-begin,
        bytes=sum((dest/n).stat().st_size for n in files)))
    print('FEATURE_COMPLETE',cell,flush=True)

if __name__=='__main__':
    try:
        for cell in CELLS:prepare(cell)
        dump(OUT/'CPU_PREPARATION_STATUS.json',dict(status='COMPLETE',cells=CELLS,at=now()))
    except BaseException as e:
        dump(OUT/'CPU_PREPARATION_STATUS.json',dict(status='BLOCKED',error=repr(e),traceback=traceback.format_exc(),at=now()));raise
