"""Actual alternate-pool reference checks on CPU; not formal evaluation."""
import sys,os,json,tempfile,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import experiment_common as cm
from experiment_common import ROOT,OUT,sha,dump,load,now
from run_supplement import Reference,ev,finish_with_reference
from train_models import Features,Arm
import numpy as np,torch

def main():
    os.chdir(ROOT);torch.set_num_threads(1);rows=[]
    gate=json.loads((OUT/'supplemental/POOL_REUSE_GATE.json').read_text());assert gate['status']=='PASS'
    for entry in gate['rows']:
        cell,pool=entry['cell'],entry['pool'];d=ev.Data(cell,device='cpu')
        ckpt=next(r for r in ev.inventory(cell) if r['architecture']=='V5' and r['training_support']=='P100')
        assert sha(ckpt['checkpoint'])==ckpt['checkpoint_sha256']
        model,heads,_,_=ev.rt.model_init(cell,0,5,'cpu');z=load(ckpt['checkpoint']);model.load_state_dict(z['model']);heads.load_state_dict(z['heads']);model.eval();heads.eval()
        for it in entry['splits']:
            z=load(it['path']);d.cpu_pool[it['split']]=z['pool_idx_global'].numpy().astype(np.int64)
        f=types.SimpleNamespace(data=d,full_trained=False,model=model,heads=heads,precision=lambda:None)
        with tempfile.TemporaryDirectory(prefix='carts_supp_cpu_') as tmp:
            ex=types.SimpleNamespace(mode='pool',d=d,f=f,support='P100',dest=Path(tmp))
            ref=Reference(ex,None);record=ref.batch('val',0,2)
            ids=record['P100_NATIVE__selected_ids'];mse=record['P100_NATIVE__selected_mse']
            assert ids.shape==(2,d.c,10) and np.isfinite(mse).all()
            oracle,_,random,_,_=ref.baseline('P100','val',0,2)
            assert oracle.shape==(2,d.c,10) and random.shape==oracle.shape
            # Verify reference error directly, independently of chunked distance adapter.
            x,y,_=d.query('val',0,2);chosen=torch.tensor(ids)
            direct=(d.values_all(chosen)+x[:,:,-1,None,None]-y[:,:,None,:]).square().mean(-1)
            torch.testing.assert_close(torch.tensor(mse),direct,atol=2e-6,rtol=2e-6)
        rows.append(dict(cell=cell,pool=pool,channels=d.c,canonical_cpu_reference_and_metric=True))
    # Confirm source patch anchors used by supplementary evaluation still exist.
    from final_evaluation import finish
    import inspect
    source=inspect.getsource(finish)
    assert source.count("official=(float(ex.f.row['canonical_retmse']) if ex.support=='P100' else float(ex.f.reference('test')['metrics']['retmse10']))")==1
    # Run the complete output pipeline on two queries, restoring already selected
    # native scorer states without calibration/refitting. Temporary outputs only.
    d=ev.Data('ETTh1_96',device='cpu');features=Features(d,OUT/'features'/d.cell)
    source=OUT/'P100'/d.cell;state=json.loads((source/'SELECTION.json').read_text())['arms']
    arms={}
    for name in cm.ARMS:
        arm=Arm(name);arm.load_state_dict(load(source/'checkpoints'/name/('epoch%02d.pt'%state[name]['best_epoch']))['model']);arms[name]=arm
    # New raw scorer values and all fixed calibration buffers stay invariant
    # when an identical candidate is drawn from Full or restricted support.
    ids=torch.arange(len(d.starts)).expand(1,d.c,-1);subset=d.pool_all('val',0,1)
    for arm in arms.values():
        with torch.no_grad():
            full=arm.raw_chunks(features,'val',np.array([0]),ids,503)
            restricted=arm.raw_chunks(features,'val',np.array([0]),subset,31)
            torch.testing.assert_close(full.gather(-1,subset),restricted,atol=1e-10,rtol=1e-10)
    class TwoQueries:
        def __init__(self,original):self.original=original;self.starts=original.starts[:2]
        def __getattr__(self,name):return getattr(self.original,name)
        def __len__(self):return 2
    for split in ('val','test'):d.datasets[split]=TwoQueries(d.datasets[split])
    ckpt=next(r for r in ev.inventory(d.cell) if r['architecture']=='V5' and r['training_support']=='P100')
    model,heads,_,_=ev.rt.model_init(d.cell,0,5,'cpu');z=load(ckpt['checkpoint']);model.load_state_dict(z['model']);heads.load_state_dict(z['heads']);model.eval();heads.eval()
    f=types.SimpleNamespace(data=d,full_trained=False,model=model,heads=heads,precision=lambda:None,row=ckpt)
    with tempfile.TemporaryDirectory(prefix='carts_supp_output_cpu_') as tmp:
        ex=types.SimpleNamespace(cell=d.cell,mode='pool',d=d,f=f,support='P100',dest=Path(tmp),features=features,arms=arms)
        ex.reference=ex.prior=Reference(ex,None)
        def batch(split,a):
            b=min(a+32,len(d.datasets[split]));ids=d.pool_all(split,a,b);mask=torch.ones_like(ids,dtype=torch.bool)
            g=ev.canonical_logits(f,split,a,b,'P100');p=ev.normalize(g,mask)
            x,y,_=d.query(split,a,b);distance=ev.candidate_distances(d,x,y,ids)
            teacher=ev.rt.normalized_teacher_prob(distance.reshape(-1,100),mask.reshape(-1,100),.1).reshape_as(distance).double()
            ex.reference.batch(split,a,b)
            return b,ids,mask,p,distance,teacher
        ex.batch=batch;finish_with_reference(ex,state)
        produced=json.loads((ex.dest/'FULL_RESULTS.json').read_text());assert len(produced)==24
        assert all(r['queries']==2 and r['channels']==7 for r in produced)
    result=dict(status='PASS',rows=rows,cross_raw_score_invariance_all_six_selected_models=True,
        two_query_complete_output_pipeline=True,sources={p.name:sha(p) for p in HERE.glob('*.py')},at=now(),scope='CPU alternate-pool audit and output pipeline probes; GPU gates remain mandatory')
    dump(OUT/'supplemental/CPU_GATE.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
