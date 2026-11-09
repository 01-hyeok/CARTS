"""Real CPU past-feature integration for native Full and transfer policies."""
import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from native_runtime import NativeContext,ROOT,OLD,OUT,Features,TrainPairPermutation,ev
import numpy as np,torch
torch.set_num_threads(1)

def main(output):
    os.chdir(ROOT);d=ev.rt.CellData('ETTh1_96','cpu');ctx=NativeContext.__new__(NativeContext);ctx.cell=d.cell;ctx.d=d
    ctx.features=Features(d,OLD/'features'/d.cell);ctx.permutation=TrainPairPermutation(len(d.starts),d.h,'cpu')
    ctx.p100_donors=np.random.RandomState(0).permutation(len(d.datasets['train']));ctx.p100_pool=np.load(d.pool_dir/'train.npy',mmap_mode='r');ctx.slot_permutation=np.random.RandomState(0).permutation(100)
    rows=np.array([0,1]);ids=torch.arange(137).expand(2,d.c,137);proof=[]
    for source in ('FULL','P100'):
        ctx.training_support=source
        full=ctx.auxiliary('val',rows,ids,True,0)
        pieces=torch.cat([ctx.auxiliary('val',rows,ids[...,a:a+19],True,a) for a in range(0,137,19)],-2)
        assert torch.allclose(full,pieces,atol=1e-12,rtol=1e-12)
        # Shuffled held-out features must only use Train donors. Corrupting every
        # held-out feature in this in-memory test must not change those scores.
        saved=ctx.features.data['val'];ctx.features.data['val']={k:torch.full_like(v,12345.) for k,v in saved.items()}
        after=ctx.auxiliary('val',rows,ids,True,0);assert torch.equal(full,after);ctx.features.data['val']=saved
        proof.append(dict(source_training_support=source,dimensions=list(full.shape),chunk_invariant=True,heldout_features_not_read_for_shuffle=True,all_finite=bool(torch.isfinite(full).all())))
    ctx.training_support='FULL';trainids=torch.arange(193,330).expand(2,d.c,137)
    mask=d.mask(rows).gather(-1,trainids[:,0]);assert mask.all()
    q,c,v=ctx.permutation.map(rows,trainids,'train');assert v.all();assert d.mask(q.flatten().numpy()).gather(1,c.flatten()[:,None]).all()
    # Check donor feature construction against canonical helper on real pairs.
    result=ctx.auxiliary('train',rows,trainids,True)
    for i in range(2):
        for channel in (0,d.c-1):
            for slot in (0,136):
                donor_q=int(q[i,channel,slot]);donor_c=int(c[i,channel,slot]);donor_ids=torch.full((1,d.c,1),donor_c,dtype=torch.long)
                expected=torch.cat([ctx.features.pair('train',np.array([donor_q]),donor_ids,'DYN'),ctx.features.pair('train',np.array([donor_q]),donor_ids,'CROSS')],-1)
                assert torch.allclose(result[i,channel,slot],expected[0,channel,0],atol=1e-12,rtol=1e-12)
    info=dict(status='PASS_REAL_TRAIN_FEATURE_INTEGRATION',cell=d.cell,torch_version=torch.__version__,gpu_used=False,canonical_encoder_gate=False,proof=proof,train_donor_eligibility_pass=True,canonical_auxiliary_pair_values_pass=True)
    with Path(output).open('x') as f:json.dump(info,f,indent=2)
    print(json.dumps(info,indent=2))
if __name__=='__main__':main(sys.argv[1])
