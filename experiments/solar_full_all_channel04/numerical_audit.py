"""Bounded CPU experiment recording rejected and retained update paths."""
import copy,json
from pathlib import Path
import numpy as np
import torch
import batch_core as f
from test_batch_core import data_fixture

def main(path):
    assert not torch.cuda.is_available(),'This audit must run with CUDA_VISIBLE_DEVICES empty'
    rows=[]
    for nh in (0,5):
      for h in (96,192,336,720):
       for qchunk,pchunk in ((3,23),(2,7)):
        d=data_fixture(h);model,heads,_,_=f.rt.model_init('Solar_'+str(h),0,nh,'cpu');model.eval()
        if heads is not None:heads.eval()
        other=copy.deepcopy(model);oh=copy.deepcopy(heads);torch.manual_seed(64);x=torch.randn(3,3,h);y=torch.randn_like(x)
        hist,vals=f.original.bank_views(d,[0,1,2]);loss=f.objective(f.original.scores_group(other,oh,x,hist),f.original.teacher_distances(vals,x[:,:,-1],y),d.mask(range(3)),oh);loss.backward()
        got=f.backward_batch(d,model,heads,x,y,np.arange(3),qchunk,pchunk)
        pp=list(model.named_parameters())+([] if heads is None else [('heads.'+n,p) for n,p in heads.named_parameters()]);qq=list(other.parameters())+([] if oh is None else list(oh.parameters()))
        maxgrad=max(float((p.grad-q.grad).abs().max()) for (_,p),q in zip(pp,qq) if p.grad is not None)
        grads=[(None,None) if p.grad is None else (p.grad.clone(),q.grad.clone()) for (_,p),q in zip(pp,qq)]
        torch.optim.Adam([p for _,p in pp],lr=.001).step();torch.optim.Adam(qq,lr=.001).step()
        diffs=[float((p-q).abs().max()) for (_,p),q in zip(pp,qq)];i=int(np.argmax(diffs));name,p=pp[i];q=qq[i];index=int((p-q).abs().reshape(-1).argmax());gp,gq=grads[i]
        passed=True
        for (_,p),q in zip(pp,qq):
            try:torch.testing.assert_close(p,q,atol=2e-6,rtol=2e-5)
            except AssertionError:passed=False
        rows.append(dict(horizon=h,heads=nh,query_chunk=qchunk,candidate_chunk=pchunk,loss_abs=abs(got-float(loss)),gradient_max_abs=maxgrad,adam_max_abs=max(diffs),adam_original_gate_pass=passed,
            max_difference_parameter=name,flattened_index=index,new_gradient_at_max=None if gp is None else float(gp.reshape(-1)[index]),reference_gradient_at_max=None if gq is None else float(gq.reshape(-1)[index]),
            allowed_for_training=qchunk==3 and pchunk==23))
    assert all(r['adam_max_abs']==0 for r in rows if r['allowed_for_training'])
    f.rt.write_json(path,dict(status='PASS_FULL_SHAPE_ONLY_CHUNKED_UPDATE_REJECTED',rows=rows,teacher_and_rng_unchanged=True,tolerance_changed=False,gpu_test='DEFERRED_UNTIL_PRIOR_FULL_COMPLETE'))

if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
