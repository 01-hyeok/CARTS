"""Actual frozen V5 checkpoint gate for alpha=0, past lag, and autograd."""
import argparse,datetime,fcntl,os,time,traceback
from pathlib import Path
import numpy as np
from server_bridge import *
from past_lag import alignment,torch_alignment
from softlag import torch_parameters,torch_forward,objective_gradient

def main(cell,support,out):
    out=Path(out);lock=out.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    if out.exists():
        prior=json.loads(out.read_text());assert prior['status']=='PASS_FIRST_TAIL_NOT_FULL_POPULATION'
        assert prior['cell']==cell and prior['support']==support
        assert sha(prior['identity']['checkpoint']['checkpoint'])==prior['identity']['checkpoint']['checkpoint_sha256']
        print('EXISTING_GPU_GATE_PRESERVED',str(out),flush=True);return
    before=start_gpu();register_owner(os.environ.get('LAG_TRACK','TRACK-SOFTLAG-HEAD-PILOT01'));begin=time.perf_counter()
    data,f,identity=frozen(cell,support);torch.cuda.reset_peak_memory_stats();rows=[]
    for split in ('val','test'):
        total=len(data.datasets[split])
        for a in sorted({0,((total-1)//32)*32}):
            b=min(a+32,total);g,mask,ids,p=native(f,split,a,b)
            x,y,starts=data.query(split,a,b);pick=ids.gather(-1,topk(p,10))
            reference=f.reference(split)
            if support=='P100':
                assert np.array_equal(pick.cpu().numpy(),reference['ids'][a:b])
            else:
                assert torch.equal(pick,f.canonical_full_ids(split,a,b))
                pred=(data.values_all(pick)+x[:,:,-1,None,None]).reshape((b-a)*data.c,10,data.h).mean(1).reshape(b-a,data.c,data.h)
                assert np.allclose(pred.cpu().numpy(),reference['prediction'][a:b],atol=1e-5,rtol=1e-5)
            # Phase-preserving sampled normalizer must match actual original rows.
            phase=torch.arange(a,b,device=data.device)%4
            pp=phased_mixture(g,mask,phase,support)
            assert torch.equal(pp,p),'sampled softmax changed canonical probabilities'
            assert torch.equal(topk(pp,10),topk(p,10))
            zero_a=g.new_zeros(g.shape[:2]+g.shape[3:]+(3,));c=g.new_ones(mask.shape)
            alpha,bias=torch_parameters(True,3,device=data.device,dtype=g.dtype)
            u,_=torch_forward(g,zero_a,c,alpha,bias,mask)
            assert torch.equal(u,g) and torch.equal(phased_mixture(u,mask,phase,support),p)
            rows.append(dict(split=split,start=a,end=b,query_shape=list(x.shape),logits_shape=list(g.shape),
                             channels_single_batch=data.c,candidate_count=g.shape[-1],alpha_zero_exact=True,
                             sampled_normalizer_bit_exact=True,canonical_ordered_top10_exact=True))
            del g,zero_a,c,u,pp,p
    # Actual past-only patch values, all channels, candidate chunking only.
    x,_,_=data.query('val',0,1);ids=torch.arange(17,device=data.device).expand(1,data.c,-1)
    hist=data.histories_all(ids);actual,lag=torch_alignment(x,hist,candidate_chunk=8)
    expected,_=alignment(x.cpu().numpy(),hist.cpu().numpy(),candidate_chunk=8)
    np.testing.assert_allclose(actual.cpu().numpy(),expected,rtol=1e-11,atol=1e-12)
    # Original teacher is used directly, including negative sign and mask.
    dd=torch.tensor([[3.,1.,2.,100.]],device=data.device);mm=torch.tensor([[True,True,True,False]],device=data.device)
    pt=rt.normalized_teacher_prob(dd,mm,.1)
    assert pt[0,1]>pt[0,2]>pt[0,0] and pt[0,3]==0
    rng=np.random.RandomState(1);g0=rng.randn(2,7,5,11);a0=rng.randn(2,7,11,5);c0=rng.rand(2,7,11)
    t0=rng.rand(2,7,11);t0/=t0.sum(-1,keepdims=True);bias0=rng.randn(5,5)*.01
    ref,ga,gb=objective_gradient(g0,a0,c0,t0,.2,bias0)
    pa,pb=torch_parameters(True,5,device=data.device);pa.data.fill_(.2);pb.data.copy_(torch.tensor(bias0,device=data.device))
    _,pm=torch_forward(torch.tensor(g0,device=data.device),torch.tensor(a0,device=data.device),torch.tensor(c0,device=data.device),pa,pb,torch.ones(2,7,11,device=data.device,dtype=torch.bool))
    tt=torch.tensor(t0,device=data.device);loss=(tt*(tt.log()-pm.log())).sum(-1).mean();loss.backward()
    np.testing.assert_allclose(pa.grad.cpu().numpy(),ga,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(pb.grad.cpu().numpy(),gb,rtol=1e-10,atol=1e-12)
    assert all(not p.requires_grad and p.grad is None for p in list(f.model.parameters())+list(f.heads.parameters()))
    torch.cuda.synchronize()
    result=dict(status='PASS_FIRST_TAIL_NOT_FULL_POPULATION',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                pid=os.getpid(),cell=cell,support=support,rows=rows,identity=identity,gpu_before=before,gpu_after=snapshot(),
                original_checkpoint_sha256_after=sha(f.row['checkpoint']),wall_seconds=time.perf_counter()-begin,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                past_only_patch_gpu_cpu_pass=True,trainable_parameter_gradient_pass=True,original_teacher_sign_pass=True,
                full_bank_count=len(data.starts),channels=data.c,seed=0,training_started=False)
    dump(out,result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cell');ap.add_argument('support');ap.add_argument('--out',required=True);a=ap.parse_args()
    try:main(a.cell,a.support,a.out)
    except BaseException as e:
        dump(Path(a.out).with_name(Path(a.out).stem+'_FAILED_'+str(os.getpid())+'.json'),dict(status='FAIL',error=repr(e),traceback=traceback.format_exc(),pid=os.getpid()))
        raise
