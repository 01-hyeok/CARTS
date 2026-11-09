import sys,time,copy,json,gc
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import runtime as rt
import torch
import numpy as np
from vector_ops import *


def sync():torch.cuda.synchronize()
def params(model,heads):return list(model.parameters())+(list(heads.parameters()) if heads is not None else [])
def legacy(data,m,h,x,y,ids):
    value=0.
    for c in range(data.c):
        sc=rt.scores(m,h,x,data,c,ids[c]);loss,_,_=rt.kl(sc,data.values(ids[c],c),x[:,-1,c],y[:,:,c])
        (loss/data.c).backward();value+=float(loss.detach())/data.c
    return value


def compare(data,x,y,ids,dropout_on):
    m,h,_,_=rt.model_init(data.cell,0,5,'cuda:0');v,vh=copy.deepcopy(m),copy.deepcopy(h)
    if not dropout_on:
        for mod in m.modules():
            if isinstance(mod,torch.nn.Dropout):mod.p=0.
        for mod in v.modules():
            if isinstance(mod,torch.nn.Dropout):mod.p=0.
    dr=install_dropout(v);m.train();v.train()
    a=torch.optim.Adam(params(m,h),lr=.001);b=torch.optim.Adam(params(v,vh),lr=.001)
    torch.cuda.manual_seed(1234);rng=torch.cuda.get_rng_state()
    lo=legacy(data,m,h,x,y,ids);endold=torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(rng);lv=batch_backward(rt,data,v,vh,dr,x,y,ids);endnew=torch.cuda.get_rng_state()
    ga=torch.cat([p.grad.flatten() for p in params(m,h) if p.grad is not None])
    gb=torch.cat([p.grad.flatten() for p in params(v,vh) if p.grad is not None])
    gmax=float((ga-gb).abs().max());grel=float((ga-gb).norm()/ga.norm().clamp_min(1e-12))
    assert torch.isfinite(ga).all() and torch.isfinite(gb).all()
    a.step();b.step()
    update=max(float((p-q).abs().max()) for p,q in zip(params(m,h),params(v,vh)))
    m.eval();v.eval()
    with torch.no_grad():
        q,target,hist,values=flat_inputs(data,x,y,ids,0,data.c)
        sv=vector_scores(v,vh,q,hist,dr)
        # Separate fresh models: forward parity must precede divergent Adam update.
        fresh,fh,_,_=rt.model_init(data.cell,0,5,'cuda:0');fresh.eval()
        expected=torch.cat([rt.scores(fresh,fh,x,data,c,ids[c]) for c in range(data.c)])
        actual=vector_scores(fresh,fh,q,hist)
        scoremax=float((expected-actual).abs().max())
        po=rt.stable_topk_indices(torch.softmax(expected/.1,-1).mean(1),10,largest=True)
        pn=rt.stable_topk_indices(torch.softmax(actual/.1,-1).mean(1),10,largest=True)
        # Exact ties can exchange ID under FP32 GEMM rounding; retain count as evidence.
        pickdiff=int((po!=pn).any(1).sum())
    result=dict(dropout_on=dropout_on,legacy_loss=lo,vector_loss=lv,loss_abs_diff=abs(lo-lv),
        gradient_max_abs_diff=gmax,gradient_relative_l2=grel,adam_update_max_abs_diff=update,
        eval_score_max_abs_diff=scoremax,eval_top10_order_diff_rows=pickdiff,
        rng_end_equal=torch.equal(endold,endnew),units=data.c*len(x),finite=True)
    print('PARITY',json.dumps(result),flush=True)
    assert abs(lo-lv)<2e-5 and grel<5e-5 and gmax<5e-5 and update<5e-5 and scoremax<3e-6,result
    assert result['rng_end_equal'],result
    return result


def bench(data,x,y,ids,kind,chunk=137):
    gc.collect();torch.cuda.empty_cache()
    m,h,_,_=rt.model_init(data.cell,0,5,'cuda:0');m.train();dr=install_dropout(m) if kind=='vector' else None
    opt=torch.optim.Adam(params(m,h),lr=.001)
    durations=[];peaks=[]
    for i in range(4):
        opt.zero_grad(set_to_none=True);sync();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        loss=batch_backward(rt,data,m,h,dr,x,y,ids,chunk) if kind=='vector' else legacy(data,m,h,x,y,ids)
        assert all(torch.isfinite(p.grad).all() for p in params(m,h) if p.grad is not None)
        opt.step();sync();elapsed=time.perf_counter()-begin
        if i:
            durations.append(elapsed);peaks.append([torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()])
    result=dict(cell=data.cell,method=kind,channel_chunk=chunk if kind=='vector' else 1,batch=32,channels=137,
        seconds=durations,median_seconds=float(np.median(durations)),peak_allocated_bytes=max(p[0] for p in peaks),
        peak_reserved_bytes=max(p[1] for p in peaks),loss=loss,finite=True)
    print('BENCH',json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    gpu=rt.gpu_guard();torch.set_num_threads(2)
    report=dict(status='RUNNING',gpu=gpu,started_at=rt.now(),parity=[],benchmarks=[],
        tolerances=dict(loss_abs=2e-5,grad_relative_l2=5e-5,grad_max_abs=5e-5,adam_max_abs=5e-5,score_max_abs=3e-6),
        benchmark_definition='1 warmup + 3 synchronized train forward/backward/Adam steps; excludes data loader, pool generation, validation, output writing')
    for horizon in (96,192,336,720):
        data=rt.CellData(f'Solar_{horizon}','cuda:0');x,y,starts=data.batch('train',range(32))
        if (data.pool_dir/'COMPLETE.json').exists():ids=pool_all(data,'train',starts)
        else:
            ids=rt.canonical_sample_pool(data,'train',np.arange(32)).permute(1,0,2).contiguous()
        assert ids.shape==(137,32,100)
        rt.validate_pool(ids.permute(1,0,2).cpu().numpy(),starts.numpy(),data.starts,horizon,horizon,'train')
        if horizon==96:
            report['parity']=[compare(data,x,y,ids,False),compare(data,x,y,ids,True)]
        report['benchmarks'].append(bench(data,x,y,ids,'legacy'))
        for chunk in (137,32):
            try:report['benchmarks'].append(bench(data,x,y,ids,'vector',chunk))
            except torch.cuda.OutOfMemoryError as exc:
                report['benchmarks'].append(dict(cell=data.cell,method='vector',channel_chunk=chunk,status='OOM',error=str(exc)))
                gc.collect();torch.cuda.empty_cache()
        rt.write_json(HERE/'BENCHMARK.json',report)
        del data,x,y,starts,ids;gc.collect();torch.cuda.empty_cache()
    report['status']='PASS';report['finished_at']=rt.now();rt.write_json(HERE/'BENCHMARK.json',report)
