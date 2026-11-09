"""Real-data, CPU initialization and GPU1 canonical-equivalence gates; no epoch."""
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from runtime import *

def max_diff(a,b):
    return float((a-b).abs().max())

def real_cell_gate(cell):
    data=CellData(cell,'cuda:0')
    cfg=config()
    info=cfg['data'][cell.rsplit('_',1)[0]]
    assert sha256_file(info['path'])==info['sha256']
    assert data.datasets['train'].base_dataset.scaler.n_samples_seen_==len(data.datasets['train'].data_x)
    checks=[]
    for split,ds in data.datasets.items():
        rows=sorted(set([0,len(ds)//2,len(ds)-1]))
        x,y,qstarts=data.batch(split,rows)
        for j,r in enumerate(rows):
            assert torch.equal(x[j].cpu(),torch.tensor(ds.data_x[r:r+data.h]).float())
            assert torch.equal(y[j].cpu(),torch.tensor(ds.data_y[r+data.h:r+2*data.h]).float())
        mask=data.mask(qstarts)
        ids_all=[]
        for c in range(data.c):
            coarse=compute_coarse_delta_last_cosine_scores(x[:,:,c:c+1],data.bank_history(c),0)
            ids=build_coarse_pool(coarse,mask,100)
            ids_all.append(ids.cpu().numpy())
        pools=np.stack(ids_all,1)
        validate_pool(pools,qstarts.numpy(),data.starts,data.h,data.h,split)
        checks.append(dict(split=split,rows=rows,pool_sha256=hashlib.sha256(pools.tobytes()).hexdigest(),
                           min_valid=int(mask.sum(-1).min()),all_channels_tested=data.c))
    init=[]
    for seed in (0,1,2):
        a,_,ah,_=model_init(cell,seed,0,'cpu')
        b,_,bh,_=model_init(cell,seed,0,'cpu')
        assert ah==bh
        h1=seeded_slot_heads(128,1,seed);h2=seeded_slot_heads(128,2,seed);h5=seeded_slot_heads(128,5,seed)
        assert torch.equal(h1.W,h5.W[:1]) and torch.equal(h2.W,h5.W[:2])
        loader1=make_training_loader(data.datasets['train'],32,seed)
        loader2=make_training_loader(data.datasets['train'],32,seed)
        # Hash full real dataset ordering without loading window tensors.
        # BatchSampler uses the same private generator, but DataLoader's iterator
        # also draws one base seed. Consume it identically before the sampler.
        def ids_order(loader):
            torch.empty((),dtype=torch.int64).random_(generator=loader.generator)
            return [data.datasets['train'].starts[np.asarray(ix)] for ix in loader.batch_sampler]
        order1=ids_order(loader1);order2=ids_order(loader2)
        assert batch_order_hash(order1)==batch_order_hash(order2)
        assert sum(map(len,order1))==len(data.datasets['train'])
        init.append(dict(seed=seed,encoder_hash=ah,head1_hash=state_hash(h1),head2_hash=state_hash(h2),
                         head5_hash=state_hash(h5),epoch1_batch_hash=batch_order_hash(order1)))
        del a,b,h1,h2,h5
    assert len({x['encoder_hash'] for x in init})==3 and len({x['head5_hash'] for x in init})==3
    # Actual B32 all-arm eval comparison to the canonical gather-all-channels path.
    x,y,qstarts=data.batch('train',range(32));mask=data.mask(qstarts)
    parity=[]
    for c in sorted(set([0,data.c-1])):
        ids=build_coarse_pool(compute_coarse_delta_last_cosine_scores(x[:,:,c:c+1],data.bank_history(c),0),mask,100)
        for nh in (0,1,2,5):
            model,heads,_,_=model_init(cell,0,nh,'cuda:0');model.eval()
            if heads is not None:heads.eval()
            with torch.no_grad():
                ref=scores(model,heads,x,data,c,ids,reference_all_channels=True)
                new=scores(model,heads,x[:,:,c:c+1],data,c,ids,query_channel=0)
                assert torch.equal(ref,new),(cell,c,nh,'score mismatch',max_diff(ref,new))
                values=data.values(ids,c)
                canonical_y=torch.from_numpy(np.array(data.sampler.memory_y[ids.cpu().numpy(),:,c],copy=True)).to(data.device)
                canonical_last=torch.from_numpy(np.array(data.sampler.memory_x[ids.cpu().numpy(),-1,c],copy=True)).to(data.device)
                assert torch.equal(values,canonical_y-canonical_last[:,:,None])
                l,d,teacher=kl(new,values,x[:,-1,c],y[:,:,c])
                valid=torch.ones_like(ids,dtype=torch.bool)
                pick,ph,pm=mean_mixture_topk_selection(new,valid,.1,10)
                order=stable_topk_indices(d,100,largest=False)
                assert torch.all(teacher.gather(1,order)[:,:-1]>=teacher.gather(1,order)[:,1:])
                last=(x[:,-1,c,None]-y[:,:,c]).square().mean(1)
                mets=metric_rows(d,values,x[:,-1,c],y[:,:,c],pick,order[:,:10],last)
                assert (mets[:,4]>=-1e-6).all()
                assert torch.equal(pick,mean_mixture_topk_selection(ref,valid,.1,10)[0])
                parity.append(dict(channel=c,model=f'V{nh}',max_score_diff=max_diff(ref,new),kl=float(l),top10_exact=True))
            del model,heads
    report=dict(status='PASS',cell=cell,data=data.data_info,loader_pool_checks=checks,initialization=init,
                canonical_actual_B32_parity=parity,training_result_reuse=False)
    write_json(HERE/'reports/runtime_cells'/f'{cell}.json',report)
    print(f'CELL GATE PASS {cell}',flush=True)
    del data
    torch.cuda.empty_cache()
    return report

def solar_gradient_gate():
    data=CellData('Solar_720','cuda:0')
    x,y,starts=data.batch('train',range(32))
    mask=data.mask(starts);pool=[]
    for c in range(data.c):
        ids=build_coarse_pool(compute_coarse_delta_last_cosine_scores(x[:,:,c:c+1],data.bank_history(c),0),mask,100)
        pool.append(ids)
    model,heads,_,_=model_init(data.cell,0,5,'cuda:0')
    reference=copy.deepcopy(model);rh=copy.deepcopy(heads)
    torch.manual_seed(1729);torch.cuda.manual_seed(1729)
    rng=torch.cuda.get_rng_state();cpu_rng=torch.get_rng_state()
    results=[]
    for is_reference,m,h in ((True,reference,rh),(False,model,heads)):
        torch.cuda.set_rng_state(rng);torch.set_rng_state(cpu_rng)
        m.train();h.train();opt=torch.optim.Adam(list(m.parameters())+list(h.parameters()),lr=.001)
        opt.zero_grad(set_to_none=True);loss_sum=0.;torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
        for c,ids in enumerate(pool):
            saved=[]
            hook=None
            if c==0:
                def capture(module,args,out):
                    out.retain_grad();saved.append(out)
                hook=m.encoder.register_forward_hook(capture)
            sc=scores(m,h,x,data,c,ids,reference_all_channels=is_reference)
            loss,_,_=kl(sc,data.values(ids,c),x[:,-1,c],y[:,:,c])
            (loss/data.c).backward();loss_sum+=float(loss.detach())/data.c
            if hook is not None:
                hook.remove()
                assert len(saved)==2 and all(z.grad is not None and z.grad.abs().sum()>0 for z in saved)
            if c%32==0:print(f'SOLAR GRADIENT {"canonical" if is_reference else "compact"} channel={c+1}/137',flush=True)
        grads={f'model.{k}':p.grad.detach().clone() for k,p in m.named_parameters() if p.grad is not None}
        grads.update({f'heads.{k}':p.grad.detach().clone() for k,p in h.named_parameters() if p.grad is not None})
        assert all(torch.isfinite(g).all() for g in grads.values())
        state_after=torch.cuda.get_rng_state().clone();opt.step();torch.cuda.synchronize()
        results.append(dict(loss=loss_sum,grads=grads,rng=state_after,
            seconds=time.monotonic()-begin,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
    a,b=results
    assert a['loss']==b['loss'],(a['loss'],b['loss'])
    assert torch.equal(a['rng'],b['rng'])
    assert a['grads'].keys()==b['grads'].keys()
    for k in a['grads']:torch.testing.assert_close(a['grads'][k],b['grads'][k],rtol=0,atol=0)
    for k,v in reference.state_dict().items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
    torch.testing.assert_close(rh.W,heads.W,rtol=0,atol=0)
    report=dict(status='PASS',cell='Solar_720',batch=32,channels=137,candidates=100,
        dropout=.1,query_gradient=True,candidate_gradient=True,optimizer_steps_per_path=1,
        exact_loss=True,exact_gradient=True,exact_optimizer_update=True,exact_cuda_rng_after=True,
        canonical={k:v for k,v in a.items() if k not in ('grads','rng')},
        compact={k:v for k,v in b.items() if k not in ('grads','rng')},
        smoke_only=True,weights_discarded=True)
    write_json(HERE/'reports/SOLAR720_GRADIENT_GATE.json',report)
    print('SOLAR720 EXACT GRADIENT/UPDATE/RNG PASS',flush=True)
    return report

def main():
    gpu=gpu_guard();begin=time.monotonic();cfg=config()
    tests=json.loads((HERE/'reports/CPU_TEST_STATUS.json').read_text())
    assert tests['failures']==0 and tests['errors']==0 and tests['tests']>=24
    cells=[]
    for cell in cfg['references']:
        cells.append(real_cell_gate(cell))
    solar=solar_gradient_gate()
    report=dict(status='PASS',checked_at=now(),gpu=gpu,source_hashes=source_hashes(),
        teacher_mode='zscore_future_mse',cpu_tests=tests,cells=[x['cell'] for x in cells],
        actual_loader_cells=12,actual_meanmix_loss_score_parity='PASS',solar_gradient_gate=solar,
        full_pool_eligibility_gate='required per cell after generation before its first training run',
        existing_training_results_reused=False,stage2=False,elapsed_seconds=time.monotonic()-begin)
    write_json(HERE/'reports/RUNTIME_GATES.json',report)
    print('ALL PRETRAINING RUNTIME GATES PASS',flush=True)

if __name__=='__main__':main()
