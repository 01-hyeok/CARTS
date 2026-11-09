"""Frozen-encoder SoftLag pilot, native support, fixed sampled timestamps.

No encoder/head updates. All channel axes retained. Candidate chunks only bound
past-feature workspace; softmax/KL normalize over the whole valid support.
"""
import argparse,csv,datetime,fcntl,os,time,traceback
from pathlib import Path
import numpy as np
from server_bridge import *
from past_lag import torch_alignment,freeze_activity_scale,indices
from softlag import torch_parameters,torch_forward

BASE=ROOT/'results/TRACK-SOFTLAG-HEAD-PILOT01'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def state(out,phase,**kw):dump(out/'CURRENT.json',dict(phase=phase,at=now(),pid=os.getpid(),**kw))
def atomic_npz(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez(f,**payload)
    os.replace(tmp,path)
def atomic_torch(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)
def write_csv(path,rows):
    assert rows
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def cache_split(data,f,split,rows,out,scale,proof):
    samples=[];byblock={}
    for index in rows:byblock.setdefault(int(index)//32*32,[]).append(int(index))
    for a,selected in sorted(byblock.items()):
        b=min(a+32,len(data.datasets[split]));todo=[]
        for i in selected:
            path=out/'cache'/split/('query%05d.npz'%i);meta=path.with_suffix('.json')
            if path.exists():
                m=json.loads(meta.read_text());assert m['identity']==proof and m['sha256']==sha(path)
                samples.append(path)
            else:todo.append(i)
        if not todo:continue
        with torch.no_grad():
            g,mask,ids,prob=native(f,split,a,b)
            x,y,starts,d=distances(data,split,a,b)
            if f.support=='P100':d=d.gather(-1,ids)
            for i in todo:
                j=i-a;qx=x[j:j+1];cid=ids[j:j+1];features=[];activities=[]
                for z in range(0,cid.shape[-1],128):
                    h=data.histories_all(cid[...,z:z+128]);aa,lag=torch_alignment(qx,h,candidate_chunk=128)
                    qs=qx.double().std(-1,unbiased=False)[...,None];cs=h.double().std(-1,unbiased=False)
                    sc=torch.as_tensor(scale,device=data.device)[None,:,None]
                    act=qs/(qs+sc)*cs/(cs+sc)
                    features.append(aa.float());activities.append(act.float())
                aa=torch.cat(features,2);act=torch.cat(activities,2)
                gg=g[j:j+1];mm=mask[j:j+1];phase=torch.tensor([i%4],device=data.device)
                actual=phased_mixture(gg,mm,phase,f.support)
                assert torch.equal(actual,prob[j:j+1]),('cache alpha0 probability mismatch',split,i)
                pick=cid.gather(-1,topk(actual,10))
                if split!='train':
                    ref=f.reference(split)
                    if f.support=='P100':assert np.array_equal(pick.cpu().numpy(),ref['ids'][i:i+1])
                    else:
                        pred=(data.values_all(pick)+qx[:,:,-1,None,None]).reshape(data.c,10,data.h).mean(1)
                        assert np.allclose(pred.cpu().numpy(),ref['prediction'][i],atol=1e-5,rtol=1e-5)
                path=out/'cache'/split/('query%05d.npz'%i)
                arrays=dict(g=gg.cpu().numpy(),alignment=aa.cpu().numpy(),activity=act.cpu().numpy(),mask=mm.cpu().numpy(),ids=cid.cpu().numpy(),
                            distance=d[j:j+1].cpu().numpy(),x=qx.cpu().numpy(),y=y[j:j+1].cpu().numpy(),phase=np.array([i%4]),
                            query_index=np.array([i]),query_start=np.array([starts[j]]),base_probability=actual.cpu().numpy())
                atomic_npz(path,arrays);dump(path.with_suffix('.json'),dict(identity=proof,sha256=sha(path),split=split,query_index=i,channels=data.c,
                      candidate_count=cid.shape[-1],alpha0_bit_exact=True,future_input_to_lag=False,at=now()))
                samples.append(path);state(out,'CACHE',split=split,completed=len(samples),total=len(rows),query_index=i,channels=data.c)
                print('SOFTLAG CACHE',data.cell,f.support,split,len(samples),'/',len(rows),'query',i,flush=True)
    assert len(samples)==len(rows)
    samples.sort();return [{k:z[k].copy() for k in z.files} for p in samples for z in [np.load(p)]]

def batch(rows,index):
    return {k:torch.as_tensor(np.concatenate([rows[int(i)][k] for i in index]),device='cuda:0') for k in rows[0]}

def probability(b,alpha,bias,support):
    u,_=torch_forward(b['g'],b['alignment'],b['activity'],alpha,bias,b['mask'])
    return u,phased_mixture(u,b['mask'],b['phase'],support)

@torch.no_grad()
def metrics(data,b,p):
    pick=topk(p.masked_fill(~b['mask'],-float('inf')),10)
    oracle=topk(-b['distance'].masked_fill(~b['mask'],float('inf')),10)
    dist=b['distance'].gather(-1,pick).double().mean(-1)
    oret=b['distance'].gather(-1,oracle).double().mean(-1)
    assert bool((dist>=oret-2e-6).all())
    ids=b['ids'].gather(-1,pick);pred=(data.values_all(ids)+b['x'][:,:,-1,None,None]).mean(2);err=pred-b['y']
    recall=(pick[...,None]==oracle[:,:,None]).any(-1).double().mean(-1)
    result=torch.stack([dist,err.square().mean(-1).double(),err.abs().mean(-1).double(),recall,oret,dist-oret],-1)
    return result.cpu().numpy(),ids.cpu().numpy()

@torch.no_grad()
def evaluation(data,rows,alpha,bias,support,method='learned'):
    allmet=[];allids=[];correction=[];overlap=[];attn=[]
    for a in range(0,len(rows),8):
        b=batch(rows,range(a,min(a+8,len(rows))))
        if method=='oracle':p=-b['distance']
        elif method=='random':
            values=[]
            for j,q in enumerate(b['query_start'].cpu().numpy()):
                rng=np.random.RandomState(int(q)%4294967295);values.append(rng.random_sample(b['mask'][j].shape))
            p=torch.as_tensor(np.stack(values),device=data.device)
        else:
            u,p=probability(b,alpha,bias,support);correction.append(float((u-b['g']).abs().mean()))
            hp=topk(u.masked_fill(~b['mask'][:,:,None],-float('inf')),10)
            overlap.extend([float((hp[:,:,h,:,None]==hp[:,:,k,None,:]).any(-1).float().mean()) for h in range(5) for k in range(h+1,5)])
            w=torch.softmax(b['alignment'][:,:,None]/.1+bias[None,None,:,None,:],-1)
            denom=b['mask'].sum((0,1)).sum().item()
            attn.append((w.masked_fill(~b['mask'][:,:,None,:,None],0).sum((0,1,3))/denom).cpu().numpy())
        m,ids=metrics(data,b,p);allmet.append(m);allids.append(ids)
    return np.concatenate(allmet),np.concatenate(allids),dict(mean_abs_score_correction=float(np.mean(correction)) if correction else None,
          mean_head_top10_overlap=float(np.mean(overlap)) if overlap else None,mean_lag_attention=np.mean(attn,axis=0).tolist() if attn else None)

def train_arm(data,f,train,val,out,headwise,proof):
    arm='B2' if headwise else 'B1';folder=out/arm;folder.mkdir(exist_ok=True)
    _,lags=indices(data.h);alpha,bias=torch_parameters(headwise,len(lags),device=data.device,dtype=torch.float32)
    optimizer=torch.optim.Adam([alpha,bias],lr=.01,weight_decay=0.)
    generator=np.random.RandomState(0);history=[];firstgrad=None;latergrad=0.
    for epoch in range(6):
        ck=folder/('epoch%02d.pth'%epoch)
        if ck.exists():
            z=torch.load(ck,map_location=data.device);assert z['identity']==proof
            alpha.data.copy_(z['alpha']);bias.data.copy_(z['bias']);optimizer.load_state_dict(z['optimizer']);generator.set_state(z['batch_rng'])
            history=z['history'];firstgrad=z['first_bias_grad'];latergrad=z['later_bias_grad_max'];continue
        losses=[]
        if epoch:
            order=generator.permutation(len(train))
            for a in range(0,len(order),8):
                b=batch(train,order[a:a+8]);optimizer.zero_grad()
                teacher=rt.normalized_teacher_prob(b['distance'].reshape(-1,b['distance'].shape[-1]),b['mask'].reshape(-1,b['mask'].shape[-1]),.1).reshape_as(b['distance'])
                _,p=probability(b,alpha,bias,f.support)
                loss=rt.kl_loss_from_prob(teacher,p,b['mask']);assert torch.isfinite(loss);loss.backward()
                assert alpha.grad is not None and bias.grad is not None and torch.isfinite(alpha.grad).all() and torch.isfinite(bias.grad).all()
                bn=float(bias.grad.norm())
                if firstgrad is None:firstgrad=bn;assert bn==0.,'alpha0 bias gradient must be zero'
                else:latergrad=max(latergrad,bn)
                optimizer.step();losses.append(float(loss));state(out,'TRAIN',arm=arm,epoch=epoch,step=a//8+1,total_steps=(len(order)+7)//8)
        vm,_,_=evaluation(data,val,alpha,bias,f.support)
        rec=dict(arm=arm,epoch=epoch,val_retmse=float(vm[...,0].mean()),val_recall=float(vm[...,3].mean()),
                 train_loss=float(np.mean(losses)) if losses else None,alpha=float(alpha.detach()),bias_norm=float(bias.detach().norm()))
        history.append(rec)
        atomic_torch(ck,dict(identity=proof,epoch=epoch,alpha=alpha.detach(),bias=bias.detach(),optimizer=optimizer.state_dict(),
                     batch_rng=generator.get_state(),history=history,first_bias_grad=firstgrad,later_bias_grad_max=latergrad))
        dump(folder/'VALIDATION_HISTORY.json',history);print('SOFTLAG EPOCH',data.cell,f.support,rec,flush=True)
    assert firstgrad==0. and latergrad>0.,'Bias trainability not established after alpha leaves zero'
    selected=min(history,key=lambda r:(r['val_retmse'],r['epoch']))
    path=folder/('epoch%02d.pth'%selected['epoch']);z=torch.load(path,map_location=data.device)
    alpha.data.copy_(z['alpha']);bias.data.copy_(z['bias'])
    dump(folder/'SELECTION.json',dict(criterion='minimum validation RetMSE; earlier epoch tie; epoch0 eligible',selected=selected,checkpoint=str(path),sha256=sha(path),test_used=False))
    return alpha,bias,history,selected

def training_smoke(data,f,train,out):
    _,lags=indices(data.h);a,b=torch_parameters(True,len(lags),device=data.device,dtype=torch.float32)
    opt=torch.optim.Adam([a,b],lr=.01);batch0=batch(train,range(min(8,len(train))))
    target=rt.normalized_teacher_prob(batch0['distance'].reshape(-1,batch0['distance'].shape[-1]),batch0['mask'].reshape(-1,batch0['mask'].shape[-1]),.1).reshape_as(batch0['distance'])
    grads=[];losses=[]
    for step in range(2):
        opt.zero_grad();u,p=probability(batch0,a,b,f.support)
        if step==0:assert torch.equal(u,batch0['g']) and torch.equal(p,batch0['base_probability'])
        loss=rt.kl_loss_from_prob(target,p,batch0['mask']);assert torch.isfinite(loss);loss.backward()
        assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
        grads.append(float(b.grad.norm()));losses.append(float(loss));opt.step()
    assert grads[0]==0. and grads[1]>0.
    assert all(not p.requires_grad and p.grad is None for p in list(f.model.parameters())+list(f.heads.parameters()))
    torch.cuda.synchronize()
    dump(out/'TRAIN_SMOKE_GATE.json',dict(status='PASS',smoke_not_official_training=True,rows=len(batch0['g']),channels=data.c,
         candidates=batch0['g'].shape[-1],teacher_source='rt.normalized_teacher_prob',loss_source='rt.kl_loss_from_prob',losses=losses,
         bias_grad_norms=grads,alpha0_bit_exact=True,frozen_encoder_and_heads=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
         peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu=snapshot()))

def main(cell,support):
    out=BASE/(cell+'_'+support);out.mkdir(parents=True,exist_ok=True)
    lock=(BASE/'00_Audit/gpu_worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'COMPLETE.json').exists(),'Completed pilot preserved'
    gate=json.loads((BASE/'00_Audit'/('GPU_GATE_'+cell+'_'+support+'.json')).read_text())
    assert gate['status']=='PASS_FIRST_TAIL_NOT_FULL_POPULATION'
    before=start_gpu();register_owner('TRACK-SOFTLAG-HEAD-PILOT01');begin=time.perf_counter();data,f,identity=frozen(cell,support)
    assert gate['identity']['checkpoint']['checkpoint_sha256']==f.row['checkpoint_sha256']
    selections={s:sample_indices(len(data.datasets[s]),n) for s,n in [('train',64),('val',32),('test',32)]}
    train_past=np.concatenate([data.query('train',int(i),int(i)+1)[0].cpu().numpy() for i in selections['train']])
    scale=freeze_activity_scale(train_past)
    assert all(int(data.datasets['train'].starts[i])+2*data.h<=data.data_info['scaler_train_rows'] for i in selections['train'])
    proof=dict(identity=identity,runner_sha256=sha(HERE/'softlag_run.py'),protocol_sha256=sha(HERE/'protocol.json'),selected_rows={s:a.tolist() for s,a in selections.items()},activity_scale=scale.tolist())
    if (out/'PROVENANCE.json').exists():assert json.loads((out/'PROVENANCE.json').read_text())==proof
    else:dump(out/'PROVENANCE.json',proof)
    torch.cuda.reset_peak_memory_stats()
    train=cache_split(data,f,'train',selections['train'],out,scale,proof)
    val=cache_split(data,f,'val',selections['val'],out,scale,proof)
    training_smoke(data,f,train,out)
    arms=[];valid=[]
    for headwise in [False,True]:
        alpha,bias,history,selected=train_arm(data,f,train,val,out,headwise,proof)
        arms.append(('B2' if headwise else 'B1',alpha,bias,selected['epoch']));valid.extend(history)
    # No test metrics were used for arm/epoch selection.
    test=cache_split(data,f,'test',selections['test'],out,scale,proof)
    _,lag=indices(data.h);zero,bz=torch_parameters(True,len(lag),device=data.device,dtype=torch.float32)
    arms.insert(0,('B0',zero,bz,0));results=[];heads=[];preferences=[];raw={}
    for name,alpha,bias,epoch in arms:
        m,ids,diag=evaluation(data,test,alpha,bias,support);raw[name]=m
        atomic_npz(out/(name+'_test_queries.npz'),dict(metrics=m,selected_ids=ids,query_indices=selections['test'],query_starts=np.array([data.datasets['test'].starts[i] for i in selections['test']]),metric_names=np.array(['RetMSE','AggMSE','AggMAE','Recall','OracleRetMSE','OracleGap'])))
        results.append(dict(method=name,selected_epoch=epoch,queries=len(test),channels=data.c,retmse=float(m[...,0].mean()),agg_mse=float(m[...,1].mean()),agg_mae=float(m[...,2].mean()),recall=float(m[...,3].mean()),oracle_gap=float(m[...,5].mean())))
        heads.append(dict(method=name,mean_abs_score_correction=diag['mean_abs_score_correction'],mean_head_top10_overlap=diag['mean_head_top10_overlap']))
        for h,row in enumerate(bias.detach().cpu().numpy()):
            for j,value in enumerate(row):preferences.append(dict(method=name,head='shared' if len(bias)==1 else h+1,lag=int(lag[j]),bias=float(value),alpha=float(alpha.detach()),attention=diag['mean_lag_attention'][h][j]))
    for name in ['random','oracle']:
        m,ids,_=evaluation(data,test,zero,bz,support,method=name)
        results.append(dict(method=name,selected_epoch='',queries=len(test),channels=data.c,retmse=float(m[...,0].mean()),agg_mse=float(m[...,1].mean()),agg_mae=float(m[...,2].mean()),recall=float(m[...,3].mean()),oracle_gap=float(m[...,5].mean())))
    last=np.concatenate([((t['x'][:,:,-1,None]-t['y'])**2).mean(-1) for t in test]);lastmae=np.concatenate([np.abs(t['x'][:,:,-1,None]-t['y']).mean(-1) for t in test])
    results.append(dict(method='LastRepeat',selected_epoch='',queries=len(test),channels=data.c,retmse='',agg_mse=float(last.mean()),agg_mae=float(lastmae.mean()),recall='',oracle_gap=''))
    write_csv(out/'TEST_RESULTS.csv',results);write_csv(out/'VALIDATION_RESULTS.csv',valid);write_csv(out/'HEAD_DIAGNOSTICS.csv',heads);write_csv(out/'LEARNED_LAG_PREFERENCES.csv',preferences)
    torch.cuda.synchronize();improvements=dict(B1_vs_B0_fraction=float(1-raw['B1'][...,0].mean()/raw['B0'][...,0].mean()),B2_vs_B1_fraction=float(1-raw['B2'][...,0].mean()/raw['B1'][...,0].mean()),B2_vs_B0_query_channel_win=float((raw['B2'][...,0]<raw['B0'][...,0]).mean()))
    assert sha(f.row['checkpoint'])==identity['checkpoint']['checkpoint_sha256']
    resource=dict(wall_seconds=time.perf_counter()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_before=before,gpu_after=snapshot(),shared_gpu='possibly concurrent D1-D4; not isolated benchmark')
    dump(out/'RESOURCE_SUMMARY.json',resource)
    dump(out/'COMPONENT_GATE.json',dict(status='PILOT_TRAIN_TEST_COMPLETE_HEAD_DETAIL_REVIEW_PENDING',at=now(),improvements=improvements,resource=resource,source_checkpoint_unchanged=True,
      pending=['unique useful head candidate breakdown','learned head bias similarity table','cross-condition verdict and final Korean report'],full_dataset_evaluation=False,sampled_queries={'train':64,'val':32,'test':32},all_channels=True))
    state(out,'PILOT_TRAIN_TEST_COMPLETE_HEAD_DETAIL_REVIEW_PENDING');print('SOFTLAG RESULT',results,improvements,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cell');p.add_argument('support');a=p.parse_args()
    try:main(a.cell,a.support)
    except BaseException as e:
        out=BASE/(a.cell+'_'+a.support);state(out,'BLOCKED',error=repr(e),traceback=traceback.format_exc(),preserved=True);raise
