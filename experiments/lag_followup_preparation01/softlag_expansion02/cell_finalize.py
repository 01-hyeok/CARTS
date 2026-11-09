"""Exact selected-test replay, head diagnostics and immutable cell gate."""
import csv,json,pickle,hashlib,time
from pathlib import Path
import numpy as np
import torch
from contracts import *
from softlag_finalize import load_rows,head_detail,bootstrap
from softlag_run import batch,probability,evaluation,write_csv,atomic_npz
from softlag import torch_parameters
from past_lag import indices
from server_bridge import topk,snapshot

def complete_cell(data,f,out,identity,before):
    if checked_gate(out/'FINAL_GATE.json','PASS_EXPANSION_CELL_COMPLETE'):return
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    component=json.loads((out/'COMPONENT_GATE.json').read_text())
    assert component['status']=='PILOT_TRAIN_TEST_COMPLETE_HEAD_DETAIL_REVIEW_PENDING'
    proof=json.loads((out/'PROVENANCE.json').read_text());assert proof['identity']==identity
    val=load_rows(out,'val');test=load_rows(out,'test');_,lags=indices(data.h)
    assert len(val)==len(test)==32 and all(r['g'].shape[1]==data.c for r in val+test)
    raw={};heads=[];biasrows=[];pairs=[];states={};audit=[]
    for arm in ['B0','B1','B2']:
        alpha,bias=torch_parameters(arm!='B1',len(lags),device=data.device,dtype=torch.float32)
        if arm!='B0':
            hist=json.loads((out/arm/'VALIDATION_HISTORY.json').read_text());choice=json.loads((out/arm/'SELECTION.json').read_text())
            assert [r['epoch'] for r in hist]==list(range(6)) and not choice['test_used']
            assert min(hist,key=lambda r:(r['val_retmse'],r['epoch']))==choice['selected']
            assert sha(choice['checkpoint'])==choice['sha256'];states[arm]=[]
            for e in range(6):
                ck=torch.load(out/arm/('epoch%02d.pth'%e),map_location=data.device)
                assert ck['epoch']==e and ck['identity']==proof and len(ck['history'])==e+1
                if e==0:assert float(ck['alpha'])==0 and not bool(ck['bias'].any())
                else:
                    assert all(int(v['step'])==e*8 for v in ck['optimizer']['state'].values())
                    assert ck['first_bias_grad']==0 and ck['later_bias_grad_max']>0
                states[arm].append(hashlib.sha256(pickle.dumps(ck['batch_rng'])).hexdigest())
            ck=torch.load(choice['checkpoint'],map_location=data.device)
            alpha.data.copy_(ck['alpha']);bias.data.copy_(ck['bias'])
            audit.append(dict(arm=arm,optimizer_steps=40,epochs=5,selected_epoch=choice['selected']['epoch'],selected_sha256=choice['sha256'],channels=data.c,train=64,val=32,test=32,validation_only=True))
        m,ids,_=evaluation(data,test,alpha,bias,f.support)
        with np.load(out/(arm+'_test_queries.npz')) as z:
            assert np.array_equal(ids,z['selected_ids']) and np.array_equal(m,z['metrics']),('Selected test replay mismatch',arm)
            starts=z['query_starts'].copy()
        raw[arm]=m
        vh,_=head_detail(data,val,alpha,bias,f.support);th,un=head_detail(data,test,alpha,bias,f.support)
        best=int(vh[...,0].mean((0,1)).argmin())
        for h in range(5):
            avg=th[:,:,h].mean((0,1));uu=un[:,:,h].mean((0,1))
            heads.append(dict(arm=arm,head=h+1,validation_selected=h==best,validation_retmse=float(vh[:,:,h,0].mean()),test_retmse=float(avg[0]),test_agg_mse=float(avg[1]),test_agg_mae=float(avg[2]),test_recall=float(avg[3]),exclusive_count=float(uu[0]),exclusive_oracle_count=float(uu[1]),exclusive_useful_1pct=float(uu[2]),exclusive_useful_5pct=float(uu[3]),exclusive_useful_10pct=float(uu[4]),queries=32,channels=data.c))
        picks=[]
        with torch.no_grad():
            for a in range(0,len(test),8):
                b=batch(test,range(a,min(a+8,len(test))));u,p=probability(b,alpha,bias,f.support)
                picks.append(topk(u.masked_fill(~b['mask'][:,:,None],-float('inf')),10).cpu().numpy())
        hids=np.concatenate(picks)
        atomic_npz(out/(arm+'_test_head_ids.npz'),dict(local_ids=hids,query_indices=np.concatenate([r['query_index'] for r in test])))
        for h in range(5):
            for k in range(h+1,5):pairs.append(dict(arm=arm,head_a=h+1,head_b=k+1,top10_overlap=float((hids[:,:,h,:,None]==hids[:,:,k,None,:]).any(-1).mean()),queries=32,channels=data.c))
        bb=bias.detach().cpu().numpy()
        for h in range(len(bb)):
            for k in range(h+1,len(bb)):
                den=float(np.linalg.norm(bb[h])*np.linalg.norm(bb[k]))
                biasrows.append(dict(arm=arm,head_a=h+1,head_b=k+1,cosine=float(np.dot(bb[h],bb[k])/den) if den else '',l2_distance=float(np.linalg.norm(bb[h]-bb[k])),zero_norm=den==0))
    assert states['B1']==states['B2'],'Batch order differs between arms'
    cis=[]
    for a,b in [('B1','B0'),('B2','B0'),('B2','B1')]:
        for mult in [1,2,4]:cis.append(dict(comparison=a+'-'+b,block_rule=str(mult)+'H',**bootstrap(raw[a][...,0]-raw[b][...,0],starts,data.h*mult)))
    for name,rows in [('HEAD_USEFULNESS.csv',heads),('HEAD_PAIR_OVERLAP.csv',pairs),('BIAS_SIMILARITY.csv',biasrows),('PAIRED_TIME_BLOCK_CI.csv',cis),('PROTOCOL_AUDIT.csv',audit)]:write_csv(out/name,rows)
    assert sha(f.row['checkpoint'])==identity['checkpoint']['checkpoint_sha256']
    torch.cuda.synchronize()
    dump(out/'FINAL_REPLAY_RESOURCE.json',dict(wall_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_before=before,gpu_after=snapshot(),shared_gpu=True))
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and 'cache' not in p.parts and p.name not in ['FINAL_GATE.json','CURRENT.json']}
    dump(out/'FINAL_GATE.json',dict(status='PASS_EXPANSION_CELL_COMPLETE',at=now(),cell=data.cell,support=f.support,seed=0,channels=data.c,samples={'train':64,'val':32,'test':32},epochs_saved=list(range(6)),optimizer_steps_per_arm=40,selected_test_ids_metrics_exact=True,source_checkpoint_unchanged=True,batch_rng_identical=True,files=files,unmeasured=['separate training GPU kernel time','separate validation wall time'],full_population=False))
    print('SOFTLAG EXPANSION CELL COMPLETE',data.cell,f.support,flush=True)
