"""CPU-only read audit: exact legacy vs vector P100 IDs and tensor inputs."""
import sys,os,json,hashlib,time,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE)]
import runtime as rt
import torch,numpy as np
from vector_ops import pool_all,flat_inputs

def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

if __name__=='__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1)
    try:os.nice(10)
    except OSError:pass
    out=HERE/'candidate_budget_audit_20261030';out.mkdir(exist_ok=True)
    start=time.monotonic();data=rt.CellData('Solar_96','cpu')
    report=dict(status='RUNNING',started_at=rt.now(),device='cpu',gpu_initialized=False,
        cell='Solar_96',bank_count=len(data.starts),bank_start_first=int(data.starts[0]),bank_start_last=int(data.starts[-1]),
        history_bank_shape=list(data.sampler.memory_x.shape),future_bank_shape=list(data.sampler.memory_y.shape),
        channels=data.c,seq_len=data.h,pred_len=data.h,counts={s:len(d) for s,d in data.datasets.items()},splits={})
    meta=json.loads((data.pool_dir/'COMPLETE.json').read_text())
    old_prov=json.loads((rt.OUT/'runs/Solar_96/seed0/V2/provenance.json').read_text())
    new_prov=json.loads((rt.OUT/'runs_vectorized03/Solar_96/seed0/V2/provenance.json').read_text())
    for split,ds in data.datasets.items():
        h_old=hashlib.sha256();h_new=hashlib.sha256();elements=0;rows_seen=0
        for first in range(0,len(ds),32):
            starts=ds.starts[first:first+32]
            old=torch.stack([data.pool(split,starts,c) for c in range(data.c)])
            new=pool_all(data,split,starts)
            assert old.shape==new.shape==(137,len(starts),100)
            assert torch.equal(old,new),'P100 ID/order mismatch'
            canonical=new.permute(1,0,2).contiguous().numpy()
            rt.validate_pool(canonical,starts,data.starts,data.h,data.h,split)
            h_old.update(old.numpy().tobytes());h_new.update(new.numpy().tobytes())
            elements+=new.numel();rows_seen+=len(starts)
        filehash=rt.sha256_file(data.pool_dir/f'{split}.npy')
        assert filehash==meta['hashes'][split]==old_prov['pool_hashes'][split]==new_prov['pool_hashes'][split]
        assert h_old.hexdigest()==h_new.hexdigest()
        report['splits'][split]=dict(query_count=rows_seen,cache_shape=[len(ds),137,100],all_ID_elements=elements,
            ordered_stream_hash_old=h_old.hexdigest(),ordered_stream_hash_new=h_new.hexdigest(),ordered_stream_layout='query batches32 -> channel137 -> query within batch -> candidate100, int64',
            cache_file_sha256=filehash,previous_and_current_provenance_sha256_match=True,
            all_ID_and_order_exact=True,all_eligible=True,all_rows_unique100=True)
        print('FULL_IDS_PASS',split,json.dumps(report['splits'][split]),flush=True)
        rt.write_json(out/'AUDIT.json',report)
    # Actual first seed0 shuffled training batch, with unchanged private loader RNG.
    x,y,starts=next(iter(data.loader('train',32,0)));x=x.float();y=y.float();ids=pool_all(data,'train',starts)
    q,target,hist,values=flat_inputs(data,x,y,ids,0,137)
    oldhist=torch.cat([data.histories(ids[c],c).reshape(-1,data.h) for c in range(137)])
    oldvalues=torch.cat([data.values(ids[c],c) for c in range(137)])
    oldq=torch.cat([x[:,:,c] for c in range(137)]);oldtarget=torch.cat([y[:,:,c] for c in range(137)])
    pairs={'history':(oldhist,hist),'aligned_future_delta':(oldvalues,values),'query_history':(oldq,q),'query_target':(oldtarget,target)}
    report['actual_first_shuffled_batch']=dict(query_starts=starts.tolist(),batch_size=32,channels=137,candidates_per_query_channel=100,
        total_candidate_encodings=32*137*100,query_channel_units=32*137,ids_shape=list(ids.shape),
        old_single_channel_history_shape=[32,100,96,1],old_single_channel_future_shape=[32,100,96],tensors={})
    for name,(old,new) in pairs.items():
        assert torch.equal(old,new),name
        report['actual_first_shuffled_batch']['tensors'][name]=dict(shape=list(new.shape),bitwise_equal=True,old_sha256=tensor_hash(old),new_sha256=tensor_hash(new))
    bank_n=len(data.starts);qs=data.datasets['train'].starts
    left=np.maximum(0,qs-2*data.h+1);right=np.minimum(bank_n-1,qs+2*data.h-1)
    counts=bank_n-(right-left+1)
    sample=np.unique([0,data.h,2*data.h,bank_n//2,bank_n-1])
    exact=data.mask(qs[sample]).sum(1).numpy()
    assert np.array_equal(exact,counts[sample])
    report['full_bank_eligibility']=dict(train_valid_count_min=int(counts.min()),train_valid_count_max=int(counts.max()),
        policy='same RAFT offline train nonoverlap: candidate_end<=query_start OR candidate_start>=query_start+L+H',
        formula_all_rows=True,canonical_mask_sample_query_indices=sample.tolist(),canonical_mask_sample_counts=exact.tolist())
    rows=json.loads((rt.HERE/'configs/run_manifest.json').read_text());solar=[r for r in rows if r['dataset']=='Solar']
    report['solar_manifest']=[{k:r[k] for k in ['run_id','status','horizon','model','num_query_heads','n_train_queries','n_val_queries','n_test_queries','optimizer_steps_per_epoch','logical_batch_size','train_epochs','channels','output_path']} for r in solar]
    for row in solar:
        assert row['train_epochs']==10 and row['logical_batch_size']==32 and row['channels']==137
        assert row['optimizer_steps_per_epoch']==math.ceil(row['n_train_queries']/32)
    histnew=json.loads((rt.OUT/'runs_vectorized03/Solar_96/seed0/V2/epochs.json').read_text())
    histold=json.loads((rt.OUT/'runs/Solar_96/seed0/V2/epochs.json').read_text())
    report['completed_new_epochs']=[dict(epoch=e['epoch'],train_queries=e['train_queries'],
        epoch_batch_hash=e['batch_hash'],matches_saved_shared_hash=e['batch_hash']==json.loads((rt.OUT/'initialization/Solar_96/seed0'/f"epoch{e['epoch']:02d}_batch_hash.json").read_text())['sha256'],
        matches_old_partial_if_available=e['batch_hash']==histold[e['epoch']-1]['batch_hash'] if e['epoch']<=len(histold) else None,
        validation_query_count=e['validation']['query_count'],validation_channels=e['validation']['channels'],validation_units=e['validation']['unit_count']) for e in histnew]
    assert all(x['matches_saved_shared_hash'] and x['train_queries']==36601 for x in report['completed_new_epochs'])
    report['loss_weighting']=dict(old='sum_c mean_b KL(q_bc || p_bc) / C',new='mean_(c,b) KL(q_bc || p_bc); partial chunks weighted C_chunk/C',
        candidates_summed_inside_KL=100,optimizer_steps_per_epoch=1144,optimizer_steps_total_planned=11440,final_k=10,
        note='Train KL consumes all100; finalK10 is retrieval evaluation selection, not training-candidate reduction')
    assert not torch.cuda.is_initialized()
    report.update(status='PASS',finished_at=rt.now(),wall_seconds=time.monotonic()-start,
        original_initial_encoder_hash_equal=old_prov['initial_model_hash']==new_prov['initial_model_hash'],head_initial_hash_equal=old_prov['head_hash']==new_prov['head_hash'])
    rt.write_json(out/'AUDIT.json',report);print('AUDIT_COMPLETE',json.dumps({k:report[k] for k in ['status','finished_at','wall_seconds','bank_count','counts','full_bank_eligibility']}),flush=True)
