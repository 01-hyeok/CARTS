"""Real 137-channel Full data/metric smoke; canonical loops are reference-only."""
import sys,time,gc,hashlib,traceback
import gate
import entry
run=entry.run;torch=run.torch;np=run.np

@torch.no_grad()
def check_block(data,split,start,end,calendar,ridge,chunk):
    run.sync();begin=time.perf_counter()
    block=run.prepare_block(data,split,start,end,'FULL',calendar,ridge,chunk)
    run.sync();production_seconds=time.perf_counter()-begin
    B=end-start;C=data.c;N=len(data.starts);H=data.h
    assert block['d'].shape==(B,137,N) and block['ids'].shape==(B,137,N)
    assert torch.equal(block['ids'],torch.arange(N,device=data.device).expand(B,C,N))
    x,y=block['x'],block['y'];refd=torch.empty_like(block['d']);refs=torch.empty(B,C,N,4,device=data.device)
    from diagnostic_core import paa16,stable_order
    import torch.nn.functional as F
    def cos(q,k):return (F.normalize(q,dim=-1)[:,None]*F.normalize(k,dim=-1)).sum(-1)
    # The historical per-channel formulas are used ONLY as independent gate evidence.
    for k in range(0,N,chunk):
        ids=block['ids'][...,k:k+chunk];values=data.values_all(ids);hist=data.histories_all(ids)
        assert values.shape[:3]==(B,C,min(chunk,N-k))
        for c in range(C):
            q=x[:,c].contiguous();v=hist[:,c].contiguous();qz=q-q.mean(-1,keepdim=True);vz=v-v.mean(-1,keepdim=True)
            qd=q-q[:,-1:];vd=v-v[...,-1:]
            refd[:,c,k:k+chunk]=run.rt.pooled_future_mse(values[:,c].contiguous(),x[:,c,-1].contiguous(),y[:,c].contiguous())
            refs[:,c,k:k+chunk]=torch.stack([cos(q,v),cos(qd,vd),cos(qz,vz),cos(paa16(qd),paa16(vd))],-1)
    assert torch.equal(refd,block['d']),'Canonical future distance differs'
    mask=block['mask'].reshape(B*C,N)
    names=run.BASE_NAMES[1:]+['IndividualMSE_Oracle'];digest=hashlib.sha256()
    for j,name in enumerate(names):
        actual=-block['d'] if j==4 else block['scores'][name]
        reference=-refd if j==4 else refs[...,j]
        assert torch.isfinite(actual).all(),name
        got=stable_order(actual.reshape(B*C,N),mask)[...,:10]
        expected=stable_order(reference.reshape(B*C,N),mask)[...,:10]
        assert torch.equal(got,expected),(name,split,start,'ordered Top10 mismatch')
        assert torch.all(got.sort(-1)[0][...,1:]!=got.sort(-1)[0][...,:-1])
        assert mask.gather(1,got).all()
        digest.update(got.cpu().numpy().tobytes())
    return dict(split=split,start=start,end=end,query_shape=[B,C,H],candidate_chunk_shape=[B,C,chunk,H],score_shape=[B,C,N],
                production_block_seconds=production_seconds,canonical_ordered_id_sha256=digest.hexdigest(),ordered_mismatches=0,
                canonical_distance_bitwise_equal=True,reference_only_channel_loop=True,production_serial_fallback=False)

def main(cell):
    gate.source_check();assert cell in gate.SOLAR
    assert run.GPU not in run.subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    run.rt.gpu_guard();out=gate.AUDIT/cell;out.mkdir(parents=True,exist_ok=True)
    if (out/'GPU_PREFLIGHT.json').exists():gate.preflight(cell);return
    begin=time.perf_counter();torch.cuda.reset_peak_memory_stats();data=run.Data(cell)
    counts={s:len(data.datasets[s]) for s in ('train','val','test')}
    gate.validate_shape(cell,data.c,len(data.starts),counts)
    inventory=run.inventory(cell,'FULL');assert len(inventory)==2 and all(r['status']=='MISSING' for r in inventory)
    calendar=run.Calendar(data);assert calendar.values is None
    ridge=run.fit_probe(data,'FULL',calendar,out)
    rows=[]
    for split,a,b in [('val',0,4),('test',counts['test']-1,counts['test'])]:
        rows.append(check_block(data,split,a,b,calendar,ridge,128));gc.collect();torch.cuda.empty_cache()
        print('SOLAR_PREFLIGHT_CANONICAL',cell,split,a,b,'137 channels; all candidates',flush=True)
    # Actual end-to-end metric/ranking/feature path at production B4/full support.
    smoke=run.evaluate_micro(data,'val',0,4,'FULL',calendar,ridge,{},None,128)
    assert smoke['metrics'].shape[:2]==(4,137) and smoke['selected_ids'].shape[:2]==(4,137)
    assert smoke['selected_ids'].shape[-1]==10 and smoke['oracle_ids'].shape==(4,137,10)
    assert len(smoke['query_start'])==4 and np.isfinite(smoke['metrics'][...,run.METRICS.index('retmse')]).all()
    done=dict(status='PASS',cell=cell,at=run.now(),all_channels_single_batch=True,channels=137,full_candidate_count=len(data.starts),query_counts=counts,
              pool_hashes=data.pool_hashes,data=data.data_info,canonical_ordered_id_mismatches=0,rows=rows,actual_metric_smoke_shape=list(smoke['metrics'].shape),
              missing_full_models=['V0','V5'],learned_model_reproduction_claimed=False,model_softmax_gate='NOT_APPLICABLE_MODELS_MISSING',
              training_performed=False,probe='existing train-only 256-query x137-channel diagnostic Ridge, not a retriever',
              total_seconds=time.perf_counter()-begin,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
              source_hashes=gate.source_check()['sources'],gpu_uuid=run.GPU,seed=0,production_candidate_chunk=128,production_query_microbatch=4)
    run.dump(out/'GPU_PREFLIGHT.json',done);print('SOLAR_PREFLIGHT_PASS',cell,done['total_seconds'],done['peak_gpu_allocated_bytes'],flush=True)

if __name__=='__main__':
    try:main(sys.argv[1])
    except BaseException as e:
        issue=dict(status='FAILED',at=run.now(),cell=sys.argv[1],phase='SOLAR_GPU_PREFLIGHT',error=repr(e),traceback=traceback.format_exc())
        run.dump(gate.AUDIT/sys.argv[1]/'PREFLIGHT_FAILED.json',issue);run.dump(gate.SOURCE/'00_Audit/IMMEDIATE_ISSUE.json',issue);raise
